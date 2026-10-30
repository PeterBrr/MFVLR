"""Opt-in skip decoder, residual stop-gradient and manual objective weights."""
import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from mfvlr.integration import MFVLRIntegration, ImageOnlyMFVLR, VisualPredictions, pairing
from mfvlr.mve import MVE, AppearanceEncoding
from mfvlr.vision_decoder import VisionDecoderOutput
from mfvlr.heads import DetectionHeads
from mfvlr.objectives import (LossValues, floating, indices,
                             forgery_detection_loss, appearance_reconstruction_loss,
                             knowledge_distillation_loss,
                             validate_mask_semantics, scalar)
from .pilot_variant import (weighted_mask_loss, language_visual_feature, ResizeConvDecoder,
                            activate_reconstruction)
from .pilot_language import (NormalizedCrossModalContrastiveLoss,
                             nonpad_language_reconstruction_loss)

LOSS_NAMES = ('fd', 'lr', 'cmc', 'fl', 'ar', 'kl')


def validate_loss_weights(weights):
    if not isinstance(weights, dict) or set(weights) != set(LOSS_NAMES):
        raise ValueError('loss_weights must specify fd, lr, cmc, fl, ar and kl')
    for name, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid nonnegative loss weight: {name}')
    if not any(weights.values()):
        raise ValueError('At least one loss weight must be positive')
    return dict(weights)


def weighted_objective_total(values, weights):
    active = [values[name] * weights[name] for name in LOSS_NAMES if weights[name] > 0]
    return scalar(sum(active), 'manual_total')


class SkipResizeStage(nn.Module):
    def __init__(self, incoming, outgoing, mode):
        super().__init__()
        self.mode = mode
        self.projection = nn.Conv2d(incoming, outgoing, 1)
        self.conv1 = nn.Conv2d(2 * outgoing, outgoing, 3, padding=1)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)

    def forward(self, state, skip):
        state = self.projection(state)
        options = {'align_corners': False} if self.mode == 'bilinear' else {}
        state = F.interpolate(state, scale_factor=2, mode=self.mode, **options)
        if skip.shape != state.shape or skip.dtype != state.dtype or skip.device != state.device:
            raise ValueError('Skip must match channels, size, dtype and device')
        return F.relu(self.conv2(F.relu(self.conv1(torch.cat((state, skip), dim=1)))))


class SkipResizeTrunk(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.stages = nn.ModuleList(SkipResizeStage(a, b, mode) for a, b in
                                   ((1024, 512), (512, 256), (256, 128), (128, 64)))

    def forward(self, local, skips):
        if not isinstance(skips, (list, tuple)) or len(skips) != 4:
            raise ValueError('Exactly four encoder skips are required')
        state = local
        for stage, skip in zip(self.stages, reversed(skips)):
            state = stage(state, skip)
        return state


class SkipResizeDecoder(nn.Module):
    def __init__(self, mode='bilinear', seed=0, reconstruction_activation='identity'):
        super().__init__()
        if reconstruction_activation not in ('identity', 'sigmoid'):
            raise ValueError('Unsupported reconstruction_activation')
        self.reconstruction_activation = reconstruction_activation
        if mode not in ('nearest', 'bilinear'):
            raise ValueError('Resize mode must be nearest or bilinear')
        self.skip_policy = 'appearance_concat'
        with torch.random.fork_rng(devices=[]):
            self.trunk = SkipResizeTrunk(mode)
            self.appearance_head = nn.Conv2d(64, 3, 1)
            self.mask_head = nn.Conv2d(64, 2, 1)
            generator = torch.Generator().manual_seed(seed)
            for stage in self.trunk.stages:
                nn.init.xavier_uniform_(stage.projection.weight, generator=generator)
                nn.init.zeros_(stage.projection.bias)
                for conv in (stage.conv1, stage.conv2):
                    nn.init.kaiming_normal_(conv.weight, nonlinearity='relu', generator=generator)
                    nn.init.zeros_(conv.bias)
            for head in (self.appearance_head, self.mask_head):
                nn.init.xavier_uniform_(head.weight, generator=generator)
                nn.init.zeros_(head.bias)

    def forward(self, local, *, skips):
        floating(local, 'decoder.local', (1024, 14, 14))
        features = self.trunk(local, skips)
        reconstruction, masks = self.appearance_head(features), self.mask_head(features)
        if not all(torch.isfinite(x).all() for x in (features, reconstruction, masks)):
            raise FloatingPointError('Nonfinite skip decoder output')
        reconstruction = activate_reconstruction(reconstruction, self.reconstruction_activation)
        return VisionDecoderOutput(features, reconstruction, masks)

    def decode_appearance(self, appearance):
        if not isinstance(appearance, AppearanceEncoding):
            raise TypeError('Expected AppearanceEncoding')
        return self(appearance.local, skips=appearance.skips)


def skip_visual_route(owner, detector, image):
    floating(image, 'image', (3, 224, 224))
    if image.min().item() < 0 or image.max().item() > 1:
        raise ValueError('Image must be in [0,1]')
    appearance = owner.mve.encode_appearance(image)
    decoded = owner.decoder.decode_appearance(appearance)
    # Only the residual/detection path is stopped; AR/FL retain their graphs.
    residual_input = decoded.reconstruction.detach() if owner.detach_reconstruction else decoded.reconstruction
    visual = owner.mve(image, residual_input, appearance_encoding=appearance)
    return VisualPredictions(visual, decoded, detector(visual.fused_global[:, 0, :]))


class ManualObjectives(nn.Module):
    def __init__(self, settings, weights):
        super().__init__()
        self.weights = validate_loss_weights(weights)
        self.cmc = NormalizedCrossModalContrastiveLoss()
        self.register_buffer('mask_class_weights', torch.tensor(
            [settings['mask_background_weight'], settings['mask_foreground_weight']], dtype=torch.float32))

    def forward(self, output, targets):
        image, predictions = output.image, output.predictions
        indices(targets.labels, image, 'labels', (image.shape[0],), 2)
        indices(targets.masks, image, 'masks', (image.shape[0], 224, 224), 2)
        validate_mask_semantics(targets.labels, targets.masks, targets.sample_kinds)
        w = self.weights
        values = {name: image.new_zeros(()) for name in LOSS_NAMES}
        if w['fd']:
            values['fd'] = forgery_detection_loss(predictions.detection_logits, targets.labels)
        if w['ar']:
            values['ar'] = appearance_reconstruction_loss(image, predictions.decoded.reconstruction)
        if w['fl']:
            values['fl'] = weighted_mask_loss(predictions.decoded.mask_logits, targets.masks, self.mask_class_weights)
        if w['lr']:
            values['lr'] = nonpad_language_reconstruction_loss(output.language.logits, targets.token_ids)
        if w['cmc']:
            visual = language_visual_feature(predictions.visual.fused_global, output.stop_language_visual)
            values['cmc'] = self.cmc(visual, output.language.global_language)
        if w['kl']:
            values['kl'] = knowledge_distillation_loss(output.language.global_language, output.predicted_language)
        return LossValues(**values, total=weighted_objective_total(values, w))


class SkipMFVLR(MFVLRIntegration):
    def __init__(self, settings, weights):
        nn.Module.__init__(self)
        self.detach_reconstruction = settings['residual_gradient'] == 'detach_reconstruction'
        self.stop_all = settings['language_visual_gradients'] == 'stop_all'
        self.mve = MVE(seed=0)
        decoder_class = SkipResizeDecoder if settings['skip_connections'] == 'concat' else ResizeConvDecoder
        self.decoder = decoder_class(settings['resize_mode'],
            reconstruction_activation=settings.get('reconstruction_activation', 'identity'))
        self.heads = DetectionHeads(seed=0)
        self.objectives = ManualObjectives(settings, weights)
        self.has_language = any(weights[name] > 0 for name in ('lr', 'cmc', 'kl'))
        self.flt = None
        if self.has_language:
            from .pilot_language import EosFineGrainedLanguageTransformer
            options = {'vim_residual': settings['vim_residual']} if 'vim_residual' in settings else {}
            if 'language_normalization' in settings:
                options['language_normalization'] = settings['language_normalization']
            self.flt = EosFineGrainedLanguageTransformer(seed=0, **options)

    def visual_only(self, image):
        return skip_visual_route(self, self.heads.detector, image)

    def with_language(self, image, token_ids, *, sample_ids):
        ordered = pairing(sample_ids, image.shape[0])
        indices(token_ids, image, 'token_ids', (image.shape[0], 308), 49408)
        predictions = self.visual_only(image)
        language = adapted = None
        if self.has_language:
            visual = language_visual_feature(predictions.visual.fused_global, self.stop_all)
            if self.objectives.weights['lr']:
                language = self.flt(token_ids, visual, return_logits=True)
            else:
                language = self.flt.encode(token_ids, visual)
            if self.objectives.weights['kl']:
                adapted = self.heads.align_language(visual)
        return SimpleNamespace(image=image, sample_ids=ordered, token_ids=token_ids,
                               predictions=predictions, language=language, predicted_language=adapted,
                               stop_language_visual=self.stop_all)

    def compute_objectives(self, output, targets):
        if pairing(targets.sample_ids, output.image.shape[0]) != output.sample_ids:
            raise ValueError('Target ordering differs from predictions')
        if not torch.equal(targets.token_ids, output.token_ids):
            raise ValueError('Target token IDs differ from input token IDs')
        return self.objectives(output, targets)


class SkipImageOnly(ImageOnlyMFVLR):
    def __init__(self, settings):
        super().__init__()
        self.detach_reconstruction = settings['residual_gradient'] == 'detach_reconstruction'
        decoder_class = SkipResizeDecoder if settings['skip_connections'] == 'concat' else ResizeConvDecoder
        self.decoder = decoder_class(settings['resize_mode'],
            reconstruction_activation=settings.get('reconstruction_activation', 'identity'))

    def forward(self, image):
        return skip_visual_route(self, self.detector, image)
