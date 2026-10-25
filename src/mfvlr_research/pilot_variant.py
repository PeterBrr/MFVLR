"""Explicit experimental decoder/language-gradient variant; does not redefine A0."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from mfvlr.integration import (MFVLRIntegration, ImageOnlyMFVLR, LanguagePredictions, pairing)
from mfvlr.objectives import SixObjectives, LossValues, floating, indices, scalar, unit_weight_sum
from mfvlr.vision_decoder import VisionDecoder


class ResizeConvStage(nn.Module):
    def __init__(self, incoming, outgoing, mode):
        super().__init__()
        self.mode = mode
        self.projection = nn.Conv2d(incoming, outgoing, 1)
        self.conv1 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)

    def forward(self, x):
        # Pointwise projection commutes with nearest/bilinear resize; do it at
        # low resolution to avoid a large high-channel intermediate on 6GB GPUs.
        x = self.projection(x)
        options = {'align_corners': False} if self.mode == 'bilinear' else {}
        x = F.interpolate(x, scale_factor=2, mode=self.mode, **options)
        return F.relu(self.conv2(F.relu(self.conv1(x))))


class ResizeConvTrunk(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.stages = nn.ModuleList(ResizeConvStage(a, b, mode) for a, b in
                                   ((1024, 512), (512, 256), (256, 128), (128, 64)))

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


class ResizeConvDecoder(VisionDecoder):
    """Retain validated VD I/O and distinct heads; replace the trunk explicitly."""
    def __init__(self, mode='nearest', seed=0):
        nn.Module.__init__(self)
        if mode not in ('nearest', 'bilinear'):
            raise ValueError('Resize mode must be nearest or bilinear')
        self.skip_policy = 'bottleneck_only'
        with torch.random.fork_rng(devices=[]):
            self.trunk = ResizeConvTrunk(mode)
            self.appearance_head = nn.Conv2d(64, 3, 1)
            self.mask_head = nn.Conv2d(64, 2, 1)
            generator = torch.Generator(device='cpu').manual_seed(seed)
            for stage in self.trunk.stages:
                nn.init.xavier_uniform_(stage.projection.weight, generator=generator)
                nn.init.zeros_(stage.projection.bias)
                for conv in (stage.conv1, stage.conv2):
                    nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu', generator=generator)
                    nn.init.zeros_(conv.bias)
            for head in (self.appearance_head, self.mask_head):
                nn.init.xavier_uniform_(head.weight, generator=generator)
                nn.init.zeros_(head.bias)


def language_visual_feature(visual, stop_all):
    return visual.detach() if stop_all else visual


def weighted_mask_loss(logits, masks, class_weights, loss_weight=1.):
    # Same weighted-mean CE, avoiding CUDA nll_loss2d's nondeterministic
    # weighted reduction. Matches the log-softmax/gather path used by A0 FL.
    log_probabilities = logits.log_softmax(dim=1)
    nll = -log_probabilities.gather(1, masks[:, None])[:, 0]
    pixel_weights = class_weights[masks]
    value = (nll * pixel_weights).sum() / pixel_weights.sum()
    return scalar(value * loss_weight, 'weighted_FL')


class VariantObjectives(SixObjectives):
    def __init__(self, settings):
        super().__init__()
        self.stop_all = settings['language_visual_gradients'] == 'stop_all'
        self.mask_loss_weight = settings['mask_loss_weight']
        self.register_buffer('mask_class_weights', torch.tensor(
            [settings['mask_background_weight'], settings['mask_foreground_weight']], dtype=torch.float32))

    def forward(self, **kwargs):
        kwargs['visual'] = language_visual_feature(kwargs['visual'], self.stop_all)
        # Reuse all A0 validation and five other objectives unchanged.
        baseline = super().forward(**kwargs)
        fl = weighted_mask_loss(kwargs['mask_logits'], kwargs['masks'],
                                self.mask_class_weights, self.mask_loss_weight)
        total = unit_weight_sum(fd=baseline.fd, lr=baseline.lr, cmc=baseline.cmc,
                               fl=fl, ar=baseline.ar, kl=baseline.kl)
        return LossValues(baseline.fd, baseline.lr, baseline.cmc, fl, baseline.ar, baseline.kl, total)


class ResizeConvMFVLR(MFVLRIntegration):
    def __init__(self, settings):
        super().__init__()
        self.stop_all = settings['language_visual_gradients'] == 'stop_all'
        self.decoder = ResizeConvDecoder(settings['resize_mode'])
        self.objectives = VariantObjectives(settings)

    def with_language(self, image, token_ids, *, sample_ids):
        floating(image, 'image', (3, 224, 224))
        ordered = pairing(sample_ids, image.shape[0])
        indices(token_ids, image, 'token_ids', (image.shape[0], 308), 49408)
        predictions = self.visual_only(image)
        iv = language_visual_feature(predictions.visual.fused_global, self.stop_all)
        language = self.flt(token_ids, iv, return_logits=True)
        adapted = self.heads.align_language(iv)
        return LanguagePredictions(image, ordered, token_ids, predictions, language, adapted)


class ResizeConvImageOnly(ImageOnlyMFVLR):
    def __init__(self, settings):
        super().__init__()
        self.decoder = ResizeConvDecoder(settings['resize_mode'])


def variant_settings(config):
    settings = config.get('model')
    if settings is None:
        return None
    expected = {'variant', 'resize_mode', 'language_visual_gradients', 'mask_background_weight',
                'mask_foreground_weight', 'mask_loss_weight'}
    if not isinstance(settings, dict) or set(settings) != expected:
        raise ValueError('Experimental model settings must explicitly specify all variant fields')
    if settings['variant'] != 'resize_conv_v1':
        raise ValueError('Unknown experimental model variant')
    if settings['resize_mode'] not in ('nearest', 'bilinear'):
        raise ValueError('Unsupported resize mode')
    if settings['language_visual_gradients'] not in ('stop_all', 'coupled'):
        raise ValueError('Unknown language visual gradient policy')
    for name in ('mask_background_weight', 'mask_foreground_weight', 'mask_loss_weight'):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be a finite positive number')
    return settings


def training_model(config):
    settings = variant_settings(config)
    if settings is None:
        from mfvlr.integration import MFVLRIntegration as Baseline
        return Baseline()
    return ResizeConvMFVLR(settings)


def visual_model(config):
    settings = variant_settings(config)
    if settings is None:
        return ImageOnlyMFVLR()
    return ResizeConvImageOnly(settings)
