"""Single config-driven trainer/evaluator for new ablations; historical runners stay frozen.

Seed, batch policy, learning rates and time budget are arguments, so a change of
recipe never needs a patched runner. Residual-to-mask fusion is an experimental
engineering choice, not part of the published MFVLR decoder.
"""
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import default_collate

from types import SimpleNamespace

from mfvlr.flt import FLTOutput
from mfvlr.heads import DetectionHeads
from mfvlr.integration import SuppliedTargets, VisualPredictions, pairing
from mfvlr.language_encoder import finite, validate_hidden
from mfvlr.mve import MVE
from mfvlr.objectives import LossValues, appearance_reconstruction_loss, forgery_detection_loss, indices
from mfvlr.vim import VIMTrace, VisionInjectionDelta
from mfvlr.vision_decoder import VisionDecoderOutput
from .pilot_language import EosFineGrainedLanguageTransformer
from .pilot_runtime import atomic_save, check_finite_tensors, metrics
from .pilot_skip import SkipMFVLR, SkipResizeDecoder
from .pilot_variant import ResizeConvDecoder, language_visual_feature, variant_settings, weighted_mask_loss
from .pretrained_encoder import ENCODERS, install_encoder
from .text_patch import TEXT_PATCH_MODES, TextPatchHead, text_patch_loss

SCHEMA = 'mfvlr-experiment-v1'
CONFIG_KEYS = {'schema', 'name', 'inputs', 'train_limit', 'val_limit', 'model', 'loss_weights',
               'optimizer', 'scheduler', 'training'}
TRAINING_KEYS = {'epochs', 'effective_batch', 'microbatch', 'checkpoint_every', 'threads',
                 'gradient_checkpointing', 'gpu_memory_fraction'}
FINAL_EPOCHS = 3


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _rate(value, name, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or (value < 0 if zero else value <= 0)):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')


def validate_config(config):
    """Unknown or missing keys fail, so a mistyped option can never be ignored."""
    if not isinstance(config, dict) or set(config) - {'train_file'} != CONFIG_KEYS:
        raise ValueError(f'Experiment config must specify exactly {sorted(CONFIG_KEYS)}')
    train_file = config.get('train_file', 'train.npz')
    if type(train_file) is not str or not re.fullmatch(r'train\w*\.npz', train_file):
        raise ValueError('train_file must name a train*.npz file in inputs')
    if config['schema'] != SCHEMA or type(config['name']) is not str or not config['name']:
        raise ValueError('Unsupported experiment schema or empty name')
    settings = dict(config['model'])
    if settings.pop('mask_residual', None) not in ('none', 'concat'):
        raise ValueError('model.mask_residual must be none or concat')
    _rate(settings.pop('mask_dice_weight', 0.), 'model.mask_dice_weight', zero=True)
    encoder = settings.pop('encoder', 'scratch')
    if encoder not in ENCODERS:
        raise ValueError(f'model.encoder must be one of {sorted(ENCODERS)}')
    vim_tokens = settings.pop('vim_tokens', 'cls')
    if vim_tokens not in ('cls', 'all'):
        raise ValueError('model.vim_tokens must be cls or all')
    if vim_tokens == 'all' and (settings.get('vim_residual') != 'single' or settings.get('language_normalization') != 'pre_norm'
                                or not any(config['loss_weights'][name] > 0 for name in ('lr', 'cmc', 'kl'))):
        raise ValueError('vim_tokens=all needs language losses with single residual and pre_norm')
    prompt = settings.pop('language_prompt', 'label')
    encoder_input = settings.pop('language_encoder_input', 'prompt')
    if prompt not in ('label', 'region') or encoder_input not in ('prompt', 'neutral'):
        raise ValueError('model.language_prompt must be label or region; model.language_encoder_input prompt or neutral')
    if (prompt, encoder_input) != ('label', 'prompt') and not language_enabled(config):
        raise ValueError('language_prompt and language_encoder_input need language losses')
    text_patch = settings.pop('text_patch', 'none')
    text_patch_weight = settings.pop('text_patch_weight', 0.)
    if text_patch not in TEXT_PATCH_MODES:
        raise ValueError(f'model.text_patch must be one of {TEXT_PATCH_MODES}')
    _rate(text_patch_weight, 'model.text_patch_weight', zero=True)
    if (text_patch == 'none') != (text_patch_weight == 0):
        raise ValueError('model.text_patch needs a positive text_patch_weight, and only then')
    if settings.get('variant') != 'skip_resize_conv_v2':
        raise ValueError('Experiments use the explicit skip_resize_conv_v2 variant')
    variant_settings({'model': settings, 'loss_weights': config['loss_weights']})
    optimizer = config['optimizer']
    if not isinstance(optimizer, dict) or set(optimizer) - {'backbone_lr'} != {'image_lr', 'language_lr', 'weight_decay'}:
        raise ValueError('optimizer must specify image_lr, language_lr and weight_decay')
    if 'backbone_lr' in optimizer:
        if encoder == 'scratch':
            raise ValueError('optimizer.backbone_lr needs a pretrained model.encoder')
        _rate(optimizer['backbone_lr'], 'backbone_lr')
    _rate(optimizer['image_lr'], 'image_lr')
    _rate(optimizer['language_lr'], 'language_lr')
    _rate(optimizer['weight_decay'], 'weight_decay', zero=True)
    schedule = config['scheduler']
    if schedule is not None:
        if not isinstance(schedule, dict) or set(schedule) != {'step_size', 'gamma'}:
            raise ValueError('scheduler must be null or specify step_size and gamma')
        _positive_int(schedule['step_size'], 'scheduler.step_size')
        _rate(schedule['gamma'], 'scheduler.gamma')
    training = config['training']
    if not isinstance(training, dict) or set(training) - {'augmentation'} != TRAINING_KEYS:
        raise ValueError(f'training must specify exactly {sorted(TRAINING_KEYS)}')
    if training.get('augmentation', 'none') not in ('none', 'flip', 'deepfakebench'):
        raise ValueError('training.augmentation must be none, flip or deepfakebench')
    for name in ('epochs', 'effective_batch', 'microbatch', 'checkpoint_every', 'threads'):
        _positive_int(training[name], f'training.{name}')
    if type(training['gradient_checkpointing']) is not bool:
        raise ValueError('training.gradient_checkpointing must be boolean')
    fraction = training['gpu_memory_fraction']
    if fraction is not None:
        _rate(fraction, 'training.gpu_memory_fraction')
        if fraction > 1:
            raise ValueError('training.gpu_memory_fraction must not exceed 1')
    if training['microbatch'] < 2 or training['effective_batch'] % training['microbatch']:
        raise ValueError('CMC needs microbatch >= 2, and it must divide effective_batch')
    return config


def language_enabled(config):
    return any(config['loss_weights'][name] > 0 for name in ('lr', 'cmc', 'kl'))


def soft_dice_loss(mask_logits, masks, labels):
    """Mean per-image soft Dice over fake images; real images have no foreground, so CE alone covers them."""
    fake = labels == 1
    if not fake.any():
        return mask_logits.new_zeros(())
    probability = mask_logits[fake].softmax(dim=1)[:, 1]
    target = masks[fake].to(probability.dtype)
    overlap = (probability * target).sum(dim=(1, 2))
    return (1 - (2 * overlap + 1) / (probability.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1)).mean()


class ResidualMaskFusion(nn.Module):
    """Add a residual-informed mask correction; zero output until it has been trained.

    The residual is detached, so localization cannot make the reconstruction worse
    in order to expose forgeries. Dividing by each image's mean error keeps the
    relative spatial pattern, which is what separates the edited region.
    """
    def __init__(self, seed=0):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.conv = nn.Conv2d(64 + 3, 32, 3, padding=1)
            self.out = nn.Conv2d(32, 2, 1)
            generator = torch.Generator().manual_seed(seed)
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu', generator=generator)
            nn.init.zeros_(self.conv.bias)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, features, residual_image):
        residual = residual_image.detach()
        relative = torch.log1p(residual / (residual.mean(dim=(1, 2, 3), keepdim=True) + 1e-6))
        return self.out(F.relu(self.conv(torch.cat((features, relative), dim=1))))


class TokenVisionInjection(VisionInjectionDelta):
    """VIM with every visual token as key/value, so attention can differ between words.

    Same parameters as the class-token VIM. A labelled variant in the spirit of the paper's
    Table XV all-token row, whose fusion is unspecified. Unlike one key, many keys make the
    scale matter; this uses sqrt(d/r) as in the authors' public CrossAttention code, not
    the printed sqrt(r/d).
    """
    def __init__(self):
        super().__init__()
        self.denominator = math.sqrt(self.head_dim)

    def forward_with_trace(self, language, visual):
        validate_hidden(language, 'language', self.q.weight, length=language.shape[1])
        validate_hidden(visual, 'visual', self.q.weight, length=visual.shape[1])
        if visual.shape[0] != language.shape[0]:
            raise ValueError('visual and language batches must match')
        batch, length, count = language.shape[0], language.shape[1], visual.shape[1]
        query = self.q(language).reshape(batch, length, 8, 64).transpose(1, 2)
        key = self.k(visual).reshape(batch, count, 8, 64).transpose(1, 2)
        value = self.v(visual).reshape(batch, count, 8, 64).transpose(1, 2)
        logits = (query @ key.transpose(-2, -1)) / self.denominator
        probabilities = logits.softmax(dim=-1)
        attended = (probabilities @ value).transpose(1, 2).reshape(batch, length, 512)
        delta = finite(self.out(attended), 'VIM.delta')
        return VIMTrace(query, key, value, logits, probabilities, attended, delta)


class TokenLanguageTransformer(EosFineGrainedLanguageTransformer):
    """Language branch conditioned on [B,197,512] visual tokens; class token stays last."""
    def __init__(self, *, seed=0, vim_residual='single', language_normalization='pre_norm'):
        super().__init__(seed=seed, vim_residual=vim_residual, language_normalization=language_normalization)
        for block in [*self.encoder.blocks, *self.decoder.blocks]:
            replacement = TokenVisionInjection()
            replacement.load_state_dict(block.vim.state_dict(), strict=True)
            block.vim = replacement

    def _validate_inputs(self, ids, visual):
        validate_hidden(visual, 'visual', self.vocabulary.weight, length=197)
        super()._validate_inputs(ids, visual[:, -1:, :])

    def decode(self, low, encoder_hidden, visual):
        validate_hidden(low, 'low', self.vocabulary.weight)
        validate_hidden(encoder_hidden, 'encoder_hidden', self.vocabulary.weight)
        validate_hidden(visual, 'visual', self.vocabulary.weight, length=197)
        if low.shape[0] != encoder_hidden.shape[0] or low.shape[0] != visual.shape[0]:
            raise ValueError('decoder input batches must match')
        shifted = finite(self.shift_low(low) + self.decoder_position, 'LD.shift_position_sum')
        return self.decoder_final_norm(self.decoder(shifted, encoder_hidden, visual))


def reconstruct_from(flt, target_ids, encoder_ids, visual):
    """FLT forward whose LE memory comes from `encoder_ids` while LD is teacher-forced on `target_ids`."""
    encoded = flt.encode(encoder_ids, visual)
    low = finite(flt.vocabulary(target_ids), 'lookup.target')
    hidden = flt.decode(low, encoded.encoder_hidden, visual)
    return FLTOutput(low, encoded.encoder_hidden, encoded.global_language, hidden, flt.project_vocabulary(hidden))


class ExperimentMFVLR(SkipMFVLR):
    """SkipMFVLR with every component seeded; seed 0 keeps the accepted initialization."""
    def __init__(self, settings, weights, *, seed=0):
        settings = dict(settings)
        mask_residual = settings.pop('mask_residual')
        self_vim_tokens = settings.pop('vim_tokens', 'cls')
        mask_dice_weight = settings.pop('mask_dice_weight', 0.)
        encoder = settings.pop('encoder', 'scratch')
        settings.pop('language_prompt', None)
        settings.pop('language_encoder_input', None)
        text_patch = settings.pop('text_patch', 'none')
        self_text_patch_weight = settings.pop('text_patch_weight', 0.)
        super().__init__(settings, weights)
        self.text_patch_weight = self_text_patch_weight
        # Set by the runner when the encoder reads a neutral prompt; not a parameter or buffer.
        self.encoder_ids = None
        self.vim_tokens = self_vim_tokens
        self.mask_dice_weight = mask_dice_weight
        options = {name: settings[name] for name in ('vim_residual', 'language_normalization') if name in settings}
        if seed:
            self.mve = MVE(seed=seed)
            decoder_class = SkipResizeDecoder if settings['skip_connections'] == 'concat' else ResizeConvDecoder
            self.decoder = decoder_class(settings['resize_mode'], seed=seed,
                reconstruction_activation=settings.get('reconstruction_activation', 'identity'))
            self.heads = DetectionHeads(seed=seed)
            if self.flt is not None:
                self.flt = EosFineGrainedLanguageTransformer(seed=seed, **options)
        if self.vim_tokens == 'all':
            self.flt = TokenLanguageTransformer(seed=seed, **options)
        self.mask_fusion = ResidualMaskFusion(seed) if mask_residual == 'concat' else None
        install_encoder(self, encoder, seed)
        self.text_patch = None if text_patch == 'none' else TextPatchHead(text_patch, seed)

    def compute_objectives(self, output, targets):
        losses = super().compute_objectives(output, targets)
        # Both extras are logged inside fl and total; evaluation keeps reporting the CE part, so fl stays
        # comparable between arms.
        extra = 0.
        if self.mask_dice_weight:
            extra = self.mask_dice_weight * self.objectives.weights['fl'] * soft_dice_loss(
                output.predictions.decoded.mask_logits, targets.masks, targets.labels)
        if self.text_patch is not None:
            extra = extra + self.text_patch_weight * self.objectives.weights['fl'] * text_patch_loss(
                self.text_patch(self.patch_tokens(output.predictions.visual)), targets.masks, targets.labels,
                targets.sample_ids)
        if not torch.is_tensor(extra):
            return losses
        return LossValues(losses.fd, losses.lr, losses.cmc, losses.fl + extra, losses.ar, losses.kl, losses.total + extra)

    @staticmethod
    def patch_tokens(visual):
        """Image plus residual tokens, the per-patch counterpart of the fused global token."""
        return visual.appearance.tokens + visual.residual_tokens

    def language_condition(self, predictions):
        """What the language branch sees of the image: the class-token sum, or every token."""
        visual = predictions.visual
        return visual.fused_global if self.vim_tokens == 'cls' else visual.appearance.tokens + visual.residual_tokens

    def with_language(self, image, token_ids, *, sample_ids):
        if self.vim_tokens == 'cls' and self.encoder_ids is None:
            return super().with_language(image, token_ids, sample_ids=sample_ids)
        ordered = pairing(sample_ids, image.shape[0])
        indices(token_ids, image, 'token_ids', (image.shape[0], 308), 49408)
        predictions = self.visual_only(image)
        condition = language_visual_feature(self.language_condition(predictions), self.stop_all)
        if self.encoder_ids is not None:
            # LE reads one neutral prompt, so LD can only learn the target text from the image.
            neutral = self.encoder_ids.expand(image.shape[0], -1)
            language = (reconstruct_from(self.flt, token_ids, neutral, condition) if self.objectives.weights['lr']
                        else self.flt.encode(neutral, condition))
        elif self.objectives.weights['lr']:
            language = self.flt(token_ids, condition, return_logits=True)
        else:
            language = self.flt.encode(token_ids, condition)
        adapted = None
        if self.objectives.weights['kl']:
            adapted = self.heads.align_language(language_visual_feature(predictions.visual.fused_global, self.stop_all))
        return SimpleNamespace(image=image, sample_ids=ordered, token_ids=token_ids, predictions=predictions,
                               language=language, predicted_language=adapted, stop_language_visual=self.stop_all)

    def visual_only(self, image):
        predictions = super().visual_only(image)
        if self.mask_fusion is None and self.text_patch is None:
            return predictions
        decoded = predictions.decoded
        logits = decoded.mask_logits
        if self.mask_fusion is not None:
            logits = logits + self.mask_fusion(decoded.features, predictions.visual.residual_image)
        if self.text_patch is not None:
            patch_logits = self.text_patch(self.patch_tokens(predictions.visual))
            logits = logits + self.text_patch.mask_bias(patch_logits, logits.shape[-2:])
        return VisualPredictions(predictions.visual,
            VisionDecoderOutput(decoded.features, decoded.reconstruction, logits), predictions.detection_logits)


def build_model(config, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    return ExperimentMFVLR(config['model'], config['loss_weights'], seed=seed)


def _checkpoint_stage(module):
    original = module.forward

    def forward(*args, **kwargs):
        if module.training and torch.is_grad_enabled():
            return checkpoint(original, *args, use_reentrant=False, **kwargs)
        return original(*args, **kwargs)

    module.forward = forward


def enable_gradient_checkpointing(model):
    """Recompute stage activations; parameters and state keys are unchanged."""
    stages = [*model.mve.shared_encoder.cnn.stages, *model.mve.shared_encoder.blocks, *model.decoder.trunk.stages]
    if model.flt is not None:
        stages += [*model.flt.encoder.blocks, *model.flt.decoder.blocks]
    for stage in stages:
        _checkpoint_stage(stage)


def load_samples(path, limit=None):
    """Read a balanced real/fake NPZ; a limit keeps evenly spaced samples of each class."""
    with np.load(path, allow_pickle=False) as data:
        labels, ids = data['labels'].tolist(), data['sample_ids'].tolist()
        selected = np.arange(len(labels)) if limit is None else np.sort(np.concatenate([
            np.array([i for i, label in enumerate(labels) if label == cls])[
                np.linspace(0, labels.count(cls) - 1, limit // 2, dtype=int)] for cls in (0, 1)]))
        # Without a limit the freshly read array is used as is: fancy indexing plus copy would hold
        # three copies of an 80k-image train split at once.
        pick = (lambda array: array) if limit is None else (lambda array: array[selected])
        images = torch.from_numpy(pick(data['images']))
        masks = torch.from_numpy(pick(data['masks']))
    return Samples(images, masks, [labels[i] for i in selected.tolist()], [ids[i] for i in selected.tolist()])


class Samples:
    """uint8 storage, float/long per sample: the same values as converting the whole array, at a quarter of the RAM."""
    def __init__(self, images, masks, labels, ids):
        self.images, self.masks, self.labels, self.ids = images, masks, labels, ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        label = self.labels[index]
        return dict(image=self.images[index].float().div_(255), mask=self.masks[index].long(), label=label,
                    sample_id=self.ids[index], kind='real' if label == 0 else 'local')

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def prompt_tokens():
    """Two [308] rows, indexed by label; the pilot has exactly one prompt per class."""
    from mfvlr.prompts import PromptBuilder, PromptMetadata
    from mfvlr.tokenizer import TokenizerAdapter
    records = [PromptMetadata('real', 'real'), PromptMetadata('fake', 'fake', 'AM', 'diffusion', 'Diffae')]
    return TokenizerAdapter().encode_levels(PromptBuilder().build(records)).ids


NEUTRAL_PROMPT = ('A photo of a face',) * 4
REAL_REGION_PROMPT = ('A photo of a real face', 'No attribute of this face is edited',
                      'No part of this image is edited', 'The edited area is empty')
ATTRIBUTE_PHRASES = {'bangs': 'bangs above the forehead', 'smile': 'smile around the mouth'}
ZONES = ('upper', 'middle', 'lower')
# Fake-mask area quartiles on the 10k validation set (25% 9.8%, 75% 16.8%), fixed before any region run.
AREA_SIZES = ((0.10, 'small'), (0.17, 'medium'), (1.01, 'large'))


def region_prompt(sample_id, label, mask):
    """Four texts describing what was edited and where; the label alone does not determine them."""
    if label == 0:
        return REAL_REGION_PROMPT
    attribute = Path(sample_id).stem.split('_')[1].casefold()
    if attribute not in ATTRIBUTE_PHRASES:
        raise ValueError(f'{sample_id}: no region text for attribute {attribute!r}')
    rows = mask.sum(dim=1).float()
    thirds = [part.sum().item() for part in rows.tensor_split(3)]
    area = mask.float().mean().item()
    size = next(name for limit, name in AREA_SIZES if area < limit)
    return ('A photo of a fake face', f'The {ATTRIBUTE_PHRASES[attribute]} of this face is edited',
            f'Most edited pixels are in the {ZONES[int(np.argmax(thirds))]} part of the image', f'The edited area is {size}')


class RegionPrompts:
    """Every region text tokenized once; row 0 is the real prompt, row 1 a fake one (for the probe)."""
    def __init__(self):
        from mfvlr.tokenizer import TokenizerAdapter
        texts = [REAL_REGION_PROMPT] + [region_text for region_text in (
            ('A photo of a fake face', f'The {phrase} of this face is edited',
             f'Most edited pixels are in the {zone} part of the image', f'The edited area is {size}')
            for phrase in ATTRIBUTE_PHRASES.values() for zone in ZONES for _, size in AREA_SIZES)]
        self.rows = {text: i for i, text in enumerate(texts)}
        self.ids = TokenizerAdapter().encode_levels(tuple(texts)).ids

    def lookup(self, batch):
        rows = [self.rows[region_prompt(name, int(label), mask)]
                for name, label, mask in zip(batch['sample_id'], batch['label'], batch['mask'])]
        return self.ids[rows]


def neutral_tokens():
    from mfvlr.tokenizer import TokenizerAdapter
    return TokenizerAdapter().encode_levels((NEUTRAL_PROMPT,)).ids


def make_schedule(count, effective_batch, epochs, seed):
    if count % effective_batch:
        raise ValueError('Training count must be divisible by effective_batch')
    rng, schedule = np.random.default_rng(seed), []
    for _ in range(epochs):
        order = rng.permutation(count).tolist()
        schedule.extend(order[i:i + effective_batch] for i in range(0, count, effective_batch))
    return schedule


def build_optimizer(model, config):
    settings = config['optimizer']
    image = [*model.mve.parameters(), *model.decoder.parameters(), *model.heads.detector.parameters()]
    if model.mask_fusion is not None:
        image += list(model.mask_fusion.parameters())
    if model.text_patch is not None:
        image += list(model.text_patch.parameters())
    known = {id(p) for p in image}
    language = [p for p in model.parameters() if id(p) not in known]
    groups = [dict(params=image, lr=settings['image_lr']), dict(params=language, lr=settings['language_lr'])]
    cnn = model.mve.shared_encoder.cnn
    if hasattr(cnn, 'pretrained_parameters'):
        # Pretrained weights get their own rate; the scratch stem and lift stay with the image group.
        backbone = {id(p) for p in cnn.pretrained_parameters()}
        groups[0]['params'] = [p for p in image if id(p) not in backbone]
        groups.append(dict(params=cnn.pretrained_parameters(), lr=settings.get('backbone_lr', settings['image_lr'])))
    optimizer = torch.optim.Adam(groups, weight_decay=settings['weight_decay'], foreach=False)
    schedule = config['scheduler']
    scheduler = None if schedule is None else torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=schedule['step_size'], gamma=schedule['gamma'])
    return optimizer, scheduler


def _batch(samples, tokens, device):
    batch = default_collate(samples)
    image, ids = batch['image'].to(device), None
    if tokens is None:
        ids = torch.zeros((len(samples), 308), dtype=torch.int64, device=device)
    elif isinstance(tokens, RegionPrompts):
        ids = tokens.lookup(batch).to(device)
    else:
        ids = tokens[batch['label']].to(device)
    names = tuple(batch['sample_id'])
    targets = SuppliedTargets(names, ids, batch['label'].to(device), batch['mask'].to(device), tuple(batch['kind']))
    return image, ids, names, targets


@torch.no_grad()
def evaluate_split(model, samples, device, class_weights, batch_size):
    """Image-only inference; fixed detection threshold 0.5 and mask argmax, no tuning."""
    was_training = model.training
    model.eval()
    labels, scores, fake_iou, fake_dice, real_area, values = [], [], [], [], [], []
    sums = dict(ar=0., fd=0., fl=0.)
    pixels = dict(tp=0, fp=0, fn=0, tn=0)
    for start in range(0, len(samples), batch_size):
        batch = default_collate(samples[start:start + batch_size])
        image, target = batch['image'].to(device), batch['mask'].to(device)
        output = model.visual_only(image)
        logits, mask_logits = output.detection_logits, output.decoded.mask_logits
        size = len(image)
        sums['ar'] += float(appearance_reconstruction_loss(image, output.decoded.reconstruction)) * size / len(samples)
        sums['fd'] += float(forgery_detection_loss(logits, batch['label'].to(device))) * size / len(samples)
        sums['fl'] += float(weighted_mask_loss(mask_logits, target, class_weights)) * size / len(samples)
        labels.extend(batch['label'].tolist())
        scores.extend(logits.softmax(-1)[:, 1].cpu().tolist())
        predicted, truth = mask_logits.argmax(1).cpu().bool(), batch['mask'].bool()
        pixels['tp'] += int((predicted & truth).sum())
        pixels['fp'] += int((predicted & ~truth).sum())
        pixels['fn'] += int((~predicted & truth).sum())
        pixels['tn'] += int((~predicted & ~truth).sum())
        for i, label in enumerate(batch['label'].tolist()):
            p, gt = predicted[i], truth[i]
            if label == 0:
                real_area.append(float(p.float().mean()))
                values.append(real_area[-1])
            else:
                union, total = int((p | gt).sum()), int(p.sum() + gt.sum())
                intersection = int((p & gt).sum())
                # Explicit empty/empty convention, separate from real images.
                fake_iou.append(intersection / union if union else 1.)
                fake_dice.append(2 * intersection / total if total else 1.)
                values.append(fake_iou[-1])
    model.train(was_training)
    # Class-wise mean over the whole set, real images included; the closest reading of the paper's mIoU.
    tp, fp, fn, tn = (pixels[name] for name in ('tp', 'fp', 'fn', 'tn'))
    class_iou = [value for value in (tp / (tp + fp + fn) if tp + fp + fn else None,
                                     tn / (tn + fp + fn) if tn + fp + fn else None) if value is not None]
    return {**metrics(labels, scores, .5), 'count': len(samples), 'losses': sums, 'pixel_confusion': pixels,
            'pixel_miou_global': float(np.mean(class_iou)) if class_iou else None,
            'pixel_precision': pixels['tp'] / max(1, pixels['tp'] + pixels['fp']),
            'pixel_recall': pixels['tp'] / max(1, pixels['tp'] + pixels['fn']),
            'fake_foreground_iou_macro': float(np.mean(fake_iou)) if fake_iou else None,
            'fake_foreground_dice_macro': float(np.mean(fake_dice)) if fake_dice else None,
            'real_false_positive_area': float(np.mean(real_area)) if real_area else None,
            # In sample order: fake probability, and IoU for a fake or predicted area for a real. Lets a
            # report be re-scored per source or on a nested subset without the checkpoint.
            'per_sample': dict(score=[round(s, 6) for s in scores], value=[round(v, 6) for v in values])}


@torch.no_grad()
def language_probe(model, samples, tokens, device, count=40):
    """Which input decides the pooled text feature, and how much each encoder block matters."""
    was_training = model.training
    model.eval()
    flt = model.flt
    picks = samples[:count // 2] + samples[-(count // 2):]
    predictions = [model.visual_only(s['image'][None].to(device)) for s in picks]
    visuals = [model.language_condition(p) for p in predictions]
    tokens = tokens.ids if isinstance(tokens, RegionPrompts) else tokens
    real, fake = tokens[:1].to(device), tokens[1:2].to(device)
    same_prompt, same_visual, matched, other_visual, other_prompt = [], [], [], [], []
    position_spread, entropy = [], []
    for index, visual in enumerate(visuals):
        other = visuals[(index + 7) % len(visuals)]
        text = flt.encode(real, visual).global_language[0, 0]
        with_other_visual = flt.encode(real, other).global_language[0, 0]
        with_other_prompt = flt.encode(fake, visual).global_language[0, 0]
        same_prompt.append(F.cosine_similarity(text, with_other_visual, dim=0).item())
        same_visual.append(F.cosine_similarity(text, with_other_prompt, dim=0).item())
        direction = predictions[index].visual.fused_global[0, 0]
        matched.append(F.cosine_similarity(direction, text, dim=0).item())
        other_visual.append(F.cosine_similarity(direction, with_other_visual, dim=0).item())
        other_prompt.append(F.cosine_similarity(direction, with_other_prompt, dim=0).item())
        first = flt.encoder.blocks[0]
        trace = first.vim.forward_with_trace(first.norm_vim(flt.vocabulary(real) + flt.encoder_position), visual)
        # Zero spread means every word received the same injection.
        position_spread.append((trace.delta.std(dim=1).mean() / trace.delta.abs().mean().clamp_min(1e-12)).item())
        if visual.shape[1] > 1:
            p = trace.probabilities.clamp_min(1e-12)
            entropy.append((-(p * p.log()).sum(-1).mean() / math.log(visual.shape[1])).item())
    start = flt.vocabulary(real) + flt.encoder_position
    baseline = flt.encoder(start, visuals[0])
    block_effect = {}
    for index in (0, 2, 5, 8, 11):
        block = flt.encoder.blocks[index]
        original, factor = block.forward, getattr(block, 'residual_factor', 1)
        block.forward = lambda state, visual, factor=factor: factor * state
        try:
            changed = flt.encoder(start, visuals[0])
        finally:
            block.forward = original
        block_effect[str(index)] = ((changed - baseline).norm() / baseline.norm()).item()
    model.train(was_training)
    return dict(text_cosine_same_prompt_other_visual=float(np.mean(same_prompt)),
                text_cosine_other_prompt_same_visual=float(np.mean(same_visual)),
                visual_to_text_matched=float(np.mean(matched)),
                visual_to_text_other_visual=float(np.mean(other_visual)),
                visual_to_text_other_prompt=float(np.mean(other_prompt)),
                vim_delta_position_spread=float(np.mean(position_spread)),
                vim_attention_entropy_ratio=float(np.mean(entropy)) if entropy else None,
                encoder_block_effect=block_effect,
                encoder_output_rms=baseline.square().mean().sqrt().item(), images=len(visuals))


def _utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_hashes(root):
    """Hash every loaded project module, helpers included; keys are portable across systems."""
    root, result = Path(root).resolve(), {}
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        # Some extension modules report a bare relative name such as torch.ops' _ops.py.
        if not file or not os.path.isabs(file):
            continue
        path = Path(file).resolve()
        if path.suffix == '.py' and path.is_relative_to(root) and '.pilot_deps' not in path.parts:
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


def _restore_history(path, step):
    """Drop updates logged after the checkpoint, so resumed steps are never duplicated."""
    if not path.exists():
        return
    kept = [line for line in path.read_text(encoding='utf-8').splitlines() if json.loads(line)['step'] <= step]
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(line + '\n' for line in kept), encoding='utf-8')
    os.replace(temporary, path)


def run_experiment(config, seed, output, *, root, device='cuda', budget_seconds=None, max_updates=None):
    """Train one seed; resumes from output/last.pt when present, otherwise starts fresh."""
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    status = dict(status='STARTING', phase='load_data', seed=seed, name=config['name'], step=0, pid=os.getpid(), error=None)
    started = time.monotonic()

    def update(**fields):
        status.update(fields, time_utc=_utc(), elapsed_seconds=round(time.monotonic() - started, 2))
        write_json(output / 'status.json', status)

    def log(event, **fields):
        with (output / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=event, time_utc=_utc(), **fields)) + '\n')

    try:
        if (output / 'status.json').exists():
            if json.loads((output / 'status.json').read_text(encoding='utf-8'))['status'] == 'COMPLETED':
                raise ValueError('Output already holds a completed run')
        update()
        training = config['training']
        torch.set_num_threads(training['threads'])
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        # Measured on a 6 GB RTX 4050: without a cap the allocator reserved up to 9 GiB and each
        # update took 5.5x longer. A cap changes the execution path, so results repeat exactly
        # under one setting but differ by rounding from another; keep it equal across paired arms.
        if device == 'cuda' and training['gpu_memory_fraction'] is not None:
            torch.cuda.set_per_process_memory_fraction(training['gpu_memory_fraction'])
        inputs = Path(root) / config['inputs']
        train_file = inputs / config.get('train_file', 'train.npz')
        train = load_samples(train_file, config['train_limit'])
        val = load_samples(inputs / 'val.npz', config['val_limit'])
        if set(train.ids) & set(val.ids):
            raise ValueError('Train and validation sample IDs overlap')
        needs_language = language_enabled(config)
        prompt_mode = config['model'].get('language_prompt', 'label')
        tokens = None if not needs_language else RegionPrompts() if prompt_mode == 'region' else prompt_tokens()
        micro, effective = training['microbatch'], training['effective_batch']
        augmentation = training.get('augmentation', 'none')
        if augmentation == 'none':
            pick = lambda _, i: train[i]
        else:
            # Imported only here, so runs without augmentation load exactly the modules they did before.
            from .augmentation import augment_sample
            pick = lambda update, i: augment_sample(train[i], augmentation, seed, update, i)
        schedule = make_schedule(len(train), effective, training['epochs'], seed)
        if max_updates is not None:
            schedule = schedule[:max_updates]
        steps_per_epoch = len(train) // effective
        torch.manual_seed(seed)
        model = build_model(config, seed)
        settings = config['model']
        class_weights = torch.tensor([settings['mask_background_weight'], settings['mask_foreground_weight']],
                                     dtype=torch.float32, device=device)
        model.to(device).train()
        if config['model'].get('language_encoder_input', 'prompt') == 'neutral':
            model.encoder_ids = neutral_tokens().to(device)
        if training['gradient_checkpointing']:
            enable_gradient_checkpointing(model)
        optimizer, scheduler = build_optimizer(model, config)
        data_hash = {s: hashlib.sha256(path.read_bytes()).hexdigest() for s, path in (('train', train_file), ('val', inputs / 'val.npz'))}
        sources = source_hashes(root)
        contract = dict(config=config, seed=seed, data=data_hash, source=sources)
        step, evaluations = 0, []
        last = output / 'last.pt'
        if last.exists():
            state = torch.load(last, map_location='cpu', weights_only=True)
            if state['schema'] != SCHEMA or state['contract'] != contract:
                raise ValueError('Resume config, seed, data or source differs from the checkpoint')
            model.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(state['scheduler'])
            torch.set_rng_state(state['torch_rng'])
            if device == 'cuda':
                torch.cuda.set_rng_state_all(state['cuda_rng'][:torch.cuda.device_count()])
            step, evaluations = state['step'], state['evaluations']
            del state
            gc.collect()
            log('resumed', step=step)
        _restore_history(output / 'history.jsonl', step)
        report = dict(status='RUNNING', **contract, schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
                      updates=len(schedule), steps_per_epoch=steps_per_epoch, test_used=False,
                      train_count=len(train), val_count=len(val), parameters=sum(p.numel() for p in model.parameters()),
                      environment=dict(torch=torch.__version__, python=sys.version, device=device,
                                       gpu=torch.cuda.get_device_name() if device == 'cuda' else None),
                      evaluations=evaluations)
        write_json(output / 'report.json', report)

        def save_state():
            atomic_save(dict(schema=SCHEMA, step=step, contract=contract, evaluations=evaluations, model=model.state_dict(),
                             optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict() if scheduler else None,
                             torch_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else []), last)
            log('checkpoint_saved', step=step)

        def run_evaluation(epoch):
            update(phase='evaluate', epoch=epoch)
            result = evaluate_split(model, val, device, class_weights, micro)
            evaluations.append(dict(epoch=epoch, step=step, val=result))
            write_json(output / 'report.json', report)
            log('metrics', epoch=epoch, step=step, val_iou=result['fake_foreground_iou_macro'], val_auc=result['auc'])

        if step == 0:
            run_evaluation(0)
        update(status='RUNNING', phase='train', step=step, updates=len(schedule))
        paused = False
        for index in range(step, len(schedule)):
            if budget_seconds is not None and time.monotonic() - started > budget_seconds:
                paused = True
                break
            optimizer.zero_grad(set_to_none=True)
            means = {}
            update_started = time.monotonic()
            for offset in range(0, effective, micro):
                image, ids, names, targets = _batch([pick(index, i) for i in schedule[index][offset:offset + micro]], tokens, device)
                result = model.with_language(image, ids, sample_ids=names)
                losses = model.compute_objectives(result, targets)
                (losses.total / (effective / micro)).backward()
                for key, value in losses.as_dict().items():
                    means[key] = means.get(key, 0.) + float(value.detach()) * micro / effective
                del result, losses
            check_finite_tensors((p.grad for p in model.parameters()), 'Nonfinite gradient')
            optimizer.step()
            check_finite_tensors(model.parameters(), 'Nonfinite parameter')
            step = index + 1
            with (output / 'history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(step=step, losses=means, seconds=round(time.monotonic() - update_started, 3),
                                             time_utc=_utc())) + '\n')
            if step % 10 == 0:
                update(step=step)
            finished_epoch = step % steps_per_epoch == 0
            if finished_epoch or step == len(schedule):
                optimizer.zero_grad(set_to_none=True)
                if finished_epoch and scheduler is not None:
                    scheduler.step()
                run_evaluation((step - 1) // steps_per_epoch + 1)
                update(phase='train', step=step)
            if finished_epoch or step == len(schedule) or step % training['checkpoint_every'] == 0:
                save_state()
        if paused:
            save_state()
            report.update(status='PAUSED_TIME_BUDGET', completed_updates=step)
            write_json(output / 'report.json', report)
            update(status='PAUSED_TIME_BUDGET', phase='paused', step=step)
            return report
        if needs_language:
            update(phase='language_probe')
            report['language_probe'] = language_probe(model, val, tokens, device)
        if source_hashes(root) != sources:
            raise RuntimeError('Source changed during the experiment')
        atomic_save(dict(schema=SCHEMA + '-final', step=step, contract=contract,
                         model={k: v.detach().cpu() for k, v in model.state_dict().items()}), output / 'final.pt')
        report.update(status='COMPLETED', completed_updates=step, source_unchanged=True)
        write_json(output / 'report.json', report)
        # The resumable state is only needed until the final weights are durable.
        if last.exists():
            last.unlink()
        update(status='COMPLETED', phase='finished', step=step)
        return report
    except BaseException as error:
        update(status='FAILED', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        model = optimizer = scheduler = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def final_iou(report, epochs=FINAL_EPOCHS):
    """Primary endpoint: mean validation IoU of the last epochs, damping epoch-to-epoch swings."""
    rows = [row for row in report['evaluations'] if row['epoch'] > 0][-epochs:]
    if not rows:
        raise ValueError('No completed epoch evaluation')
    return float(np.mean([row['val']['fake_foreground_iou_macro'] for row in rows]))


def compare_reports(baseline, candidate):
    """Paired by seed; the same initialization and batch order differ only by the treatment."""
    seeds = sorted(set(baseline) & set(candidate))
    if not seeds:
        raise ValueError('No paired seeds to compare')
    rows = {}
    for seed in seeds:
        rows[str(seed)] = dict(baseline=final_iou(baseline[seed]), candidate=final_iou(candidate[seed]))
        rows[str(seed)]['difference'] = rows[str(seed)]['candidate'] - rows[str(seed)]['baseline']
    differences = [row['difference'] for row in rows.values()]
    return dict(seeds=rows, mean_difference=float(np.mean(differences)),
                sample_sd=float(np.std(differences, ddof=1)) if len(differences) > 1 else None,
                candidate_better_in=sum(d > 0 for d in differences), paired_seeds=len(differences))
