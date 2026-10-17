"""Approved Phase 2D.1 independent executable interpretations of six objectives.

Original equations remain unchanged in docs/phase2d_losses_contract.md.
MFVLR III-E p. 7 Eq. (17), p. 8 Eqs. (18)-(27). Scalarizations, log-softmax
interpretations, raw-dot similarity, log-temperature and two-sided KL gradients
are scoped engineering assumptions P2D06-P2D14/P2D16. No optimizer/data pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


def finite(value: Tensor, name: str) -> Tensor:
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"Nonfinite objective tensor: {name}")
    return value


def floating(value: Tensor, name: str, tail: tuple[int, ...]) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != len(tail) + 1 or value.shape[0] < 1 or tuple(value.shape[1:]) != tail:
        raise ValueError(f"{name} must have positive [B,{','.join(map(str, tail))}] shape")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be FP32 (FP64 analytical fixtures allowed)")
    if value.device.type == "meta":
        raise ValueError(f"{name} must contain materialized values")
    finite(value, name)


def matching(actual: Tensor, reference: Tensor, name: str) -> None:
    if actual.dtype != reference.dtype:
        raise TypeError(f"{name} dtype must match")
    if actual.device != reference.device:
        raise ValueError(f"{name} device must match")
    if actual.shape[0] != reference.shape[0]:
        raise ValueError(f"{name} batch must match")


def indices(target: Tensor, reference: Tensor, name: str, shape: tuple[int, ...], classes: int) -> None:
    if not isinstance(target, Tensor):
        raise TypeError(f"{name} must be a tensor; no missing-target fallback")
    if tuple(target.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if target.dtype != torch.int64:
        raise TypeError(f"{name} must be int64 class indices")
    if target.device != reference.device:
        raise ValueError(f"{name} device must match")
    if target.min().item() < 0 or target.max().item() >= classes:
        raise ValueError(f"{name} must be in [0,{classes})")


def scalar(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 0:
        raise ValueError(f"{name} must be a rank-zero tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be floating point")
    return finite(value, name)


def forgery_detection_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Eq. (26) interpretation: class log-softmax and batch mean."""
    floating(logits, "detection_logits", (2,))
    indices(labels, logits, "labels", (logits.shape[0],), 2)
    logp = finite(logits.log_softmax(dim=1), "FD.log_probabilities")
    nll = -logp.gather(1, labels[:, None])[:, 0]
    return scalar(nll.mean(), "L_fd")


def language_token_nll(logits: Tensor, ids: Tensor) -> Tensor:
    """Unshifted targets; includes every pad/BOS/EOS/content token."""
    floating(logits, "language_logits", (308, 49408))
    indices(ids, logits, "token_ids", (logits.shape[0], 308), 49408)
    logp = finite(logits.log_softmax(dim=-1), "LR.log_probabilities")
    return finite(-logp.gather(-1, ids[..., None])[..., 0], "LR.token_nll")


def language_reconstruction_loss(logits: Tensor, ids: Tensor) -> Tensor:
    """Eq. (25): token sum, then batch mean; no ignore-index or length mean."""
    per_image = finite(language_token_nll(logits, ids).sum(dim=1), "LR.token_sum")
    return scalar(per_image.mean(), "L_lr")


class CrossModalContrastiveLoss(nn.Module):
    """Eqs. (20)-(23): raw dot, diagonal targets, symmetric local-batch CE."""

    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.07), dtype=torch.float32, device="cpu"))

    @property
    def temperature(self) -> Tensor:
        scalar(self.log_temperature, "CMC.log_temperature")
        tau = finite(self.log_temperature.exp(), "CMC.temperature")
        if tau.item() <= 0:
            raise FloatingPointError("CMC effective temperature must be positive; underflow is not clamped")
        return tau

    def scores(self, visual: Tensor, language: Tensor) -> Tensor:
        floating(visual, "visual", (1, 512))
        floating(language, "global_language", (1, 512))
        matching(language, visual, "global_language")
        if visual.dtype != self.log_temperature.dtype:
            raise TypeError("CMC input dtype must match temperature (explicit FP64 copy allowed)")
        if visual.device != self.log_temperature.device:
            raise ValueError("CMC input device must match temperature")
        raw = finite(visual[:, 0, :] @ language[:, 0, :].T, "CMC.raw_dot")
        return finite(raw / self.temperature, "CMC.scores")

    def forward(self, visual: Tensor, language: Tensor) -> Tensor:
        scores = self.scores(visual, language)
        forward = finite(scores.log_softmax(dim=1), "CMC.visual_to_language")
        reverse = finite(scores.T.log_softmax(dim=1), "CMC.language_to_visual")
        # Formula executes at B=1, preserving connected, allocated zero derivatives.
        l_v2l = -forward.diagonal().mean()
        l_l2v = -reverse.diagonal().mean()
        return scalar((l_v2l + l_l2v) / 2, "L_cmc")


def forgery_localization_loss(logits: Tensor, masks: Tensor) -> Tensor:
    """Eq. (18) interpretation: class CE, mean batch and spatial dimensions."""
    floating(logits, "mask_logits", (2, 224, 224))
    indices(masks, logits, "masks", (logits.shape[0], 224, 224), 2)
    logp = finite(logits.log_softmax(dim=1), "FL.log_probabilities")
    nll = -logp.gather(1, masks[:, None, :, :])[:, 0, :, :]
    return scalar(nll.mean(), "L_fl")


def appearance_reconstruction_loss(image: Tensor, reconstruction: Tensor) -> Tensor:
    """Eq. (17) interpretation: RGB/all-pixel mean; no clipping or detach."""
    floating(image, "image", (3, 224, 224))
    floating(reconstruction, "reconstruction", (3, 224, 224))
    matching(reconstruction, image, "reconstruction")
    if image.min().item() < 0 or image.max().item() > 1:
        raise ValueError("image target must lie in [0,1]; reconstruction is unrestricted")
    difference = finite(image - reconstruction, "AR.difference")
    squared = finite(difference.square(), "AR.squared_difference")
    return scalar(squared.mean(), "L_ar")


def knowledge_distillation_loss(language: Tensor, predicted_language: Tensor) -> Tensor:
    """Eq. (19): KL(text||adapted visual), T=.5, feature sum, two-sided grads."""
    floating(language, "global_language", (1, 512))
    floating(predicted_language, "predicted_language", (1, 512))
    matching(predicted_language, language, "predicted_language")
    text_scores = finite(language / 0.5, "KL.text_scores")
    visual_scores = finite(predicted_language / 0.5, "KL.visual_scores")
    logp = finite(text_scores.log_softmax(dim=-1), "KL.text_log_probabilities")
    logq = finite(visual_scores.log_softmax(dim=-1), "KL.visual_log_probabilities")
    p = finite(logp.exp(), "KL.text_probabilities")
    difference = finite(logp - logq, "KL.log_ratio")
    pointwise = finite(p * difference, "KL.pointwise")
    per_image = finite(pointwise.sum(dim=-1)[:, 0], "KL.feature_sum")
    return scalar(per_image.mean(), "L_kl")


def validate_mask_semantics(labels: Tensor, masks: Tensor, sample_kinds: Sequence[str] | None = None) -> dict:
    """Validate supplied targets only (P2D14); no mask generation/resampling.

    Without explicit sample kinds a fake mask may be empty, mixed or full.
    Kind-specific validation never infers generation category from a class label.
    """
    if not isinstance(masks, Tensor) or masks.ndim != 3 or masks.shape[0] < 1 or tuple(masks.shape[1:]) != (224, 224):
        raise ValueError("supplied masks must have positive [B,224,224] shape")
    indices(masks, masks, "masks", tuple(masks.shape), 2)
    indices(labels, masks, "labels", (masks.shape[0],), 2)
    if bool((masks[labels == 0] != 0).any()):
        raise ValueError("real image targets require all-background masks")
    if sample_kinds is None:
        return {"empty_local_mask_indices": [], "sample_kinds_supplied": False}
    if isinstance(sample_kinds, (str, bytes)) or not isinstance(sample_kinds, Sequence) or len(sample_kinds) != masks.shape[0]:
        raise ValueError("sample_kinds must supply one real/efs/local category per image")
    empty = []
    for index, kind in enumerate(sample_kinds):
        if kind not in ("real", "efs", "local"):
            raise ValueError("unknown sample kind; no category fallback")
        expected = 0 if kind == "real" else 1
        if labels[index].item() != expected:
            raise ValueError("sample kind and detection target disagree")
        if kind == "efs" and not bool((masks[index] == 1).all()):
            raise ValueError("EFS supplied masks must be all foreground")
        if kind == "local" and masks[index].count_nonzero().item() == 0:
            empty.append(index)
    return {"empty_local_mask_indices": empty, "sample_kinds_supplied": True}


def unit_weight_sum(*, fd: Tensor, lr: Tensor, cmc: Tensor, fl: Tensor, ar: Tensor, kl: Tensor) -> Tensor:
    """Literal Eq. (27) order and coefficients, without averaging by six."""
    values = (fd, lr, cmc, fl, ar, kl)
    for name, value in zip(("fd", "lr", "cmc", "fl", "ar", "kl"), values):
        scalar(value, name)
        if value.dtype != fd.dtype or value.device != fd.device:
            raise ValueError("all objective scalars must have the same dtype/device")
    total = fd
    for value in values[1:]:
        total = scalar(total + value, "total_intermediate")
    return scalar(total, "total")


@dataclass(frozen=True)
class LossValues:
    fd: Tensor
    lr: Tensor
    cmc: Tensor
    fl: Tensor
    ar: Tensor
    kl: Tensor
    total: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {name: getattr(self, name) for name in ("fd", "lr", "cmc", "fl", "ar", "kl", "total")}


class SixObjectives(nn.Module):
    """Independent loss owner; only CMC theta is registered, no model modules."""

    def __init__(self) -> None:
        super().__init__()
        self.cmc = CrossModalContrastiveLoss()

    @classmethod
    def from_config(cls, config):
        config.require_approval()
        return cls()

    def forward(self, *, detection_logits: Tensor, labels: Tensor, language_logits: Tensor,
                token_ids: Tensor, visual: Tensor, global_language: Tensor, mask_logits: Tensor,
                masks: Tensor, image: Tensor, reconstruction: Tensor, predicted_language: Tensor,
                sample_kinds: Sequence[str] | None = None) -> LossValues:
        # Require consistent independent fixtures; no accidental cross-batch broadcasting.
        floating(image, "image", (3, 224, 224))
        for name, value, tail in (
            ("detection_logits", detection_logits, (2,)),
            ("language_logits", language_logits, (308, 49408)),
            ("visual", visual, (1, 512)), ("global_language", global_language, (1, 512)),
            ("mask_logits", mask_logits, (2, 224, 224)),
            ("reconstruction", reconstruction, (3, 224, 224)),
            ("predicted_language", predicted_language, (1, 512)),
        ):
            floating(value, name, tail)
            matching(value, image, name)
        validate_mask_semantics(labels, masks, sample_kinds)
        fd = forgery_detection_loss(detection_logits, labels)
        lr = language_reconstruction_loss(language_logits, token_ids)
        cmc = self.cmc(visual, global_language)
        fl = forgery_localization_loss(mask_logits, masks)
        ar = appearance_reconstruction_loss(image, reconstruction)
        kl = knowledge_distillation_loss(global_language, predicted_language)
        total = unit_weight_sum(fd=fd, lr=lr, cmc=cmc, fl=fl, ar=ar, kl=kl)
        return LossValues(fd, lr, cmc, fl, ar, kl, total)
