"""P2D01-P2D04 independent detector and visual-language adapter.

MFVLR III-E p. 8, Eqs. (19)/(26): one FC per role. Bias, class order,
initialization and exact interfaces are approved engineering assumptions.
No visual/language backbone or localization/reconstruction head is owned here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _finite(value: Tensor, name: str) -> Tensor:
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"Nonfinite head tensor: {name}")
    return value


def _visual(value: Tensor, parameter: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError("fused_global must be a tensor")
    if value.ndim != 3 or value.shape[0] < 1 or tuple(value.shape[1:]) != (1, 512):
        raise ValueError("fused_global must have positive [B,1,512] shape")
    if value.dtype not in (torch.float32, torch.float64) or value.dtype != parameter.dtype:
        raise TypeError("fused_global must match FP32 heads (explicit FP64 copies allowed)")
    if value.device != parameter.device:
        raise ValueError("fused_global device must match heads")
    _finite(value, "fused_global")


@dataclass(frozen=True)
class HeadOutput:
    detection_logits: Tensor
    predicted_language: Tensor


class DetectionHeads(nn.Module):
    """Two independent affine maps; a single dedicated initialization draw stream."""

    class_order = ("Real", "Fake")

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__()
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0,2**63)")
        with torch.random.fork_rng(devices=[]):
            self.detector = nn.Linear(512, 2, bias=True, device="cpu", dtype=torch.float32)
            self.adapter = nn.Linear(512, 512, bias=True, device="cpu", dtype=torch.float32)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            for layer in (self.detector, self.adapter):
                nn.init.xavier_uniform_(layer.weight, gain=1.0, generator=generator)
                nn.init.zeros_(layer.bias)

    @classmethod
    def from_config(cls, config):
        config.require_approval()
        return cls(seed=config.data["initialization"]["seed"])

    def detect(self, fused_global: Tensor) -> Tensor:
        _visual(fused_global, self.detector.weight)
        return _finite(self.detector(fused_global[:, 0, :]), "detection_logits")

    def align_language(self, fused_global: Tensor) -> Tensor:
        _visual(fused_global, self.adapter.weight)
        return _finite(self.adapter(fused_global), "predicted_language")

    def forward(self, fused_global: Tensor) -> HeadOutput:
        return HeadOutput(self.detect(fused_global), self.align_language(fused_global))

    @staticmethod
    def probabilities(logits: Tensor) -> Tensor:
        if not isinstance(logits, Tensor):
            raise TypeError("logits must be a tensor")
        if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] != 2:
            raise ValueError("logits must have positive [B,2] shape")
        if logits.dtype not in (torch.float32, torch.float64):
            raise TypeError("logits must be FP32 (FP64 analytical copies allowed)")
        return _finite(_finite(logits, "logits").softmax(dim=1), "probabilities")

    @staticmethod
    def fake_probability(logits: Tensor) -> Tensor:
        return DetectionHeads.probabilities(logits)[:, 1]

    @staticmethod
    def predicted_class(logits: Tensor) -> Tensor:
        # Validate without changing the raw-logit decision or applying a threshold.
        DetectionHeads.probabilities(logits)
        return logits.argmax(dim=1)


def head_parameter_report(module: DetectionHeads) -> dict:
    parameters = list(module.parameters())
    return {
        "detector": sum(p.numel() for p in module.detector.parameters()),
        "adapter": sum(p.numel() for p in module.adapter.parameters()),
        "total_unique": sum(p.numel() for p in parameters),
        "parameter_tensors": len(parameters),
        "unique_storage_elements": sum(p.numel() for p in {p.data_ptr(): p for p in parameters}.values()),
        "class_order": list(module.class_order),
    }
