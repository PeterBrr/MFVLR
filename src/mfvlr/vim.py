"""Standalone L09 VIM delta, MFVLR III-D p. 7 Eqs. (4)-(11).

Engineering assumptions: d=512, r=8, bias-free independent projections,
key-axis softmax, Xavier initialization, residual owned by the caller.
The published sqrt(r/d) denominator and complete Q/K graph are retained.
No LE/LD blocks, normalization, gating, or residual addition are implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class VIMTrace:
    query: Tensor
    key: Tensor
    value: Tensor
    logits: Tensor
    probabilities: Tensor
    attended: Tensor
    delta: Tensor


class VisionInjectionDelta(nn.Module):
    def __init__(self, *, seed: int = 0) -> None:
        super().__init__()
        if type(seed) is not int or seed < 0 or seed >= 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        self.embed_dim = 512
        self.heads = 8
        self.head_dim = 64
        self.denominator = math.sqrt(self.heads / self.embed_dim)
        with torch.random.fork_rng(devices=[]):
            self.q = nn.Linear(512, 512, bias=False, device="cpu", dtype=torch.float32)
            self.k = nn.Linear(512, 512, bias=False, device="cpu", dtype=torch.float32)
            self.v = nn.Linear(512, 512, bias=False, device="cpu", dtype=torch.float32)
            self.out = nn.Linear(512, 512, bias=False, device="cpu", dtype=torch.float32)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            for layer in (self.q, self.k, self.v, self.out):
                nn.init.xavier_uniform_(layer.weight, gain=1.0, generator=generator)

    @classmethod
    def from_config(cls, config):
        config.require_approval()
        return cls(seed=config.data["initialization"]["seed"])

    def _validate(self, language: Tensor, visual: Tensor) -> None:
        for value, name in ((language, "language"), (visual, "visual")):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.ndim != 3 or value.shape[-1] != 512 or min(value.shape[:2]) < 1:
                raise ValueError(f"{name} must have positive [batch,length,512] shape")
            if value.dtype not in (torch.float32, torch.float64) or value.dtype != self.q.weight.dtype:
                raise TypeError(f"{name} must match the FP32 module (FP64 allowed for analytical copies)")
            if value.device != self.q.weight.device:
                raise ValueError(f"{name} must match the module device")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must contain only finite values")
        if visual.shape[1] != 1:
            raise ValueError("visual must contain exactly one fused class token")
        if visual.shape[0] != language.shape[0]:
            raise ValueError("visual and language batches must match")

    def forward(self, language: Tensor, visual: Tensor) -> Tensor:
        return self.forward_with_trace(language, visual).delta

    def forward_with_trace(self, language: Tensor, visual: Tensor) -> VIMTrace:
        """Explicit per-call diagnostic graph; no persistent trace/cache is stored."""
        self._validate(language, visual)
        batch, length, _ = language.shape
        query = self.q(language).reshape(batch, length, 8, 64).transpose(1, 2)
        key = self.k(visual).reshape(batch, 1, 8, 64).transpose(1, 2)
        value = self.v(visual).reshape(batch, 1, 8, 64).transpose(1, 2)
        # Intentionally NOT sqrt(d/r), and NOT a simplified broadcast shortcut.
        logits = (query @ key.transpose(-2, -1)) / self.denominator
        probabilities = logits.softmax(dim=-1)
        attended = (probabilities @ value).transpose(1, 2).reshape(batch, length, 512)
        delta = self.out(attended)
        # Surface finite-input overflow instead of hiding it with a constant softmax.
        if not torch.isfinite(logits).all().item() or not torch.isfinite(delta).all().item():
            raise FloatingPointError("VIM projection/attention overflow: nonfinite logits or delta")
        return VIMTrace(query, key, value, logits, probabilities, attended, delta)


def parameter_report(model: VisionInjectionDelta) -> dict[str, int]:
    report = {name: sum(p.numel() for p in layer.parameters())
              for name, layer in (("query", model.q), ("key", model.k),
                                  ("value", model.v), ("output", model.out))}
    report["total_unique"] = sum(p.numel() for p in model.parameters())
    report["singleton_zero_task_gradient_qk"] = report["query"] + report["key"]
    return report
