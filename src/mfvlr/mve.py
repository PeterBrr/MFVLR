"""Approved M1--M5 MVE, not an assertion of recovered author internals.

Paper anchors: III-B pp. 5--6, Eqs. (1)--(2), IV-A p. 8.
Exact internal layers/initialization: docs/implementation_contract.md.
No decoder, classifier, language module, VIM, or training loss is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .config import ResolvedConfig


@dataclass(frozen=True)
class LocalEncoding:
    local: Tensor
    skips: tuple[Tensor, ...]


@dataclass(frozen=True)
class GlobalEncoding:
    tokens: Tensor
    global_token: Tensor


@dataclass(frozen=True)
class AppearanceEncoding:
    local: Tensor
    skips: tuple[Tensor, ...]
    tokens: Tensor
    global_token: Tensor


@dataclass(frozen=True)
class MVEOutput:
    appearance: AppearanceEncoding
    residual_image: Tensor
    residual_tokens: Tensor
    residual_global: Tensor
    fused_global: Tensor


def _validate_tensor(
    tensor: Tensor, name: str, shape: tuple[int, ...], device: torch.device,
) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != len(shape) + 1 or tuple(tensor.shape[1:]) != shape:
        raise ValueError(f"{name} must have shape [b,{','.join(map(str, shape))}]")
    if tensor.shape[0] < 1:
        raise ValueError(f"{name} batch must be nonempty")
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must be float32; implicit conversion is disabled")
    if tensor.device != device:
        raise ValueError(f"{name} device {tensor.device} must match encoder device {device}")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")


class DoubleConv(nn.Sequential):
    """A02: two padded biased 3x3 convolutions, each followed by ReLU."""

    def __init__(self, incoming: int, outgoing: int) -> None:
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, stride=1, padding=1, bias=True, device="cpu"),
            nn.ReLU(inplace=False),
            nn.Conv2d(outgoing, outgoing, 3, stride=1, padding=1, bias=True, device="cpu"),
            nn.ReLU(inplace=False),
        )


class UNetLocalEncoder(nn.Module):
    """Contracting path only; skips are interfaces, not a production decoder."""

    def __init__(self) -> None:
        super().__init__()
        widths = (3, 64, 128, 256, 512, 1024)
        self.stages = nn.ModuleList(DoubleConv(widths[i], widths[i + 1]) for i in range(5))
        self.pools = nn.ModuleList(
            nn.MaxPool2d(2, stride=2, padding=0, dilation=1, ceil_mode=False)
            for _ in range(4)
        )

    def forward(self, image: Tensor, return_skips: bool = True) -> LocalEncoding:
        skips = []
        state = image
        for index, stage in enumerate(self.stages):
            state = stage(state)
            if index < 4:
                if return_skips:
                    skips.append(state)
                state = self.pools[index](state)
        return LocalEncoding(local=state, skips=tuple(skips))


class VisualSelfAttention(nn.Module):
    """A03 visual MHA: separate Q/K/V, key softmax and sqrt(64) scaling.

    This is visual self-attention; MFVLR's separate VIM Eq. (10) is untouched.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 512
        self.heads = 8
        self.head_dim = 64
        self.q = nn.Linear(512, 512, bias=True, device="cpu")
        self.k = nn.Linear(512, 512, bias=True, device="cpu")
        self.v = nn.Linear(512, 512, bias=True, device="cpu")
        self.out = nn.Linear(512, 512, bias=True, device="cpu")

    def forward(self, state: Tensor) -> Tensor:
        batch, length, _ = state.shape

        def heads(projection: nn.Linear) -> Tensor:
            return projection(state).reshape(batch, length, 8, 64).transpose(1, 2)

        query, key, value = heads(self.q), heads(self.k), heads(self.v)
        logits = (query @ key.transpose(-2, -1)) / math.sqrt(64)
        probabilities = logits.softmax(dim=-1)
        attended = (probabilities @ value).transpose(1, 2).reshape(batch, length, 512)
        return self.out(attended)


class VisualTransformerBlock(nn.Module):
    """A03: pre-LN MHA residual, then pre-LN 2048-wide exact GELU FFN residual."""

    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(512, eps=1e-5, elementwise_affine=True, device="cpu")
        self.attention = VisualSelfAttention()
        self.ln2 = nn.LayerNorm(512, eps=1e-5, elementwise_affine=True, device="cpu")
        self.fc1 = nn.Linear(512, 2048, bias=True, device="cpu")
        self.activation = nn.GELU(approximate="none")
        self.fc2 = nn.Linear(2048, 512, bias=True, device="cpu")

    def forward(self, state: Tensor) -> Tensor:
        attended = state + self.attention(self.ln1(state))
        return attended + self.fc2(self.activation(self.fc1(self.ln2(attended))))


class SharedVisualEncoder(nn.Module):
    """Single owner of CNN/projection/CLS/positions/four independent blocks.

    Final approved initialization uses a private CPU generator, in contract order.
    Construction defaults are overwritten once; no role-specific reinitialization.
    Constructor allocation does not change the caller's global CPU RNG state.
    """

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        with torch.random.fork_rng(devices=[]):
            self.cnn = UNetLocalEncoder()
            self.projection = nn.Linear(1024, 512, bias=True, device="cpu")
            self.class_token = nn.Parameter(torch.empty(1, 1, 512, device="cpu"))
            self.position = nn.Parameter(torch.empty(1, 197, 512, device="cpu"))
            self.blocks = nn.ModuleList(VisualTransformerBlock() for _ in range(4))
            self._initialize(seed)

    def _initialize(self, seed: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for stage in self.cnn.stages:
            for layer in stage:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_in", nonlinearity="relu", generator=generator,
                    )
                    nn.init.zeros_(layer.bias)

        def linear(layer: nn.Linear) -> None:
            nn.init.xavier_uniform_(layer.weight, gain=1.0, generator=generator)
            nn.init.zeros_(layer.bias)

        def norm(layer: nn.LayerNorm) -> None:
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

        linear(self.projection)
        nn.init.normal_(self.class_token, mean=0.0, std=0.02, generator=generator)
        nn.init.normal_(self.position, mean=0.0, std=0.02, generator=generator)
        for block in self.blocks:
            norm(block.ln1)
            for layer in (block.attention.q, block.attention.k, block.attention.v, block.attention.out):
                linear(layer)
            norm(block.ln2)
            linear(block.fc1)
            linear(block.fc2)

    @property
    def device(self) -> torch.device:
        return self.projection.weight.device

    def encode_local(self, image: Tensor, return_skips: bool = True) -> LocalEncoding:
        _validate_tensor(image, "image", (3, 224, 224), self.device)
        return self.cnn(image, return_skips=return_skips)

    def tokenize(self, local: Tensor) -> Tensor:
        _validate_tensor(local, "local", (1024, 14, 14), self.device)
        spatial = self.projection(local.flatten(2).transpose(1, 2))
        tokens = torch.cat((spatial, self.class_token.expand(local.shape[0], -1, -1)), dim=1)
        return tokens + self.position

    def encode_global(self, local: Tensor) -> GlobalEncoding:
        tokens = self.tokenize(local)
        for block in self.blocks:
            tokens = block(tokens)
        return GlobalEncoding(tokens=tokens, global_token=tokens[:, -1:, :])


class MVE(nn.Module):
    """Image/reconstruction -> absolute residual -> shared encoding -> sum.

    The external reconstruction can be produced from encode_appearance's local
    features. Passing that AppearanceEncoding avoids recomputing the image role.
    Records belong to the caller and must not cross inputs or parameter updates.
    """

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.shared_encoder = SharedVisualEncoder(seed=seed)

    @classmethod
    def from_config(cls, config: ResolvedConfig) -> MVE:
        config.require_mve_approval()
        return cls(seed=config.proposal["initialization"]["seed"])

    def encode_appearance(self, image: Tensor) -> AppearanceEncoding:
        local = self.shared_encoder.encode_local(image, return_skips=True)
        global_encoding = self.shared_encoder.encode_global(local.local)
        return AppearanceEncoding(
            local=local.local, skips=local.skips, tokens=global_encoding.tokens,
            global_token=global_encoding.global_token,
        )

    def _validate_appearance(self, appearance: AppearanceEncoding, batch: int) -> None:
        if not isinstance(appearance, AppearanceEncoding):
            raise TypeError("appearance_encoding must be an AppearanceEncoding")
        device = self.shared_encoder.device
        values = [(appearance.local, "appearance.local", (1024, 14, 14)),
                  (appearance.tokens, "appearance.tokens", (197, 512)),
                  (appearance.global_token, "appearance.global_token", (1, 512))]
        if len(appearance.skips) != 4:
            raise ValueError("appearance_encoding must retain all four appearance skips")
        for index, (channels, size) in enumerate(((64, 224), (128, 112), (256, 56), (512, 28))):
            values.append((appearance.skips[index], f"appearance.skip{index}", (channels, size, size)))
        for tensor, name, shape in values:
            _validate_tensor(tensor, name, shape, device)
            if tensor.shape[0] != batch:
                raise ValueError(f"{name} batch must match image batch")

    def forward(
        self, image: Tensor, reconstruction: Tensor,
        appearance_encoding: AppearanceEncoding | None = None,
    ) -> MVEOutput:
        device = self.shared_encoder.device
        _validate_tensor(image, "image", (3, 224, 224), device)
        _validate_tensor(reconstruction, "reconstruction", (3, 224, 224), device)
        if reconstruction.shape != image.shape:
            raise ValueError("image and reconstruction must have identical batch/shape")
        if appearance_encoding is None:
            appearance_encoding = self.encode_appearance(image)
        else:
            self._validate_appearance(appearance_encoding, image.shape[0])
        residual = torch.abs(reconstruction - image)
        local_residual = self.shared_encoder.encode_local(residual, return_skips=False)
        global_residual = self.shared_encoder.encode_global(local_residual.local)
        return MVEOutput(
            appearance=appearance_encoding, residual_image=residual,
            residual_tokens=global_residual.tokens, residual_global=global_residual.global_token,
            fused_global=appearance_encoding.global_token + global_residual.global_token,
        )


def parameter_report(model: MVE) -> dict[str, int]:
    """Counts unique owned parameters; never a paper-fidelity criterion."""
    encoder = model.shared_encoder
    count = lambda module: sum(parameter.numel() for parameter in module.parameters())
    report = {
        "cnn": count(encoder.cnn), "projection": count(encoder.projection),
        "class_token": encoder.class_token.numel(), "position": encoder.position.numel(),
    }
    report.update({f"transformer_block_{i + 1}": count(block) for i, block in enumerate(encoder.blocks)})
    report["total_unique"] = count(model)
    return report
