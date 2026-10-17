"""Phase 2B approved Option B, not a claim of recovered author internals.

MFVLR III-C p. 6 / Fig. 4 p. 5: shared AD/MD decoder and distinct heads.
Exact engineering topology/initialization: phase2b_decoder_contract.md D01-D08.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .mve import AppearanceEncoding

if TYPE_CHECKING:
    from .decoder_config import DecoderConfig


@dataclass(frozen=True)
class VisionDecoderOutput:
    features: Tensor
    reconstruction: Tensor
    mask_logits: Tensor


class UpsamplingStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2,
                                    padding=0, output_padding=0, bias=True, device="cpu")
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True, device="cpu")
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True, device="cpu")
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, state: Tensor) -> Tensor:
        return self.relu2(self.conv2(self.relu1(self.conv1(self.up(state)))))


class DecoderTrunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.ModuleList(UpsamplingStage(cin, cout) for cin, cout in
                                    ((1024, 512), (512, 256), (256, 128), (128, 64)))

    def forward(self, local: Tensor) -> Tensor:
        for stage in self.stages:
            local = stage(local)
        return local


class VisionDecoder(nn.Module):
    """One deterministic trunk, separate identity AD and raw-logit MD heads.

Policy must be explicit. Only approved bottleneck_only is implemented.
The wrapper deliberately reads only appearance.local, leaving accepted MVE
record validation and all skip/token/global tensors untouched.
"""

    def __init__(self, *, skip_policy: str, seed: int = 0) -> None:
        super().__init__()
        if skip_policy != "bottleneck_only":
            raise ValueError("Only approved Option B bottleneck_only is implemented")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.skip_policy = skip_policy
        with torch.random.fork_rng(devices=[]):
            self.trunk = DecoderTrunk()
            self.appearance_head = nn.Conv2d(64, 3, 1, bias=True, device="cpu")
            self.mask_head = nn.Conv2d(64, 2, 1, bias=True, device="cpu")
            generator = torch.Generator(device="cpu").manual_seed(seed)
            for stage in self.trunk.stages:
                nn.init.xavier_uniform_(stage.up.weight, gain=1.0, generator=generator)
                nn.init.zeros_(stage.up.bias)
                for layer in (stage.conv1, stage.conv2):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu", generator=generator)
                    nn.init.zeros_(layer.bias)
            for head in (self.appearance_head, self.mask_head):
                nn.init.xavier_uniform_(head.weight, gain=1.0, generator=generator)
                nn.init.zeros_(head.bias)

    @classmethod
    def from_config(cls, config: DecoderConfig) -> VisionDecoder:
        config.require_approval()
        return cls(skip_policy=config.proposal["decoder"]["skip_policy"],
                   seed=config.proposal["initialization"]["seed"])

    def forward(self, local: Tensor, *, skips=None) -> VisionDecoderOutput:
        if skips is not None:
            raise ValueError("bottleneck_only rejects explicit skip inputs")
        if not isinstance(local, Tensor):
            raise TypeError("local must be a tensor")
        if local.ndim != 4 or tuple(local.shape[1:]) != (1024, 14, 14) or local.shape[0] < 1:
            raise ValueError("local must have shape [b,1024,14,14] with b >= 1")
        if local.dtype != torch.float32 or self.appearance_head.weight.dtype != torch.float32:
            raise TypeError("VD requires float32; implicit conversion is disabled")
        if local.device != self.appearance_head.weight.device:
            raise ValueError("local device must match VD device")
        if not torch.isfinite(local).all().item():
            raise ValueError("local must be finite")
        features = self.trunk(local)
        reconstruction = self.appearance_head(features)
        mask_logits = self.mask_head(features)
        if not all(torch.isfinite(tensor).all().item() for tensor in (features, reconstruction, mask_logits)):
            raise ValueError("VD outputs must be finite")
        return VisionDecoderOutput(features, reconstruction, mask_logits)

    def decode_appearance(self, appearance: AppearanceEncoding) -> VisionDecoderOutput:
        if not isinstance(appearance, AppearanceEncoding):
            raise TypeError("appearance must be an AppearanceEncoding")
        return self(appearance.local)


def decoder_parameter_report(decoder: VisionDecoder) -> dict[str, int]:
    def count(module):
        return sum(p.numel() for p in module.parameters())
    result = {f"U{i+1}": count(stage) for i, stage in enumerate(decoder.trunk.stages)}
    result.update(trunk=count(decoder.trunk), appearance_head=count(decoder.appearance_head),
                  mask_head=count(decoder.mask_head), total_unique=count(decoder))
    return result
