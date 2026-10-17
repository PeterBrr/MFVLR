"""Approved Phase 2C.2 LE engineering baseline; MFVLR III-D pp. 6--7.

No pretrained tensors. Exactly one pre-attention LN per encoder block.
The accepted standalone VIM is reused unchanged, with caller-owned residuals.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .vim import VisionInjectionDelta


def finite(value: Tensor, name: str) -> Tensor:
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"Nonfinite language tensor: {name}")
    return value


def validate_hidden(value: Tensor, name: str, parameter: Tensor, *, length: int = 308) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 3 or value.shape[0] < 1 or tuple(value.shape[1:]) != (length, 512):
        raise ValueError(f"{name} must have positive [batch,{length},512] shape")
    if value.dtype not in (torch.float32, torch.float64) or value.dtype != parameter.dtype:
        raise TypeError(f"{name} must match FP32 module dtype (FP64 analytical copies allowed)")
    if value.device != parameter.device:
        raise ValueError(f"{name} must match module device")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain finite values")


class LanguageAttention(nn.Module):
    """Ordinary attention: separate biased maps, eight heads, sqrt(64) scale.

This scale is for ordinary MHA only. VIM retains its published sqrt(r/d).
No padding mask, dropout, fused kernel, normalization or cross-batch mixing.
"""
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(512, 512, bias=True, device="cpu", dtype=torch.float32)
        self.k = nn.Linear(512, 512, bias=True, device="cpu", dtype=torch.float32)
        self.v = nn.Linear(512, 512, bias=True, device="cpu", dtype=torch.float32)
        self.out = nn.Linear(512, 512, bias=True, device="cpu", dtype=torch.float32)

    def forward(self, query: Tensor, memory: Tensor | None = None, *, causal: bool = False) -> Tensor:
        return self.forward_with_trace(query, memory, causal=causal)[0]

    def forward_with_trace(self, query: Tensor, memory: Tensor | None = None, *, causal: bool = False):
        # Low-level primitive permits shorter analytical fixtures; public LE fixes 308.
        for tensor, name in ((query, "query"), (query if memory is None else memory, "memory")):
            if not isinstance(tensor, Tensor):
                raise TypeError(f"attention {name} must be a tensor")
            if tensor.ndim != 3 or tensor.shape[0] < 1 or tensor.shape[1] < 1 or tensor.shape[2] != 512:
                raise ValueError(f"attention {name} must be positive [batch,length,512]")
            if tensor.dtype != self.q.weight.dtype or tensor.dtype not in (torch.float32,torch.float64):
                raise TypeError(f"attention {name} dtype must match the module")
            if tensor.device != self.q.weight.device:
                raise ValueError(f"attention {name} device must match the module")
            finite(tensor, f"attention.{name}")
        if type(causal) is not bool:
            raise TypeError("causal must be boolean")
        memory = query if memory is None else memory
        if memory.shape[0] != query.shape[0]:
            raise ValueError("attention batches must match")
        if causal and memory.shape[1] != query.shape[1]:
            raise ValueError("causal self-attention lengths must match")
        batch, nq, _ = query.shape
        nk = memory.shape[1]
        q = self.q(query).reshape(batch,nq,8,64).transpose(1,2)
        k = self.k(memory).reshape(batch,nk,8,64).transpose(1,2)
        v = self.v(memory).reshape(batch,nk,8,64).transpose(1,2)
        logits = finite((q @ k.transpose(-2,-1)) / math.sqrt(64), "attention.raw_logits")
        if causal:
            future = torch.ones((nq,nk), dtype=torch.bool, device=query.device).triu(1)
            logits = logits.masked_fill(future, float("-inf"))
        probabilities = finite(logits.softmax(dim=-1), "attention.probabilities")
        context = (probabilities @ v).transpose(1,2).reshape(batch,nq,512)
        output = finite(self.out(context), "attention.output")
        return output, {"query":q,"key":k,"value":v,"logits":logits,
                        "probabilities":probabilities,"context":context}


class LanguageFeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(512,2048,bias=True,device="cpu",dtype=torch.float32)
        self.gelu = nn.GELU(approximate="none")
        self.fc2 = nn.Linear(2048,512,bias=True,device="cpu",dtype=torch.float32)

    def forward(self, state: Tensor) -> Tensor:
        hidden = finite(self.gelu(self.fc1(state)), "ff.gelu")
        return finite(self.fc2(hidden), "ff.output")


class LanguageEncoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(512,eps=1e-5,elementwise_affine=True,device="cpu",dtype=torch.float32)
        self.attention = LanguageAttention()
        self.vim = VisionInjectionDelta()
        self.ff = LanguageFeedForward()

    def forward(self, state: Tensor, visual: Tensor) -> Tensor:
        tokens = finite(state + self.attention(self.ln(state)), "LE.attention_residual")
        injected = finite(tokens + self.vim(tokens,visual), "LE.vim_residual")
        return finite(injected + self.ff(injected), "LE.ff_residual")


class LanguageEncoder(nn.Module):
    """Twelve independent blocks, full bidirectional 308-position context."""
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(LanguageEncoderBlock() for _ in range(12))

    def forward(self, state: Tensor, visual: Tensor) -> Tensor:
        for index, block in enumerate(self.blocks,1):
            try:
                state = block(state,visual)
            except FloatingPointError as error:
                raise FloatingPointError(f"LE block {index}: {error}") from error
        return state


def initialize_linear(layer: nn.Linear, generator: torch.Generator) -> None:
    nn.init.xavier_uniform_(layer.weight,gain=1.0,generator=generator)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def initialize_attention(module, generator):
    for layer in (module.q,module.k,module.v,module.out):
        initialize_linear(layer,generator)


@dataclass(frozen=True)
class LanguageEncoding:
    low: Tensor
    encoder_hidden: Tensor
    global_language: Tensor


class LanguageEncoderStage(nn.Module):
    """Stage-1 owner with the exact full-FLT vocabulary/position draw prefix.

Pd is reserved and initialized here to preserve Wvoc,Pe,Pd,LE draw ordering.
It is not consumed by LE and no Language Decoder exists in this stage.
"""
    def __init__(self, *, seed: int = 0) -> None:
        super().__init__()
        self._validate_seed(seed)
        with torch.random.fork_rng(devices=[]):
            self._build_encoder_owner()
            self._initialize_encoder_owner(torch.Generator(device="cpu").manual_seed(seed))

    @staticmethod
    def _validate_seed(seed):
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0,2**63)")

    def _build_encoder_owner(self):
        self.vocabulary = nn.Embedding(49408,512,padding_idx=None,device="cpu",dtype=torch.float32)
        self.encoder_position = nn.Parameter(torch.empty(1,308,512,dtype=torch.float32,device="cpu"))
        self.decoder_position = nn.Parameter(torch.empty(1,308,512,dtype=torch.float32,device="cpu"))
        self.encoder = LanguageEncoder()

    def _initialize_encoder_owner(self,generator):
        for parameter in (self.vocabulary.weight,self.encoder_position,self.decoder_position):
            nn.init.normal_(parameter,mean=0.0,std=0.02,generator=generator)
        for block in self.encoder.blocks:
            nn.init.ones_(block.ln.weight)
            nn.init.zeros_(block.ln.bias)
            initialize_attention(block.attention,generator)
            initialize_attention(block.vim,generator)
            initialize_linear(block.ff.fc1,generator)
            initialize_linear(block.ff.fc2,generator)

    def _validate_inputs(self, ids: Tensor, visual: Tensor):
        if not isinstance(ids, Tensor):
            raise TypeError("ids must be a tensor")
        if ids.ndim != 2 or ids.shape[0] < 1 or ids.shape[1] != 308:
            raise ValueError("ids must have positive [batch,308] shape")
        if ids.dtype != torch.int64:
            raise TypeError("ids must be int64")
        if ids.device != self.vocabulary.weight.device:
            raise ValueError("ids must match module device")
        if ids.min().item() < 0 or ids.max().item() >= 49408:
            raise ValueError("ids outside vocabulary [0,49408)")
        validate_hidden(visual,"visual",self.vocabulary.weight,length=1)
        if visual.shape[0] != ids.shape[0]:
            raise ValueError("ids and visual batches must match")

    def encode(self, ids: Tensor, visual: Tensor) -> LanguageEncoding:
        self._validate_inputs(ids,visual)
        low = finite(self.vocabulary(ids),"lookup")
        initial = finite(low+self.encoder_position,"LE.position_sum")
        hidden = self.encoder(initial,visual)
        return LanguageEncoding(low,hidden,hidden[:,307:308,:])

    def forward(self, ids: Tensor, visual: Tensor) -> LanguageEncoding:
        return self.encode(ids,visual)
