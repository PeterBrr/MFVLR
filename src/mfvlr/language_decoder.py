"""Phase 2C.2 LD baseline, MFVLR III-D p. 7 Eqs. (12)--(16).

Causal self-attention, full LE-memory attention, FF, then VIM; four residuals.
No LD normalization. Vocabulary projection/shift ownership is external.
"""
from torch import Tensor, nn

from .language_encoder import LanguageAttention, LanguageFeedForward, finite
from .vim import VisionInjectionDelta


class LanguageDecoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attention = LanguageAttention()
        self.memory_attention = LanguageAttention()
        self.ff = LanguageFeedForward()
        self.vim = VisionInjectionDelta()

    def forward(self, state: Tensor, memory: Tensor, visual: Tensor) -> Tensor:
        masked = finite(state+self.self_attention(state,causal=True),"LD.self_residual")
        communicated = finite(masked+self.memory_attention(masked,memory),"LD.memory_residual")
        fed = finite(communicated+self.ff(communicated),"LD.ff_residual")
        return finite(fed+self.vim(fed,visual),"LD.vim_residual")


class LanguageDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(LanguageDecoderBlock() for _ in range(7))

    def forward(self, state: Tensor, memory: Tensor, visual: Tensor) -> Tensor:
        for index,block in enumerate(self.blocks,1):
            try:
                state = block(state,memory,visual)
            except FloatingPointError as error:
                raise FloatingPointError(f"LD block {index}: {error}") from error
        return state
