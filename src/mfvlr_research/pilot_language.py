"""Agreed language ablations: last EOS, target-PAD masking, normalized CMC."""
import torch
from torch import nn
from torch.nn import functional as F

from mfvlr.flt import FineGrainedLanguageTransformer
from mfvlr.language_encoder import LanguageEncoding, LanguageEncoderBlock, finite
from mfvlr.language_decoder import LanguageDecoderBlock
from mfvlr.vim import VisionInjectionDelta
from mfvlr.objectives import (CrossModalContrastiveLoss, floating, language_token_nll, scalar)

# IDs of the pinned CLIP tokenizer (mfvlr.tokenizer); no tokenizer initialization.
EOS_ID, PAD_ID = 49407, 0


def last_eos_pool(hidden, token_ids):
    """Return [B,1,D] at the last actual EOS, including multi-segment prompts."""
    if hidden.ndim != 3 or token_ids.ndim != 2 or hidden.shape[:2] != token_ids.shape:
        raise ValueError('Hidden and token IDs must have matching [batch, sequence] dimensions')
    if token_ids.dtype != torch.int64 or token_ids.device != hidden.device:
        raise ValueError('Token IDs must be int64 on the hidden-state device')
    if not token_ids.shape[0] or not token_ids.shape[1]:
        raise ValueError('EOS pooling requires a nonempty batch and sequence')
    positions = torch.arange(token_ids.shape[1], device=token_ids.device).expand_as(token_ids)
    last = positions.masked_fill(token_ids != EOS_ID, -1).amax(dim=1)
    if (last < 0).any().item():
        raise ValueError('Each prompt sequence must contain at least one actual EOS')
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last].unsqueeze(1)


class ResidualVisionInjection(VisionInjectionDelta):
    """Return x+delta; existing LE/LD caller adds x, giving exactly 2x+delta."""
    def forward(self, language, visual):
        return language + super().forward(language, visual)


def configure_vim_residuals(flt, mode):
    if mode not in ('single', 'double'):
        raise ValueError('vim_residual must be single or double')
    cls = ResidualVisionInjection if mode == 'double' else VisionInjectionDelta
    for block in [*flt.encoder.blocks, *flt.decoder.blocks]:
        old = block.vim
        if type(old) is cls:
            continue
        replacement = cls().to(device=old.q.weight.device, dtype=old.q.weight.dtype)
        replacement.load_state_dict(old.state_dict(), strict=True)
        replacement.train(old.training)
        block.vim = replacement


class PreNormEncoderBlock(LanguageEncoderBlock):
    def __init__(self, factor):
        super().__init__()
        self.norm_vim = nn.LayerNorm(512, eps=1e-5)
        self.norm_ff = nn.LayerNorm(512, eps=1e-5)
        self.residual_factor = factor

    def forward(self, state, visual):
        tokens = finite(state + self.attention(self.ln(state)), 'LE.attention_residual')
        delta = self.vim.forward_with_trace(self.norm_vim(tokens), visual).delta
        injected = finite(self.residual_factor * tokens + delta, 'LE.vim_residual')
        return finite(injected + self.ff(self.norm_ff(injected)), 'LE.ff_residual')


class PreNormDecoderBlock(LanguageDecoderBlock):
    def __init__(self, factor):
        super().__init__()
        self.norm_self = nn.LayerNorm(512, eps=1e-5)
        self.norm_memory = nn.LayerNorm(512, eps=1e-5)
        self.norm_ff = nn.LayerNorm(512, eps=1e-5)
        self.norm_vim = nn.LayerNorm(512, eps=1e-5)
        self.residual_factor = factor

    def forward(self, state, memory, visual):
        masked = finite(state + self.self_attention(self.norm_self(state), causal=True), 'LD.self_residual')
        communicated = finite(masked + self.memory_attention(self.norm_memory(masked), memory), 'LD.memory_residual')
        fed = finite(communicated + self.ff(self.norm_ff(communicated)), 'LD.ff_residual')
        delta = self.vim.forward_with_trace(self.norm_vim(fed), visual).delta
        return finite(self.residual_factor * fed + delta, 'LD.vim_residual')


def configure_language_normalization(flt, mode, residual):
    if mode not in ('baseline', 'pre_norm'):
        raise ValueError('language_normalization must be baseline or pre_norm')
    if mode == 'baseline':
        return
    factor = 2 if residual == 'double' else 1
    with torch.random.fork_rng(devices=[]):
        for owner, cls in ((flt.encoder, PreNormEncoderBlock), (flt.decoder, PreNormDecoderBlock)):
            for i, old in enumerate(owner.blocks):
                block = cls(factor).to(device=old.vim.q.weight.device, dtype=old.vim.q.weight.dtype)
                mismatch = block.load_state_dict(old.state_dict(), strict=False)
                assert not mismatch.unexpected_keys and all(k.startswith('norm_') for k in mismatch.missing_keys)
                owner.blocks[i] = block
    flt.encoder_final_norm = nn.LayerNorm(512, eps=1e-5)
    flt.decoder_final_norm = nn.LayerNorm(512, eps=1e-5)


class EosFineGrainedLanguageTransformer(FineGrainedLanguageTransformer):
    """Keep all LE/LD operations and parameters; change only the global readout."""
    def __init__(self, *, seed=0, vim_residual='single', language_normalization='baseline'):
        super().__init__(seed=seed)
        configure_vim_residuals(self, vim_residual)
        configure_language_normalization(self, language_normalization, vim_residual)

    def encode(self, ids, visual):
        encoded = super().encode(ids, visual)
        hidden = self.encoder_final_norm(encoded.encoder_hidden) if hasattr(self, 'encoder_final_norm') else encoded.encoder_hidden
        return LanguageEncoding(encoded.low, hidden, last_eos_pool(hidden, ids))

    def decode(self, low, encoder_hidden, visual):
        hidden = super().decode(low, encoder_hidden, visual)
        return self.decoder_final_norm(hidden) if hasattr(self, 'decoder_final_norm') else hidden


def nonpad_language_reconstruction_loss(logits, token_ids):
    """Exclude only target PAD; retain token SUM then batch MEAN, BOS and EOS."""
    nll = language_token_nll(logits, token_ids)
    return scalar(nll.masked_fill(token_ids == PAD_ID, 0.).sum(dim=1).mean(), 'LR_nonpad_sum')


class NormalizedCrossModalContrastiveLoss(CrossModalContrastiveLoss):
    """Preserve learned temperature/diagonal targets; L2 only the CMC operands."""
    def scores(self, visual, language):
        floating(visual, 'visual', (1, 512))
        floating(language, 'global_language', (1, 512))
        v = F.normalize(visual, p=2, dim=-1, eps=1e-12)
        t = F.normalize(language, p=2, dim=-1, eps=1e-12)
        return super().scores(v, t)
