"""A1-D01 research assumptions: A0/P3/Q3, A1.1 canonical affine LayerNorm.

MFVLR III-D p.7 Eqs.(12)--(16) is the reference, not evidence for added norms.
No production file, initializer or ordinary attention/FF/VIM is redefined.
"""
from __future__ import annotations

import copy
import hashlib

import torch
from torch import nn

from mfvlr.flt import FineGrainedLanguageTransformer
from mfvlr.integration import MFVLRIntegration
from mfvlr.language_decoder import LanguageDecoder, LanguageDecoderBlock
from mfvlr.language_encoder import finite

VARIANTS = ("A0", "P3", "Q3")


def require_variant(variant):
    if type(variant) is not str or variant not in VARIANTS:
        raise ValueError("A1-D01 permits only A0, P3, Q3; I1/R2 are unapproved")


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class NormalizedDecoderBlock(nn.Module):
    """Exactly three independent norms; VIM delta remains outside every norm."""

    def __init__(self, owned_block: LanguageDecoderBlock, variant: str):
        super().__init__()
        if variant not in ("P3", "Q3"):
            raise ValueError("normalized block must be P3 or Q3")
        if type(owned_block) is not LanguageDecoderBlock:
            raise TypeError("requires an accepted decoder block owned by the builder")
        self.variant = variant
        self.self_attention = owned_block.self_attention
        self.memory_attention = owned_block.memory_attention
        self.ff = owned_block.ff
        self.vim = owned_block.vim
        parameter = self.self_attention.q.weight
        # LayerNorm reset is constant (ones/zeros), with no random draws.
        for name in ("ln1", "ln2", "ln3"):
            setattr(self, name, nn.LayerNorm(512, eps=1e-5, elementwise_affine=True,
                                            device=parameter.device, dtype=parameter.dtype))

    @classmethod
    def from_reference(cls, reference, variant):
        return cls(copy.deepcopy(reference), variant)

    def forward(self, state, memory, visual):
        if self.variant == "P3":
            a = finite(state + self.self_attention(self.ln1(state), causal=True), "P3.self_residual")
            b = finite(a + self.memory_attention(self.ln2(a), memory), "P3.memory_residual")
            c = finite(b + self.ff(self.ln3(b)), "P3.ff_residual")
        else:
            a = finite(self.ln1(state + self.self_attention(state, causal=True)), "Q3.self_residual")
            b = finite(self.ln2(a + self.memory_attention(a, memory)), "Q3.memory_residual")
            c = finite(self.ln3(b + self.ff(b)), "Q3.ff_residual")
        return finite(c + self.vim(c, visual), self.variant + ".vim_residual")


def build_variant(reference, variant):
    """Copy accepted FP32 state; the caller's reference/RNG are never modified.

    Frozen means unchanged reference values, not requires_grad=False: all task
    gradients remain available. Only exact accepted root/FLT types are allowed.
    Common names/shapes/bytes and independent ownership are verified, not inferred
    from aggregate counts. No checkpoint permissiveness or random reinitialization.
    """
    require_variant(variant)
    if type(reference) not in (MFVLRIntegration, FineGrainedLanguageTransformer):
        raise TypeError("requires an accepted root or full FLT reference")
    if hasattr(reference, "research_variant"):
        raise ValueError("use the original accepted reference, not another research variant")
    if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in reference.parameters()):
        raise TypeError("copy CPU FP32-initialized reference first, promote copies later")
    result = copy.deepcopy(reference)
    flt = result.flt if type(result) is MFVLRIntegration else result
    if type(flt.decoder) is not LanguageDecoder:
        raise TypeError("accepted decoder required")
    if variant != "A0":
        flt.decoder.blocks = nn.ModuleList(NormalizedDecoderBlock(b, variant) for b in flt.decoder.blocks)
    result.research_variant = variant
    parity_report(reference, result)
    return result


def parity_report(reference, candidate):
    source = dict(reference.named_parameters(remove_duplicate=False))
    target = dict(candidate.named_parameters(remove_duplicate=False))
    source_ptrs = {p.data_ptr() for p in source.values()}
    if len({id(p) for p in target.values()}) != len(target) or len({p.data_ptr() for p in target.values()}) != len(target):
        raise AssertionError("duplicate candidate parameter/storage ownership")
    common = {}
    for name, original in source.items():
        copied = target[name]
        if original.shape != copied.shape or original.dtype != copied.dtype:
            raise AssertionError("common name/shape/dtype mismatch: " + name)
        digest = tensor_digest(original)
        if digest != tensor_digest(copied) or copied is original or copied.data_ptr() in source_ptrs:
            raise AssertionError("common bytes/ownership mismatch: " + name)
        common[name] = {"shape": list(original.shape), "sha256": digest, "byte_equal": True,
                        "independent_parameter_and_storage": True}
    extra = sorted(set(target) - set(source))
    prefix = "flt." if type(candidate) is MFVLRIntegration else ""
    expected = [] if candidate.research_variant == "A0" else sorted(
        f"{prefix}decoder.blocks.{i}.ln{j}.{suffix}" for i in range(7)
        for j in (1, 2, 3) for suffix in ("weight", "bias"))
    if extra != expected:
        raise AssertionError("unexpected added parameter inventory")
    for name in extra:
        p = target[name]
        if p.shape != (512,) or not torch.equal(p, torch.full_like(p, 1 if name.endswith("weight") else 0)):
            raise AssertionError("norm initialization mismatch: " + name)
    flt = candidate.flt if type(candidate) is MFVLRIntegration else candidate
    norms = [m for m in flt.decoder.modules() if isinstance(m, nn.LayerNorm)]
    if len(norms) != (0 if candidate.research_variant == "A0" else 21):
        raise AssertionError("norm module inventory mismatch")
    total = sum(p.numel() for p in target.values())
    root = type(candidate) is MFVLRIntegration
    expected_total = (154156232 if root else 112758272) + (21504 if extra else 0)
    expected_tensors = (508 if root else 387) + len(extra)
    if total != expected_total or len(target) != expected_tensors:
        raise AssertionError("parameter count mismatch")
    if flt.vocabulary_output_weight is not flt.vocabulary.weight:
        raise AssertionError("vocabulary tie broken")
    return {"variant": candidate.research_variant, "common": common, "new_names": extra,
            "norm_modules": len(norms), "parameter_tensors": len(target), "parameters": total,
            "vocabulary_tied": True, "vim_roles": [f"LE.{i}" for i in range(12)] + [f"LD.{i}" for i in range(7)]}


def save_checkpoint(model, path):
    """Variant identity is required in addition to strict tensor keys."""
    require_variant(model.research_variant)
    with open(path, "xb") as handle:
        torch.save({"schema": "a11-decoder-v1", "variant": model.research_variant,
                    "state_dict": model.state_dict()}, handle)


def load_checkpoint(model, path):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if set(record) != {"schema", "variant", "state_dict"} or record["schema"] != "a11-decoder-v1":
        raise ValueError("invalid research checkpoint schema")
    if record["variant"] != model.research_variant:
        raise ValueError("checkpoint variant mismatch")
    # No strict=False fallback; A0 lacks the mandatory P3/Q3 LN keys.
    model.load_state_dict(record["state_dict"], strict=True)
