import copy
import gc
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from mfvlr.integration import MFVLRIntegration
from mfvlr.language_encoder import LanguageAttention, LanguageFeedForward
from mfvlr.vim import VisionInjectionDelta
from mfvlr_research.decoder_a11 import build_variant, parity_report, save_checkpoint, load_checkpoint
from mfvlr_research.a11_config import FIXED, load_config, validate_config
from mfvlr_research.a11_evidence import state_hash


@pytest.mark.parametrize("variant", ["A0", "P3", "Q3"])
def test_a1_t01_t07_full_ownership_initialization_and_serialization(variant, tmp_path, evidence):
    reference = MFVLRIntegration()
    before = state_hash(reference); rng = torch.random.get_rng_state().clone()
    candidate = build_variant(reference, variant)
    assert torch.equal(rng, torch.random.get_rng_state())
    report = parity_report(reference, candidate)
    assert report["parameters"] == (154156232 if variant == "A0" else 154177736)
    assert report["parameter_tensors"] == (508 if variant == "A0" else 550)
    flt = candidate.flt
    assert len(flt.encoder.blocks) == 12 and len(flt.decoder.blocks) == 7
    vims = [m for m in flt.modules() if isinstance(m, VisionInjectionDelta)]
    assert len(vims) == 19 and len({id(m) for m in vims}) == 19
    for block in flt.decoder.blocks:
        assert type(block.self_attention) is type(block.memory_attention) is LanguageAttention
        assert type(block.ff) is LanguageFeedForward and type(block.vim) is VisionInjectionDelta
        assert len(list(block.parameters())) == (24 if variant == "A0" else 30)
        assert sum(p.numel() for p in block.parameters()) == (5249536 if variant == "A0" else 5252608)
        assert not any(isinstance(m, nn.Dropout) for m in block.modules())
    if variant != "A0":
        with pytest.raises(RuntimeError, match="Missing key"):
            candidate.load_state_dict(reference.state_dict(), strict=True)
    path = tmp_path / (variant + ".pt")
    save_checkpoint(candidate, path)
    with pytest.raises(FileExistsError): save_checkpoint(candidate, path)
    fingerprint = state_hash(candidate)
    with torch.no_grad(): candidate.flt.decoder.blocks[0].ff.fc1.weight.add_(1)
    load_checkpoint(candidate, path)
    assert state_hash(candidate) == fingerprint
    candidate.research_variant = "Q3" if variant != "Q3" else "P3"
    with pytest.raises(ValueError, match="variant"): load_checkpoint(candidate, path)
    candidate.research_variant = variant
    assert state_hash(reference) == before and torch.equal(rng, torch.random.get_rng_state())
    assert flt.vocabulary_output_weight is flt.vocabulary.weight
    report.update(status="PASS", reference_sha256=before, candidate_sha256=fingerprint,
                  reference_unchanged=True, caller_rng_unchanged=True, serialization="strict round trip PASS")
    evidence("ownership_"+variant, report)
    del reference, candidate
    gc.collect()


@pytest.mark.parametrize("variant", ["A0", "P3", "Q3"])
def test_a1_t07_strict_configuration(variant, tmp_path):
    data = load_config(Path("configs/research_a11") / (variant+".yaml"))
    for field, value in (("seed", True), ("full_experiments_enabled", True), ("unknown", 1),
                         ("variant", "I1"), ("variant", "R2"), ("initialization", "random")):
        bad = copy.deepcopy(data); bad[field] = value
        with pytest.raises(ValueError): validate_config(bad)
    for field, value in (("ln_eps", 1e-6), ("ln_axis", 0), ("ln_gamma", 2.0), ("fourth_norm", True)):
        bad = copy.deepcopy(data); bad["architecture"][field] = value
        with pytest.raises(ValueError): validate_config(bad)
    bad = copy.deepcopy(data); del bad["seed"]
    with pytest.raises(ValueError): validate_config(bad)
    path = tmp_path / "duplicate.yaml"
    path.write_text(yaml.safe_dump(data)+"variant: A0\n")
    with pytest.raises(ValueError, match="duplicate"): load_config(path)


def test_a1_t08_scope_only_i1_r2_not_implemented(accepted_flt):
    # T08's transformation/oracle is NOT APPLICABLE. This is a scope guard only.
    for variant in ("I1", "R2", "P3+I1", None):
        with pytest.raises(ValueError): build_variant(accepted_flt, variant)
    with pytest.raises(TypeError): build_variant(torch.nn.Linear(2,2), "P3")
