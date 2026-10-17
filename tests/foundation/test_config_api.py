"""Scoped L11--L13 approval/configuration/serialization tests, no full FLT."""
import copy
import hashlib
import io
import json
import shutil

import pytest
import torch
import yaml

from mfvlr.config import ConfigError
from mfvlr.foundation_config import load_foundation_config
from mfvlr.prompts import PromptBuilder, PromptMetadata
from mfvlr.tokenizer import TokenizerAdapter
from mfvlr.vim import VisionInjectionDelta


def test_fdn_c01_profiles(root):
    reference = load_foundation_config(root/"configs/foundation_reference.yaml")
    debug = load_foundation_config(root/"configs/foundation_debug.yaml")
    reference.require_approval()
    debug.require_approval()
    left, right = reference.data, debug.data
    for key in ("profile","diagnostics"):
        left.pop(key)
        right.pop(key)
    assert left == right
    assert left["approval"]["full_flt_approved"] is False
    assert "standalone" in left["approval"]["approved_portions"]["L09"]
    mutable = reference.data
    mutable["vim"]["heads"] = 16
    assert reference.data["vim"]["heads"] == 8
    record = PromptBuilder.from_config(debug).build([PromptMetadata("real","real")])
    assert TokenizerAdapter.from_config(debug).encode_levels(record).ids.shape == (1,308)
    assert sum(p.numel() for p in VisionInjectionDelta.from_config(debug).parameters()) == 1048576


@pytest.mark.parametrize("mutation", ["unknown","missing","bool_int","heads","scope","tolerance","asset","bias","sequence","seed"])
def test_fdn_c02_reference_rejects_changes(root,tmp_path,mutation):
    data = load_foundation_config(root/"configs/foundation_reference.yaml").data
    if mutation == "unknown": data["language_encoder"] = {"blocks":12}
    elif mutation == "missing": data.pop("vim")
    elif mutation == "bool_int": data["schema_version"] = True
    elif mutation == "heads": data["vim"]["heads"] = 4
    elif mutation == "scope": data["approval"]["full_flt_approved"] = True
    elif mutation == "tolerance": data["tolerances"]["fp32_atol"] = 5e-6
    elif mutation == "asset": data["tokenizer"]["asset_manifest_sha256"] = "changed"
    elif mutation == "bias": data["vim"]["bias"] = True
    elif mutation == "sequence": data["tokenizer"]["sequence_length"] = 77
    elif mutation == "seed": data["initialization"]["seed"] = True
    path = tmp_path/"bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError): load_foundation_config(path)


@pytest.mark.parametrize("mutation", ["topology","approval","base","cycle","nested","duplicate","null","diagnostic_type"])
def test_fdn_c03_strict_overlay(root,tmp_path,mutation):
    shutil.copyfile(root/"configs/foundation_reference.yaml",tmp_path/"foundation_reference.yaml")
    data = yaml.safe_load((root/"configs/foundation_debug.yaml").read_text())
    if mutation == "topology": data["overrides"]["vim"] = {"heads":4}
    elif mutation == "approval": data["approval"]["decisions"]["L04"] = "APPROVED"
    elif mutation == "base": data["base"] = "../foundation_reference.yaml"
    elif mutation in ("cycle","nested"):
        (tmp_path/"foundation_reference.yaml").write_text(yaml.safe_dump(data))
    elif mutation == "null": data["overrides"]["diagnostics"] = None
    elif mutation == "diagnostic_type": data["overrides"]["diagnostics"]["batch_size"] = True
    path = tmp_path/("foundation_reference.yaml" if mutation == "cycle" else "debug.yaml")
    text = yaml.safe_dump(data)
    if mutation == "duplicate": text += "profile: debug\n"
    path.write_text(text)
    with pytest.raises(ConfigError): load_foundation_config(path)


def test_fdn_c04_serialization_and_eval():
    module = VisionInjectionDelta()
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(2,308,512,generator=generator)
    v = torch.randn(2,1,512,generator=generator)
    before = module(x,v)
    buffer = io.BytesIO()
    torch.save(module.state_dict(),buffer)
    buffer.seek(0)
    loaded = VisionInjectionDelta(seed=9)
    loaded.load_state_dict(torch.load(buffer,weights_only=True))
    assert torch.equal(before,loaded(x,v))
    assert torch.equal(before,loaded.eval()(x,v))
    assert len(loaded.state_dict()) == 4
    assert not any('trace' in name or 'cache' in name for name in vars(loaded))


def test_fdn_c05_accepted_visual_preservation(root):
    manifest = json.loads((root/"artifacts/phase2b/manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name


def test_fdn_c06_visual_feature_boundary_and_detach(model):
    # Uses the accepted output tensor boundary; no production visual module is edited.
    g = torch.Generator().manual_seed(121)
    image = torch.rand(1,3,224,224,generator=g)
    reconstruction = (image*0.9+0.02).detach().requires_grad_()
    appearance = model.encode_appearance(image)
    visual = model(image,reconstruction,appearance_encoding=appearance)
    module = VisionInjectionDelta()
    language = torch.ones(1,308,512)
    module(language,visual.fused_global).square().mean().backward()
    assert reconstruction.grad is not None and reconstruction.grad.isfinite().all() and reconstruction.grad.norm() > 0
    assert module.q.weight.grad.count_nonzero() == module.k.weight.grad.count_nonzero() == 0
    model.zero_grad(set_to_none=True)
    reconstruction.grad = None
    module.zero_grad(set_to_none=True)
    module(language,visual.fused_global.detach()).square().mean().backward()
    assert reconstruction.grad is None
    assert all(p.grad is None for p in model.parameters())
    assert module.v.weight.grad.norm() > 0
