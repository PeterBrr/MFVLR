"""Applicable FLT-T02--T05 foundation cases; no embedding/LE/LD creation."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import socket

import pytest
import torch

from mfvlr.prompts import PromptBuilder, PromptMetadata, GENERATOR_REGISTRY
from mfvlr.tokenizer import TokenizerAdapter, verify_assets, ASSET_ROOT, ASSET_MANIFEST_SHA256

FIXTURES = json.loads((Path(__file__).parent / "fixtures/golden_tokens.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tokenizer():
    return TokenizerAdapter()


def test_flt_t02_prompt_golden_and_order():
    cases = FIXTURES["prompt_cases"]
    records = [PromptMetadata(**case["metadata"]) for case in cases]
    builder = PromptBuilder()
    assert builder.build(records) == tuple(tuple(case["prompts"]) for case in cases)
    assert len(GENERATOR_REGISTRY) == 9
    real = records[0]
    assert builder.build([replace(real, generator="unused", family="ignored", manipulation="unknown")]) == builder.build([real])
    assert builder.build(list(reversed(records))) == tuple(reversed(builder.build(records)))


@pytest.mark.parametrize("changes", [
    {"authenticity":True}, {"authenticity":"unknown"}, {"sample_id":""},
    {"generator":None}, {"generator":"LatDiff"}, {"generator":"DDPM "},
    {"generator":"<|endoftext|>"}, {"family":None}, {"family":"GAN"},
    {"family":"Diffusion"}, {"manipulation":"FS"}, {"manipulation":False},
])
def test_flt_t02_invalid_metadata(changes):
    valid = PromptMetadata("sample","fake","EFS","diffusion","DDPM")
    with pytest.raises(ValueError):
        PromptBuilder().build([replace(valid,**changes)])


def test_flt_t02_invalid_batch():
    for value in ("text", [dict(authenticity="real")], 3):
        with pytest.raises(TypeError):
            PromptBuilder().build(value)
    with pytest.raises(ValueError):
        PromptBuilder().build([])


def test_flt_t03_pins_and_offline(tokenizer, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("runtime tokenizer attempted network access")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    manifest = verify_assets()
    assert manifest["commit"] == "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
    assert manifest["vocabulary"]["size"] == 49408
    assert manifest["vocabulary"]["bos"] == 49406 and manifest["vocabulary"]["eos"] == 49407
    assert manifest["learned_model_weights_downloaded"] is False
    assert FIXTURES["asset_manifest_sha256"] == ASSET_MANIFEST_SHA256
    assert hashlib.sha256((Path(__file__).parent/"fixtures/golden_tokens.json").read_bytes()).hexdigest() == "1e8950d386d765c5cd13e628e751356afe4ac11e35482701eadc7a696761d0e6"
    assert TokenizerAdapter().encode_text("A photo of a real face") == tokenizer.encode_text("A photo of a real face")


@pytest.mark.parametrize("index", range(len(FIXTURES["text_cases"])))
def test_flt_t03_independent_payload_golden(tokenizer, index):
    case = FIXTURES["text_cases"][index]
    assert tokenizer.encode_text(case["text"]) == case["payload_ids"]
    if case["overflow"]:
        with pytest.raises(ValueError, match="truncation forbidden"):
            tokenizer.encode_levels([[case["text"]]*4])
    else:
        encoded = tokenizer.encode_levels([[case["text"]]*4])
        assert encoded.segment_ids[0,0].tolist() == case["segment"]


def test_flt_t03_tampered_and_missing_assets(tmp_path):
    root = tmp_path/"assets"
    shutil.copytree(ASSET_ROOT, root)
    bpe = root/"clip/bpe_simple_vocab_16e6.txt.gz"
    bpe.write_bytes(bpe.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        TokenizerAdapter(asset_root=root)
    with pytest.raises(ValueError, match="missing"):
        TokenizerAdapter(asset_root=tmp_path/"missing")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        (root/"manifest.json").write_text("{}")
        TokenizerAdapter(asset_root=root)


def test_flt_t03_dependency_pin(monkeypatch):
    monkeypatch.setattr("mfvlr.tokenizer.importlib.metadata.version", lambda name: "unapproved")
    with pytest.raises(ValueError, match="requires"):
        verify_assets()


def test_flt_t04_four_segments_and_padding(tokenizer):
    cases = FIXTURES["prompt_cases"]
    encoded = tokenizer.encode_levels([case["prompts"] for case in cases])
    assert tuple(encoded.ids.shape) == (10,308)
    assert tuple(encoded.segment_ids.shape) == (10,4,77)
    assert encoded.ids.dtype == torch.int64 and encoded.ids.device.type == "cpu"
    assert encoded.ids.tolist() == [case["ids"] for case in cases]
    assert encoded.segment_ids.tolist() == [case["segments"] for case in cases]
    assert torch.equal(encoded.ids.reshape(10,4,77), encoded.segment_ids)
    for row, case in enumerate(cases):
        for level, length in enumerate(case["payload_lengths"]):
            start = level*77
            assert encoded.payload_lengths[row,level] == length
            assert encoded.eos_positions[row,level] == start+length+1
            assert encoded.ids[row,start] == 49406
            assert encoded.ids[row,start+length+1] == 49407
            assert not encoded.padding_mask[row,start:start+length+2].any()
            assert encoded.padding_mask[row,start+length+2:start+77].all()
            assert (encoded.ids[row,start+length+2:start+77] == 0).all()
    joined = [49406]+tokenizer.encode_text(" ".join(cases[1]["prompts"]))+[49407]
    joined += [0]*(308-len(joined))
    assert joined != encoded.ids[1].tolist()


def test_flt_t05_boundaries_and_invalid_inputs(tokenizer):
    fit = "face "*75
    overflow = "face "*76
    assert len(tokenizer.encode_text(fit)) == 75
    encoded = tokenizer.encode_levels([[fit,"","face",""]])
    assert encoded.ids[0,76] == 49407
    assert not encoded.padding_mask[0,:77].any()
    assert encoded.segment_ids[0,1,:2].tolist() == [49406,49407]
    assert encoded.padding_mask[0,79:154].all()
    with pytest.raises(ValueError, match="exceeds 75"):
        tokenizer.encode_levels([[overflow,"","",""]])
    for bad in ([], [["a"]*3], [["a"]*5], ["abcd"]):
        with pytest.raises(ValueError):
            tokenizer.encode_levels(bad)
    for bad in ("text", 1):
        with pytest.raises(TypeError):
            tokenizer.encode_levels(bad)
    with pytest.raises(TypeError):
        tokenizer.encode_levels([["a",None,"b","c"]])
