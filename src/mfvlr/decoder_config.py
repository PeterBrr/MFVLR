"""Dedicated Phase 2B schema/approval loader; accepted MVE loader unchanged."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path

from .config import ConfigError, _read, _validate_schema, load_config, merge_overlay

_CONTRACT = json.loads(Path(__file__).with_name("decoder_contract_snapshot.json").read_text(encoding="utf-8"))


def _review(review):
    immutable = _CONTRACT["review_immutable"]
    for key, value in immutable.items():
        if review[key] != value:
            raise ConfigError(f"Invalid Phase 2B approval scope/value: {key}")
    if review["status"] not in ("APPROVED", "PROPOSED"):
        raise ConfigError("Invalid decoder review status")
    if set(review["decisions"]) != {f"D{i:02}" for i in range(1, 9)}:
        raise ConfigError("Decoder decisions must identify D01-D08 exactly")
    if any(status not in ("APPROVED", "PROPOSED") for status in review["decisions"].values()):
        raise ConfigError("Invalid decoder decision status")


def _validate(data):
    _validate_schema(data, _CONTRACT["reference_schema"], "decoder_config")
    if data["schema_version"] != 1:
        raise ConfigError("Unsupported decoder schema version")
    _review(data["review"])
    for key, expected in _CONTRACT["locked_proposal"].items():
        if data["proposal"][key] != expected:
            raise ConfigError(f"Decoder {key} differs from approved D01-D08")
    runtime = data["proposal"]["runtime"]
    if runtime != _CONTRACT["runtime"]:
        raise ConfigError("Phase 2B runtime must preserve CPU FP32/deterministic settings")
    diagnostics = data["proposal"]["diagnostics"]
    if diagnostics["optimizer_updates"] or diagnostics["is_paper_loss"]:
        raise ConfigError("Diagnostics cannot authorize training/paper losses")
    if type(diagnostics["input_seed"]) is not int or diagnostics["input_seed"] < 0:
        raise ConfigError("Diagnostic seed must be nonnegative")
    if diagnostics["enabled"]:
        if diagnostics["iterations"] != 3 or diagnostics["batch_size"] != 2:
            raise ConfigError("Approved debug workload requires batch 2 and three iterations")
    elif diagnostics["iterations"] != 0 or diagnostics["batch_size"] != 1:
        raise ConfigError("Reference diagnostics must be disabled")


@dataclass(frozen=True)
class DecoderConfig:
    path: Path
    _data: dict
    review_chain: tuple[dict, ...]
    mve_path: Path

    @property
    def data(self):
        return copy.deepcopy(self._data)

    @property
    def proposal(self):
        return copy.deepcopy(self._data["proposal"])

    def require_approval(self):
        _validate(self._data)
        for record in self.review_chain:
            _review(record)
            if record["status"] != "APPROVED" or set(record["decisions"].values()) != {"APPROVED"}:
                raise ConfigError("Explicit decoder approval required in every profile")
        load_config(self.mve_path).require_mve_approval()


def load_decoder_config(path: str | Path) -> DecoderConfig:
    def resolve(current, stack):
        current = current.resolve()
        if current in stack:
            raise ConfigError("Decoder configuration base cycle")
        raw = _read(current)
        if "base" not in raw:
            _validate(raw)
            return raw, (copy.deepcopy(raw["review"]),), (current.parent / raw["mve_config"]).resolve()
        _validate_schema(raw, _CONTRACT["overlay_schema"], "decoder_overlay")
        if raw["schema_version"] != 1 or raw["merge_policy"] != _CONTRACT["merge_policy"]:
            raise ConfigError("Decoder overlay must use strict contract merge policy")
        _review(raw["review"])
        base, reviews, mve_path = resolve(current.parent / raw["base"], stack + (current,))
        _validate_schema(raw["overrides"], _CONTRACT["reference_schema"]["keys"]["proposal"], "overrides", partial=True)
        result = copy.deepcopy(base)
        result["proposal"] = merge_overlay(base["proposal"], raw["overrides"])
        result["profile"] = raw["profile"]
        _validate(result)
        return result, reviews + (copy.deepcopy(raw["review"]),), mve_path

    actual = Path(path).resolve()
    data, reviews, mve_path = resolve(actual, ())
    mve_config = load_config(mve_path)
    mve_config.require_mve_approval()
    for subtree, value in _CONTRACT["mve_settings"].items():
        if mve_config.proposal[subtree] != value:
            raise ConfigError(f"Accepted MVE {subtree} must remain unchanged")
    return DecoderConfig(actual, copy.deepcopy(data), reviews, mve_path)
