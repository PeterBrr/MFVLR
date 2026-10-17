"""Closed Phase 2C.1 scope; diagnostics-only, single-level overlay.

Accepted MVE/VD schemas are imported for YAML parsing only and never changed.
Approval of this configuration cannot authorize a Language Encoder/Decoder.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path

from .config import ConfigError, _read

_REFERENCE = json.loads(Path(__file__).with_name("foundation_contract_snapshot.json").read_text(encoding="utf-8"))


def _exact(actual, expected, path="foundation"):
    if type(actual) is not type(expected):
        raise ConfigError(f"{path}: wrong type")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ConfigError(f"{path}: unknown or missing keys")
        for key, value in expected.items():
            _exact(actual[key], value, f"{path}.{key}")
    elif actual != expected:
        raise ConfigError(f"{path}: differs from scoped approved contract")


def _validate(data):
    if not isinstance(data, dict) or set(data) != set(_REFERENCE):
        raise ConfigError("foundation: unknown or missing keys")
    for key in _REFERENCE:
        if key not in ("profile", "diagnostics"):
            _exact(data[key], _REFERENCE[key], key)
    if data["profile"] not in ("reference", "debug"):
        raise ConfigError("profile must be reference or debug")
    expected = {"enabled": False, "batch_size": 1, "iterations": 0} if data["profile"] == "reference" else {
        "enabled": True, "batch_size": 2, "iterations": 3}
    _exact(data["diagnostics"], expected, "diagnostics")


@dataclass(frozen=True)
class FoundationConfig:
    path: Path
    _data: dict

    @property
    def data(self):
        return copy.deepcopy(self._data)

    def require_approval(self):
        _validate(self._data)


def load_foundation_config(path: str | Path) -> FoundationConfig:
    path = Path(path).resolve()
    raw = _read(path)
    if "base" not in raw:
        _validate(raw)
        return FoundationConfig(path, copy.deepcopy(raw))
    if set(raw) != {"schema_version", "phase", "profile", "base", "merge_policy", "approval", "overrides"}:
        raise ConfigError("foundation overlay: unknown or missing keys")
    for key, expected in (("schema_version", 1), ("phase", "2C.1"), ("profile", "debug"),
                          ("base", "foundation_reference.yaml"), ("merge_policy", "diagnostics_only"),
                          ("approval", _REFERENCE["approval"])):
        _exact(raw[key], expected, f"overlay.{key}")
    if type(raw["overrides"]) is not dict or set(raw["overrides"]) != {"diagnostics"}:
        raise ConfigError("overlay may only override diagnostics; no approval/topology/tolerance overrides")
    base_path = path.parent / raw["base"]
    if base_path.resolve() == path:
        raise ConfigError("foundation overlay cycle")
    base = _read(base_path)
    if "base" in base:
        raise ConfigError("only one overlay level is permitted")
    _validate(base)
    if base["profile"] != "reference":
        raise ConfigError("overlay base must be the reference profile")
    data = copy.deepcopy(base)
    data["profile"] = "debug"
    data["diagnostics"] = copy.deepcopy(raw["overrides"]["diagnostics"])
    _validate(data)
    return FoundationConfig(path, data)
