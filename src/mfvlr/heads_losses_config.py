"""Closed Phase 2D.1 schema; approval cannot authorize G2/data/optimizer work."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path

from .config import ConfigError, _read

_REFERENCE = json.loads(Path(__file__).with_name("heads_losses_contract_snapshot.json").read_text(encoding="utf-8"))


def _exact(actual, expected, path):
    if type(actual) is not type(expected):
        raise ConfigError(f"{path}: wrong scalar/container type")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ConfigError(f"{path}: unknown or missing fields")
        for key in expected:
            _exact(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ConfigError(f"{path}: sequence length differs")
        for index, (value, reference) in enumerate(zip(actual, expected)):
            _exact(value, reference, f"{path}[{index}]")
    elif actual != expected:
        raise ConfigError(f"{path}: differs from scoped contract")


def _validate(data: dict) -> None:
    if type(data) is not dict or set(data) != set(_REFERENCE):
        raise ConfigError("heads/objectives: unknown or missing fields")
    for key, expected in _REFERENCE.items():
        if key not in ("profile", "diagnostics"):
            _exact(data[key], expected, f"heads_objectives.{key}")
    if type(data["profile"]) is not str or data["profile"] not in ("reference", "debug"):
        raise ConfigError("heads/objectives profile must be reference/debug")
    expected = {"enabled": False, "batch_size": 1, "optimizer_updates": False, "full_model_graph": False}
    if data["profile"] == "debug":
        expected.update(enabled=True, batch_size=2)
    _exact(data["diagnostics"], expected, "heads_objectives.diagnostics")


@dataclass(frozen=True)
class HeadsLossesConfig:
    path: Path
    _data: dict

    @property
    def data(self):
        return copy.deepcopy(self._data)

    def require_approval(self):
        _validate(self._data)


def load_heads_losses_config(path: str | Path) -> HeadsLossesConfig:
    path = Path(path).resolve()
    raw = _read(path)
    if "base" in raw:
        if set(raw) != {"schema_version", "phase", "profile", "base", "merge_policy", "approval", "overrides"}:
            raise ConfigError("heads/objectives overlay: unknown or missing fields")
        for key, expected in (("schema_version", 1), ("phase", "2D.1"), ("profile", "debug"),
                              ("base", "heads_losses_reference.yaml"), ("merge_policy", "diagnostics_only"),
                              ("approval", _REFERENCE["approval"])):
            _exact(raw[key], expected, f"heads_objectives.overlay.{key}")
        if type(raw["overrides"]) is not dict or set(raw["overrides"]) != {"diagnostics"}:
            raise ConfigError("heads/objectives overlay only replaces diagnostics")
        base_path = (path.parent / raw["base"]).resolve()
        if base_path == path:
            raise ConfigError("heads/objectives overlay cycle")
        base = _read(base_path)
        if "base" in base:
            raise ConfigError("only one overlay level is permitted")
        _validate(base)
        if base["profile"] != "reference":
            raise ConfigError("overlay base must be reference")
        resolved = copy.deepcopy(base)
        resolved["profile"] = "debug"
        resolved["diagnostics"] = copy.deepcopy(raw["overrides"]["diagnostics"])
        raw = resolved
    _validate(raw)
    return HeadsLossesConfig(path, copy.deepcopy(raw))
