"""Strict configuration loading for the approved MVE-only contract.

Maps merge recursively, sequences/scalars/null replace, and unknown keys fail.
Review metadata never merges or confers approval to another scope. No full-model
runner is provided. The shipped snapshot fixes the M1--M5 engineering choices.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Invalid schema, overlay, approval scope or approved-MVE configuration."""


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigError("Configuration mapping keys must be strings")
        if key in result:
            raise ConfigError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)
_SNAPSHOT = json.loads(Path(__file__).with_name("contract_snapshot.json").read_text(encoding="utf-8"))
_BUNDLES = ["M1", "M2", "M3", "M4", "M5"]
_POLICY = {
    "target": "proposal", "mappings": "recursive", "sequences": "replace",
    "scalars": "replace", "explicit_null": "replace", "unknown_keys": "reject",
    "propagate_approval": False,
}


def _read(path: Path) -> dict:
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"Cannot read configuration {path}: {error}") from error
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration {path} must be a mapping")
    return data


def _validate_schema(value: Any, schema: dict, path: str, *, partial: bool = False) -> None:
    kind = schema["kind"]
    if kind == "partial_map":
        kind, partial = "map", True
    if kind == "nullable_string":
        if value is None:
            return
        kind = "string"
    if kind == "map":
        if not isinstance(value, dict):
            raise ConfigError(f"{path} must be a mapping")
        keys = schema["keys"]
        unknown = set(value) - set(keys)
        missing = set(keys) - set(value)
        if unknown:
            raise ConfigError(f"Unknown keys at {path}: {sorted(unknown)}")
        if missing and not partial:
            raise ConfigError(f"Missing keys at {path}: {sorted(missing)}")
        for key, child in value.items():
            _validate_schema(child, keys[key], f"{path}.{key}", partial=partial)
    elif kind == "list":
        if not isinstance(value, list):
            raise ConfigError(f"{path} must be a sequence")
        for index, child in enumerate(value):
            _validate_schema(child, schema["item"], f"{path}[{index}]", partial=partial)
    elif kind == "number":
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ConfigError(f"{path} must be a finite number")
    elif kind == "int":
        if type(value) is not int:
            raise ConfigError(f"{path} must be an integer")
    elif kind == "bool":
        if type(value) is not bool:
            raise ConfigError(f"{path} must be a boolean")
    elif kind == "string":
        if not isinstance(value, str):
            raise ConfigError(f"{path} must be a string")
    elif kind == "nullable":
        # Unresolved later-component placeholders may stay null or become scalars.
        if value is not None and type(value) not in (str, int, float, bool):
            raise ConfigError(f"{path} is an unresolved scalar placeholder")
        if type(value) is float and not math.isfinite(value):
            raise ConfigError(f"{path} must be finite")
    else:
        raise ConfigError(f"Unknown schema kind {kind}")


def merge_overlay(base: dict, patch: dict, path: str = "proposal") -> dict:
    """Pure merge: no input mutation; null replaces, never silently inherits."""
    if not isinstance(base, dict) or not isinstance(patch, dict):
        raise ConfigError(f"{path}: base and overlay must be mappings")
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if key not in base:
            raise ConfigError(f"Unknown overlay key: {path}.{key}")
        if isinstance(value, dict):
            if not isinstance(base[key], dict):
                raise ConfigError(f"Cannot overlay a mapping onto scalar {path}.{key}")
            result[key] = merge_overlay(base[key], value, f"{path}.{key}")
        else:
            result[key] = copy.deepcopy(value)
    return result


def _validate_review(review: dict) -> None:
    if review["training_authorized"] is not False:
        raise ConfigError("Phase 2A does not authorize full-model training")
    if review["authorized_implementation_scopes"] != ["mve"]:
        raise ConfigError("Only the MVE implementation scope is approved")
    if review["approved_decisions"] != []:
        raise ConfigError("Whole A01-A35 assumptions must not be blanket-approved")
    if review["numerical_amendments"] != _SNAPSHOT["approved_numerical_amendments"]:
        raise ConfigError("N1 approval is limited to MVE-T12 cross-batch-size CPU FP32 comparison")
    approval = review["mve_approval"]
    if approval["scope"] != "mve_only" or approval["classification"] != "ASSUMED":
        raise ConfigError("MVE approval must be scoped engineering assumptions")
    if approval["confirms_author_architecture"] or approval["production_skip_decoder_approved"]:
        raise ConfigError("Author-fidelity and production-decoder approval are not granted")
    if approval["bundles"] != _BUNDLES:
        raise ConfigError("MVE approval must identify exactly M1-M5")
    if "decisions" in review:
        if set(review["decisions"]) != {f"A{i:02}" for i in range(1, 36)}:
            raise ConfigError("Decision register must contain A01-A35 exactly")
        if set(review["decisions"].values()) != {"PROPOSED"}:
            raise ConfigError("Unrelated assumption portions must remain PROPOSED")


def _validate_resolved(data: dict) -> None:
    _validate_schema(data, _SNAPSHOT["reference_schema"], "config")
    if data["schema_version"] != 1:
        raise ConfigError("Unsupported schema version")
    _validate_review(data["review"])
    proposal = data["proposal"]
    for subtree, approved in _SNAPSHOT["approved_subtrees"].items():
        if proposal[subtree] != approved:
            raise ConfigError(f"{subtree} differs from approved M1-M5 contract")
    for path in data["evidence"]["explicit_paths"] | data["evidence"]["derived_paths"]:
        value = data
        for segment in path.split("."):
            if not isinstance(value, dict) or segment not in value:
                raise ConfigError(f"Missing evidence path {path}")
            value = value[segment]
    if data["paper_explicit"] != _SNAPSHOT["paper_explicit"]:
        raise ConfigError("Immutable paper facts/equations differ from the recorded source")
    if proposal["execution"]["components"] == ["mve"]:
        if any(proposal[section]["enabled"] for section in
               ("vision_decoder", "heads", "language", "losses", "training", "evaluation")):
            raise ConfigError("MVE diagnostics cannot enable later components or training")
    diagnostic = proposal["diagnostics"]
    if diagnostic["enabled"]:
        if not diagnostic["synthetic_input"] or diagnostic["optimizer_updates"]:
            raise ConfigError("Phase 2A diagnostics must be synthetic without optimizer updates")
        if diagnostic["objective_is_paper_loss"]:
            raise ConfigError("Diagnostic scalar must not be represented as a paper loss")
        if diagnostic["batch_size"] != 2 or diagnostic["iterations"] != 3:
            raise ConfigError("Approved debug workload is batch 2 for three checks")
    if proposal["runtime"]["precision"] != "float32":
        raise ConfigError("Approved verification precision is float32")


@dataclass(frozen=True)
class ResolvedConfig:
    path: Path
    _data: dict
    review_chain: tuple[dict, ...]

    @property
    def proposal(self) -> dict:
        return copy.deepcopy(self._data["proposal"])

    @property
    def data(self) -> dict:
        return copy.deepcopy(self._data)

    def require_mve_approval(self) -> None:
        # Every profile must carry its own explicit scoped record; no inheritance.
        for review in self.review_chain:
            _validate_review(review)
            if review["implementation_authorized"] is not True:
                raise ConfigError("MVE implementation approval is missing")
            if review["mve_approval"]["status"] != "APPROVED":
                raise ConfigError("M1-M5 have not been approved for this profile")
        _validate_resolved(self._data)


def load_config(path: str | Path) -> ResolvedConfig:
    """Resolve relative overlay bases; reject cycles, unknown keys and drift."""

    def resolve(current: Path, stack: tuple[Path, ...]) -> tuple[dict, tuple[dict, ...]]:
        current = current.resolve()
        if current in stack:
            raise ConfigError(f"Configuration base cycle: {current}")
        raw = _read(current)
        if "base" not in raw:
            _validate_resolved(raw)
            return raw, (copy.deepcopy(raw["review"]),)
        _validate_schema(raw, _SNAPSHOT["overlay_schema"], "overlay")
        if raw["schema_version"] != 1 or raw["merge_policy"] != _POLICY:
            raise ConfigError("Overlay must use the contract's strict merge policy")
        _validate_review(raw["review"])
        base_path = current.parent / raw["base"]
        base, chain = resolve(base_path, stack + (current,))
        _validate_schema(raw["overrides"], _SNAPSHOT["reference_schema"]["keys"]["proposal"],
                         "overrides", partial=True)
        result = copy.deepcopy(base)
        result["proposal"] = merge_overlay(base["proposal"], raw["overrides"])
        result["profile"] = raw["profile"]
        # Base review is retained as base metadata; raw review is checked separately.
        _validate_resolved(result)
        for key in raw["acceptance"]["unchanged_proposal_subtrees"]:
            if key not in base["proposal"] or result["proposal"][key] != base["proposal"][key]:
                raise ConfigError(f"Debug must preserve subtree {key}")
        return result, chain + (copy.deepcopy(raw["review"]),)

    actual = Path(path).resolve()
    resolved, chain = resolve(actual, ())
    return ResolvedConfig(actual, copy.deepcopy(resolved), chain)
