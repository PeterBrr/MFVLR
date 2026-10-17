"""Closed A1.1 research schema; cannot authorize an experiment through YAML."""
from pathlib import Path
import yaml
from .decoder_a11 import require_variant

FIXED = {
    "schema": "mfvlr-research-a11-v1", "scope": "A1-D01-module-correctness-only",
    "architecture": {"blocks": 7, "dimensions": 512, "heads": 8, "ffn": 2048,
                     "ln_eps": 1e-5, "ln_variance": "population", "ln_axis": -1,
                     "ln_gamma": 1.0, "ln_beta": 0.0, "vim_residual": "once-outside-norm"},
    "initialization": "copy-accepted-FP32-common-state-constant-new-norms",
    "full_experiments_enabled": False, "seed": 0,
}


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node):
    result = {}
    for k, v in node.value:
        key = loader.construct_object(k)
        if key in result:
            raise ValueError("duplicate YAML key: " + str(key))
        result[key] = loader.construct_object(v)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def validate_config(data):
    if type(data) is not dict or set(data) != set(FIXED) | {"variant"}:
        raise ValueError("unknown or missing A1.1 configuration keys")
    require_variant(data["variant"])
    def exact(actual, expected, key):
        if type(actual) is not type(expected):
            raise ValueError("invalid configuration type: " + key)
        if isinstance(expected, dict):
            if set(actual) != set(expected):
                raise ValueError("unknown or missing architecture keys")
            for child in expected: exact(actual[child], expected[child], key + "." + child)
        elif actual != expected:
            raise ValueError("unapproved configuration value: " + key)
    for key, value in FIXED.items(): exact(data[key], value, key)
    return data


def load_config(path):
    return validate_config(yaml.load(Path(path).read_text(encoding="utf-8"), Loader=UniqueLoader))
