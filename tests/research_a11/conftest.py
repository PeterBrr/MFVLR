"""Module correctness fixtures, isolated from the production MVE autouse owner."""
import json
import os
from pathlib import Path
import pytest
import torch
from mfvlr.flt import FineGrainedLanguageTransformer


@pytest.fixture(autouse=True)
def reset_gradients():
    # Overrides only the inherited MVE fixture for this new test directory.
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")


@pytest.fixture(scope="module")
def accepted_flt():
    return FineGrainedLanguageTransformer(seed=0)


@pytest.fixture
def evidence(tmp_path):
    root = Path(os.environ.get("MFVLR_A11_UNIT_EVIDENCE", str(tmp_path)))
    root.mkdir(parents=True, exist_ok=True)
    def write(name, value):
        with (root / (name + ".json")).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
    return write
