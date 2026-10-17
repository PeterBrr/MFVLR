"""Fixed-runtime, full-topology Phase 2A CPU fixtures."""

from pathlib import Path
import platform

import pytest
import torch
import yaml

from mfvlr.config import load_config
from mfvlr.mve import MVE

ROOT = Path(__file__).resolve().parents[1]


def pytest_sessionstart(session):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    print(f"Phase 2A: Python={platform.python_version()} torch={torch.__version__} "
          f"PyYAML={yaml.__version__} CPU threads={torch.get_num_threads()} "
          f"CUDA={torch.cuda.is_available()}")


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def model(root):
    result = MVE.from_config(load_config(root / "configs/debug.yaml"))
    yield result
    result.zero_grad(set_to_none=True)


@pytest.fixture(autouse=True)
def reset_gradients(model):
    model.zero_grad(set_to_none=True)
    model.train()
    yield
    model.zero_grad(set_to_none=True)


@pytest.fixture
def image():
    return torch.rand(1, 3, 224, 224, generator=torch.Generator().manual_seed(17))
