from pathlib import Path

import pytest
import torch

from mfvlr.decoder_config import load_decoder_config
from mfvlr.vision_decoder import VisionDecoder


@pytest.fixture(scope="session")
def decoder():
    root = Path(__file__).resolve().parents[2]
    return VisionDecoder.from_config(load_decoder_config(root / "configs/decoder_debug.yaml"))


@pytest.fixture(autouse=True)
def reset_decoder(decoder):
    decoder.zero_grad(set_to_none=True)
    decoder.train()
    yield
    decoder.zero_grad(set_to_none=True)


@pytest.fixture
def local():
    return torch.rand(1, 1024, 14, 14, generator=torch.Generator().manual_seed(71))
