import gc
from pathlib import Path

import pytest
import torch

from mfvlr.language_encoder import LanguageEncoderStage


@pytest.fixture(scope="module")
def encoder_stage():
    module = LanguageEncoderStage()
    yield module
    module.zero_grad(set_to_none=True)
    del module
    gc.collect()


@pytest.fixture
def language_inputs():
    # Accepted independent oracle IDs, never shortened for a full-topology test.
    import json
    root = Path(__file__).resolve().parents[2]
    cases = json.loads((root/"tests/foundation/fixtures/golden_tokens.json").read_text())["prompt_cases"]
    ids = torch.tensor([cases[1]["ids"],cases[5]["ids"]],dtype=torch.int64)
    visual = torch.randn(2,1,512,generator=torch.Generator().manual_seed(341))*0.1
    return ids,visual


@pytest.fixture(scope="module")
def flt():
    from mfvlr.flt import FineGrainedLanguageTransformer
    from mfvlr.flt_config import load_flt_config
    root=Path(__file__).resolve().parents[2]
    module=FineGrainedLanguageTransformer.from_config(load_flt_config(root/'configs/flt_debug.yaml'))
    yield module
    module.zero_grad(set_to_none=True)
    del module
    gc.collect()


@pytest.fixture(autouse=True)
def reset_full_language(request):
    module=request.getfixturevalue('flt') if 'flt' in request.fixturenames else None
    if module is not None:
        module.zero_grad(set_to_none=True)
        module.train()
    yield
    if module is not None: module.zero_grad(set_to_none=True)
    gc.collect()
