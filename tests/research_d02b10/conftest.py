import sys
import pytest

@pytest.fixture(autouse=True)
def reset_gradients(): yield

@pytest.fixture(autouse=True)
def scope_guard(monkeypatch):
    import torch
    def forbidden(*a,**k):raise AssertionError('Optimizer forbidden')
    monkeypatch.setattr(torch.optim.Optimizer,'__init__',forbidden)
    monkeypatch.setattr(torch.optim.SGD,'step',forbidden)
    names={'mfvlr.integration','mfvlr.mve','mfvlr.flt','mfvlr.vision_decoder'}
    assert not names.intersection(sys.modules)
    yield
    assert not names.intersection(sys.modules)

def pytest_collection_modifyitems(session,config,items):
    if not items or any(not i.nodeid.replace('\\','/').startswith('tests/research_d02b10/') for i in items):
        raise pytest.UsageError('Only isolated D02B10 controls authorized')
