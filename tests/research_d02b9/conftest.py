import sys
import pytest

@pytest.fixture(autouse=True)
def reset_gradients():
    yield

@pytest.fixture(autouse=True)
def no_production_graph(monkeypatch):
    import torch
    def unauthorized(*args, **kwargs):
        raise AssertionError('Optimizer construction/steps unauthorized')
    monkeypatch.setattr(torch.optim.Optimizer, '__init__', unauthorized)
    monkeypatch.setattr(torch.optim.SGD, 'step', unauthorized)
    monkeypatch.setattr(torch.optim.Adam, 'step', unauthorized)
    forbidden = {'mfvlr.integration','mfvlr.mve','mfvlr.flt','mfvlr.vision_decoder'}
    assert not forbidden.intersection(sys.modules)
    yield
    assert not forbidden.intersection(sys.modules)

def pytest_collection_modifyitems(session, config, items):
    if not items or any(not i.nodeid.replace('\\','/').startswith('tests/research_d02b9/') for i in items):
        raise pytest.UsageError('D02B9 isolated tests only; legacy optimizer tests excluded')
