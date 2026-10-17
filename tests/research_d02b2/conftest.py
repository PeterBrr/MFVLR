import sys
import pytest

@pytest.fixture(autouse=True)
def reset_gradients():
    pass  # Override root's model-owning fixture, even on accidental broad collection.

def pytest_collection_modifyitems(session,config,items):
    if not items or any(not x.nodeid.replace('\\','/').startswith('tests/research_d02b2/') for x in items):
        raise pytest.UsageError('Only isolated D02B2 static/mock tests authorized')

@pytest.fixture(autouse=True)
def no_model_import():
    assert 'torch' not in sys.modules
    assert not any(n=='mfvlr' or n.startswith('mfvlr.') for n in sys.modules)
    yield
    assert 'torch' not in sys.modules
