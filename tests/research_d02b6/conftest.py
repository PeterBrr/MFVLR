import sys
import pytest

@pytest.fixture(autouse=True)
def reset_gradients():
    pass

@pytest.fixture(autouse=True)
def no_model_imports():
    assert 'torch' not in sys.modules
    yield
    assert 'torch' not in sys.modules

def pytest_collection_modifyitems(session,config,items):
    if not items or any(not x.nodeid.replace('\\','/').startswith('tests/research_d02b6/') for x in items):
        raise pytest.UsageError('Only D02B6 isolated mock tests permitted; legacy optimizer tests excluded')
