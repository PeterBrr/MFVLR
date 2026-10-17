"""Run with --confcutdir=tests/research_d02b1: no parent model fixtures."""
import sys
import pytest

@pytest.fixture(autouse=True)
def reset_gradients():
    # Defense in depth: override the root fixture without a model dependency.
    pass

@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    from mfvlr_research.d02b1.runner import assert_mock_selection
    assert_mock_selection([item.nodeid for item in items])

@pytest.fixture(autouse=True)
def no_graph_imports():
    # A parent fixture/model import is a hard error, not an implicit full graph.
    assert 'torch' not in sys.modules
    assert not any(n == 'mfvlr' or n.startswith('mfvlr.') for n in sys.modules)
    yield
    assert 'torch' not in sys.modules

