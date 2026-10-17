import sys,pytest
@pytest.fixture(autouse=True)
def reset_gradients():pass
@pytest.fixture(autouse=True)
def no_torch():
    assert 'torch' not in sys.modules
    yield
    assert 'torch' not in sys.modules
def pytest_collection_modifyitems(session,config,items):
    if not items or any(not x.nodeid.replace('\\','/').startswith('tests/research_d02b8/') for x in items):raise pytest.UsageError('D02B8 isolated tests only')
