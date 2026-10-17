import sys,pytest
@pytest.fixture(autouse=True)
def reset_gradients():pass
@pytest.fixture(autouse=True)
def no_full_model_import():
    for name in ('mfvlr.integration','mfvlr.mve','mfvlr.flt','mfvlr.vision_decoder'):
        assert name not in sys.modules
    yield
    for name in ('mfvlr.integration','mfvlr.mve','mfvlr.flt','mfvlr.vision_decoder'):
        assert name not in sys.modules
def pytest_collection_modifyitems(session,config,items):
    if not items or any(not x.nodeid.replace('\\','/').startswith('tests/research_d02b7/') for x in items):raise pytest.UsageError('D02B7 isolated tests only')
