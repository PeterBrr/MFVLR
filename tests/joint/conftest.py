import gc
from pathlib import Path
import pytest
from mfvlr.integration import MFVLRIntegration


@pytest.fixture(autouse=True)
def reset_gradients():
    # Namespace-local: do not create an unrelated root MVE fixture.
    yield
    gc.collect()


@pytest.fixture(scope='module')
def graph():
    root=MFVLRIntegration.from_config(Path(__file__).resolve().parents[2]/'configs/integration_debug.yaml')
    yield root
    root.zero_grad(set_to_none=True)
    del root
    gc.collect()


@pytest.fixture(autouse=True)
def clean_graph(request):
    root=request.getfixturevalue('graph') if 'graph' in request.fixturenames else None
    if root:root.zero_grad(set_to_none=True)
    yield
    if root:root.zero_grad(set_to_none=True)

