import gc
import pytest


@pytest.fixture(autouse=True)
def reset_gradients():
    """Standalone tests do not request the ancestor fixture's MVE instance."""
    yield
    gc.collect()
