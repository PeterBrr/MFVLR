"""No-update pytest sentinel: allows local models but never legacy update tests."""
import torch
from tools.d02a.common import configure


def pytest_sessionstart(session):configure(4,False)


def pytest_runtest_setup(item):
    if item.name.split("[")[0] in ("test_mve_t10_shared_gradient_sum_and_single_update",
                                  "test_vd_t12_head_gradient_sum_and_unique_update"):
        raise AssertionError("legacy optimizer-containing tests forbidden in D02A")
