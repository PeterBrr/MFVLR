"""Process-local no-update/no-model safety sentinels for reconciliation only.

Never edits accepted files. Not a numerical policy or architecture change.
"""
import json
import os
from pathlib import Path

STATE = {"optimizer_step_attempts": [], "model_construction_attempts": [],
         "forbidden_test_attempts": [], "executed_test_ids": []}
FORBIDDEN = {
    "test_vd_t12_head_gradient_sum_and_unique_update",
    "test_mve_t10_shared_gradient_sum_and_single_update",
}


def pytest_sessionstart(session):
    import torch
    from mfvlr.mve import MVE
    from mfvlr.vision_decoder import VisionDecoder
    from mfvlr.flt import FineGrainedLanguageTransformer
    from mfvlr.integration import MFVLRIntegration
    # Guard all exported optimizer classes. No optimizer is instantiated.
    for name in dir(torch.optim):
        cls = getattr(torch.optim, name)
        if isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer):
            def deny_step(*args, _name=name, **kwargs):
                STATE["optimizer_step_attempts"].append(_name)
                raise AssertionError("reconciliation forbids every optimizer step")
            cls.step = deny_step
    for cls in (MVE, VisionDecoder, FineGrainedLanguageTransformer, MFVLRIntegration):
        def deny_model(*args, _name=cls.__name__, **kwargs):
            STATE["model_construction_attempts"].append(_name)
            raise AssertionError("guard verification must not instantiate a model")
        cls.__init__ = deny_model


def pytest_runtest_setup(item):
    if item.name.split("[")[0] in FORBIDDEN:
        STATE["forbidden_test_attempts"].append(item.nodeid)
        raise AssertionError("neither legacy update test may execute in reconciliation")
    STATE["executed_test_ids"].append(item.nodeid)


def pytest_sessionfinish(session, exitstatus):
    path = Path(os.environ["MFVLR_RECONCILIATION_GUARD_RECORD"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump({**STATE, "pytest_exit_code":int(exitstatus),
                   "collect_only":bool(session.config.option.collectonly),
                   "scope":"process-local instrumentation; no accepted file changes"}, handle, indent=2)
