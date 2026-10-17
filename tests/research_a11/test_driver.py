import json
from pathlib import Path
import uuid

import pytest

from tools.research_phase2d2_a11 import prepare, execute, fixture_spec, BACKENDS
from tools.run_phase2d2_a11 import audit_collected_updates


def test_a1_t10_runner_optimizer_scope_guard():
    with pytest.raises(PermissionError, match="VD-T12"):
        audit_collected_updates("tests/decoder/test_unit.py::test_vd_t12_head_gradient_sum_and_unique_update\n")
    with pytest.raises(RuntimeError, match="no collected"):
        audit_collected_updates("collection failed")
    # Guard validation only; neither optimizer test is executed by this test.
    selected = audit_collected_updates("tests/test_mve.py::test_mve_t10_shared_gradient_sum_and_single_update\n")
    assert len(selected) == 1


def test_a1_t10_driver_preparation_and_execution_lock():
    dest = Path("artifacts/phase2d2_a11/prepared") / ("unit_"+uuid.uuid4().hex)
    selected = BACKENDS[0]
    result = prepare("unit_plan", "P3", "NLD-0", selected, dest)
    record = json.loads((result/"request.json").read_text())["record"]
    assert record["status"] == "PREPARED_NOT_RUN_A1_D02_BLOCKED"
    assert record["model_instantiated"] is record["fixture_generated"] is record["execution_enabled"] is False
    assert record["fixture_spec"]["FLT_seed"] == 6101
    assert len(record["fixture_spec"]["multipliers"]) == 8
    with pytest.raises(FileExistsError): prepare("unit_plan", "P3", "NLD-0", selected, dest)
    for path in (Path("artifacts/phase2d2_n31"), dest/"../../../../phase2d2_n31"):
        with pytest.raises(ValueError, match="isolated"): prepare("unsafe", "A0", "NLD-0", selected, path)
    for variant in ("I1", "R2", "../A0"):
        with pytest.raises(ValueError): prepare("bad", variant, "NLD-0", selected, dest)
    with pytest.raises(ValueError): prepare("../bad", "A0", "NLD-0", selected, dest)
    with pytest.raises(ValueError): prepare("bad", "A0", "unknown", selected, dest)
    with pytest.raises(ValueError): prepare("bad", "A0", "G2-real", "cpu-fp64-threads4-mkldnn", dest)
    with pytest.raises(PermissionError, match="A1-D02"): execute()


@pytest.mark.parametrize("name", ["NLD-0","NLD-1","NLD-2","NATT-0","NATT-1","NATT-2",
                                  "NG2-0","NG2-1","NG2-2","N2-F0","N2-F1","N2-F2",
                                  "G2-real","G2-efs","G2-local","G2-real_efs","G2-real_local",
                                  "G2-local_duplicate","N3-same-forward","N3-CMC-chain","N31-locked-local"])
def test_a1_t10_fixture_registry_only(name):
    # Metadata resolution only: no fixture tensors, models or ablation executions.
    assert fixture_spec(name)["family"]
