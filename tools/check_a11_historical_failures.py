"""Reassert A0's eight unchanged gates; no new G2/NG2 sweep or variant study.

I08 recomputes its single existing A0 CMC chain-rule fixture. I10 and six READY
cases reassert the preserved N3.1 G2 measurements, copied byte-for-byte to a new
evidence location. These seven are not described as freshly computed G2 runs.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_phase2d2_a11 import run


def main():
    destination = ROOT/"artifacts/phase2d2_a11/a0_gate_reassertion"
    destination.mkdir(exist_ok=False)
    copied = {}
    source = ROOT/"artifacts/phase2d2_n31"
    for folder in (source/"g2").iterdir():
        if not folder.is_dir(): continue
        for path in folder.glob("*.json"):
            target = destination/path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
            copied[path.relative_to(ROOT).as_posix()] = digest
    (destination/"provenance.json").write_text(json.dumps({"copied_historical_G2":copied,
        "G2_regenerated":False, "I08":"fresh original single A0 fixture; original threshold unchanged",
        "I10_READY_N02":"original tests reassert historical G2 records; no reclassification"}, indent=2)+"\n")
    selected = ["tests/joint/test_joint.py::test_joint_i08_cmc_chain_rule_and_batch_coupling",
                "tests/joint/test_joint.py::test_joint_i10_fp64_oracle_and_full_fp32_gradient_additivity",
                "tests/joint/test_readiness.py::test_ready_n02_full_g2_no_update"]
    return run("a0_eight_gates_01", selected, {"MFVLR_PHASE2D2_EVIDENCE_ROOT":str(destination)})


if __name__ == "__main__": raise SystemExit(main())
