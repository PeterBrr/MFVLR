"""Append-only A1.1 unit/regression evidence runner; no ablation scheduling."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/phase2d2_a11"


def audit_collected_updates(collection_text):
    selected = {line.strip() for line in collection_text.splitlines() if line.startswith("tests/") and "::" in line}
    if not selected:
        raise RuntimeError("no collected test identities: refuse unaudited execution")
    vd = "tests/decoder/test_unit.py::test_vd_t12_head_gradient_sum_and_unique_update"
    if vd in selected:
        raise PermissionError("VD-T12 optimizer step is not authorized in A1.1; do not rerun it")
    mve = "tests/test_mve.py::test_mve_t10_shared_gradient_sum_and_single_update"
    if mve in selected:
        approval = json.loads((OUT/"legacy_mve_t10_approval.json").read_text())
        if approval.get("verbatim_answer") != "Allow unchanged MVE-T10 only":
            raise PermissionError("unchanged MVE-T10 needs its explicit scoped approval")
        import hashlib
        before = json.loads((OUT/"preservation_before.json").read_text())
        if hashlib.sha256((ROOT/"tests/test_mve.py").read_bytes()).hexdigest() != before["tests/test_mve.py"]:
            raise PermissionError("MVE-T10 source is not the approved unchanged file")
    return sorted(selected)


def run(name, paths, extra_env=None):
    if not name or not all(c.isalnum() or c == "_" for c in name):
        raise ValueError("choose an explicit safe run identity")
    dest = OUT / "runs" / name
    dest.mkdir(parents=True, exist_ok=False)
    historical = {p:p.read_bytes() for p in (ROOT / "artifacts/phase2c2/gradients").glob("*.json")}
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(map(str, [ROOT/".phase2c1_deps", ROOT/"src", ROOT]))
    env["MFVLR_A11_UNIT_EVIDENCE"] = str(dest/"evidence")
    if extra_env: env.update(extra_env)
    collection = subprocess.run([sys.executable, "-m", "pytest", *paths, "--collect-only", "-q"],
                                cwd=ROOT, env=env, capture_output=True, text=True)
    (dest/"collection.stdout.log").write_text(collection.stdout, encoding="utf-8")
    (dest/"collection.stderr.log").write_text(collection.stderr, encoding="utf-8")
    if collection.returncode:
        raise RuntimeError("test collection failed; no tests executed")
    selected = audit_collected_updates(collection.stdout)
    (dest/"selected_test_ids.json").write_text(json.dumps(selected, indent=2)+"\n")
    cmd = [sys.executable, "-m", "pytest", *paths, "-vv", "-s", f"--junitxml={dest/'junit.xml'}"]
    samples = []; stop = threading.Event(); started = time.perf_counter()
    available_start = psutil.virtual_memory().available
    result = None
    try:
        with (dest/"stdout.log").open("x", encoding="utf-8") as stdout, (dest/"stderr.log").open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            def monitor():
                while not stop.wait(.05):
                    try:
                        parent = psutil.Process(process.pid)
                        rss = parent.memory_info().rss
                        for child in parent.children(recursive=True):
                            try: rss += child.memory_info().rss
                            except psutil.Error: pass
                        samples.append(rss)
                    except psutil.Error: pass
            thread = threading.Thread(target=monitor, daemon=True); thread.start()
            result = process.wait()
            stop.set(); thread.join()
    finally:
        stop.set()
        for path, data in historical.items():
            fresh = dest/"regression_gradients"/path.name
            fresh.parent.mkdir(exist_ok=True)
            fresh.write_bytes(path.read_bytes())
            (dest/"historical_gradients").mkdir(exist_ok=True)
            (dest/"historical_gradients"/path.name).write_bytes(data)
            path.write_bytes(data)
        record = {"command":cmd, "exit_code":result, "elapsed_seconds":time.perf_counter()-started,
                  "sampled_peak_process_tree_rss_bytes":max(samples, default=0), "rss_samples":len(samples),
                  "sampling_seconds":.05, "available_RAM_start_bytes":available_start,
                  "available_RAM_end_bytes":psutil.virtual_memory().available,
                  "PYTHONPATH":env["PYTHONPATH"], "restored_historical_gradient_records":len(historical)}
        (dest/"execution.json").write_text(json.dumps(record, indent=2)+"\n")
        print(json.dumps(record), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("paths", nargs="+")
    args, extra = parser.parse_known_args()
    raise SystemExit(run(args.name, args.paths + extra))
