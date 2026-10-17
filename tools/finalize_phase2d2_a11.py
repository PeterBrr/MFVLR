"""Aggregate executed A1.1 evidence; preserve historical artifacts, no model runs."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/phase2d2_a11"
sys.path[:0] = [str(ROOT/".phase2c1_deps"), str(ROOT/"src"), str(ROOT)]


def digest(path):
    with open(path, "rb") as handle: return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, data):
    with path.open("x", encoding="utf-8") as handle: json.dump(data, handle, indent=2, allow_nan=False)


def cases(path):
    return {(c.attrib["classname"], c.attrib["name"]):
            ("FAIL" if c.find("failure") is not None else "ERROR" if c.find("error") is not None
             else "SKIP" if c.find("skipped") is not None else "PASS")
            for c in ET.parse(path).iter("testcase")}


def main():
    import torch
    from mfvlr_research.a11_evidence import backend
    before = json.loads((OUT/"preservation_before.json").read_text())
    after = {name:digest(ROOT/name) if (ROOT/name).is_file() else None for name in before}
    changed = {name:{"before":before[name], "after":after[name]} for name in before if before[name] != after[name]}
    write(OUT/"preservation_audit.json", {"status":"PASS" if not changed else "FAIL", "files":len(before),
          "changed":changed, "before_manifest_sha256":digest(OUT/"preservation_before.json"), "after":after,
          "scope":"all pre-existing production source/config/tests/assets/docs/tools and historical artifacts in the before inventory",
          "note":"Four fixed-path FLT gradient records written by unchanged tests were archived and restored after each invocation; original bytes verified here."})
    assert not changed, "baseline preservation failure"
    accepted = cases(ROOT/"artifacts/phase2d1/junit/complete_01.xml")
    main_regression = cases(OUT/"runs/regression_no_optimizer_01/junit.xml")
    mve = cases(OUT/"runs/mve_t10_approved_01/junit.xml")
    union = {**main_regression, **mve}
    assert len(accepted) == 276 and set(union) == set(accepted) and all(x == "PASS" for x in union.values())
    research = cases(OUT/"runs/research_complete_01/junit.xml")
    assert len(research) == 54 and all(x == "PASS" for x in research.values())
    historical = cases(ROOT/"artifacts/phase2d2_n31/junit/complete_01.xml")
    failures = cases(OUT/"runs/a0_eight_gates_01/junit.xml")
    assert len(failures) == 8 and all(v == "FAIL" for v in failures.values())
    assert set(failures) == {key for key, value in historical.items() if value == "FAIL"}
    runs = {}
    for directory in sorted((OUT/"runs").iterdir()):
        if not (directory/"junit.xml").exists(): continue
        result = cases(directory/"junit.xml")
        runs[directory.name] = {**json.loads((directory/"execution.json").read_text()),
            "cases":len(result), "passed":sum(v == "PASS" for v in result.values()),
            "failed":sum(v == "FAIL" for v in result.values()), "skipped":sum(v == "SKIP" for v in result.values()),
            "errors":sum(v == "ERROR" for v in result.values())}
    verification = {"phase":"BLOCKED", "A1_D01_module_correctness":"PASS", "A1_D02":"NOT RUN / UNAPPROVED",
        "research_cases":54, "research_passed":54, "accepted_regression_cases":276,
        "accepted_regressions_passed_across_two_runs":276, "test_identity_coverage_exact":True,
        "original_failure_cases":8, "original_failures_reasserted":8,
        "failure_scope":"one fresh A0 I08 fixture; I10 and six READY gates reassert copied historical G2, no G2 regeneration",
        "historical_328_suite_fresh_full_rerun":"NOT RUN; never claimed fully passing",
        "old_cases_outside_this_rerun":len(set(historical)-set(union)-set(failures)),
        "unique_executed_final_cases":len(union)+len(research)+len(failures),
        "new_research_optimizer_steps":0, "approved_legacy_MVE_T10_steps":1,
        "unapproved_restored_legacy_VD_T12_steps":1, "scope_compliance":"DEVIATION_REQUIRES_REVIEW",
        "hidden_skips_or_xfails":False, "initial_explicit_MVE_T10_deselection":"later run unchanged after user approval",
        "runs":runs, "accepted_case_ids":[list(k) for k in sorted(union)],
        "original_failure_ids":[list(k) for k in sorted(failures)]}
    write(OUT/"verification.json", verification)
    reports = {}
    for variant in ("A0", "P3", "Q3"):
        report = json.loads((OUT/f"runs/research_complete_01/evidence/ownership_{variant}.json").read_text())
        reports[variant] = {k:v for k,v in report.items() if k != "common"}
        reports[variant]["common_tensors_verified_by_name_shape_hash"] = len(report["common"])
        reports[variant]["detailed_report"] = f"runs/research_complete_01/evidence/ownership_{variant}.json"
    write(OUT/"parameter_initialization_report.json", {"classification":"DERIVED counts / ASSUMED research architecture",
         "variants":reports, "additional_parameters":21504, "additional_parameter_tensors":42,
         "new_LayerNorm_modules":21, "FLT_counts":{"A0":112758272,"P3":112779776,"Q3":112779776}})
    gradients = {}
    for variant in reports:
        r = json.loads((OUT/f"runs/research_complete_01/evidence/T05_{variant}.json").read_text())
        gradients[variant] = {"connected_parameter_tensors":len(r["parameters"]),
                             "all_finite":all(v["finite"] for v in r["parameters"].values()),
                             "all_38_QK_allocated_exact_zero":r["all_38_QK_allocated_exact_zero"],
                             "all_38_VO_nonzero":r["all_38_VO_nonzero"], "visual_gradient_l2":r["visual_gradient_l2"],
                             "state_unchanged":r["state_unchanged"], "scope":r["scope"]}
    write(OUT/"gradient_report.json", gradients)
    # Archive the new mandatory N2 diagnostic created by the unchanged regression.
    log = (OUT/"runs/regression_no_optimizer_01/stdout.log").read_text(encoding="utf-8")
    paths = re.findall(r"N2_DIAGNOSTIC: ([^\r\n]+)", log)
    records = []
    for i, name in enumerate(paths):
        path = Path(name.strip()); target = OUT/f"accepted_N2_regression_diagnostic_{i}.json"
        with target.open("xb") as handle: handle.write(path.read_bytes())
        records.append({"original_path":str(path), "copy":target.name, "sha256":digest(target)})
    write(OUT/"accepted_N2_diagnostic_provenance.json", records)
    write(OUT/"environment.json", {"backend":backend(), "Python":sys.version, "platform":platform.platform(),
        "versions":{n:importlib.metadata.version(n) for n in ("torch","numpy","pytest","PyYAML","psutil","ftfy","regex","wcwidth")},
        "torch_build":torch.__config__.show(), "GPU":"NOT RUN (CUDA unavailable)", "batch8":"NOT RUN (not authorized)",
        "timing_warning":"Recorded RSS/timings are process-tree samples, not guaranteed peak or capacity certification. System memory pressure occurred."})
    print(json.dumps({"preservation_files":len(before), "research_passed":54, "regression_passed":276,
                      "genuine_A0_failures":8, "phase":"BLOCKED", "scope_deviation":True}))


if __name__ == "__main__": main()
