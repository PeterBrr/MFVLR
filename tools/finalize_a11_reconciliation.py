"""Read-only technical evidence reconciliation; writes new closure artifacts only."""
from pathlib import Path
import ast
import hashlib
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/phase2d2_a11_reconciliation"
OLD = ROOT/"artifacts/phase2d2_a11"


def sha(path):
    with path.open("rb") as handle: return hashlib.file_digest(handle,"sha256").hexdigest()


def write(name, value):
    path = OUT/name
    with path.open("x",encoding="utf-8") as handle: json.dump(value,handle,indent=2,allow_nan=False)
    return sha(path)


def cases(path):
    return {(c.attrib["classname"],c.attrib["name"]):
            "FAIL" if c.find("failure") is not None else "ERROR" if c.find("error") is not None
            else "SKIP" if c.find("skipped") is not None else "PASS"
            for c in ET.parse(path).iter("testcase")}


def main():
    frozen = json.loads((OUT/"freeze_before.json").read_text())["files"]
    changed = [p for p,r in frozen.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=r["sha256"]]
    assert not changed, changed
    executed = cases(OUT/"junit.xml")
    assert len(executed)==25 and set(executed.values())=={"PASS"}
    runtime = json.loads((OUT/"runtime_guard.json").read_text())
    assert len(runtime["executed_test_ids"])==25
    for name in ("runtime_guard.json","collection_runtime_guard.json","runner_collection_runtime_guard.json"):
        record = json.loads((OUT/name).read_text())
        assert not record["optimizer_step_attempts"] and not record["model_construction_attempts"] and not record["forbidden_test_attempts"]
        if record["collect_only"]: assert not record["executed_test_ids"]
    old_research=cases(OLD/"runs/research_complete_01/junit.xml")
    regressions={**cases(OLD/"runs/regression_no_optimizer_01/junit.xml"),
                 **cases(OLD/"runs/mve_t10_approved_01/junit.xml")}
    accepted=cases(ROOT/"artifacts/phase2d1/junit/complete_01.xml")
    failure_cases=cases(OLD/"runs/a0_eight_gates_01/junit.xml")
    assert len(old_research)==54 and set(old_research.values())=={"PASS"}
    assert len(regressions)==276 and set(regressions)==set(accepted) and set(regressions.values())=={"PASS"}
    assert len(failure_cases)==8 and set(failure_cases.values())=={"FAIL"}
    historical_failures={k for k,v in cases(ROOT/"artifacts/phase2d2_n31/junit/complete_01.xml").items() if v=="FAIL"}
    assert set(failure_cases)==historical_failures
    launcher=ast.parse((ROOT/"tools/research_phase2d2_a11.py").read_text())
    execute=next(n for n in launcher.body if isinstance(n,ast.FunctionDef) and n.name=="execute")
    assert len(execute.body)==1 and isinstance(execute.body[0],ast.Raise)
    step_sites=[]
    for file in (ROOT/"tests").rglob("*.py"):
        tree=ast.parse(file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="step":
                step_sites.append({"path":file.relative_to(ROOT).as_posix(),"line":node.lineno,
                                   "expression":ast.unparse(node)})
    assert len(step_sites)==2
    guard={"status":"PASS","new_cases":25,"pytest_exit_code":0,"skips":0,"xfails":0,
           "existing_guard_cases":23,"new_reconciliation_cases":2,
           "optimizer_steps":0,"optimizer_containing_tests_executed":False,
           "model_constructions":0,"VD_actual_runner_rejected_before_execution":True,
           "MVE_historical_exception":"parser checked only; current authorization forbids rerun",
           "ablation_CLI":{"exit_code":1,"reason":"PermissionError: A1-D02 unapproved",
                           "expected_refusal_test_passed":True,"experiment_output_created":False},
           "launcher_execute_AST":"single unconditional Raise; accepted source unchanged",
           "known_step_call_sites_from_static_scan":step_sites,
           "limits":"Canonical pytest node IDs verified; this guard is not a security sandbox or approval to execute tests by another route.",
           "evidence":["junit.xml","runtime_guard.json","collection_runtime_guard.json",
                       "runner_collection_runtime_guard.json","runner_rejection.json",
                       "blocked_ablation_cli.execution.json","stdout.log","stderr.log"]}
    write("guard_verification.json",guard)
    evidence={"directly_executed_now":["25 no-update guard/metadata cases", "actual optimizer-test collection only",
              "runner VD rejection before execution", "CLI execution refusal", "SHA256 comparison of frozen files",
              "static inspection of step sites and restoration", "submitted JUnit identity/outcome reconciliation"],
              "submitted_logs_only":{"research_cases_passed":54,"regression_cases_passed":276,
                   "A0_original_failures":8,"parameter_inventory_and_common_state_parity":"recorded, not regenerated",
                   "incident_restoration":"source finally block plus submitted post-restoration stdout/PASS; no dedicated incident tensor hash"},
              "user_reported_only":{"Linux_selected_passes":51,"local_linux_JUnit_provided":False},
              "not_rerun":["MVE-T10","VD-T12","full research numerical suite","276 regressions",
                            "eight failing assertions","A1-D02","training"],
              "source_and_historical_artifact_preservation":{"files":len(frozen),"changed":changed}}
    write("verification_basis.json",evidence)
    subset={p:r for p,r in frozen.items() if p.startswith(("src/","configs/research_a11/","tests/research_a11/"))
            or p in ("tools/run_phase2d2_a11.py","tools/research_phase2d2_a11.py",
                      "docs/phase2d2_a1_ablation_preregistration_v1_1.md","docs/phase2d2_a11_erratum.md")}
    source_digest=hashlib.sha256(json.dumps(subset,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    write("source_hashes.json",{"files":subset,"canonical_source_inventory_sha256":source_digest,
                               "hash_definition":"SHA256 UTF-8 JSON of files mapping; sort_keys=True,separators=(',',':')"})
    manifest={"schema":"mfvlr-a11-acceptance-snapshot-v1","date":"2026-10-12","timezone":"Asia/Saigon",
              "scope":"A1-D01 research implementation, conditionally technically reviewed; closure submitted for user review",
              "incident":"execution-scope deviation retained; no retroactive VD-T12 authorization",
              "A1_D02":"UNAPPROVED / NOT RUN","Phase_2D_2":"BLOCKED",
              "files":frozen,"file_count":len(frozen),"changed_since_freeze":changed,
              "canonical_source_inventory_sha256":source_digest,
              "verification_basis":evidence,
              "snapshot_semantics":"Existing files frozen by content, not filesystem permission changes. New closure documents and guard logs are separately hashed. No original logs or approval records overwritten."}
    digest=write("acceptance_manifest.json",manifest)
    with (OUT/"acceptance_manifest.sha256").open("x") as handle:handle.write(digest+"  acceptance_manifest.json\n")
    print(json.dumps({"status":"PASS","frozen_files":len(frozen),"new_cases_passed":25,
                     "acceptance_manifest_sha256":digest,"source_inventory_sha256":source_digest}))


if __name__=="__main__":main()
