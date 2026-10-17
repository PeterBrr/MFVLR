"""Build and verify a deterministic source/artifact review archive.

Does not install dependencies, modify accepted visual files or run models.
"""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/phase2c1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = json.loads((ROOT/"artifacts/phase2b/manifest.json").read_text())
    for name, expected in old["files_sha256"].items():
        if sha(ROOT/name) != expected:
            raise RuntimeError(f"Accepted Phase 2B file changed: {name}")
    preservation = {"baseline_manifest":"artifacts/phase2b/manifest.json","files_verified":len(old["files_sha256"]),
                    "mismatches":[],"sha256":old["files_sha256"]}
    (OUT/"preservation.json").write_text(json.dumps(preservation,indent=2)+"\n")
    reports = {}
    for path in sorted((OUT/"junit").glob("*.xml")):
        tree = ET.parse(path).getroot()
        suite = tree.find("testsuite") if tree.tag == "testsuites" else tree
        cases = [{"name":c.attrib["name"],"class":c.attrib["classname"],
                  "status":"FAIL" if c.find("failure") is not None or c.find("error") is not None else
                           ("SKIP" if c.find("skipped") is not None else "PASS")} for c in suite.findall("testcase")]
        reports[path.stem] = {**suite.attrib,"cases":cases}
    final_name = sorted(name for name in reports if name.startswith("complete_"))[-1]
    final = reports[final_name]
    if int(final["tests"]) != 119 or any(c["status"] != "PASS" for c in final["cases"]):
        raise RuntimeError("Final complete suite has not passed all 119 cases")
    vim_ids = set()
    for case in final["cases"]:
        match = re.match(r"test_vim_t(\d\d)_",case["name"])
        if match: vim_ids.add(match.group(1))
    if vim_ids != {f"{i:02}" for i in range(1,11)}:
        raise RuntimeError("Incomplete VIM-T01--T10 coverage")
    summary = {"client_date":"2026-10-11","phase":"2C.1","all_new_cases":79,"vim_test_ids":10,
               "vim_cases":12,"accepted_visual_cases":40,"complete_cases":119,
               "failed":0,"skipped":0,"xfail":0,"n1_extended":False,"final_report":final_name,"reports":reports}
    (OUT/"test_summary.json").write_text(json.dumps(summary,indent=2)+"\n")

    files = {ROOT/name for name in old["files_sha256"]}
    files.add(ROOT/"artifacts/phase2b/manifest.json")
    files.update((ROOT/"docs").glob("phase2c*.md"))
    files.update((ROOT/"configs").glob("foundation*"))
    files.update((ROOT/"tools").glob("*phase2c*.py"))
    for name in ("prompts.py","tokenizer.py","vim.py","foundation_config.py","foundation_contract_snapshot.json"):
        files.add(ROOT/"src/mfvlr"/name)
    for folder in (ROOT/"src/mfvlr/tokenizer_assets",ROOT/"tests/foundation",OUT):
        files.update(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    excluded = {OUT/name for name in ("manifest.json","phase2c1_review_package.zip",
                                      "phase2c1_review_package.zip.sha256","package_verification.json")}
    files.difference_update(excluded)
    paths = sorted(files,key=lambda p:p.relative_to(ROOT).as_posix())
    hashes = {p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    manifest = {"phase":"2C.1","client_date":"2026-10-11","report":"docs/phase2c1_implementation_report.md",
                "approval_scope":"L01-L03 and standalone L09, foundation portions of L11-L13 only",
                "author_architecture_confirmed":False,"full_flt_approved":False,
                "training_authorized":False,"full_flt_implemented":False,
                "complete_suite":{"cases":119,"passed":119,"failed":0,"skipped":0,"xfail":0},
                "standalone_vim_unique_parameters":1048576,"accepted_visual_unique_parameters":41134277,
                "accepted_phase2b_files_preserved":len(old["files_sha256"]),
                "files_sha256":hashes,"manifest_and_zip_self_hashes_excluded":True}
    manifest_path = OUT/"manifest.json"
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    archive = OUT/"phase2c1_review_package.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as package:
        for path in sorted(paths+[manifest_path],key=lambda p:p.relative_to(ROOT).as_posix()):
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(),date_time=(2026,10,11,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            package.writestr(info,path.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        assert len(package.namelist()) == len(hashes)+1
        for name,expected in hashes.items():
            assert hashlib.sha256(package.read(name)).hexdigest() == expected,name
        assert package.read("artifacts/phase2c1/manifest.json") == manifest_path.read_bytes()
    checksum = sha(archive)
    (OUT/"phase2c1_review_package.zip.sha256").write_text(checksum+"  "+archive.name+"\n")
    result = {"zip":archive.relative_to(ROOT).as_posix(),"sha256":checksum,"bytes":archive.stat().st_size,
              "entries":len(hashes)+1,"all_payload_hashes_verified":True,
              "accepted_files_preserved":len(old["files_sha256"]),"fixed_zip_timestamps":True}
    (OUT/"package_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
