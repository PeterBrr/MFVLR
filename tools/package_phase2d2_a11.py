"""Deterministic-order, hashed review archive. No source or historical edits."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/phase2d2_a11"


def sha(path):
    with open(path, "rb") as handle: return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    before = json.loads((OUT/"preservation_before.json").read_text())
    changed = [p for p, h in before.items() if not (ROOT/p).is_file() or sha(ROOT/p) != h]
    if changed: raise AssertionError("baseline changed before packaging: " + repr(changed))
    files = {ROOT/p for p in before if not p.endswith(".zip")}
    for folder in ("src", "configs", "tests", "docs", "tools", ".phase2c1_deps", "artifacts/phase2d2_a11"):
        for path in (ROOT/folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".zip", ".pt"):
                files.add(path)
    for name in ("pyproject.toml", "README.md"):
        if (ROOT/name).exists(): files.add(ROOT/name)
    excluded_outputs = {OUT/"review_manifest.json", OUT/"package_verification.json", OUT/"phase2d2_a11_review.zip.sha256"}
    files -= excluded_outputs
    files = sorted(files)
    records = {p.relative_to(ROOT).as_posix():{"sha256":sha(p), "bytes":p.stat().st_size} for p in files}
    manifest_path = OUT/"review_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump({"format":"a11-review-v1", "files":records,
            "excluded_historical_nested_archives":{p:h for p,h in before.items() if p.endswith(".zip")},
            "exclusions":"compiled caches and temporary checkpoints; package manifest cannot hash itself",
            "baseline_preservation_checked_before_packaging":True}, handle, indent=2)
    files.append(manifest_path); files.sort()
    destination = OUT/"phase2d2_a11_review.zip"
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026,10,12,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3; info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        expected = set(records) | {manifest_path.relative_to(ROOT).as_posix()}
        assert set(archive.namelist()) == expected
        for name, record in records.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == record["sha256"], name
    digest = sha(destination)
    with (OUT/"phase2d2_a11_review.zip.sha256").open("x") as handle:
        handle.write(digest+"  "+destination.name+"\n")
    result = {"status":"PASS", "archive":destination.name, "bytes":destination.stat().st_size,
              "members":len(files), "sha256":digest, "all_member_hashes_verified":True,
              "archive_CRC_check":"PASS", "baseline_files_rechecked":len(before), "changed":changed,
              "fixed_entry_timestamps":True, "sorted_member_order":True}
    with (OUT/"package_verification.json").open("x") as handle: json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == "__main__": main()
