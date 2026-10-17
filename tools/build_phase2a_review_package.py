"""Validate recorded Phase 2A acceptance evidence and create a review archive.

Does not run or simulate tests. Run the documented pytest command first.
Deterministic archive entries package the current sources and recorded evidence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from mfvlr.config import load_config


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    artifact_root = ROOT / "artifacts/phase2a"
    review = artifact_root / "n1_review"
    previous = read_json(artifact_root / "manifest.json")
    evidence = read_json(review / "numerical_evidence.json")
    assert digest(ROOT / "src/mfvlr/mve.py") == previous["files_sha256"]["src/mfvlr/mve.py"]
    assert evidence["mve_source_sha256"] == digest(ROOT / "src/mfvlr/mve.py")
    for name, record in previous["phase1_preservation"].items():
        assert digest(ROOT / name) == record["sha256"], name
    assert digest(ROOT / "papers/MFVLR.pdf") == previous["paper_sha256"]
    for name, expected in previous["files_sha256"].items():
        if name.startswith("artifacts/phase2a/"):
            assert digest(ROOT / name) == expected, f"Original evidence changed: {name}"

    assert evidence["n1_threshold_passes"] and not evidence["original_threshold_passes"]
    assert evidence["batch_permutation_exact"] and evidence["parameter_identities_unchanged"]
    assert all(item["exactly_equal"] and item["max_abs"] == 0
               for item in evidence["fixed_batch_perturbations"].values())
    assert all(item["sample_1_nonzero_gradient_elements"] == 0
               for item in evidence["cross_sample_jacobian_probe"].values())
    assert all(item["inputs_exactly_equal"] and item["fp64_max_abs"] < item["fp32_max_abs"]
               for item in evidence["isolated_block_precision_replay"])
    assert evidence["parameters"]["total_unique"] == 32078912

    suite = ET.parse(review / "final_pytest.xml").getroot().find("testsuite")
    assert suite is not None
    assert {key: int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")} == {
        "tests": 15, "failures": 0, "errors": 0, "skipped": 0,
    }
    outcomes = []
    for case in suite.findall("testcase"):
        assert not any(case.find(key) is not None for key in ("failure", "error", "skipped"))
        outcomes.append({"name": case.attrib["name"], "status": "PASS", "seconds": float(case.attrib["time"])})
    names = [item["name"] for item in outcomes]
    for index in range(1, 15):
        matches = [name for name in names if name.startswith(f"test_mve_t{index:02}_")]
        assert len(matches) == (2 if index == 1 else 1)
    assert any(name.endswith("[1]") for name in names) and any(name.endswith("[2]") for name in names)

    # Convert only newly captured PowerShell logs; never modify historical logs.
    for log in review.glob("*.log"):
        raw = log.read_bytes()
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            log.write_bytes(raw.decode("utf-16").encode("utf-8"))

    resolved_dir = review / "resolved_configs"
    resolved_dir.mkdir(exist_ok=True)
    chains = {}
    for name in ("paper_reference", "debug"):
        config = load_config(ROOT / f"configs/{name}.yaml")
        config.require_mve_approval()
        export = resolved_dir / f"{name}.yaml"
        export.write_text(yaml.safe_dump(config.data, sort_keys=False), encoding="utf-8")
        reloaded = load_config(export)
        reloaded.require_mve_approval()
        assert reloaded.proposal == config.proposal
        chains[name] = {"authored_config": f"configs/{name}.yaml", "review_chain": config.review_chain}
    write_json(resolved_dir / "review_chains.json", chains)
    versions = {name: importlib.metadata.version(name) for name in ("torch", "pytest", "PyYAML")}
    write_json(review / "environment.json", evidence["environment"] | {
        "core_package_versions": versions,
        "installed_distributions": dict(sorted((dist.metadata["Name"], dist.version)
                                               for dist in importlib.metadata.distributions())),
    })

    manifest_path = review / "manifest.json"
    archive_path = artifact_root / "phase2a_review_package.zip"
    paths = [ROOT / "README.md", ROOT / "pyproject.toml", ROOT / ".gitignore", ROOT / "papers/MFVLR.pdf"]
    for folder in ("src/mfvlr", "tests", "configs", "docs", "tools", "artifacts/phase2a"):
        paths.extend(path for path in (ROOT / folder).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts
                     and path not in (manifest_path, archive_path)
                     and path.suffix not in (".zip", ".sha256", ".pyc"))
    paths = sorted(set(paths), key=lambda p: p.relative_to(ROOT).as_posix())
    manifest = {
        "phase": "2A", "implementation_scope": ["mve"],
        "approved_bundles": ["M1", "M2", "M3", "M4", "M5"],
        "numerical_amendment": chains["debug"]["review_chain"][-1]["numerical_amendments"]["N1"],
        "acceptance_satisfied_under_m1_m5_plus_n1": True,
        "author_fidelity_established": False, "training_authorized": False,
        "final_suite": {"path": "artifacts/phase2a/n1_review/final_pytest.xml", "test_ids": 14,
                        "cases": 15, "passed": 15, "failed": 0, "errors": 0, "skipped": 0,
                        "expected_failures": 0, "junit_seconds": float(suite.attrib["time"]),
                        "host_timestamp": suite.attrib["timestamp"], "outcomes": outcomes},
        "parameters": evidence["parameters"], "mve_source_unchanged": True,
        "phase1_documents_and_original_evidence_unchanged": True,
        "supersedes_acceptance_state_of": "artifacts/phase2a/manifest.json",
        "original_threshold_failure_preserved": "artifacts/phase2a/logs/07_approved_suite.log",
        "environment": evidence["environment"] | {"package_versions": versions},
        "resolved_export_note": "Flattened effective configs retain base review metadata; independent "
                                "authored-profile approval chains are in resolved_configs/review_chains.json.",
        "files_sha256": {path.relative_to(ROOT).as_posix(): digest(path) for path in paths},
        "manifest_and_archive_self_hashes_excluded": True,
    }
    write_json(manifest_path, manifest)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(paths + [manifest_path], key=lambda p: p.relative_to(ROOT).as_posix()):
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for name, expected in manifest["files_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    archive_path.with_suffix(".zip.sha256").write_text(
        f"{digest(archive_path)}  {archive_path.name}\n", encoding="utf-8")
    print(f"Acceptance: 14 IDs / 15 cases passed; no failures, errors, skips or expected failures.")
    print(f"Original source/evidence unchanged; resolved configs reload consistently; parameters=32,078,912.")
    print(f"Archive: {archive_path}; files={len(paths) + 1}; bytes={archive_path.stat().st_size}")
    print(f"SHA-256: {digest(archive_path)}")


if __name__ == "__main__":
    main()
