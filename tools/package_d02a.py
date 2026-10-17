"""Freeze the completed pilot without modifying any accepted or historical file."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.d02a.common import OUT,sha,write


def main(analysis):
    coverage=json.loads((analysis/'coverage.json').read_text())
    hard_stops=[j for j in coverage['jobs'] if j['execution']=='FAIL']
    if coverage['complete']!=414 and not hard_stops:
        raise RuntimeError('Full registered local grid is incomplete; final packaging requires completion or an explicit hard-stop report.')
    for d in (OUT/'jobs').iterdir():
        if not (d/'summary.json').exists():raise RuntimeError('Unfinished worker evidence: '+str(d))
    baseline=json.loads((OUT/'preservation_before.json').read_text())
    after={};drift=[]
    for name,entry in baseline.items():
        p=ROOT/name
        record={'sha256':sha(p),'bytes':p.stat().st_size} if p.is_file() else None
        after[name]=record
        if record!=entry:drift.append(name)
    write(OUT/'preservation_after.json',after)
    write(OUT/'preservation_verification.json',{'status':'FAIL' if drift else 'PASS',
        'frozen_files':len(baseline),'changed_or_missing':drift,
        'before_manifest_sha256':sha(OUT/'preservation_before.json'),
        'after_manifest_sha256':sha(OUT/'preservation_after.json'),
        'verified_utc':datetime.now(timezone.utc).isoformat(),
        'production_architecture_changes':False,'threshold_changes':False,
        'historical_VD_T12_incident':'preserved; not retroactively authorized'})
    if drift:raise RuntimeError('Frozen baseline/historical drift: '+repr(drift))
    reproduction='''# D02A numerical pilot review package

This package records local no-update A0/P3/Q3 comparisons. Phase 2D.2 remains BLOCKED.
Read docs/phase2d2_d02a_numerical_pilot_report.md and artifacts/phase2d2_d02a/analysis/ first.
Do not run the complete pytest suite: legacy MVE-T10 and VD-T12 are forbidden in this pilot.
No full-model ablation, optimizer, training, GPU or batch-8 execution is authorized here.

## Integrity

Verify every entry in artifacts/phase2d2_d02a/review_manifest.json against the extracted
bytes. The SHA-256 of that manifest and the archive itself are supplied separately in
review_package_record.json. preservation_before/after.json seal the 1,594 original
accepted/historical files in the original workspace; historical ZIP payloads are not
nested in this archive. Original logs are retained in the workspace.

The nine frozen .pt decoder states and ALL files of the interrupted attempts,
including their temporary gradients, are included byte-for-byte in this ZIP.
Generated canonical .npz/.npy activation caches are omitted from this ZIP;
cache_manifest.json records every omitted path, size and SHA-256. They remain on disk
in the original workspace. FP32 decoder states can also be regenerated from frozen
source and seeds6101..6103 using prepare(); inputs use the exact registered seeds.
Raw per-job comparisons, input/state tensor hashes, full CSV/JSON tables, logs and
JUnits are included. Exclusion is for archive size, not selective numerical reporting.

## Reproduction in a separate fresh workspace

Use the recorded Windows/Python/Torch/NumPy dependency versions in environment_detail.json
and dependency_versions.json. pyproject.toml minimum versions alone are not a numerical
reproduction pin. Cross-platform bitwise equivalence is not assumed. Install only the
needed dependencies; no model or tokenizer pretrained weights are loaded by this pilot.

Keep the submitted artifacts unchanged in the review copy. Create a second clean
workspace containing src/, tests/, tools/, docs/, configs/, assets/, pyproject.toml and
the baseline historical JSON files under artifacts/phase2d2_a11_reconciliation/.
Create a fresh artifacts/phase2d2_d02a/ containing only copies of preservation_before.json,
driver_manifest_v2.json and LN_preexecution_input_manifest.json from the review package.
Do not overwrite an existing job directory or historical log.

From the fresh workspace:

    python -m pip install -e . --no-deps
    python tools/d02a/run.py prepare
    python -m pytest tests/research_a11/test_primitives.py -p tools.d02a.safety -vv -s --junitxml=oracle_replay.xml

The reviewed driver tests live in artifacts/phase2d2_d02a/test_driver_oracles.py;
copy that test file (not its old result directories) to the new workspace and run:

    python -m pytest artifacts/phase2d2_d02a/test_driver_oracles.py -p tools.d02a.safety -vv -s --junitxml=driver_replay.xml

For numerical workers, use this PowerShell host loop in the fresh workspace.
It keeps the original driver unchanged, releases one registered worker at a time,
and records the 3 GiB available-RAM gate before every release. Preserve its console
output alongside the driver's per-job logs. Stop on any nonzero driver exit.

    for ($pilotIndex = 0; $pilotIndex -lt 414; $pilotIndex++) {
        do {
            $pilotAvailable = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1024
            Write-Output ([pscustomobject]@{utc=[DateTime]::UtcNow.ToString('o'); index=$pilotIndex; available_RAM=$pilotAvailable; minimum_RAM=3GB} | ConvertTo-Json -Compress)
            if ($pilotAvailable -lt 3GB) { Start-Sleep -Seconds 5 }
        } while ($pilotAvailable -lt 3GB)
        python tools/d02a/run.py run --start $pilotIndex --stop ($pilotIndex + 1)
        if ($LASTEXITCODE -ne 0) { throw 'Pilot stopped; inspect before any further execution.' }
    }

This is the registered **local** schedule only. It can take many CPU hours. Preserve
all stdout/stderr, exit codes and JSON records. A hard stop forbids silently retrying
or reducing fixture dimensions. Finite diagnostic mismatches remain FAIL comparisons.
Do not modify tolerances or invoke the still-locked full A1-D02 launcher.

After all jobs complete, aggregate into a new directory:

    python tools/summarize_d02a.py --out artifacts/phase2d2_d02a/replay_analysis

The original run's 26 pytest cases plus eight versioned RAM-guard checks are selected helper checks,
not 276 fresh regression cases. Any historical results are explicitly labeled in the
report. Diagnostic exit-zero is not equivalent to passing all numerical comparisons.
'''
    with (OUT/'REPRODUCE.md').open('x',encoding='utf-8') as handle:handle.write(reproduction)
    files=set()
    # Include all original baseline logs/manifests and non-archive evidence too;
    # keep historical ZIP payloads on disk rather than nesting archives.
    files.update(ROOT/name for name in baseline if not name.endswith('.zip'))
    for folder in ('src','tests','tools','configs','docs','assets','papers'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:files.add(p)
    for pattern in ('pyproject.toml','requirements*.txt','README*'):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    # Include frozen acceptance/reconciliation metadata needed by the local driver,
    # without nesting historical archives or launching their experiments.
    for folder in ('artifacts/phase2d2_a11_reconciliation',):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix!='.zip' and '__pycache__' not in p.parts:files.add(p)
    excluded={}
    for p in OUT.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or '.pytest_cache' in p.parts:continue
        if p.suffix in ('.pt','.npz','.npy') and 'history' not in p.relative_to(OUT).parts and 'states' not in p.relative_to(OUT).parts:
            excluded[p.relative_to(ROOT).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
        elif p.suffix!='.zip':files.add(p)
    write(OUT/'cache_manifest.json',{'reason':'Generated tensor caches omitted from ZIP; full raw numerical records retained.',
        'files':excluded,'bytes':sum(x['bytes'] for x in excluded.values())})
    files.add(OUT/'cache_manifest.json')
    entries={p.relative_to(ROOT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)}
    manifest=OUT/'review_manifest.json'
    write(manifest,{'scope':'A1-D02A local no-update pilot','phase2d2_status':'BLOCKED',
        'files':entries,'count':len(entries),'included_bytes':sum(x['bytes'] for x in entries.values()),
        'cache_manifest_sha256':sha(OUT/'cache_manifest.json'),'manifest_self_included':False})
    files.add(manifest);archive=OUT/'phase2d2_d02a_review.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
    # Independently read and SHA every archived entry, not just ZIP CRC metadata.
    with zipfile.ZipFile(archive) as z:
        for name,entry in entries.items():
            with z.open(name) as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
            if actual!=entry['sha256']:raise AssertionError('Archive SHA mismatch: '+name)
        with z.open(manifest.relative_to(ROOT).as_posix()) as handle:
            assert hashlib.file_digest(handle,'sha256').hexdigest()==sha(manifest)
    write(OUT/'review_package_record.json',{'archive':archive.relative_to(ROOT).as_posix(),
        'bytes':archive.stat().st_size,'sha256':sha(archive),'manifest_sha256':sha(manifest),
        'archive_entry_SHA256_verification':'PASS','included_files':len(files),
        'omitted_generated_cache_files':len(excluded),'original_artifacts_preserved':True})
    print(json.dumps({'zip':str(archive),'bytes':archive.stat().st_size,'baseline_preservation':'PASS'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--analysis',type=Path,required=True)
    main(parser.parse_args().analysis)
