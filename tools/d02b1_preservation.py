"""File-only preservation audit. Does not import any model or numerical driver."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/phase2d2_d02b1'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}

def main(mode):
    OUT.mkdir(parents=True, exist_ok=True)
    if mode == 'before':
        historical = json.loads((ROOT / 'artifacts/phase2d2_d02a/preservation_before.json').read_text())
        mismatches = [p for p, record in historical.items() if digest(ROOT / p) != record]
        if mismatches:
            raise RuntimeError(mismatches)
        paths = set(historical)
        for folder in ('src', 'configs', 'tests', 'tools', 'docs'):
            paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / folder).rglob('*')
                         if p.is_file() and '__pycache__' not in p.parts and 'd02b1' not in p.as_posix())
        paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / 'artifacts/phase2d2_d02b0').rglob('*') if p.is_file())
        for folder, names in {
            'phase2d2_d02a': ('review_manifest.json', 'pilot_protocol.json', 'driver_manifest_v2.json', 'cache_manifest.json', 'phase2d2_d02a_review.zip'),
            'phase2d2_d02a_review_lite': ('review_lite_manifest.json', 'review_lite_verification.json', 'phase2d2_d02a_review_lite.zip'),
        }.items():
            paths.update(f'artifacts/{folder}/{n}' for n in names)
        record = {'historical_checked': len(historical), 'historical_mismatches': mismatches,
                  'files': {p: digest(ROOT / p) for p in sorted(paths)}}
        with (OUT / 'baseline_before.json').open('x') as f:
            json.dump(record, f, indent=2)
    elif mode == 'after':
        before = json.loads((OUT / 'baseline_before.json').read_text())
        mismatches = [p for p, v in before['files'].items() if digest(ROOT / p) != v]
        record = {'checked': len(before['files']), 'mismatches': mismatches,
                  'status': 'FAIL' if mismatches else 'PASS', 'baseline_sha256': digest(OUT / 'baseline_before.json')['sha256']}
        with (OUT / 'baseline_after.json').open('x') as f:
            json.dump(record, f, indent=2)
        if mismatches:
            raise RuntimeError(mismatches)
    else:
        raise ValueError(mode)
    print(json.dumps({k: v for k, v in record.items() if k != 'files'}))

if __name__ == '__main__':
    main(sys.argv[1])
