"""Continue the existing local registration and summarize it; no scientific changes.

Each subprocess has separate exclusive raw logs and an exit record. An error
stops this wrapper. It never retries a scientific failure or runs pytest suites.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/phase2d2_d02a'
RECORDS = OUT / 'continuation_0122'


def write(path, data):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)


def run(label, args):
    command = [sys.executable, *args]
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    with (RECORDS / (label + '.stdout.log')).open('x') as stdout, (RECORDS / (label + '.stderr.log')).open('x') as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    write(RECORDS / (label + '.execution.json'), {
        'command': command, 'started_utc': started,
        'elapsed_seconds': time.perf_counter() - clock, 'exit_code': result.returncode,
    })
    if result.returncode:
        raise RuntimeError(f'{label} exited {result.returncode}; no automatic retry')


def main():
    RECORDS.mkdir(exist_ok=False)
    replay = json.loads((OUT / 'history/interruption_0121_20261013/replay_comparison.json').read_text())
    assert replay['status'] == 'PASS', 'Replay discrepancy requires review before continuation'
    for job in (OUT / 'jobs').iterdir():
        summary = json.loads((job / 'summary.json').read_text())
        assert summary['status'] == 'COMPLETED_LOCAL_DIAGNOSTIC', job
    write(RECORDS / 'scope.json', {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'registered_range': [122, 414], 'stop_exclusive': True,
        'completed_cells_skipped': True, 'retries': False,
        'scope': 'Existing locked local NLD/NATT only; no full-model sweep or optimizer',
        'MVE_T10': 'NOT RUN', 'VD_T12': 'NOT RUN',
    })
    run('registered_local', ['tools/d02a/run.py', 'run', '--start', '122', '--stop', '414'])
    run('aggregation', ['tools/summarize_d02a.py', '--out', 'artifacts/phase2d2_d02a/analysis'])
    run('report', ['tools/report_d02a.py', '--analysis', 'artifacts/phase2d2_d02a/analysis', '--report', 'docs/phase2d2_d02a_numerical_pilot_report.md'])
    write(RECORDS / 'complete.json', {'status': 'COMPLETED_PENDING_REPORT_REVIEW_AND_PACKAGING', 'phase_2D2': 'BLOCKED'})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if RECORDS.is_dir():
            write(RECORDS / 'stopped.json', {'error': repr(exc), 'automatic_retry': False})
        raise
