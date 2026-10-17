"""Resume after a missing output directory, preserving the failed execution."""
from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'.phase2c1_deps'), str(ROOT/'src'), str(ROOT)]
from tools.n31_evidence import read, write, sources, sha, seal, negative_tests, validate
from tools.run_phase2d2_n31 import command
from tools import run_phase2d2 as runner


def main():
    out = ROOT/'artifacts/phase2d2_n31'
    start = read(out/'workflow_start.json')
    assert sources() == start['sources'], 'Pinned sources changed'
    assert not (out/'evidence_seal.json').exists(), 'Already sealed'
    failure = out/'logs/fresh_n2_fixture0_float32_threads4.stderr.log'
    assert 'FileNotFoundError' in failure.read_text()
    assert not (out/'n2').exists()
    write(out/'resume_record.json', {
        'reason': 'Original runner omitted creation of n2 output directory; worker failed only when writing its result.',
        'original_stderr_sha256': sha(failure), 'original_run_id': start['run_id'],
        'sources_unchanged': True, 'fixture_or_threshold_changes': False,
        'resume_source_sha256': sha(__file__),
        'action': 'Create missing directory and rerun all six N2 workers with distinct retry log names; retain original failure.'})
    (out/'n2').mkdir()
    runner.OUT = out
    for index in range(3):
        for dtype in ('float32', 'float64'):
            name = f'fixture{index}_{dtype}_threads4'
            code = command(out, 'retry_n2_'+name, [sys.executable,
                str(ROOT/'tools/review_phase2c2_n2.py'), '--worker', str(index),
                '--dtype', dtype, '--threads', '4', '--output', str(out/'n2'/f'{name}.json')])
            if code: raise RuntimeError('N2 retry failed; preserve all evidence')
    seal(out)
    negative_tests(out)
    os.environ.update(MFVLR_N3_REVIEW_ROOT=str(ROOT), MFVLR_N3_EVIDENCE_ROOT=str(out),
        MFVLR_N31_RUN_ID=start['run_id'], MFVLR_PHASE2D2_EVIDENCE_ROOT=str(out))
    code = runner.pytest_run('supplemental_01', ['tests/joint/test_n3_controls.py',
        '-p', 'tools.n31_evidence', '-p', 'no:cacheprovider'])
    if code: raise RuntimeError('Supplemental tests failed')
    validate(out, start['run_id'])
    code = runner.pytest_run('complete_01', ['tests', '-p', 'tools.n31_evidence', '-p', 'no:cacheprovider'])
    write(out/'suite_exit.json', {'complete_exit_code': code, 'supplemental_exit_code': 0,
        'acceptance': 'BLOCKED', 'no_existing_assertions_removed': True, 'no_threshold_changes': True})
    print('Verification completed; inspect actual case results. Phase 2D.2 BLOCKED.', flush=True)


if __name__ == '__main__': main()
