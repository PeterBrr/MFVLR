"""Continue frozen Kaggle mask-weight runs locally, one GPU job at a time."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REMOTE = ROOT / 'artifacts/language_mask_weights_kaggle_20261029'
BASE = ROOT / 'artifacts/language5000_scratch_kaggle_20261028'
OUT = ROOT / 'artifacts/language_mask_weights_local_20261029/run_001'
PYTHON = Path(r'C:\Users\NHA\miniconda3\envs\my_env\python.exe')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_bytes(json.dumps(value, indent=2, ensure_ascii=False).encode('utf-8'))


WRAPPER = '''import gc,json,runpy,sys,hashlib
from pathlib import Path
import torch
runner=Path(sys.argv[1]).resolve()
resume=Path(sys.argv[sys.argv.index('--resume')+1])
checkpoint=torch.load(resume,map_location='cpu',weights_only=False)
assert checkpoint['schema']=='mfvlr-language5000-stage1-v1'
assert len(checkpoint['cuda_rng']) in (1,2)
for name,expected in checkpoint['provenance']['source'].items():
    file=runner.parents[1]/name.replace(chr(92),'/')
    assert hashlib.sha256(file.read_bytes()).hexdigest()==expected
print(json.dumps(dict(event='local_checkpoint_checked',step=checkpoint['step'],
    torch=torch.__version__,gpu=torch.cuda.get_device_name(),
    saved_cuda_devices=len(checkpoint['cuda_rng']),active_training_device=0,
    policy='Restore trained GPU0 RNG; ignore unused GPU1 RNG on a one-GPU host')),flush=True)
del checkpoint;gc.collect()
original_load=torch.load
def load_with_local_path_keys(path,*args,**kwargs):
    state=original_load(path,*args,**kwargs)
    if Path(path).resolve()==resume.resolve():
        source=state['provenance']['source']
        state['provenance']['source']={str(Path(k)) if k.replace(chr(92),'/').startswith('src/') else k:v for k,v in source.items()}
        print(json.dumps(dict(event='source_path_separator_compatibility',
            policy='Source file hashes already verified; normalize src dictionary keys for Windows only')),flush=True)
    return state
torch.load=load_with_local_path_keys
original=torch.cuda.set_rng_state_all
def restore_active_gpu(states):
    assert torch.cuda.device_count()==1 and len(states) in (1,2)
    original(states[:1])
torch.cuda.set_rng_state_all=restore_active_gpu
sys.path.insert(0,str(runner.parent))
sys.argv=sys.argv[1:]
runpy.run_path(str(runner),run_name='__main__')
'''


def prepare():
    assert not OUT.exists(), 'Use a fresh local run directory'
    OUT.mkdir(parents=True)
    bundle = BASE / 'package_001/dataset_upload/mfvlr_language_assets.bin'
    manifest = json.loads((BASE / 'package_001/package_manifest.json').read_text())
    assert sha(bundle) == manifest['bundle_sha256']
    data_bundle = ROOT / 'artifacts/genface5000_epoch30_kaggle_20261027/package_001/dataset_upload/mfvlr_fd_validation_bundle.bin'
    assert sha(data_bundle) == manifest['original_bundle_sha256']
    with zipfile.ZipFile(data_bundle) as archive:
        for name in ('inputs/train.npz', 'inputs/val.npz'):
            archive.extract(name, OUT)
    assert all(sha(OUT / 'inputs' / (s + '.npz')) == h for s, h in manifest['npz_sha256'].items())
    for weight, step in ((2, 11818), (1, 12106)):
        project = OUT / f'weight_{weight}/project'
        with zipfile.ZipFile(bundle) as archive:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in archive.namelist())
            archive.extractall(project)
        config_path = project / 'assets/config.json'
        config = json.loads(config_path.read_text())
        config['model']['mask_foreground_weight'] = float(weight)
        write(config_path, config)
        assets_path = project / 'assets/manifest.json'
        assets = json.loads(assets_path.read_text())
        assets['files']['config.json'] = sha(config_path)
        write(assets_path, assets)
        runner = project / 'tools/run_language5000.py'
        code = runner.read_text(encoding='utf-8')
        old = "weights=torch.tensor([1.,4.],device=device)"
        assert code.count(old) == code.count('time.monotonic()-start>8*3600') == 1
        code = code.replace(old, "weights=torch.tensor([config['model']['mask_background_weight'],config['model']['mask_foreground_weight']],device=device)")
        code = code.replace('time.monotonic()-start>8*3600', 'time.monotonic()-start>8600')
        runner.write_bytes(code.encode('utf-8'))
        report_path = REMOTE / f'weight_{weight}_stage_2/downloaded/language5000_results/report.json'
        report = json.loads(report_path.read_text())
        assert report['completed_updates'] == step and report['status'] == 'PARTIAL_TIME_BUDGET'
        assert report['config'] == config and report['asset_sha256'] == assets['files']
        assert report['data_sha256'] == manifest['npz_sha256']
        assert all(sha(project / p) == h for p, h in report['source_sha256'].items())
        write(project / 'source_sha256.json', {**manifest['source_sha256'], **report['source_sha256']})
        write(OUT / f'weight_{weight}/origin.json', dict(remote_report=str(report_path),
            start_step=step, checkpoint=str(report_path.parent / 'last.pt'),
            source_sha256=report['source_sha256'], config=config, assets=assets['files']))
    (OUT / 'resume_wrapper.py').write_bytes(WRAPPER.encode('utf-8'))
    write(OUT / 'plan.json', dict(weights=[2, 1], target_epochs=15, target_updates=18750,
        execution='Sequential, alternate weights after each 8600-second segment until completed',
        test_used=False, source_policy='Byte-identical Kaggle source/assets; no provenance rewrite',
        environment_change='Kaggle T4 torch2.11 to local RTX4050 torch2.7; not bitwise equivalent',
        rng_policy='Restore CPU and trained CUDA0 RNG; unused CUDA1 RNG omitted on local single GPU'))
    print('Prepared frozen local projects and verified all source/data/assets hashes.', flush=True)


def run(restart_failed=False, resume_stopped=False, resume_interrupted=False):
    assert (OUT / 'plan.json').exists()
    marker = OUT / 'queue_started.json'
    assert not marker.exists() or restart_failed or resume_stopped or resume_interrupted, 'Queue already started; inspect status before relaunching'
    import os
    import psutil
    if marker.exists():
        previous_pid = json.loads(marker.read_text(encoding='utf-8'))['pid']
        if psutil.pid_exists(previous_pid):
            command = ' '.join(psutil.Process(previous_pid).cmdline())
            assert 'continue_mask_weights_local.py' not in command, 'Previous supervisor is still running'
    if restart_failed or resume_stopped or resume_interrupted:
        states = json.loads((OUT / 'queue_status.json').read_text(encoding='utf-8'))
        if not resume_interrupted:
            expected = 'STOPPED_BY_USER' if resume_stopped else 'FAILED'
            assert any(s['status'] == expected for s in states.values())
            assert not any(s['status'] == 'RUNNING' for s in states.values())
        else:
            assert not any(s['status'] == 'STOPPED_BY_USER' for s in states.values()), 'User-stopped runs need --resume-stopped'
        for state in states.values():
            pid = state.get('worker_pid')
            if pid and psutil.pid_exists(pid):
                assert 'resume_wrapper.py' not in ' '.join(psutil.Process(pid).cmdline()), 'Previous worker is still running'
            if resume_interrupted and state.get('output'):
                saved = Path(state['output']) / 'last.pt'
                assert saved.exists(), 'Missing durable checkpoint; do not guess resume step'
                import torch
                import gc
                checkpoint = torch.load(saved, map_location='cpu', weights_only=False)
                assert checkpoint['schema'] == 'mfvlr-language5000-stage1-v1'
                assert checkpoint['step'] >= state['step']
                state.update(step=checkpoint['step'], checkpoint=str(saved))
                del checkpoint
                gc.collect()
            if state['step'] < 18750:
                state['status'] = 'QUEUED'
            state.pop('exit_code', None)
            state.pop('worker_pid', None)
    else:
        states = {}
        for weight in (2, 1):
            origin = json.loads((OUT / f'weight_{weight}/origin.json').read_text())
            checkpoint = Path(origin['checkpoint'])
            states[str(weight)] = dict(step=origin['start_step'], checkpoint=str(checkpoint),
                                      status='QUEUED', segment=0)
    if restart_failed:
        (OUT / 'resume_wrapper.py').write_bytes(WRAPPER.encode('utf-8'))
    write(marker, dict(pid=os.getpid(), wrapper_sha256=sha(OUT / 'resume_wrapper.py'),
                       supervisor_sha256=sha(Path(__file__))))
    def save():
        write(OUT / 'queue_status.json', states)
    save()
    while any(s['step'] < 18750 for s in states.values()):
        for weight in (2, 1):
            state = states[str(weight)]
            if state['step'] >= 18750:
                continue
            checkpoint = Path(state['checkpoint'])
            while not checkpoint.exists() or checkpoint.stat().st_size < 1_000_000_000 or checkpoint.with_name(checkpoint.name + '.kaggle-partial').exists():
                state['status'] = 'WAITING_FOR_DOWNLOAD'
                save()
                time.sleep(15)
            if state['segment'] == 0:
                state['input_checkpoint_sha256'] = sha(checkpoint)
            state['segment'] += 1
            folder = OUT / f'weight_{weight}'
            output = folder / f'segment_{state["segment"]:03d}'
            assert not output.exists()
            project = folder / 'project'
            command = [str(PYTHON), '-u', str(OUT / 'resume_wrapper.py'),
                str(project / 'tools/run_language5000.py'), '--inputs', str(OUT / 'inputs'),
                '--assets', str(project / 'assets'), '--epochs', '15', '--end-epoch', '15',
                '--resume', state['checkpoint'], '--output', str(output)]
            state.update(status='RUNNING', output=str(output))
            save()
            print(json.dumps(dict(event='local_segment_start', weight=weight, **state)), flush=True)
            with (folder / f'segment_{state["segment"]:03d}.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state['worker_pid'] = process.pid
                save()
                exit_code = process.wait()
            if exit_code:
                state.update(status='FAILED', exit_code=exit_code)
                save()
                raise RuntimeError(f'Weight {weight} failed; inspect {folder}')
            report = json.loads((output / 'report.json').read_text())
            assert report['completed_updates'] > state['step']
            assert report['source_unchanged'] and not report['test_used']
            state.update(status=report['status'], step=report['completed_updates'],
                         checkpoint=str(output / 'last.pt'))
            save()
    print('Both weights completed 15 epochs.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--restart-failed', action='store_true')
    parser.add_argument('--resume-stopped', action='store_true')
    parser.add_argument('--resume-interrupted', action='store_true')
    args = parser.parse_args()
    assert sum((args.restart_failed, args.resume_stopped, args.resume_interrupted)) <= 1
    prepare() if args.prepare else run(args.restart_failed, args.resume_stopped, args.resume_interrupted)
