"""Launch experiment configs in one private Kaggle session, or collect its reports.

The notebook carries the current source, so nothing unreviewed is uploaded, and takes
its data from the private bundle already on the account after checking its hash. One
worker runs per GPU and every run gets its own process, as in the local queue.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    'artifacts/genface_fd_validation_kaggle_20261026/package_001': (
        'nha4von/mfvlr-genface-fd-pilot-20261026', '9bc64eb4e32c1dea5d15a08fe34de6df9a7048c510e3dd84d17e8295e928f073'),
    'artifacts/genface5000_epoch30_kaggle_20261027/package_001': (
        'nha4von/mfvlr-genface5000-epoch30-20261027', '1a372662af9f3e9f3208baef47fea5fda0eb9d218ea403dad6c57771c21bdc1b'),
    'artifacts/genface10k_20261103/package_001': (
        'v5nhdf/mfvlr-genface10k-20261103', '804020c7fab0ee21398b5b34310a49725e55997ac269a97e007d3024a7a6f18c'),
}
# Pretrained encoders by config name: (project path, private dataset, sha256).
WEIGHTS = {
    'resnet50_imagenet': ('artifacts/pretrained/resnet50-11ad3fa6.bin', 'nha4von/mfvlr-resnet50-imagenet',
                          '11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'),
}
BOOTSTRAP = r'''# Private MFVLR experiment; no credentials or test images.
import base64, hashlib, io, json, os, queue, subprocess, sys, threading, time, zipfile
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONUNBUFFERED'] = '1'
PAYLOAD = '@PAYLOAD@'
CONFIGS, SEEDS, BUDGET, MAX_UPDATES = @CONFIGS@, @SEEDS@, @BUDGET@, @MAX_UPDATES@
started = time.time()
inputs = Path(os.environ.get('MFVLR_INPUT', '/kaggle/input'))
project = Path(os.environ.get('MFVLR_PROJECT', '/tmp/mfvlr_project'))
results = Path(os.environ.get('MFVLR_RESULTS', '/kaggle/working/results'))
project.mkdir(parents=True, exist_ok=True)
results.mkdir(parents=True, exist_ok=True)
payload = base64.b64decode(PAYLOAD)
assert hashlib.sha256(payload).hexdigest() == '@PAYLOAD_SHA256@'
zipfile.ZipFile(io.BytesIO(payload)).extractall(project)
candidates = list(inputs.rglob('mfvlr_fd_validation_bundle.bin'))
assert len(candidates) == 1, 'Missing or ambiguous private data bundle'
bundle = candidates[0].read_bytes()
assert hashlib.sha256(bundle).hexdigest() == '@DATA_SHA256@'
folder = project / '@DATA_FOLDER@'
folder.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
    for name in archive.namelist():
        if name.startswith('inputs/') and name.endswith('.npz'):
            (folder / name[len('inputs/'):]).write_bytes(archive.read(name))
del bundle
for relative, digest in @WEIGHTS@:
    found = list(inputs.rglob(Path(relative).name))
    assert len(found) == 1, f'Missing or ambiguous weights {relative}'
    weights = found[0].read_bytes()
    assert hashlib.sha256(weights).hexdigest() == digest
    (project / relative).parent.mkdir(parents=True, exist_ok=True)
    (project / relative).write_bytes(weights)
import cv2, torch
assert torch.cuda.is_available(), 'GPU unavailable: stop rather than silently use CPU'
gpus = torch.cuda.device_count()
log_lock = threading.Lock()


def log(**fields):
    line = json.dumps(dict(fields, elapsed=round(time.time() - started)))
    with log_lock:
        print(line, flush=True)
        with (results / 'queue.jsonl').open('a') as stream:
            stream.write(line + '\n')


log(event='ready', torch=torch.__version__, cv2=cv2.__version__, gpu=torch.cuda.get_device_name(), gpus=gpus)
jobs = queue.Queue()
for seed in SEEDS:
    for name in CONFIGS:
        jobs.put((seed, name))


def command(names, seeds, budget=None):
    line = [sys.executable, '-u', 'tools/run_experiment.py', '--configs', *[f'configs/experiments/{n}.json' for n in names],
            '--seeds', *map(str, seeds), '--output', str(results)]
    if budget is not None:
        line += ['--budget-seconds', str(budget)]
    if MAX_UPDATES is not None:
        line += ['--max-updates', str(MAX_UPDATES)]
    return line


def worker(gpu):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
    while True:
        try:
            seed, name = jobs.get_nowait()
        except queue.Empty:
            return
        # One retry from the last checkpoint, then move on; a pause near the budget keeps its state.
        for attempt in (1, 2):
            left = int(BUDGET - (time.time() - started))
            if left < 300:
                log(event='skipped_no_budget', name=name, seed=seed)
                break
            log(event='start', name=name, seed=seed, gpu=gpu, attempt=attempt)
            with (results / f'{name}_seed{seed}.log').open('a') as stream:
                code = subprocess.run(command([name], [seed], left), cwd=project, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
            log(event='end', name=name, seed=seed, gpu=gpu, attempt=attempt, code=code)
            if code == 0:
                break


threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in range(gpus)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
states = {}
for seed in SEEDS:
    for name in CONFIGS:
        path = results / name / f'seed_{seed}/status.json'
        states[f'{name}/seed_{seed}'] = json.loads(path.read_text())['status'] if path.exists() else 'NOT_STARTED'
log(event='finished', states=states)
if all(state == 'COMPLETED' for state in states.values()):
    # Every run is skipped now, so this only writes the paired comparison.
    subprocess.run(command(CONFIGS, SEEDS), cwd=project, check=True, stdout=subprocess.DEVNULL)
    log(event='summary_written')
'''


def data_folder(names):
    """All arms of one notebook must read the same known dataset."""
    folders = {json.loads((ROOT / f'configs/experiments/{name}.json').read_text(encoding='utf-8'))['inputs'] for name in names}
    if len(folders) != 1 or not folders <= set(DATASETS):
        raise ValueError(f'Configs need one known dataset, got {sorted(folders)}')
    return folders.pop()


def encoders(names):
    found = {json.loads((ROOT / f'configs/experiments/{name}.json').read_text(encoding='utf-8'))['model'].get('encoder', 'scratch')
             for name in names} - {'scratch'}
    if not found <= set(WEIGHTS):
        raise ValueError(f'No Kaggle weights for {sorted(found - set(WEIGHTS))}')
    return sorted(found)


def source_files(names):
    # Tokenizer assets exceed the notebook size limit, so only image-only configs can run this way.
    for name in names:
        weights = json.loads((ROOT / f'configs/experiments/{name}.json').read_text(encoding='utf-8'))['loss_weights']
        if any(weights[key] > 0 for key in ('lr', 'cmc', 'kl')):
            raise ValueError(f'{name} uses language losses, which need the tokenizer assets')
    files = [path for folder in ('src/mfvlr', 'src/mfvlr_research') for path in sorted((ROOT / folder).rglob('*'))
             if path.is_file() and not {'__pycache__', 'tokenizer_assets'} & set(path.parts) and not path.parent.name.startswith('d02b')]
    files += [ROOT / 'tools/run_experiment.py', *[ROOT / f'configs/experiments/{name}.json' for name in names]]
    return files


def build(names, seeds, budget, max_updates):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in source_files(names):
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix())
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    payload = buffer.getvalue()
    script = BOOTSTRAP
    folder = data_folder(names)
    for key, value in (('@PAYLOAD@', base64.b64encode(payload).decode()), ('@PAYLOAD_SHA256@', hashlib.sha256(payload).hexdigest()),
                       ('@DATA_SHA256@', DATASETS[folder][1]), ('@DATA_FOLDER@', folder), ('@CONFIGS@', repr(names)),
                       ('@SEEDS@', repr(seeds)), ('@BUDGET@', str(budget)), ('@MAX_UPDATES@', repr(max_updates)),
                       ('@WEIGHTS@', repr([(WEIGHTS[e][0], WEIGHTS[e][2]) for e in encoders(names)]))):
        script = script.replace(key, value)
    compile(script, 'run.py', 'exec')
    if len(script.encode()) >= 1000000:
        raise ValueError(f'Notebook too large: {len(script.encode())} bytes')
    return script


def api_client():
    sys.path.insert(0, str(ROOT / 'tmp/kaggle_current_deps'))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slug', required=True, help='Kernel slug under the owner')
    parser.add_argument('--owner', default='nha4von')
    parser.add_argument('--output', type=Path, required=True, help='Local record folder')
    parser.add_argument('--configs', nargs='+', help='Config names; the first is the paired baseline')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--budget-seconds', type=int, default=37800, help='Stop starting work after this long')
    parser.add_argument('--max-updates', type=int, help='Smoke limit')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    ref = f'{args.owner}/{args.slug}'
    args.output.mkdir(parents=True, exist_ok=True)
    if args.collect:
        api = api_client()
        print(json.dumps(dict(status=str(api.kernels_status(ref).status)), indent=2))
        files, _ = api.kernels_output(ref, str(args.output / 'downloaded'), quiet=True,
                                      file_pattern=r'results/.*(report\.json|status\.json|summary\.json|history\.jsonl|events\.jsonl|queue\.jsonl|\.log)$')
        print(json.dumps(dict(downloaded=len(files)), indent=2))
        return
    if not args.configs:
        parser.error('--configs is required unless collecting')
    folder = args.output / 'notebook'
    folder.mkdir(exist_ok=True)
    script = build(args.configs, args.seeds, args.budget_seconds, args.max_updates)
    (folder / 'run.py').write_text(script, encoding='utf-8')
    metadata = dict(id=ref, title=args.slug.replace('-', ' ').title(), code_file='run.py', language='python',
                    kernel_type='script', is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=False,
                    machine_shape='NvidiaTeslaT4',
                    dataset_sources=[DATASETS[data_folder(args.configs)][0], *(WEIGHTS[e][1] for e in encoders(args.configs))])
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(dict(notebook_bytes=len(script.encode()), ref=ref, private=True, configs=args.configs, seeds=args.seeds), indent=2), flush=True)
    if args.launch:
        record = args.output / 'launch_record.json'
        if record.exists():
            raise ValueError('Already launched; inspect the existing notebook first')
        response = api_client().kernels_push(str(folder), timeout=str(args.budget_seconds + 1800), acc='NvidiaTeslaT4')
        result = dict(url=response.url, version=response.version_number, ref=ref, configs=args.configs, seeds=args.seeds)
        record.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
