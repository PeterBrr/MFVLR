"""Prepare and launch private FD-weight ablation using the already-approved bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tmp/kaggle_current_deps'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    folder = ROOT / 'artifacts/genface_fd_weight_kaggle_20261026/notebook_001'
    folder.mkdir(parents=True, exist_ok=True)
    wrapper = (ROOT / 'tools/run_fd_weight_ablation.py').read_text(encoding='utf-8')
    digest = hashlib.sha256(wrapper.encode()).hexdigest()
    script = f'''# Private GenFace FD weight experiment, 16 fixed training images.
import hashlib, io, json, os, subprocess, sys, zipfile
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONUNBUFFERED'] = '1'
print('BOOTSTRAP: looking for the private 16-image bundle', flush=True)
candidates = list(Path('/kaggle/input').rglob('mfvlr_loss_bundle.bin'))
assert len(candidates) == 1, 'Missing or ambiguous private bundle'
payload = candidates[0].read_bytes()
assert hashlib.sha256(payload).hexdigest() == 'f5e3db0c5bb6df066af13e49c4e859eb362adfc81a39a74b710ad4f36edae0b9'
project = Path('/kaggle/working/mfvlr_fd_project')
project.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in archive.namelist())
    archive.extractall(project)
source = json.loads((project/'inputs/source_sha256.json').read_text())
assert all(hashlib.sha256((project/p).read_bytes()).hexdigest() == h for p,h in source.items())
print('BOOTSTRAP: bundle and original source hashes verified', flush=True)
wrapper = {wrapper!r}
assert hashlib.sha256(wrapper.encode()).hexdigest() == {digest!r}
(project/'tools/run_fd_weight_ablation.py').write_text(wrapper, encoding='utf-8')
import torch
assert torch.cuda.is_available(), 'GPU unavailable: stop rather than silently use CPU'
print(json.dumps({{'event':'gpu_ready','torch':torch.__version__,'cuda':torch.version.cuda,
                   'gpu':torch.cuda.get_device_name(),'gpu_count':torch.cuda.device_count()}}), flush=True)
subprocess.run([sys.executable, '-u', str(project/'tools/run_fd_weight_ablation.py'),
                '--data', str(project/'inputs/train16.npz'), '--config', str(project/'inputs/config.json'),
                '--output', '/kaggle/working/fd_weight_results', '--updates', '400', '--gradient-pairs', '4'],
               cwd=project, check=True)
print('EXPERIMENT COMPLETED: see fd_weight_results/report.json', flush=True)
'''
    compile(script, 'kaggle_fd_bootstrap.py', 'exec')
    assert len(script.encode()) < 1000000
    (folder / 'run.py').write_text(script, encoding='utf-8')
    metadata = dict(id='nha4von/mfvlr-genface-fd-weights-20261026-001',
                    title='MFVLR GenFace FD Weights 20261026 001', code_file='run.py',
                    language='python', kernel_type='script', is_private=True,
                    enable_gpu=True, enable_tpu=False, enable_internet=False,
                    machine_shape='NvidiaTeslaT4',
                    dataset_sources=['nha4von/mfvlr-genface-loss16-20261026'])
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(dict(prepared=str(folder), wrapper_sha256=digest,
                          notebook_bytes=len(script.encode()), private=True), indent=2), flush=True)
    if args.prepare_only:
        return
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    response = api.kernels_push(str(folder), timeout='7200', acc='NvidiaTeslaT4')
    result = dict(url=response.url, version_number=response.version_number,
                  ref=metadata['id'], wrapper_sha256=digest)
    (folder.parent / 'launch_record.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
