"""Bundle only selected GenFace tensors and required source into a private Kaggle script."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.pilot_deps'), str(ROOT / 'src'), str(ROOT / 'tools')]
import numpy as np
import torch
from mfvlr_research.pilot_datasets import dataset_for_config
from test_split_mask_encoder_pilot import balanced_subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    args.output.mkdir(parents=True)
    config = json.loads((ROOT / 'configs/diffae_pilot_skip_detach.json').read_text())
    config['model']['reconstruction_activation'] = 'identity'
    samples = balanced_subset(dataset_for_config(ROOT, config, 'train', limit=1000), 16)
    images = np.stack([(s['image'] * 255).round().byte().numpy() for s in samples])
    masks = np.stack([s['mask'].byte().numpy() for s in samples])
    assert all(torch.equal(torch.from_numpy(rgb).float().div_(255), s['image']) for rgb, s in zip(images, samples))
    assert all(masks[i].any() for i in range(1, 16, 2))
    np.savez_compressed(args.output / 'train16.npz', images=images, masks=masks,
                        labels=np.array([s['label'] for s in samples]),
                        sample_ids=np.array([s['sample_id'] for s in samples]))
    # Local machine paths/manifests are not needed in the portable training script.
    portable = {key: config[key] for key in ('model', 'loss_weights')}
    (args.output / 'config.json').write_text(json.dumps(portable, indent=2), encoding='utf-8')
    tools = ['run_loss_overfit.py', 'measure_ar_fl_gradients.py', 'test_split_mask_encoder_pilot.py',
             'test_ad_overfit.py', 'export_ad_reconstructions.py', 'pilot_checkpointing.py']
    files = [*ROOT.joinpath('src/mfvlr').glob('*.py'), *ROOT.joinpath('src/mfvlr').glob('*.json'),
             *ROOT.joinpath('src/mfvlr_research').glob('pilot_*.py'), ROOT / 'src/mfvlr_research/__init__.py',
             *(ROOT / 'tools' / name for name in tools)]
    hashes = {}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            blob = path.read_bytes()
            hashes[relative] = hashlib.sha256(blob).hexdigest()
            archive.writestr(relative, blob)
        for name in ('train16.npz', 'config.json'):
            archive.write(args.output / name, 'inputs/' + name)
        archive.writestr('inputs/source_sha256.json', json.dumps(hashes, indent=2))
    blob = stream.getvalue()
    digest = hashlib.sha256(blob).hexdigest()
    script = '''# Private MFVLR GenFace training-set loss diagnostic; embedded 16-image input.
import base64, hashlib, io, json, os, subprocess, sys, zipfile
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
payload = base64.b64decode(PAYLOAD_LITERAL)
assert hashlib.sha256(payload).hexdigest() == DIGEST_LITERAL
project = Path('/kaggle/working/mfvlr_loss_project')
project.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in archive.namelist())
    archive.extractall(project)
source = json.loads((project/'inputs/source_sha256.json').read_text())
assert all(hashlib.sha256((project/p).read_bytes()).hexdigest() == h for p,h in source.items())
import torch, numpy, PIL, yaml, tqdm
assert torch.cuda.is_available(), 'GPU unavailable; stop rather than train on CPU'
print(json.dumps({'torch':torch.__version__, 'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(),
                  'gpu_count':torch.cuda.device_count(), 'payload_sha256':DIGEST_LITERAL}), flush=True)
subprocess.run([sys.executable, str(project/'tools/run_loss_overfit.py'),
                '--data', str(project/'inputs/train16.npz'), '--config', str(project/'inputs/config.json'),
                '--output', '/kaggle/working/loss_overfit_results', '--updates', '400', '--gradient-pairs', '4'],
               cwd=project, check=True)
'''
    script = script.replace('PAYLOAD_LITERAL', repr(base64.b64encode(blob).decode('ascii')))
    script = script.replace('DIGEST_LITERAL', repr(digest))
    (args.output / 'notebook.py').write_text(script, encoding='utf-8')
    (args.output / 'notebook_base64.txt').write_text(base64.b64encode(script.encode()).decode('ascii'))
    manifest = dict(privacy='Private notebook only', sample_ids=[s['sample_id'] for s in samples],
                    train_count=16, class_counts={'real': 8, 'Diffae': 8},
                    payload_sha256=digest, payload_bytes=len(blob), script_bytes=len(script.encode()),
                    transport_bytes=(args.output / 'notebook_base64.txt').stat().st_size,
                    source_sha256=hashes, script_sha256=hashlib.sha256(script.encode()).hexdigest(),
                    training='4 trials, 400 updates each, fresh seed0, linear RGB, active lambdas1, pixel weights[1,4], Adam1e-4 WD0')
    (args.output / 'package_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k,v in manifest.items() if k not in ('source_sha256','sample_ids')}, indent=2))


if __name__ == '__main__':
    main()
