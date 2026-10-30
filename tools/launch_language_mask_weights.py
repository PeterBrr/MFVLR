"""Matched GenFace foreground-weight ablations from frozen language source/assets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/language5000_scratch_kaggle_20261028'
ART = ROOT / 'artifacts/language_mask_weights_kaggle_20261029'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def patch_project(project, weight):
    """Change only foreground weight and evaluator's previously hardcoded weight."""
    config_path = project / 'assets/config.json'
    config = json.loads(config_path.read_text())
    assert config['model']['mask_foreground_weight'] == 4.
    config['model']['mask_foreground_weight'] = float(weight)
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    assets_path = project / 'assets/manifest.json'
    assets = json.loads(assets_path.read_text())
    assets['files']['config.json'] = sha(config_path)
    assets_path.write_text(json.dumps(assets, indent=2), encoding='utf-8')
    runner = project / 'tools/run_language5000.py'
    code = runner.read_text(encoding='utf-8')
    old = "weights=torch.tensor([1.,4.],device=device)"
    assert code.count(old) == 1
    code = code.replace(old, "weights=torch.tensor([config['model']['mask_background_weight'],config['model']['mask_foreground_weight']],device=device)")
    runner.write_text(code, encoding='utf-8')
    source_path = project / 'source_sha256.json'
    source = json.loads(source_path.read_text())
    source['tools/run_language5000.py'] = sha(runner)
    source_path.write_text(json.dumps(source, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weight', type=int, choices=(1, 2), required=True)
    parser.add_argument('--end-epoch', type=int, default=5)
    parser.add_argument('--previous', help='Own previous notebook output to resume')
    parser.add_argument('--start-step', type=int, default=0)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--budget-seconds', type=int, default=0,
                        help='Checkpoint and exit before the per-session GPU budget is exhausted')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.end_epoch <= 15
    assert bool(args.previous) == (args.start_step > 0)
    assert args.start_step >= 0 and args.budget_seconds >= 0
    assert args.start_step < args.end_epoch * 1250
    ART.mkdir(parents=True, exist_ok=True)
    m = json.loads((BASE / 'package_001/package_manifest.json').read_text())
    assert sha(BASE / 'package_001/dataset_upload/mfvlr_language_assets.bin') == m['bundle_sha256']
    folder = ART / f'weight_{args.weight}_stage_{args.stage}'
    folder.mkdir(exist_ok=True)
    ref = f'nha4von/mfvlr-lang5000-fw{args.weight}-s{args.stage}-20261029-train'
    # Bootstrap verifies the original frozen package before applying the explicit changes.
    script = (BASE / 'notebook_001/run.py').read_text(encoding='utf-8')
    script = script.replace("'--end-epoch','1'", f"'--end-epoch','{args.end_epoch}'")
    script = script.replace('stage_end_epoch=1,language=', f'stage_end_epoch={args.end_epoch},foreground_weight={args.weight},language=')
    patch_code = "\n" + __import__('inspect').getsource(patch_project)
    patch_code += "\ndef sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()\n"
    patch_code += f"patch_project(project,{args.weight})\n"
    if args.budget_seconds:
        # Preserve recipe and record the sole execution-policy change in provenance.
        patch_code += f"""budget_runner=project/'tools/run_language5000.py'
old_runner_sha=sha(budget_runner)
budget_code=budget_runner.read_text(encoding='utf-8')
assert budget_code.count('time.monotonic()-start>8*3600')==1
budget_code=budget_code.replace('time.monotonic()-start>8*3600','time.monotonic()-start>{args.budget_seconds}')
budget_runner.write_text(budget_code,encoding='utf-8')
source_path=project/'source_sha256.json'
source=json.loads(source_path.read_text())
source['tools/run_language5000.py']=sha(budget_runner)
source_path.write_text(json.dumps(source,indent=2),encoding='utf-8')
"""
    patch_code += "print(json.dumps(dict(event='weight_patch_verified',config=json.loads((project/'assets/config.json').read_text()),runner_sha256=sha(project/'tools/run_language5000.py'))),flush=True)\n"
    assert script.count('import torch\n') == 1
    script = script.replace('import torch\n', patch_code + 'import torch\n')
    if args.previous:
        resume = f"""paths=list(Path('/kaggle/input').rglob('last.pt'))
assert len(paths)==1,'Missing or ambiguous own checkpoint'
resume_path=paths[0]
state=torch.load(resume_path,map_location='cpu',weights_only=False)
assert state['schema']=='mfvlr-language5000-stage1-v1' and state['step']=={args.start_step}
assert state['config']['model']['mask_foreground_weight']=={float(args.weight)}
"""
        if args.budget_seconds:
            resume += """original_checkpoint_sha=hashlib.sha256(resume_path.read_bytes()).hexdigest()
old_source=state['provenance']['source']
assert old_source['tools/run_language5000.py']==old_runner_sha
new_source={p:sha(project/p) for p in old_source}
assert all(new_source[p]==h for p,h in old_source.items() if p!='tools/run_language5000.py')
state['provenance']['source']=new_source
migration=dict(event='resume_budget_policy_verified',input_checkpoint_sha256=original_checkpoint_sha,
    original_source=old_source,new_source=new_source,
    change='Only per-session elapsed-time budget; model, optimizer, RNG, schedule, config and losses unchanged')
(project/'resume_budget_audit.json').write_text(json.dumps(migration,indent=2),encoding='utf-8')
resume_path=project/'budget_resume.pt'
torch.save(state,resume_path)
print(json.dumps(dict(event=migration['event'],input_checkpoint_sha256=original_checkpoint_sha,
    old_runner_sha256=old_runner_sha,new_runner_sha256=new_source['tools/run_language5000.py'])),flush=True)
"""
        resume += "del state\nprint('OWN_WEIGHT_RUN_RESUME_VERIFIED',flush=True)\n"
        script = script.replace('subprocess.run([sys.executable', resume + 'subprocess.run([sys.executable')
        script = script.replace("'--output','/kaggle/working/language5000_results'", "'--resume',str(resume_path),'--output','/kaggle/working/language5000_results'")
        if args.budget_seconds:
            script += "\nresume_path.unlink()\n"
    compile(script, 'bootstrap.py', 'exec')
    (folder / 'run.py').write_text(script, encoding='utf-8')
    metadata = json.loads((BASE / 'notebook_001/kernel-metadata.json').read_text())
    metadata.update(id=ref, title=f'MFVLR Lang5000 FW{args.weight} S{args.stage} 20261029 Train',
                    kernel_sources=[args.previous] if args.previous else [])
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    plan = dict(ref=ref, target_epochs=15, end_epoch=args.end_epoch, start_step=args.start_step,
                foreground_weight=args.weight, seed=0, train=5000, val=400, test_used=False,
                control='Existing scratch language weight4 run; same initialization/schedule',
                existing_datasets=metadata['dataset_sources'], new_images_uploaded=False,
                previous=args.previous, budget_seconds=args.budget_seconds,
                bootstrap_sha256=sha(folder / 'run.py'))
    (folder / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps(plan, indent=2), flush=True)
    if args.smoke:
        project = folder / 'smoke_project'
        assert not project.exists()
        with zipfile.ZipFile(BASE / 'package_001/dataset_upload/mfvlr_language_assets.bin') as z:
            z.extractall(project)
        patch_project(project, args.weight)
        original = ROOT / 'artifacts/genface5000_epoch30_kaggle_20261027/package_001/dataset_upload/mfvlr_fd_validation_bundle.bin'
        with zipfile.ZipFile(original) as z:
            for name in ('inputs/train.npz', 'inputs/val.npz'):
                z.extract(name, project)
        subprocess.run([r'C:\Users\NHA\miniconda3\envs\my_env\python.exe', '-u',
            str(project / 'tools/run_language5000.py'), '--inputs', str(project / 'inputs'),
            '--assets', str(project / 'assets'), '--output', str(folder / 'smoke_result'),
            '--epochs', '15', '--smoke'], check=True)
    if args.launch:
        record = folder / 'launch_record.json'
        assert not record.exists(), 'Already launched'
        sys.path.insert(0, str(ROOT / 'tmp/kaggle_current_deps'))
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        response = api.kernels_push(str(folder), timeout='36000', acc='NvidiaTeslaT4')
        result = dict(ref=ref, url=response.url, version=response.version_number, **{k: v for k, v in plan.items() if k != 'ref'})
        record.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
