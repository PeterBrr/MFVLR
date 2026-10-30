"""Freeze local continuations, privately transfer to Kaggle2, and resume there."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'artifacts/language_mask_weights_local_20261029/run_001'
ART = ROOT / 'artifacts/language_mask_weights_kaggle2_20261030'
DATASET = 'v5nhdf/mfvlr-fw-resume-20261030'
SPECS = {2: ('segment_004', 14600, 6300), 1: ('segment_001', 12700, 6600)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    path.write_bytes(json.dumps(obj, indent=2, ensure_ascii=False).encode('utf-8'))


def prepare():
    package = ART / 'package_001'
    assert not package.exists()
    upload = package / 'upload'
    upload.mkdir(parents=True)
    import torch
    audit = dict(dataset=DATASET, private=True, train=5000, val=400, test_used=False, weights={})
    data = ROOT / 'artifacts/genface5000_epoch30_kaggle_20261027/package_001/dataset_upload/mfvlr_fd_validation_bundle.bin'
    expected = json.loads((data.parents[1] / 'package_manifest.json').read_text())
    assert sha(data) == expected['bundle_sha256']
    # Explicit files only; no credentials, logs from other projects or test set.
    os.link(data, upload / 'genface5000_data.bin')
    for weight, (segment, step, budget) in SPECS.items():
        origin = LOCAL / f'weight_{weight}'
        checkpoint_path = origin / segment / 'last.pt'
        original_sha = sha(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert checkpoint['schema'] == 'mfvlr-language5000-stage1-v1'
        assert checkpoint['step'] == step and len(checkpoint['history']) == step
        assert checkpoint['config']['model']['mask_foreground_weight'] == weight
        assert checkpoint['provenance']['data'] == expected['npz_sha256']
        old_source = checkpoint['provenance']['source']
        canonical = {p.replace('\\', '/'): h for p, h in old_source.items()}
        assert all(sha(origin / 'project' / p) == h for p, h in canonical.items())
        project = package / f'project_fw{weight}'
        shutil.copytree(origin / 'project', project, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        runner = project / 'tools/run_language5000.py'
        code = runner.read_text(encoding='utf-8')
        assert code.count('time.monotonic()-start>8600') == 1
        runner.write_bytes(code.replace('time.monotonic()-start>8600', f'time.monotonic()-start>{budget}').encode('utf-8'))
        new_source = dict(canonical)
        new_source['tools/run_language5000.py'] = sha(runner)
        assert all(sha(project / p) == h for p, h in new_source.items())
        checkpoint['provenance']['source'] = new_source
        write(project / 'source_sha256.json', new_source)
        transferred = upload / f'resume_fw{weight}.bin'
        torch.save(checkpoint, transferred)
        project_bundle = upload / f'project_fw{weight}.bin'
        with zipfile.ZipFile(project_bundle, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in project.rglob('*'):
                if p.is_file():
                    z.write(p, p.relative_to(project).as_posix())
        audit['weights'][str(weight)] = dict(start_step=step, budget_seconds=budget,
            original_checkpoint=str(checkpoint_path), original_checkpoint_sha256=original_sha,
            transferred_checkpoint_sha256=sha(transferred), project_bundle_sha256=sha(project_bundle),
            old_source=old_source, new_source=new_source,
            data_sha256=checkpoint['provenance']['data'], config=checkpoint['config'],
            migration='Normalize source path keys for Linux; only runner time budget changed. Model/optimizer/RNG/schedule/history unchanged.')
        del checkpoint
        gc.collect()
    audit['files'] = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in upload.iterdir() if p.is_file()}
    audit['total_bytes'] = sum(v['bytes'] for v in audit['files'].values())
    write(package / 'migration_audit.json', audit)
    write(upload / 'dataset-metadata.json', dict(id=DATASET, title='MFVLR FW Local Resume 20261030',
        licenses=[dict(name='other')], description='Private GenFace MFVLR continuation: fixed 5000 train/400val images and masks, frozen source/assets, full FW2/FW1 optimizer checkpoints. No test set.'))
    print(json.dumps(dict(dataset=DATASET, private=True, total_bytes=audit['total_bytes'],
        files={k:v['bytes'] for k,v in audit['files'].items()}, start_steps={w:v['start_step'] for w,v in audit['weights'].items()})), flush=True)


def notebooks():
    package = ART / 'package_001'
    audit = json.loads((package / 'migration_audit.json').read_text(encoding='utf-8'))
    for weight in (2, 1):
        spec = audit['weights'][str(weight)]
        ref = f'v5nhdf/mfvlr-fw{weight}-resume-20261030-train'
        folder = ART / f'notebook_fw{weight}'
        folder.mkdir(exist_ok=True)
        script = f'''import gc,hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONUNBUFFERED']='1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def find(name,expected):
    found=list(Path('/kaggle/input').rglob(name));assert len(found)==1
    p=found[0];assert sha(p)==expected
    return p
project=Path('/kaggle/working/fw{weight}_project');project.mkdir()
data=find('genface5000_data.bin',{audit['files']['genface5000_data.bin']['sha256']!r})
with zipfile.ZipFile(data) as z:
    for name in ('inputs/train.npz','inputs/val.npz'):z.extract(name,project)
with zipfile.ZipFile(find('project_fw{weight}.bin',{spec['project_bundle_sha256']!r})) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(project)
source=json.loads((project/'source_sha256.json').read_text())
assert source=={spec['new_source']!r}
assert all(sha(project/p)==h for p,h in source.items())
for s,h in {spec['data_sha256']!r}.items():assert sha(project/'inputs'/(s+'.npz'))==h
resume=find('resume_fw{weight}.bin',{spec['transferred_checkpoint_sha256']!r})
import torch
assert torch.cuda.is_available()
state=torch.load(resume,map_location='cpu',weights_only=False)
assert state['step']=={spec['start_step']} and state['config']['model']['mask_foreground_weight']=={weight}
assert state['provenance']['source']==source
del state;gc.collect()
print(json.dumps(dict(event='kaggle2_resume_input_verified',weight={weight},start_step={spec['start_step']},
    gpu=torch.cuda.get_device_name(),torch=torch.__version__,budget_seconds={spec['budget_seconds']},train=5000,val=400,test_used=False)),flush=True)
subprocess.run([sys.executable,'-u',str(project/'tools/run_language5000.py'),
    '--inputs',str(project/'inputs'),'--assets',str(project/'assets'),'--epochs','15','--end-epoch','15',
    '--resume',str(resume),'--output','/kaggle/working/language5000_results'],cwd=project,check=True)
'''
        compile(script, 'bootstrap.py', 'exec')
        (folder / 'run.py').write_bytes(script.encode('utf-8'))
        write(folder / 'kernel-metadata.json', dict(id=ref,title=f'MFVLR FW{weight} Resume 20261030 Train',
            code_file='run.py',language='python',kernel_type='script',is_private=True,
            enable_gpu=True,enable_tpu=False,enable_internet=False,machine_shape='NvidiaTeslaT4',
            dataset_sources=[DATASET],kernel_sources=[]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=('prepare','upload','launch','upload-and-launch'), required=True)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare()
        notebooks()
        return
    os.environ['KAGGLE_API_TOKEN'] = os.environ['KAGGLE_API_TOKEN_2']
    sys.path.insert(0,str(ROOT/'tmp/kaggle_current_deps'))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    assert {d.ref.split('/')[0] for d in api.dataset_list(mine=True)} == {'v5nhdf'}
    def status(phase, **fields):
        value=dict(phase=phase,pid=os.getpid(),time_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),**fields)
        write(ART/'pipeline_status.json',value)
        print(json.dumps(value),flush=True)
    try:
        if args.action in ('upload','upload-and-launch'):
            assert not (ART/'upload_record.json').exists()
            status('VERIFYING_UPLOAD')
            audit=json.loads((ART/'package_001/migration_audit.json').read_text(encoding='utf-8'))
            assert all(sha(ART/'package_001/upload'/name)==v['sha256'] for name,v in audit['files'].items())
            status('UPLOADING',bytes=audit['total_bytes'],destination=DATASET,private=True)
            r=api.dataset_create_new(str(ART/'package_001/upload'),public=False,quiet=False,convert_to_csv=False)
            write(ART/'upload_record.json',dict(url=r.url,status=r.status,private_requested=True))
        if args.action in ('launch','upload-and-launch'):
            for attempt in range(120):
                current=api.dataset_status(DATASET)
                status('WAITING_DATASET_READY',dataset_status=current)
                if current=='ready':break
                assert current not in ('error','failed'),current
                time.sleep(15)
            else:raise TimeoutError('Dataset not ready after 30 minutes')
            matches=[d for d in api.dataset_list(mine=True,search='mfvlr-fw-resume-20261030') if d.ref==DATASET]
            assert len(matches)==1 and matches[0].is_private is True,'Private destination not verified'
            quota=api.quota_view().gpu_quota
            remaining=(quota.total_time_allowed-quota.time_used-quota.time_reserved).total_seconds()
            assert remaining>=13200,f'Quota changed: only {remaining} seconds remain; adjust budgets before launch'
            status('LAUNCHING',private_verified=True,quota_remaining_seconds=remaining)
            notebooks()
            launched=[]
            for weight in (2,1):
                folder=ART/f'notebook_fw{weight}'
                record=folder/'launch_record.json'
                if record.exists():
                    launched.append(json.loads(record.read_text()))
                    continue
                r=api.kernels_push(str(folder),timeout='18000',acc='NvidiaTeslaT4')
                result=dict(ref=f'v5nhdf/mfvlr-fw{weight}-resume-20261030-train',url=r.url,version=r.version_number)
                write(record,result)
                launched.append(result)
                print(json.dumps(result),flush=True)
            status('SUBMITTED',notebooks=launched)
    except Exception as error:
        status('FAILED',error=str(error))
        raise


if __name__ == '__main__':main()
