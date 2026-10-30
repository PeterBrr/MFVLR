"""Prepare, upload privately, or launch the approved GenFace5000/epoch30 seed0 job."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
REF='nha4von/mfvlr-genface5000-epoch30-20261027'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=('prepare','upload','launch'),default='prepare')
    args=p.parse_args()
    root=ROOT/'artifacts/genface5000_epoch30_kaggle_20261027'
    package=root/'package_001';m=json.loads((package/'package_manifest.json').read_text())
    bundle=package/'dataset_upload/mfvlr_fd_validation_bundle.bin'
    assert hashlib.sha256(bundle.read_bytes()).hexdigest()==m['bundle_sha256']
    assert m['dataset']==REF and m['train_count']==5000 and m['val_count']==400
    assert m['planned_trials']==dict(fd_weights=[.1],seeds=[0],epochs=30,updates_each=37500,total_updates=37500)
    folder=root/'notebook_001';folder.mkdir(exist_ok=True)
    script=f'''# Private GenFace5000 epoch30; no language/test; same FD0.1 visual baseline.
import hashlib,io,json,os,subprocess,sys,zipfile
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONUNBUFFERED']='1'
files=list(Path('/kaggle/input').rglob('mfvlr_fd_validation_bundle.bin'))
assert len(files)==1,'Missing or ambiguous bundle'
payload=files[0].read_bytes()
assert hashlib.sha256(payload).hexdigest()=={m['bundle_sha256']!r}
project=Path('/kaggle/working/mfvlr5000_project');project.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(project)
del payload
source=json.loads((project/'inputs/source_sha256.json').read_text())
assert all(hashlib.sha256((project/p).read_bytes()).hexdigest()==h for p,h in source.items())
for split,h in {m['npz_sha256']!r}.items():
    assert hashlib.sha256((project/'inputs'/f'{{split}}.npz').read_bytes()).hexdigest()==h
print('BOOTSTRAP: source/data verified; train5000 val400; no test; seed0 epochs30 FD0.1',flush=True)
import torch
assert torch.cuda.is_available()
print(json.dumps(dict(event='gpu_ready',gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda)),flush=True)
subprocess.run([sys.executable,'-u',str(project/'tools/run_fd_validation_pilot.py'),
    '--inputs',str(project/'inputs'),'--output','/kaggle/working/fd_validation_results',
    '--epochs','30','--seeds','0','--fd-weights','0.1','--gradient-pairs','4',
    '--save-epoch-checkpoints'],cwd=project,check=True)
print('EXPERIMENT COMPLETED: fd_validation_results/report.json',flush=True)
'''
    compile(script,'bootstrap5000.py','exec')
    (folder/'run.py').write_text(script,encoding='utf-8')
    metadata=dict(id=REF+'-train',title='MFVLR GenFace5000 Epoch30 20261027 Train',code_file='run.py',language='python',
        kernel_type='script',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,
        machine_shape='NvidiaTeslaT4',dataset_sources=[REF])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(dict(action=args.action,ref=REF,private=True,bundle_bytes=m['bundle_bytes'],
                         planned_trials=m['planned_trials']),indent=2),flush=True)
    if args.action=='prepare':return
    sys.path.insert(0,str(ROOT/'tmp/kaggle_current_deps'))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    if args.action=='upload':
        assert not (root/'upload_record.json').exists(),'Already uploaded'
        response=api.dataset_create_new(str(package/'dataset_upload'),public=False,quiet=True,convert_to_csv=False)
        result=dict(url=response.url,status=response.status,private_requested=True,bytes=m['bundle_bytes'])
        record=root/'upload_record.json'
    else:
        assert not (root/'launch_record.json').exists(),'Already launched'
        try:
            response=api.kernels_push(str(folder),timeout='36000',acc='NvidiaTeslaT4')
        except Exception as error:
            response_error=getattr(error,'response',None)
            if response_error is not None: print(response_error.text[:3000],flush=True)
            raise
        result=dict(url=response.url,version=response.version_number,ref='/'.join(response.url.rstrip('/').split('/')[-2:]),
                    planned_trials=m['planned_trials'],bundle_sha256=m['bundle_sha256'],timeout_seconds=36000)
        record=root/'launch_record.json'
    record.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
