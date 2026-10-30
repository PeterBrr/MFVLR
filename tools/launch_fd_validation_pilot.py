"""Prepare/upload/launch the private 1000-train400-validation Kaggle comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/kaggle_current_deps'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('prepare','upload','launch'),default='prepare')
    args=parser.parse_args()
    root=ROOT/'artifacts/genface_fd_validation_kaggle_20261026'
    package=root/'package_001'
    manifest=json.loads((package/'package_manifest.json').read_text())
    bundle=package/'dataset_upload/mfvlr_fd_validation_bundle.bin'
    assert hashlib.sha256(bundle.read_bytes()).hexdigest()==manifest['bundle_sha256']
    folder=root/'notebook_001';folder.mkdir(exist_ok=True)
    script=f'''# Private paired-seed FD pilot,1000 train and400 held-out validation.
import hashlib,io,json,os,subprocess,sys,zipfile
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONUNBUFFERED']='1'
print('BOOTSTRAP: locating private train1000/val400 bundle',flush=True)
files=list(Path('/kaggle/input').rglob('mfvlr_fd_validation_bundle.bin'))
assert len(files)==1,'Missing or ambiguous data bundle'
payload=files[0].read_bytes()
assert hashlib.sha256(payload).hexdigest()=={manifest['bundle_sha256']!r}
project=Path('/kaggle/working/mfvlr_fd_validation_project');project.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in archive.namelist())
    archive.extractall(project)
source=json.loads((project/'inputs/source_sha256.json').read_text())
assert all(hashlib.sha256((project/p).read_bytes()).hexdigest()==h for p,h in source.items())
for split,expected in {manifest['npz_sha256']!r}.items():
    assert hashlib.sha256((project/'inputs'/f'{{split}}.npz').read_bytes()).hexdigest()==expected
print('BOOTSTRAP: source/data hashes verified; test not included',flush=True)
import torch
assert torch.cuda.is_available(),'GPU unavailable'
print(json.dumps({{'event':'gpu_ready','torch':torch.__version__,'cuda':torch.version.cuda,
    'gpu':torch.cuda.get_device_name(),'gpu_count':torch.cuda.device_count()}}),flush=True)
subprocess.run([sys.executable,'-u',str(project/'tools/run_fd_validation_pilot.py'),
    '--inputs',str(project/'inputs'),'--output','/kaggle/working/fd_validation_results',
    '--epochs','5','--seeds','0','1','2','--gradient-pairs','4'],cwd=project,check=True)
print('EXPERIMENT COMPLETED: fd_validation_results/report.json',flush=True)
'''
    compile(script,'fd_pilot_bootstrap.py','exec')
    (folder/'run.py').write_text(script,encoding='utf-8')
    metadata=dict(id='nha4von/mfvlr-genface-fd-pilot-20261026-001',
        title='MFVLR GenFace FD Pilot 20261026 001',code_file='run.py',language='python',kernel_type='script',
        is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,machine_shape='NvidiaTeslaT4',
        dataset_sources=[manifest['dataset']])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(dict(action=args.action,dataset=manifest['dataset'],notebook=metadata['id'],
        bundle_bytes=manifest['bundle_bytes'],private=True,planned_trials=manifest['planned_trials']),indent=2),flush=True)
    if args.action=='prepare': return
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    if args.action=='upload':
        response=api.dataset_create_new(str(package/'dataset_upload'),public=False,quiet=True,convert_to_csv=False)
        result=dict(url=response.url,status=response.status,private_requested=True)
        (root/'upload_record.json').write_text(json.dumps(result,indent=2))
    else:
        response=api.kernels_push(str(folder),timeout='10800',acc='NvidiaTeslaT4')
        result=dict(url=response.url,version=response.version_number,ref=metadata['id'],
                    data_sha256=manifest['bundle_sha256'],planned_trials=manifest['planned_trials'])
        (root/'launch_record.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
