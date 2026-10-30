"""Freeze language stage1 code/tokens/warm-start weights, privately upload, launch Kaggle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/language5000_scratch_kaggle_20261028'
DATASET='nha4von/mfvlr-language5000-scratch-assets-20261028'
NOTEBOOK='nha4von/mfvlr-language5000-scratch15-20261028-train'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    package=ART/'package_001';assert not package.exists()
    assets=package/'assets';assets.mkdir(parents=True)
    sys.path[:0]=[str(ROOT/'.pilot_deps'),str(ROOT/'src')]
    import numpy as np
    from mfvlr.prompts import PromptBuilder,PromptMetadata
    from mfvlr.tokenizer import TokenizerAdapter,verify_assets
    from mfvlr_research.pilot_variant import visual_model
    from test_ad_overfit import state_hash
    import torch
    adapter=TokenizerAdapter();verify_assets()
    prompts=PromptBuilder().build([PromptMetadata('real','real'),PromptMetadata('fake','fake','AM','diffusion','Diffae')])
    batch=adapter.encode_levels(prompts)
    np.save(assets/'tokens.npy',batch.ids.numpy(),allow_pickle=False)
    (assets/'token_audit.json').write_text(json.dumps(dict(prompts=prompts,payload_lengths=batch.payload_lengths.tolist(),
        eos_positions=batch.eos_positions.tolist(),policy='Locally verified pinned CLIP tokenizer; precomputed2 class rows; no remote tokenizer installation'),indent=2),encoding='utf-8')
    config=json.loads((ROOT/'configs/genface_language_readiness.json').read_text())
    config['model']['language_visual_gradients']='coupled'
    config['protocol']['scope']='Scratch image vs language; 5000 train/400val; target15 epochs; no test'
    (assets/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    base=ROOT/'artifacts/genface5000_epoch30_kaggle_20261027'
    old=json.loads((base/'downloaded/fd_validation_results/report.json').read_text())
    torch.manual_seed(0);local_reference=visual_model(config)
    initial_hash=next(iter(old['trials'].values()))['initial_state_sha256']
    by_torch={old['environment']['torch']:initial_hash,torch.__version__:state_hash(local_reference)}
    del local_reference
    (assets/'initialization_reference.json').write_text(json.dumps(dict(
        initial_visual_sha256=initial_hash,by_torch=by_torch,
        policy='Only initial random weight hash; no trained tensors included'),indent=2),encoding='utf-8')
    asset_hashes={p.name:sha(p) for p in assets.iterdir() if p.is_file()}
    (assets/'manifest.json').write_text(json.dumps(dict(files=asset_hashes),indent=2),encoding='utf-8')
    helpers=['run_language5000.py','run_fd_validation_pilot.py','run_loss_overfit.py','run_genface_ablation.py',
        'run_fd_weight_ablation.py','measure_ar_fl_gradients.py','pilot_checkpointing.py','test_ad_overfit.py',
        'export_ad_reconstructions.py','test_split_mask_encoder_pilot.py']
    files=[*ROOT.joinpath('src/mfvlr').glob('*.py'),*ROOT.joinpath('src/mfvlr').glob('*.json'),
        *ROOT.joinpath('src/mfvlr/tokenizer_assets').rglob('*'),
        *ROOT.joinpath('src/mfvlr_research').glob('pilot_*.py'),ROOT/'src/mfvlr_research/__init__.py',
        *(ROOT/'tools'/name for name in helpers)]
    files=[p for p in files if p.is_file() and '__pycache__' not in p.parts]
    source={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    (package/'source_sha256.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
    upload=package/'dataset_upload';upload.mkdir()
    bundle=upload/'mfvlr_language_assets.bin'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        for p in assets.iterdir():z.write(p,'assets/'+p.name)
        z.write(package/'source_sha256.json','source_sha256.json')
    (upload/'dataset-metadata.json').write_text(json.dumps(dict(id=DATASET,title='MFVLR Language5000 Scratch Assets 20261028',
        licenses=[dict(name='other')],description='Private MFVLR scratch experiment code and pinned tokenizer tokens/assets only. No trained weights, images or test data.'),indent=2),encoding='utf-8')
    original=json.loads((base/'package_001/package_manifest.json').read_text())
    manifest=dict(dataset=DATASET,notebook=NOTEBOOK,private=True,bundle_bytes=bundle.stat().st_size,bundle_sha256=sha(bundle),
        original_dataset=original['dataset'],original_bundle_sha256=original['bundle_sha256'],npz_sha256=original['npz_sha256'],
        source_sha256=source,epochs=15,stage_end_epoch=1,seed=0,updates=18750,stage_updates=1250,test_used=False)
    (package/'package_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='source_sha256'},indent=2),flush=True)


def notebook():
    package=ART/'package_001';m=json.loads((package/'package_manifest.json').read_text())
    folder=ART/'notebook_001';folder.mkdir(exist_ok=True)
    script=f'''import hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONUNBUFFERED']='1'
project=Path('/kaggle/working/language5000_project');project.mkdir(parents=True,exist_ok=True)
def bundle(name,expected):
    paths=list(Path('/kaggle/input').rglob(name));assert len(paths)==1
    p=paths[0];assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
    return zipfile.ZipFile(p)
with bundle('mfvlr_fd_validation_bundle.bin',{m['original_bundle_sha256']!r}) as z:
    for name in ('inputs/train.npz','inputs/val.npz'):z.extract(name,project)
with bundle('mfvlr_language_assets.bin',{m['bundle_sha256']!r}) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(project)
source=json.loads((project/'source_sha256.json').read_text())
assert all(hashlib.sha256((project/p).read_bytes()).hexdigest()==h for p,h in source.items())
for split,h in {m['npz_sha256']!r}.items():assert hashlib.sha256((project/'inputs'/f'{{split}}.npz').read_bytes()).hexdigest()==h
import torch
assert torch.cuda.is_available()
print(json.dumps(dict(event='bootstrap_verified',gpu=torch.cuda.get_device_name(),torch=torch.__version__,
    train=5000,val=400,target_epochs=15,stage_end_epoch=1,language='scratch_double_pre_norm_coupled',test_used=False)),flush=True)
subprocess.run([sys.executable,'-u',str(project/'tools/run_language5000.py'),
    '--inputs',str(project/'inputs'),'--assets',str(project/'assets'),'--epochs','15','--end-epoch','1',
    '--output','/kaggle/working/language5000_results'],cwd=project,check=True)
'''
    compile(script,'bootstrap.py','exec')
    (folder/'run.py').write_text(script,encoding='utf-8')
    metadata=dict(id=NOTEBOOK,title='MFVLR Language5000 Scratch15 20261028 Train',code_file='run.py',language='python',kernel_type='script',
        is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,machine_shape='NvidiaTeslaT4',dataset_sources=[m['original_dataset'],DATASET])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return m,folder


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=('prepare','upload','launch'),default='prepare');args=p.parse_args()
    if args.action=='prepare':prepare()
    m,folder=notebook()
    if args.action=='prepare':return
    assert sha(ART/'package_001/dataset_upload/mfvlr_language_assets.bin')==m['bundle_sha256']
    sys.path.insert(0,str(ROOT/'tmp/kaggle_current_deps'))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    record=ART/(args.action+'_record.json');assert not record.exists()
    if args.action=='upload':
        r=api.dataset_create_new(str(ART/'package_001/dataset_upload'),public=False,quiet=True,convert_to_csv=False)
        result=dict(url=r.url,status=r.status,private_requested=True,bundle_bytes=m['bundle_bytes'])
    else:
        r=api.kernels_push(str(folder),timeout='36000',acc='NvidiaTeslaT4')
        result=dict(url=r.url,version=r.version_number,ref=NOTEBOOK,target_epochs=15,stage_end_epoch=1,
                    total_updates=18750,stage_updates=1250,timeout_seconds=36000)
    record.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
