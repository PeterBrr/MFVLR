"""Continue the same scratch language experiment using Kaggle output as an input."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/language5000_scratch_kaggle_20261028'
PREVIOUS='nha4von/mfvlr-language5000-scratch15-20261028-train'
REF='nha4von/mfvlr-language5000-scratch15-stage2-20261028-train'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--launch',action='store_true');args=p.parse_args()
    prior=json.loads((ART/'downloaded_stage1/language5000_results/report.json').read_text())
    assert prior['status']=='STAGE_COMPLETED' and prior['completed_updates']==1250 and prior['target_epochs']==15
    assert prior['source_unchanged'] and prior['initialization'].startswith('scratch')
    assert not prior['test_used'] and len(prior['train_ids'])==5000 and len(prior['val_ids'])==400
    folder=ART/'notebook_stage2';folder.mkdir(exist_ok=True)
    script=(ART/'notebook_001/run.py').read_text(encoding='utf-8')
    assert script.count("'--end-epoch','1'")==1
    script=script.replace("'--end-epoch','1'","'--end-epoch','15'")
    marker="subprocess.run([sys.executable"
    assert script.count(marker)==1
    check="""checkpoints=list(Path('/kaggle/input').rglob('last.pt'))
assert len(checkpoints)==1,'Missing or ambiguous previous-stage checkpoint'
resume_path=checkpoints[0]
state=torch.load(resume_path,map_location='cpu',weights_only=False)
assert state['schema']=='mfvlr-language5000-stage1-v1' and state['step']==1250
assert state['provenance']['data']==%r
del state
print('RESUME: own scratch epoch1 checkpoint; continue step1251, target15epochs',flush=True)
""" % prior['data_sha256']
    script=script.replace(marker,check+marker)
    assert "'--output','/kaggle/working/language5000_results'" in script
    script=script.replace("'--output','/kaggle/working/language5000_results'", "'--resume',str(resume_path),'--output','/kaggle/working/language5000_results'")
    script=script.replace('stage_end_epoch=1,language=', 'stage_end_epoch=15,language=')
    compile(script,'bootstrap_stage2.py','exec')
    (folder/'run.py').write_text(script,encoding='utf-8')
    metadata=json.loads((ART/'notebook_001/kernel-metadata.json').read_text())
    metadata.update(id=REF,title='MFVLR Language5000 Scratch15 Stage2 20261028 Train',kernel_sources=[PREVIOUS])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(dict(ref=REF,resume_from=PREVIOUS,start_step=1250,target_epochs=15,new_images_uploaded=False),indent=2),flush=True)
    if args.launch:
        record=ART/'stage2_launch_record.json';assert not record.exists(),'Already launched'
        sys.path.insert(0,str(ROOT/'tmp/kaggle_current_deps'))
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
        r=api.kernels_push(str(folder),timeout='36000',acc='NvidiaTeslaT4')
        result=dict(url=r.url,version=r.version_number,ref=REF,previous=PREVIOUS,start_step=1250,target_epochs=15)
        record.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
