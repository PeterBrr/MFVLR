"""Eight-update train-only smoke: pretrained visual, fresh normalized language, stop vs coupled."""
import copy
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.pilot_deps'),str(ROOT/'src'),str(ROOT/'tools')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from mfvlr.integration import SuppliedTargets
from mfvlr_research.pilot_variant import training_model
from mfvlr_research.pilot_runtime import check_finite_tensors
from run_loss_overfit import load_samples, collate, write_json, export_preview, state_hash
from run_genface_ablation import MetricView
from test_split_mask_encoder_pilot import evaluate
from pilot_checkpointing import enable_checkpointing
from mfvlr.prompts import PromptBuilder, PromptMetadata
from mfvlr.tokenizer import TokenizerAdapter

def norm(params):
    return sum(float(p.grad.detach().double().square().sum()) for p in params if p.grad is not None)**.5

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--policies',choices=('stop_all','coupled'),nargs='+',default=['stop_all','coupled'])
    parser.add_argument('--reference-control',type=Path)
    args=parser.parse_args()
    out=args.output
    assert not out.exists();out.mkdir(parents=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    cfg=json.loads((ROOT/'configs/genface_language_readiness.json').read_text())
    path=ROOT/'artifacts/genface5000_epoch30_kaggle_20261027/downloaded/fd_validation_results/seed_0_fd_0p1/final.pt'
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    samples=load_samples(ROOT/'artifacts/kaggle_loss_overfit_20261026/package_001/train16.npz')
    prompts=PromptBuilder().build([PromptMetadata('real','real'),PromptMetadata('fake','fake','AM','diffusion','Diffae')])
    tokens=TokenizerAdapter().encode_levels(prompts).ids
    for s in samples:s['token_ids']=tokens[s['label']].clone()
    rng=np.random.default_rng(0);schedule=[]
    for _ in range(2):
        order=rng.permutation(16).tolist();schedule.extend(order[i:i+4] for i in range(0,16,4))
    source=[*ROOT.joinpath('src/mfvlr').glob('*.py'),*ROOT.joinpath('src/mfvlr_research').glob('pilot_*.py'),Path(__file__),ROOT/'configs/genface_language_readiness.json']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
    report=dict(status='RUNNING',scope='16 TRAIN images, 8 updates; no validation/test; warm visual epoch30, fresh language seed0',
        source_sha256=hashes,config=cfg,visual_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sample_ids=[s['sample_id'] for s in samples],schedule=schedule,microbatch=2,effective_batch=4,
        cmc_batch=2,optimizer=dict(name='Adam',lr=1e-5,weight_decay=0.),trials={})
    write=lambda:write_json(out/'report.json',report)
    write();initial_hashes=[]
    for policy in args.policies:
        print('START '+policy,flush=True)
        config=copy.deepcopy(cfg);config['model']['language_visual_gradients']=policy
        torch.manual_seed(0);model=training_model(config)
        state=model.state_dict()
        for key,value in ckpt['model'].items():
            dest='heads.'+key if key.startswith('detector.') else key
            assert dest in state and state[dest].shape==value.shape
            state[dest]=value
        model.load_state_dict(state,strict=True);del state
        initial_hashes.append(state_hash(model))
        model.cuda().train();enable_checkpointing(model)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-5,weight_decay=0.,foreach=False)
        row=dict(status='RUNNING',history=[],initial_state_sha256=initial_hashes[-1]);report['trials'][policy]=row
        row['initial_train'],_=evaluate(MetricView(model),samples,'cuda',torch.tensor([1.,4.],device='cuda'),2)
        for step,indices in enumerate(schedule,1):
            optimizer.zero_grad(set_to_none=True);means={}
            for offset in (0,2):
                batch=collate([samples[i] for i in indices[offset:offset+2]])
                image,ids=batch['image'].cuda(),batch['token_ids'].cuda();names=tuple(batch['sample_id'])
                output=model.with_language(image,ids,sample_ids=names)
                losses=model.compute_objectives(output,SuppliedTargets(names,ids,batch['label'].cuda(),batch['mask'].cuda(),tuple(batch['kind'])))
                (losses.total/2).backward()
                for k,v in losses.as_dict().items():means[k]=means.get(k,0.)+float(v.detach())/2
                del output,losses
            check_finite_tensors((p.grad for p in model.parameters()),'Language pilot gradient')
            grad=dict(encoder=norm(model.mve.parameters()),decoder=norm(model.decoder.parameters()),language=norm(model.flt.parameters()))
            optimizer.step();check_finite_tensors(model.parameters(),'Language pilot parameters')
            row['history'].append(dict(step=step,losses=means,total_gradient_norms=grad))
            print(json.dumps(dict(policy=policy,**row['history'][-1])),flush=True);write()
        optimizer.zero_grad(set_to_none=True)
        row['final_train'],preview=evaluate(MetricView(model),samples,'cuda',torch.tensor([1.,4.],device='cuda'),2)
        folder=out/policy;folder.mkdir();export_preview(folder,preview)
        row['status']='COMPLETED';write()
        del optimizer,model;gc.collect();torch.cuda.empty_cache()
    assert len(set(initial_hashes))==1
    if args.reference_control:
        control=json.loads(args.reference_control.read_text())
        assert control['trials']['stop_all']['status']=='COMPLETED'
        assert control['trials']['stop_all']['initial_state_sha256']==initial_hashes[0]
        assert control['schedule']==schedule and control['config']==cfg
        assert control['sample_ids']==report['sample_ids']
        assert control['visual_checkpoint_sha256']==report['visual_checkpoint_sha256']
        for p,h in control['source_sha256'].items():
            if p.replace('\\','/')!=Path(__file__).relative_to(ROOT).as_posix(): assert hashes[p]==h
        report['reference_control']=str(args.reference_control)
        report['reference_control_verified']=True
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report.update(status='COMPLETED',matched_initialization=len(initial_hashes)>1 or bool(args.reference_control),source_unchanged=True);write()
    print('COMPLETED '+str(out),flush=True)

if __name__=='__main__':main()
