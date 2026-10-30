"""Scratch image/language comparison on GenFace5000; resume only this experiment."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.pilot_deps'),str(ROOT/'src'),str(ROOT/'tools')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from mfvlr.integration import SuppliedTargets
from mfvlr_research.pilot_variant import training_model, visual_model
from mfvlr_research.pilot_runtime import check_finite_tensors
from mfvlr_research.pilot_language import nonpad_language_reconstruction_loss, PAD_ID
from run_fd_validation_pilot import samples_from_npz
from run_loss_overfit import collate, write_json, export_preview, state_hash
from run_genface_ablation import MetricView
from test_split_mask_encoder_pilot import evaluate
from pilot_checkpointing import enable_checkpointing


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=15)
    parser.add_argument('--end-epoch',type=int)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--image-only',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    assert args.epochs>0 and not args.output.exists()
    assert args.end_epoch is None or 1<=args.end_epoch<=args.epochs
    args.output.mkdir(parents=True)
    start=time.monotonic()
    def log(event,**kw):
        record=dict(event=event,elapsed_seconds=round(time.monotonic()-start,2),**kw)
        line=json.dumps(record);print(line,flush=True)
        with (args.output/'live_log.jsonl').open('a',encoding='utf-8') as f:f.write(line+'\n')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    device='cuda'
    config=json.loads((args.assets/'config.json').read_text())
    assert config['model']['language_visual_gradients']=='coupled'
    assert config['model']['vim_residual']=='double' and config['model']['language_normalization']=='pre_norm'
    if args.image_only:
        config['loss_weights'].update(lr=0.,cmc=0.,kl=0.)
    assets=json.loads((args.assets/'manifest.json').read_text())
    assert all(sha(args.assets/name)==h for name,h in assets['files'].items())
    train=samples_from_npz(args.inputs/'train.npz',16 if args.smoke else None)
    val=samples_from_npz(args.inputs/'val.npz',16 if args.smoke else None)
    assert len(train)==(16 if args.smoke else 5000) and len(val)==(16 if args.smoke else 400)
    assert not ({s['sample_id'] for s in train}&{s['sample_id'] for s in val})
    tokens=torch.from_numpy(np.load(args.assets/'tokens.npy',allow_pickle=False)).long()
    assert tokens.shape==(2,308) and (tokens==49407).sum(1).tolist()==[4,4]
    for s in [*train,*val]:s['token_ids']=tokens[s['label']].clone()
    log('data_ready',train_count=len(train),val_count=len(val),test_used=False)
    torch.manual_seed(0)
    model=training_model(config)
    reference=visual_model(config)
    for name,actual,expected in [('encoder',model.mve,reference.mve),('decoder',model.decoder,reference.decoder),
                                ('detector',model.heads.detector,reference.detector)]:
        assert state_hash(actual)==state_hash(expected),f'Initial visual mismatch: {name}'
    initial_visual_hash=state_hash(reference)
    expected=json.loads((args.assets/'initialization_reference.json').read_text())
    expected_hash=expected['by_torch'].get(torch.__version__)
    if expected_hash is not None:assert initial_visual_hash==expected_hash
    del reference;gc.collect()
    initial_hash=state_hash(model)
    model.cuda().train();enable_checkpointing(model)
    image_params=[*model.mve.parameters(),*model.decoder.parameters(),*model.heads.detector.parameters()]
    image_ids={id(p) for p in image_params}
    language_params=[p for p in model.parameters() if id(p) not in image_ids]
    optimizer=torch.optim.Adam([dict(params=image_params,lr=1e-4),dict(params=language_params,lr=1e-5)],
        weight_decay=0.,foreach=False)
    weights=torch.tensor([1.,4.],device=device)
    source={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mfvlr','src/mfvlr_research') for p in (ROOT/folder).glob('*.py')}
    source['tools/run_language5000.py']=sha(Path(__file__))
    rng=np.random.default_rng(0);schedule=[]
    for _ in range(args.epochs):
        order=rng.permutation(len(train)).tolist();schedule.extend(order[i:i+4] for i in range(0,len(train),4))
    if args.smoke:schedule=schedule[:1]
    report=dict(status='RUNNING',config=config,seed=0,smoke=args.smoke,epochs=args.epochs,test_used=False,
        train_ids=[s['sample_id'] for s in train],val_ids=[s['sample_id'] for s in val],
        scope='From scratch seed0; matched initial image weights and data schedule; fixed detection0.5/mask argmax',
        initialization='scratch; no previous trained model loaded',image_only=args.image_only,
        initial_visual_sha256=initial_visual_hash,
        historical_initial_match=initial_visual_hash==expected['initial_visual_sha256'],
        initial_state_sha256=initial_hash,asset_sha256=assets['files'],source_sha256=source,
        data_sha256={s:sha(args.inputs/(s+'.npz')) for s in ('train','val')},
        effective_batch=4,microbatch=2,cmc_batch=2,schedule=schedule,
        optimizer=dict(name='Adam',image_lr=1e-4,language_lr=1e-5,weight_decay=0.,foreach=False),
        environment=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name()),history=[],evaluations=[])
    write=lambda:write_json(args.output/'report.json',report)
    write()

    @torch.no_grad()
    def language_eval(samples):
        if not model.has_language:return None
        model.eval();sums=dict(lr=0.,cmc=0.,kl=0.);correct=valid=0
        for i in range(0,len(samples),2):
            batch=collate(samples[i:i+2]);image,ids=batch['image'].cuda(),batch['token_ids'].cuda();names=tuple(batch['sample_id'])
            output=model.with_language(image,ids,sample_ids=names)
            losses=model.compute_objectives(output,SuppliedTargets(names,ids,batch['label'].cuda(),batch['mask'].cuda(),tuple(batch['kind'])))
            for k in sums:sums[k]+=float(getattr(losses,k))*len(names)/len(samples)
            mask=ids!=PAD_ID;valid+=int(mask.sum());correct+=int(((output.language.logits.argmax(-1)==ids)&mask).sum())
        model.train();return dict(losses=sums,nonpad_token_accuracy=correct/max(1,valid),count=len(samples))

    @torch.no_grad()
    def evaluate_epoch(epoch,step):
        log('evaluate_start',epoch=epoch,step=step)
        visual,preview=evaluate(MetricView(model),val,device,weights,2)
        language=language_eval(val)
        result=dict(epoch=epoch,step=step,val=visual,language_val=language)
        report['evaluations'].append(result)
        folder=args.output/'val';folder.mkdir(exist_ok=True);export_preview(folder,preview)
        write();log('metrics',**result)

    def save_checkpoint(step,epoch):
        temporary=args.output/'last.tmp.pt'
        torch.save(dict(schema='mfvlr-language5000-stage1-v1',config=config,step=step,epoch=epoch,
            model=model.state_dict(),optimizer=optimizer.state_dict(),schedule=schedule,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            provenance=dict(source=source,data=report['data_sha256'],assets=assets['files']),
            history=report['history'],evaluations=report['evaluations']),temporary)
        temporary.replace(args.output/'last.pt');log('checkpoint_saved',step=step,epoch=epoch)

    start_step=0
    if args.resume:
        resume=torch.load(args.resume,map_location='cpu',weights_only=False)
        assert resume['schema']=='mfvlr-language5000-stage1-v1' and resume['config']==config
        assert resume['schedule']==schedule and resume['provenance']==dict(source=source,data=report['data_sha256'],assets=assets['files'])
        model.load_state_dict(resume['model'],strict=True);optimizer.load_state_dict(resume['optimizer'])
        torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
        start_step=resume['step'];report['history']=resume['history'];report['evaluations']=resume['evaluations']
        report['resume_checkpoint_sha256']=sha(args.resume)
        del resume;gc.collect()
        log('resume_verified',step=start_step,target_epochs=args.epochs)
    else:
        evaluate_epoch(0,0)
    log('scratch_initialization_verified',visual_sha256=initial_visual_hash,image_only=args.image_only)
    report['status']='TRAINING';write()
    paused=False;stage_complete=False
    for step,selected in enumerate(schedule,1):
        if step<=start_step:continue
        if args.end_epoch is not None and step>args.end_epoch*(len(train)//4):
            stage_complete=True;break
        epoch=(step-1)//(len(train)//4)+1
        if time.monotonic()-start>8*3600:
            paused=True;save_checkpoint(step-1,epoch);break
        optimizer.zero_grad(set_to_none=True);means={}
        for offset in (0,2):
            batch=collate([train[i] for i in selected[offset:offset+2]])
            image,ids=batch['image'].cuda(),batch['token_ids'].cuda();names=tuple(batch['sample_id'])
            output=model.with_language(image,ids,sample_ids=names)
            losses=model.compute_objectives(output,SuppliedTargets(names,ids,batch['label'].cuda(),batch['mask'].cuda(),tuple(batch['kind'])))
            (losses.total/2).backward()
            for k,v in losses.as_dict().items():means[k]=means.get(k,0.)+float(v.detach())/2
            del output,losses
        check_finite_tensors((p.grad for p in model.parameters()),'Language5000 gradient')
        optimizer.step();check_finite_tensors(model.parameters(),'Language5000 parameters')
        report['history'].append(dict(step=step,epoch=epoch,losses=means))
        log('train',step=step,total=len(schedule),epoch=epoch,losses=means)
        if step%10==0:write()
        if step%(len(train)//4)==0 or step==len(schedule):
            optimizer.zero_grad(set_to_none=True);evaluate_epoch(epoch,step);save_checkpoint(step,epoch)
        elif step%100==0:save_checkpoint(step,epoch)
    assert all(sha(ROOT/p)==h for p,h in source.items())
    report.update(status='PARTIAL_TIME_BUDGET' if paused else 'STAGE_COMPLETED' if stage_complete else 'COMPLETED',
        completed_updates=len(report['history']),target_epochs=args.epochs,source_unchanged=True)
    if not paused and not stage_complete:
        torch.save(dict(schema='mfvlr-language5000-stage1-final-v1',config=config,step=len(schedule),
            model={k:v.detach().cpu() for k,v in model.state_dict().items()},provenance=dict(source=source,data=report['data_sha256'])),args.output/'final.pt')
    write();log('experiment_complete',status=report['status'])

if __name__=='__main__':main()
