"""Fresh-process independent loss diagnostics, never a full model/train step.

Run from source: python tools/diagnose_phase2d1.py (four B/dtype processes).
No optimizer, parameter update, full-model composition or production detach.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d1'
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def worker(batch,dtype_name):
    import psutil
    import torch
    from mfvlr.objectives import (forgery_detection_loss as fd,language_reconstruction_loss as lr,
        language_token_nll,CrossModalContrastiveLoss,forgery_localization_loss as fl,
        appearance_reconstruction_loss as ar,knowledge_distillation_loss as kl)
    from tests.objectives.oracles import cmc_fixture,kl_fixture,cmc_reference,kl_reference,FP32,FP64
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    dtype=getattr(torch,dtype_name);process=psutil.Process();samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            samples.append(process.memory_info().rss);stop.wait(.005)
    baseline=process.memory_info().rss;thread=threading.Thread(target=monitor,daemon=True);thread.start()
    started=time.perf_counter();records={}
    def stats(tensor):
        if tensor is None:return {'allocated':False}
        value=tensor.detach().double()
        return {'allocated':True,'finite':bool(value.isfinite().all()),'shape':list(value.shape),
                'l2':float(value.norm()),'max_abs':float(value.abs().max()),
                'nonzero':int(value.count_nonzero())}
    def probe(name,call,variables,extra=None):
        start=time.perf_counter();loss=call()
        assert loss.ndim==0 and bool(loss.isfinite())
        loss.backward()
        gradients={key:stats(tensor.grad) for key,tensor in variables.items()}
        assert all(g['allocated'] and g['finite'] for g in gradients.values())
        records[name]={'loss':loss.item(),'rank':loss.ndim,'finite':True,'gradients':gradients,
                       'elapsed_forward_backward_seconds':time.perf_counter()-start,**(extra or {})}
        return loss.item()
    try:
        # FD full dimensions, uniform classes, then independent direct leaf grads.
        logits=torch.zeros(batch,2,dtype=dtype,requires_grad=True)
        targets=torch.arange(batch,dtype=torch.int64)%2
        probe('FD',lambda:fd(logits,targets),{'logits':logits},{'oracle':math.log(2),'class_order':['Real','Fake']})
        del logits,targets;gc.collect()
        logits=torch.zeros(batch,2,224,224,dtype=dtype,requires_grad=True)
        masks=torch.zeros(batch,224,224,dtype=torch.int64)
        if batch==2:masks[1]=1
        probe('FL',lambda:fl(logits,masks),{'logits':logits},{'oracle':math.log(2),'mask_categories':['real_zero']+(['efs_one'] if batch==2 else [])})
        del logits,masks;gc.collect()
        # Golden IDs are read verbatim; no tokenizer or target policy changed.
        golden_path=ROOT/'tests/foundation/fixtures/golden_tokens.json'
        golden=json.loads(golden_path.read_text())
        case=golden['prompt_cases'][0]
        ids=torch.tensor(case['ids'],dtype=torch.int64).reshape(1,308).repeat(batch,1)
        logits=torch.zeros(batch,308,49408,dtype=dtype,requires_grad=True)
        with torch.no_grad():uniform_value=lr(logits,ids).item()
        # Known target-score offsets exercise actual pad/special/content sums.
        scores={0:-.5,49406:1.,49407:2.}
        with torch.no_grad():
            for b in range(batch):
                for position,target in enumerate(ids[b].tolist()):logits[b,position,target]=scores.get(target,.25)
            nll=language_token_nll(logits,ids)
            category_masks={'padding':ids==0,'special':(ids==49406)|(ids==49407),
                            'content':(ids!=0)&(ids!=49406)&(ids!=49407)}
            categories={name:{'count':int(mask.sum()),'batch_mean_nll_contribution':float(nll[mask].sum()/batch)}
                        for name,mask in category_masks.items()}
            del nll
        expected=math.fsum(math.log(49407+math.exp(scores.get(t,.25)))-scores.get(t,.25) for t in ids[0].tolist())
        value=probe('LR',lambda:lr(logits,ids),{'logits':logits},
                    {'uniform_value':uniform_value,'uniform_oracle':308*math.log(49408),'oracle':expected,
                     'categories':categories,'categories_sum':sum(v['batch_mean_nll_contribution'] for v in categories.values()),
                     'golden_case':0,'golden_sha256':digest(golden_path),'targets':'unshifted_all_308'})
        tolerance=FP64 if dtype==torch.float64 else FP32
        assert math.isclose(sum(v['batch_mean_nll_contribution'] for v in categories.values()),value,
                            abs_tol=tolerance['atol'],rel_tol=tolerance['rtol'])
        del logits,ids;gc.collect()
        module=CrossModalContrastiveLoss().to(dtype=dtype)
        v,l=cmc_fixture(batch=batch,dtype=dtype,scale=.1)
        with torch.no_grad():
            raw=v[:,0]@l[:,0].T;scores_tensor=module.scores(v,l)
            rows=scores_tensor.softmax(1);columns=scores_tensor.T.softmax(1)
            def entropy(p):
                return (-(p.double()*p.double().log()).sum(-1)).tolist()
            # Explicit duplicated-language experiment: diagonal labels retained.
            dv=torch.ones(batch,1,512,dtype=dtype);dl=torch.ones_like(dv)
            duplicate_value=module(dv,dl).item()
            cmc_extra={'tau':module.temperature.item(),'theta':module.log_temperature.item(),
                       'raw_scores':raw.tolist(),'scaled_scores':scores_tensor.tolist(),'raw_score_max':raw.abs().max().item(),
                       'scaled_score_max':scores_tensor.abs().max().item(),'row_entropy':entropy(rows),
                       'column_entropy':entropy(columns),'duplicate_identical_feature_loss':duplicate_value,
                       'duplicate_prompt_count':max(batch-1,0),'duplicate_feature_fixture':'all_one_features_language_and_visual',
                       'duplicate_oracle':math.log(batch),'diagonal_positives':list(range(batch)),
                       'oracle':cmc_reference(v[:,0].tolist(),l[:,0].tolist(),module.log_temperature.item())[0]}
        probe('CMC',lambda:module(v,l),{'visual':v,'language':l,'theta':module.log_temperature},cmc_extra)
        del module,v,l,raw,scores_tensor,rows,columns,dv,dl;gc.collect()
        text,adapted=kl_fixture(batch=batch,dtype=dtype)
        with torch.no_grad():
            p=(text/.5).softmax(-1);q=(adapted/.5).softmax(-1)
            extra={'oracle':kl_reference(text[:,0].tolist(),adapted[:,0].tolist())[0],
                   'teacher_entropy':entropy(p[:,0]),'student_entropy':entropy(q[:,0]),
                   'teacher_probability_sum':p.sum(-1).tolist(),'student_probability_sum':q.sum(-1).tolist(),
                   'temperature':.5,'direction':'text||adapted_vision','both_distributions_differentiated':True}
        probe('KL',lambda:kl(text,adapted),{'text':text,'adapted_vision':adapted},extra)
        del text,adapted,p,q;gc.collect()
        image=torch.zeros(batch,3,224,224,dtype=dtype,requires_grad=True)
        reconstruction=torch.full_like(image,2.,requires_grad=True)
        probe('AR',lambda:ar(image,reconstruction),{'image':image,'reconstruction':reconstruction},
              {'oracle':4.,'image_range':[0.,0.],'reconstruction_range':[2.,2.],'clipping':False})
        del image,reconstruction;gc.collect()
        # CPU diagnostic tolerances identical to new unit tests; no N1/N2 scope.
        tolerance=FP64 if dtype==torch.float64 else FP32
        for name,record in records.items():
            assert math.isclose(record['loss'],record['oracle'],abs_tol=tolerance['atol'],rel_tol=tolerance['rtol']),name
        lr_record=records['LR']
        assert math.isclose(lr_record['uniform_value'],lr_record['uniform_oracle'],abs_tol=tolerance['atol'],rel_tol=tolerance['rtol'])
        status='PASS'
    finally:
        stop.set();thread.join();samples.append(process.memory_info().rss)
    elapsed=time.perf_counter()-started
    record={'status':status,'batch_size':batch,'dtype':dtype_name,'scope':'independent_leaf_loss_fixtures_not_full_model',
            'optimizer_updates':0,'graph_composition_with_accepted_modules':False,'records':records,
            'resources':{'rss_baseline_bytes':baseline,'sampled_peak_rss_bytes':max(samples),
                         'sampled_rss_increase_bytes':max(samples)-baseline,'rss_sampling_interval_seconds':.005,
                         'rss_samples':len(samples),'elapsed_seconds':elapsed,'includes_fixture_build_and_gc':True},
            'backend':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
                       'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
                       'deterministic':torch.are_deterministic_algorithms_enabled(),
                       'matmul_precision':torch.get_float32_matmul_precision(),'cuda_available':torch.cuda.is_available(),
                       'cuda_build':torch.version.cuda,'torch_config':torch.__config__.show()},
            'fixture_provenance':{'reference_fixtures_sha256':digest(ROOT/'tests/objectives/reference_fixtures.json'),
                                  'oracle_source_sha256':digest(ROOT/'tests/objectives/oracles.py'),
                                  'script_sha256':digest(Path(__file__))}}
    write(OUT/f'diagnostics/b{batch}_{dtype_name}.json',record)
    print(json.dumps({'status':status,'batch':batch,'dtype':dtype_name,'elapsed_seconds':elapsed}),flush=True)


def driver():
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)])
    for batch in (1,2):
        for dtype in ('float32','float64'):
            logs=OUT/'logs';logs.mkdir(parents=True,exist_ok=True)
            stem=f'diagnostics_b{batch}_{dtype}'
            if (logs/f'{stem}.stdout.log').exists():
                raise FileExistsError('Preserve existing diagnostics; choose a new evidence directory before rerunning')
            command=[sys.executable,str(Path(__file__)),'--batch',str(batch),'--dtype',dtype]
            with (logs/f'{stem}.stdout.log').open('w',encoding='utf-8') as out,(logs/f'{stem}.stderr.log').open('w',encoding='utf-8') as err:
                started=time.perf_counter();result=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err)
            write(logs/f'{stem}.execution.json',{'command':command,'exit_code':result.returncode,'elapsed_seconds':time.perf_counter()-started})
            if result.returncode:raise SystemExit(result.returncode)
    print('All four standalone B/dtype diagnostic processes passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=int,choices=(1,2));parser.add_argument('--dtype',choices=('float32','float64'))
    args=parser.parse_args()
    if args.batch is None and args.dtype is None:driver()
    elif args.batch is not None and args.dtype is not None:worker(args.batch,args.dtype)
    else:parser.error('Specify both batch and dtype, or neither')
