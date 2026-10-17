"""N2 read-only model review: new evidence only, no test/policy amendment.

FP64 workers promote the exact seed-0 FP32 weights and FP32 fixture values.
All statistics/perturbations are diagnostic instrumentation, not model edits.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src')]
import torch
from mfvlr.flt import FineGrainedLanguageTransformer,parameter_report

OUT=ROOT/'artifacts/phase2c2_n2'
FIELDS=('low','encoder_hidden','global_language','decoder_hidden','logits')


def write(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def tensor_hash(tensor):
    return hashlib.sha256(memoryview(tensor.detach().contiguous().numpy()).cast('B')).hexdigest()


def state_hash(model):
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        digest.update(name.encode());digest.update(str(tuple(p.shape)).encode())
        digest.update(memoryview(p.detach().numpy()).cast('B'))
    return digest.hexdigest()


def stats(tensor):
    x=tensor.detach()
    isfinite=bool(x.isfinite().all())
    if not isfinite:raise FloatingPointError('Nonfinite diagnostic activation/gradient')
    norm=torch.linalg.vector_norm(x,dtype=torch.float64).item()
    return {'shape':list(x.shape),'finite':isfinite,'l2_norm':norm,
            'rms':norm/math.sqrt(x.numel()),'max_abs':x.abs().max().item()}


def compare(actual,reference,atol,rtol):
    """Chunked metrics; legacy isclose counts retain original dtype arithmetic.

    Relative error uses |reference|, without an epsilon floor. Zero-reference
    nonzero discrepancies are counted explicitly instead of emitting JSON inf.
    """
    assert actual.shape==reference.shape and actual.dtype==reference.dtype
    a=actual.detach().reshape(-1);b=reference.detach().reshape(-1)
    maximum=relative=0.;sum_diff=sum_ref=0.;outside=zero_nonzero=0
    all_equal=True
    for start in range(0,a.numel(),262144):
        aa=a[start:start+262144];bb=b[start:start+262144]
        if not bool(aa.isfinite().all() and bb.isfinite().all()):
            raise FloatingPointError('Nonfinite comparison input')
        da=aa.double();db=bb.double();delta=(da-db).abs()
        maximum=max(maximum,delta.max().item())
        nonzero=db!=0
        if bool(nonzero.any()):relative=max(relative,(delta[nonzero]/db[nonzero].abs()).max().item())
        zero_nonzero+=int(((~nonzero)&(delta!=0)).sum())
        outside+=int((~torch.isclose(aa,bb,atol=atol,rtol=rtol)).sum())
        sum_diff+=delta.square().sum().item();sum_ref+=db.square().sum().item()
        all_equal=all_equal and torch.equal(aa,bb)
    return {'shape':list(actual.shape),'elements':a.numel(),'finite':True,'exact_equal':all_equal,
        'atol':atol,'rtol':rtol,'outside_tolerance_count':outside,'allclose':outside==0,
        'max_absolute_difference':maximum,'max_relative_difference_nonzero_reference':relative,
        'zero_reference_nonzero_difference_count':zero_nonzero,
        'max_relative_difference_is_infinite':zero_nonzero>0,
        'rms_absolute_difference':math.sqrt(sum_diff/a.numel()),
        'relative_l2_difference':math.sqrt(sum_diff/sum_ref) if sum_ref else None}


def fixture(index):
    cases=json.loads((ROOT/'tests/foundation/fixtures/golden_tokens.json').read_text())['prompt_cases']
    indices=((1,5),(0,4),(8,9))[index]
    ids=torch.tensor([cases[i]['ids'] for i in indices],dtype=torch.int64)
    if index<2:
        seed=(341,642)[index]
        v=torch.randn(2,1,512,generator=torch.Generator().manual_seed(seed))*0.1
        description=f'FP32 normal visual fixture, seed={seed}, std=0.1'
    else:
        from mfvlr.config import load_config
        from mfvlr.decoder_config import load_decoder_config
        from mfvlr.mve import MVE
        from mfvlr.vision_decoder import VisionDecoder
        mve=MVE.from_config(load_config(ROOT/'configs/debug.yaml'))
        vd=VisionDecoder.from_config(load_decoder_config(ROOT/'configs/decoder_debug.yaml'))
        image=torch.rand(2,3,224,224,generator=torch.Generator().manual_seed(412))
        with torch.no_grad():
            appearance=mve.encode_appearance(image)
            reconstruction=vd.decode_appearance(appearance).reconstruction
            v=mve(image,reconstruction,appearance_encoding=appearance).fused_global.clone()
        description='Accepted FP32 MVE+VD fused visual, random image seed=412, batch=2'
        del mve,vd;gc.collect()
    return ids,v,{'fixture_index':index,'golden_prompt_indices':indices,
        'metadata':[cases[i]['metadata'] for i in indices],'visual_generation':description,
        'ids_sha256':tensor_hash(ids),'fp32_visual_sha256':tensor_hash(v),
        'ids':ids.tolist(),'fp32_visual':v.tolist(),'visual_statistics':stats(v)}


class Capture:
    def __init__(self,model,gradient=False,detail=False):
        self.values={};self.hooks=[];self.gradient=gradient
        def keep(name,value):
            if gradient:
                value.retain_grad();self.values[name]=value
            else:self.values[name]=value.detach().clone()
        for name,module in model.named_modules():
            block=name.startswith(('encoder.blocks.','decoder.blocks.')) and name.count('.')==2
            primitive=detail and name.startswith('encoder.blocks.0.') and isinstance(module,(torch.nn.Linear,torch.nn.LayerNorm,torch.nn.GELU))
            if name=='vocabulary' or block or primitive:
                self.hooks.append(module.register_forward_hook(lambda m,args,value,n=name:keep(n,value)))
        self.hooks.append(model.encoder.register_forward_pre_hook(lambda m,args:keep('encoder.input',args[0])))
        self.hooks.append(model.decoder.register_forward_pre_hook(lambda m,args:keep('decoder.input',args[0])))
    def close(self):
        for h in self.hooks:h.remove()


def forward_capture(model,ids,v,detail=False):
    capture=Capture(model,detail=detail)
    try:
        with torch.no_grad():output=model(ids,v)
    finally:capture.close()
    for name in FIELDS:capture.values['output.'+name]=getattr(output,name).detach()
    return capture.values


def backend():
    return {'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),
        'processor':platform.processor(),'torch':torch.__version__,'torch_git_version':torch.version.git_version,
        'device':'cpu','cuda_available':torch.cuda.is_available(),'cuda_version':torch.version.cuda,
        'num_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'mkl_available':torch.backends.mkl.is_available(),'mkldnn_available':torch.backends.mkldnn.is_available(),
        'mkldnn_enabled':torch.backends.mkldnn.enabled,'build_config':torch.__config__.show(),
        'selected_environment':{key:os.environ.get(key) for key in
          ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_CBWR','CUBLAS_WORKSPACE_CONFIG')}}


def full_backward(model,ids,v):
    model.zero_grad(set_to_none=True)
    visual=v.clone().requires_grad_();capture=Capture(model,gradient=True)
    try:
        output=model(ids,visual)
        output.logits.retain_grad()
        probe=output.logits.square().mean()+output.decoder_hidden.square().mean()+output.global_language.square().mean()
        probe.backward()
    finally:capture.close()
    parameters={name:stats(p.grad) for name,p in model.named_parameters() if p.grad is not None}
    missing=[name for name,p in model.named_parameters() if p.grad is None]
    qk={name:bool(p.grad is not None and p.grad.count_nonzero()==0) for name,p in model.named_parameters()
        if '.vim.q.' in name or '.vim.k.' in name}
    result={'probe':'mean(logits^2)+mean(LD_hidden^2)+mean(LE_global^2), no production loss',
        'probe_value':probe.item(),'parameter_gradients':parameters,'missing_gradients':missing,
        'vim_qk_allocated_exact_zero':qk,'visual_gradient':stats(visual.grad),
        'logits_gradient':stats(output.logits.grad),
        'stage_gradients':{name:stats(value.grad) for name,value in capture.values.items()},
        'parameter_gradient_global_l2':math.sqrt(sum(value['l2_norm']**2 for value in parameters.values()))}
    del output,probe,capture;model.zero_grad(set_to_none=True);gc.collect()
    return result


def cross_sample_gradients(model,ids,v):
    # Integer token IDs have no derivative; low is their differentiable lookup boundary.
    visual=v.clone().requires_grad_();output=model(ids,visual)
    reports=[]
    for sample in (0,1):
        probe=output.logits[sample].square().mean()+output.decoder_hidden[sample].square().mean()+output.global_language[sample].square().mean()
        gv,gl=torch.autograd.grad(probe,(visual,output.low),retain_graph=sample==0)
        reports.append({'probe_sample':sample,'other_sample':1-sample,
            'other_visual_gradient_exact_zero':bool(gv[1-sample].count_nonzero()==0),
            'other_lookup_gradient_exact_zero':bool(gl[1-sample].count_nonzero()==0),
            'own_visual_gradient':stats(gv[sample]),'own_lookup_gradient':stats(gl[sample])})
    del output,probe;gc.collect()
    return reports


def perturbation(model,ids,v):
    # Fixed-batch FP64 finite perturbations isolate LD sensitivity, with memory/Iv fixed.
    with torch.no_grad():
        encoded=model.encode(ids,v)
        initial=model.shift_low(encoded.low)+model.decoder_position
        direction=torch.randn(initial.shape,generator=torch.Generator().manual_seed(123),dtype=torch.float64)
        direction=direction/torch.linalg.vector_norm(direction)*math.sqrt(direction.numel())
        base=[];x=initial
        for block in model.decoder.blocks:
            x=block(x,encoded.encoder_hidden,v);base.append(x)
        records=[]
        for epsilon in (1e-6,1e-7):
            x=initial+epsilon*direction;actual_input=stats(x-initial)
            stages=[]
            for i,block in enumerate(model.decoder.blocks):
                x=block(x,encoded.encoder_hidden,v)
                delta=stats(x-base[i])
                stages.append({'block':i+1,'difference':delta,
                    'rms_gain_from_actual_input_perturbation':delta['rms']/actual_input['rms']})
            records.append({'epsilon':epsilon,'actual_input_perturbation':actual_input,'stages':stages})
    return {'dtype':'float64','same_batch_shape':True,'memory_and_visual_held_fixed':True,
        'direction_seed':123,'direction_rms':1.0,'experiments':records,
        'normalization_counterfactual_run':False}


def worker(index,dtype,threads,destination):
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    result={'review_only':True,'N2_status':'PROPOSED','N2_applied':False,'N1_applied_to_FLT':False,
        'fixture':index,'dtype':dtype,'backend':backend(),'status':'RUNNING'}
    try:
        ids,visual,description=fixture(index)
        model=FineGrainedLanguageTransformer(seed=0)
        original_hash=state_hash(model)
        result.update(fixture_description=description,fp32_initialized_state_sha256=original_hash,
            parameter_report=parameter_report(model))
        if dtype=='float64':model=model.double();visual=visual.double()
        identities=[id(p) for p in model.parameters()]
        atol,rtol=(1e-6,1e-5) if dtype=='float32' else (1e-10,1e-8)
        batch=forward_capture(model,ids,visual,detail=index==0)
        result['activations']={name:stats(value) for name,value in batch.items()}
        result['cross_batch']=[]
        for sample in (0,1):
            single=forward_capture(model,ids[sample:sample+1],visual[sample:sample+1],detail=index==0)
            result['cross_batch'].append({'sample':sample,'stages':{
                name:compare(value[sample:sample+1],single[name],atol,rtol) for name,value in batch.items()}})
            del single
        perm=torch.tensor([1,0])
        swapped=forward_capture(model,ids[perm],visual[perm],detail=index==0)
        result['permutation']={name:compare(swapped[name][perm],value,atol,rtol) for name,value in batch.items()}
        del swapped
        result['fixed_batch_independence']=[]
        for changed_sample in (0,1):
            ci=ids.clone();ci[changed_sample,250]=1234
            cv=visual.clone();cv[changed_sample]+=0.5
            changed=forward_capture(model,ci,cv,detail=index==0)
            other=1-changed_sample
            result['fixed_batch_independence'].append({'changed_sample':changed_sample,'unaffected_sample':other,
                'exact_equality':{name:torch.equal(value[other],changed[name][other]) for name,value in batch.items()}})
            del changed
        with torch.no_grad():
            projected=model.project_vocabulary(batch['output.decoder_hidden'][:1])
        result['projection_only_cross_batch']=compare(batch['output.logits'][:1],projected,atol,rtol)
        del projected,batch;gc.collect()
        if dtype=='float32':
            result['full_backward']=full_backward(model,ids,visual)
            result['cross_sample_gradients']=cross_sample_gradients(model,ids,visual)
        else:
            result['decoder_perturbation']=perturbation(model,ids,visual)
        result['parameter_objects_unchanged']=[id(p) for p in model.parameters()]==identities
        result['vocabulary_tie_identity']=model.vocabulary_output_weight is model.vocabulary.weight
        result['state_unchanged_after_probes']=state_hash(model.float())==original_hash
        result['status']='COMPLETED'
    except Exception as error:
        result['status']='FAIL';result['exception']=repr(error);result['traceback']=traceback.format_exc()
        result['nonfinite_detected']=isinstance(error,FloatingPointError)
    result['elapsed_seconds']=time.perf_counter()-start
    write(destination,result)
    print(json.dumps({k:result[k] for k in ('fixture','dtype','status','elapsed_seconds')},indent=2),flush=True)
    if result['status']!='COMPLETED':print(result['traceback']);return 1
    return 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int,choices=(0,1,2))
    parser.add_argument('--dtype',choices=('float32','float64'),default='float32')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--output')
    args=parser.parse_args()
    if args.worker is not None:raise SystemExit(worker(args.worker,args.dtype,args.threads,args.output))
    OUT.mkdir(exist_ok=True)
    attempt=1
    while (OUT/f'run_{attempt:02}').exists():attempt+=1
    run=OUT/f'run_{attempt:02}';run.mkdir()
    jobs=[(i,d,4) for i in range(3) for d in ('float32','float64')]+[(0,'float32',1)]
    records=[]
    for index,dtype,threads in jobs:
        name=f'fixture{index}_{dtype}_threads{threads}'
        path=run/f'{name}.json'
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',str(index),'--dtype',dtype,
                 '--threads',str(threads),'--output',str(path)]
        completed=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding='utf-8',errors='replace')
        (run/f'{name}.log').write_text('COMMAND: '+json.dumps(command)+'\n'+completed.stdout+
            f'\nEXIT_CODE: {completed.returncode}\n',encoding='utf-8')
        print(completed.stdout,flush=True)
        if not path.is_file():raise RuntimeError(f'Worker exited without evidence: {completed.returncode}')
        record=json.loads(path.read_text());records.append({'file':path.relative_to(ROOT).as_posix(),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':record['status']})
        write(run/'index.json',{'status':'RUNNING','cases':records})
        if completed.returncode or record['status']!='COMPLETED':
            raise RuntimeError('Diagnostic failed; stop and inspect its preserved log before continuing')
    write(run/'index.json',{'status':'COMPLETED','policy_applied':False,'cases':records})


if __name__=='__main__':main()
