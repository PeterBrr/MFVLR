"""Fresh-process CPU resource/stability probes; no losses or optimizer.

Mandatory b1/b2 FLT and integrated routes retain full logits and autograd.
Batch8/GPU capacity is reported explicitly; conservative preflight is visible.
"""
from pathlib import Path
import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import threading
import time
import weakref

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src')]
import psutil
import torch

from mfvlr.flt import FineGrainedLanguageTransformer,parameter_report
from mfvlr.flt_config import load_flt_config
from mfvlr.config import load_config
from mfvlr.decoder_config import load_decoder_config
from mfvlr.mve import MVE
from mfvlr.vision_decoder import VisionDecoder
from mfvlr.visual_language import VisualLanguageOrchestrator

OUT=ROOT/'artifacts/phase2c2'


def stats(t):
    v=t.detach()
    return {'shape':list(v.shape),'dtype':str(v.dtype),'finite':bool(v.isfinite().all()),
            'max_abs':v.abs().max().item(),'norm_float64':torch.linalg.vector_norm(v,dtype=torch.float64).item()}


def worker(route,batch,destination):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    process=psutil.Process()
    baseline=process.memory_info().rss
    samples=[baseline]
    stop=threading.Event()
    def sample():
        while not stop.wait(0.002):samples.append(process.memory_info().rss)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    result={'route':route,'batch':batch,'dtype':'float32','device':'cpu','rss_baseline_bytes':baseline,
            'memory_sampling_interval_seconds':0.002,'optimizer_present':False,'status':'RUNNING'}
    try:
        flt=FineGrainedLanguageTransformer.from_config(load_flt_config(ROOT/'configs/flt_reference.yaml'))
        modules=[flt]
        if route=='integrated':
            mve=MVE.from_config(load_config(ROOT/'configs/debug.yaml'))
            vd=VisionDecoder.from_config(load_decoder_config(ROOT/'configs/decoder_debug.yaml'))
            wrapper=VisualLanguageOrchestrator(mve,vd,flt);modules+=[mve,vd]
        fixture=json.loads((ROOT/'tests/foundation/fixtures/golden_tokens.json').read_text())['prompt_cases']
        ids=torch.tensor([fixture[1 if i%2==0 else 5]['ids'] for i in range(batch)],dtype=torch.int64)
        g=torch.Generator().manual_seed(341)
        visual=(torch.randn(batch,1,512,generator=g)*0.1).requires_grad_()
        image=torch.rand(batch,3,224,224,generator=torch.Generator().manual_seed(412))
        traces=[];weak=[];hooks=[]
        for name,module in flt.named_modules():
            if name in ('vocabulary',) or name.startswith(('encoder.blocks.','decoder.blocks.')):
                # Block outputs, FF intermediate tensors and VIM values cover every layer.
                if name.count('.')==2 or name.endswith(('ff.fc1','ff.gelu','vim')):
                    def observe(m,args,out,n=name):
                        traces.append({'module':n,**stats(out)});weak.append(weakref.ref(out))
                    hooks.append(module.register_forward_hook(observe))
        def forward():
            return wrapper.with_language(image,ids) if route=='integrated' else flt(ids,visual)
        def lang(output):return output.language if route=='integrated' else output
        def reset():
            for module in modules:module.zero_grad(set_to_none=True)
            visual.grad=None
        def iteration(backward):
            reset();traces.clear();weak.clear()
            start=time.perf_counter()
            if backward:
                output=forward();language=lang(output)
                probe=language.logits.square().mean()+language.decoder_hidden.square().mean()+language.global_language.square().mean()
                probe.backward()
            else:
                with torch.no_grad():output=forward();language=lang(output)
            elapsed=time.perf_counter()-start
            assert language.logits.shape==(batch,308,49408)
            assert language.logits.isfinite().all()
            metadata={'logits_shape':list(language.logits.shape),'logits_bytes':language.logits.numel()*language.logits.element_size(),
                      'logits':stats(language.logits),'encoder':stats(language.encoder_hidden),'decoder':stats(language.decoder_hidden)}
            if route=='integrated':metadata['fused_visual']=stats(output.visual.fused_global)
            gradients={}
            if backward:
                for group,module in zip(('flt','mve','vd'),modules):
                    for name,p in module.named_parameters():
                        if p.grad is not None:
                            assert p.grad.isfinite().all(),f'nonfinite gradient {group}.{name}'
                            gradients[f'{group}.{name}']=stats(p.grad)
                        else:gradients[f'{group}.{name}']={'gradient':None}
                for name,p in flt.named_parameters():
                    assert p.grad is not None
                    if '.vim.q.' in name or '.vim.k.' in name:assert p.grad.count_nonzero()==0
                if route=='flt':assert visual.grad.isfinite().all() and visual.grad.norm()>0
                del probe
            final_trace=list(traces)
            del output,language
            gc.collect()
            metadata['live_output_refs_after_cleanup']=sum(ref() is not None for ref in weak)
            return elapsed,metadata,gradients,final_trace
        # One full warmup and three full measured iterations in each mode.
        iteration(False)
        forward_times=[]
        for _ in range(3):
            elapsed,metadata,_,_=iteration(False);forward_times.append(elapsed)
        iteration(True)
        backward_times=[];live=[]
        for _ in range(3):
            elapsed,metadata,gradients,trace=iteration(True);backward_times.append(elapsed)
            live.append(metadata['live_output_refs_after_cleanup'])
        assert live==[0,0,0],live
        result.update(metadata)
        result['forward']={'warmup':1,'iterations':3,'seconds':forward_times}
        result['forward_backward']={'warmup':1,'iterations':3,'seconds':backward_times}
        result['live_refs_after_each_backward']=live
        result['trace']=trace;result['gradients']=gradients
        result['parameter_report']=parameter_report(flt)
        result['parameter_bytes']=sum(p.numel()*p.element_size() for m in modules for p in m.parameters())
        result['gradient_bytes']=sum(p.grad.numel()*p.grad.element_size() for m in modules for p in m.parameters() if p.grad is not None)
        result['status']='PASS'
        for hook in hooks:hook.remove()
    except (MemoryError,torch.OutOfMemoryError) as error:
        result.update(status='FAIL/OOM',exception=repr(error))
    except Exception as error:
        result.update(status='FAIL',exception=repr(error))
        if isinstance(error,FloatingPointError) or 'nonfinite' in str(error):result['nonfinite_detected']=True
    finally:
        stop.set();thread.join()
        info=process.memory_info()
        result['rss_sampled_peak_bytes']=max(samples+[info.rss])
        result['rss_final_bytes']=info.rss
        result['platform_lifetime_peak_bytes']=getattr(info,'peak_wset',None)
        Path(destination).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('route','batch','status','rss_sampled_peak_bytes')},indent=2))
    return 0 if result['status']=='PASS' else 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',choices=('flt','integrated'))
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--output')
    args=parser.parse_args()
    if args.worker:raise SystemExit(worker(args.worker,args.batch,args.output))
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'resources').mkdir(exist_ok=True)
    cases=[]
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(ROOT/'.phase2c1_deps'),str(ROOT/'src')])
    env['PYTHONIOENCODING']='utf-8'
    for route in ('flt','integrated'):
        for batch in (1,2):
            destination=OUT/f'resources/{route}_b{batch}.json'
            command=[sys.executable,str(Path(__file__).resolve()),'--worker',route,'--batch',str(batch),'--output',str(destination)]
            completed=subprocess.run(command,env=env,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
            (OUT/f'resources/{route}_b{batch}.log').write_text('COMMAND: '+json.dumps(command)+'\n'+completed.stdout+f'\nEXIT_CODE: {completed.returncode}\n')
            if destination.is_file():case=json.loads(destination.read_text())
            else:case={'route':route,'batch':batch,'status':'FAIL/OOM','exit_code':completed.returncode,'exception':'worker exited without evidence'}
            cases.append(case);print(route,batch,case['status'],flush=True)
            if case.get('nonfinite_detected'):
                raise RuntimeError('Nonfinite detected: stop subsequent execution and inspect worker report')
    available=psutil.virtual_memory().available
    # Conservative full integrated b8 preflight; no claimed observed peak/capacity.
    required=8*1024**3
    batch8={'status':'NOT RUN','reason':'available RAM below conservative 8-GiB full integrated FP32/autograd preflight',
            'available_bytes':available,'preflight_required_available_bytes':required,'estimate_is_measurement':False}
    if available>=required:
        destination=OUT/'resources/integrated_b8.json'
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','integrated','--batch','8','--output',str(destination)]
        completed=subprocess.run(command,env=env,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        (OUT/'resources/integrated_b8.log').write_text(completed.stdout+f'\nEXIT_CODE: {completed.returncode}\n')
        batch8=json.loads(destination.read_text()) if destination.is_file() else {'status':'FAIL/OOM','exit_code':completed.returncode}
    result={'environment':{'python':platform.python_version(),'torch':torch.__version__,'platform':platform.platform(),
                          'cpu_threads':4,'total_ram_bytes':psutil.virtual_memory().total},
            'cases':cases,'batch8':batch8,
            'gpu':{'status':'NOT RUN','reason':'CPU-only PyTorch build; CUDA unavailable','cuda_available':torch.cuda.is_available()}}
    (OUT/'resources.json').write_text(json.dumps(result,indent=2)+'\n')
    if any(c['status']!='PASS' for c in cases):raise SystemExit(1)


if __name__=='__main__':main()
