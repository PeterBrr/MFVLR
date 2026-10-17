"""Synthetic Phase 2B measurements only; no optimizer or paper objective."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import psutil
import torch
from torch import nn

from mfvlr.config import load_config
from mfvlr.decoder_config import load_decoder_config
from mfvlr.mve import MVE,parameter_report
from mfvlr.vision_decoder import VisionDecoder,decoder_parameter_report


def shape(value):
    if isinstance(value,torch.Tensor):return list(value.shape)
    if isinstance(value,(tuple,list)):return [shape(item) for item in value]
    return None


def run(args):
    config=load_decoder_config(ROOT/'configs/decoder_debug.yaml')
    config.require_approval()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    process=psutil.Process()
    rss_import=process.memory_info().rss
    decoder=VisionDecoder.from_config(config)
    model=MVE.from_config(load_config(config.mve_path)) if args.component=='integrated' else None
    decoder.train(args.backward)
    if model is not None:model.train(args.backward)
    counts=decoder_parameter_report(decoder)
    assert counts['total_unique']==9055365
    rss_model=process.memory_info().rss
    events=[];handles=[];calls={}
    for prefix,owner in [('vd',decoder)]+([] if model is None else [('mve',model)]):
        for name,module in owner.named_modules():
            if isinstance(module,(nn.Conv2d,nn.ConvTranspose2d,nn.ReLU,nn.MaxPool2d,nn.Linear,nn.LayerNorm,nn.GELU)):
                full=f'{prefix}.{name}'
                def forward(m,inputs,out,name=full):
                    calls[name]=calls.get(name,0)+1
                    events.append({'direction':'forward','module':name,'call':calls[name],'input':shape(inputs),'output':shape(out)})
                handles.append(module.register_forward_hook(forward))
                if args.backward:
                    def backward(m,inputs,out,name=full):
                        events.append({'direction':'backward','module':name,'input_gradient':shape(inputs),'output_gradient':shape(out)})
                    handles.append(module.register_full_backward_hook(backward))
    samples=[rss_model];stop=threading.Event()
    def monitor():
        while not stop.wait(.01):samples.append(process.memory_info().rss)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    generator=torch.Generator().manual_seed(17)
    checks=[];started=time.perf_counter()
    try:
        for index in range(args.iterations):
            decoder.zero_grad(set_to_none=True)
            if model is not None:model.zero_grad(set_to_none=True)
            iteration_started=time.perf_counter()
            with torch.set_grad_enabled(args.backward):
                appearance=None;encoded=None
                if model is None:
                    value=torch.rand(args.batch,1024,14,14,generator=generator,requires_grad=args.backward)
                    decoded=decoder(value)
                    probe=decoded.reconstruction.square().mean()+decoded.mask_logits.square().mean()
                    watched={'local':value,'features':decoded.features,'reconstruction':decoded.reconstruction,'mask_logits':decoded.mask_logits}
                else:
                    value=torch.rand(args.batch,3,224,224,generator=generator,requires_grad=args.backward)
                    appearance=model.encode_appearance(value)
                    decoded=decoder.decode_appearance(appearance)
                    encoded=model(value,decoded.reconstruction,appearance_encoding=appearance)
                    probe=encoded.fused_global.square().mean()+decoded.mask_logits.square().mean()
                    watched={'image':value,'local':appearance.local,'features':decoded.features,'reconstruction':decoded.reconstruction,
                             'mask_logits':decoded.mask_logits,'residual':encoded.residual_image,'residual_tokens':encoded.residual_tokens,
                             'residual_global':encoded.residual_global,'fused_global':encoded.fused_global}
                check={'iteration':index+1,'shapes':{k:shape(v) for k,v in watched.items()},'probe_scalar':probe.item(),
                       'is_paper_loss':False}
                if args.backward:
                    for tensor in watched.values():tensor.retain_grad()
                    probe.backward()
                    check['gradient_shapes']={k:shape(v.grad) for k,v in watched.items()}
                    check['gradient_norms']={k:None if v.grad is None else v.grad.norm().item() for k,v in watched.items()}
                    gradients=[p.grad for p in decoder.parameters()]
                    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
                    check['finite_vd_parameter_gradient_tensors']=len(gradients)
                    if model is not None:
                        mve_gradients=[p.grad for p in model.parameters()]
                        assert all(g is not None and torch.isfinite(g).all() for g in mve_gradients)
                        check['finite_mve_parameter_gradient_tensors']=len(mve_gradients)
                        del mve_gradients
                    del gradients
                check['seconds']=time.perf_counter()-iteration_started
                checks.append(check)
                print(f'{args.component} b={args.batch} backward={args.backward} iteration={index+1}: '
                      f'VD={counts["total_unique"]:,}, seconds={check["seconds"]:.6f}',flush=True)
                del watched,probe,decoded,encoded,appearance,value
            samples.append(process.memory_info().rss)
    finally:
        stop.set();thread.join()
        for handle in handles:handle.remove()
    elapsed=time.perf_counter()-started
    memory=process.memory_info()
    summary={
        'component':args.component,'batch':args.batch,'backward':args.backward,'iterations':args.iterations,
        'optimizer_updates':0,'environment':{'python':platform.python_version(),'torch':torch.__version__,
            'psutil':psutil.__version__,'platform':platform.platform(),'threads':torch.get_num_threads(),
            'device':'cpu','precision':'float32','cuda_available':torch.cuda.is_available()},
        'vd_parameters':counts,'mve_parameters':None if model is None else parameter_report(model),
        'timing':{'elapsed_seconds':elapsed,'includes_trace_hooks':True,'includes_model_construction':False,'warmup_iterations':0},
        'memory':{'rss_after_import_bytes':rss_import,'rss_after_models_bytes':rss_model,
            'rss_after_execution_bytes':memory.rss,'sampled_interval_peak_rss_bytes':max(samples),
            'sample_interval_seconds':.01,'samples':len(samples),'process_lifetime_peak_working_set_bytes':getattr(memory,'peak_wset',None),
            'method':'psutil RSS polling; Windows peak_wset when available; fresh process per case; not tensor-only memory',
            'gpu_status':'NOT RUN: CUDA unavailable' if not torch.cuda.is_available() else 'Available; CPU measurement only'},
        'checks':checks,
    }
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    (args.output_dir/'tensor_trace.json').write_text(json.dumps(events,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary['memory'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',choices=['vd','integrated'],required=True)
    parser.add_argument('--batch',type=int,choices=[1,2],required=True)
    parser.add_argument('--backward',action='store_true')
    parser.add_argument('--iterations',type=int,default=1)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.iterations not in (1,3):parser.error('Only one check or three debug checks are supported')
    run(args)
