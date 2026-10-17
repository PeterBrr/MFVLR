"""All-depth local decoder VJPs; original frozen block forwards are called."""
import copy
import gc
import math
import time

import numpy as np
import torch
from torch import nn

from mfvlr.language_decoder import LanguageDecoder
from mfvlr_research.decoder_a11 import NormalizedDecoderBlock
from mfvlr_research.a11_evidence import state_hash
from tools.d02a.common import OUT,MULTIPLIERS,SCALES,write,array,stats,compare,tensor_hash


def load_decoder(j,variant):
    with torch.random.fork_rng(devices=[]):
        module=LanguageDecoder()
        if variant!="A0":module.blocks=nn.ModuleList(NormalizedDecoderBlock.from_reference(b,variant) for b in module.blocks)
    record=torch.load(OUT/f"states/j{j}_{variant}.pt",map_location="cpu",weights_only=True)
    module.load_state_dict(record,strict=True)
    return module


def fixture(j,batch,index):
    result=[]
    for seed,shape,multiplier in zip((6201+j,6301+j,6401+j),((2,308,512),(2,308,512),(2,1,512)),MULTIPLIERS[index]):
        master=torch.randn(shape,generator=torch.Generator().manual_seed(seed))
        result.append((master[:batch].clone().contiguous()*multiplier).contiguous())
    u,v=[torch.randn(batch,308,512,generator=torch.Generator().manual_seed(seed))/math.sqrt(batch*308*512)
         for seed in (6501+j,6601+j)]
    return result,u,v


def capture(model,x,memory,visual):
    stages={"state":x.detach().clone(),"memory":memory.detach().clone(),"visual":visual.detach().clone()}
    outputs=[];handles=[]
    for i,block in enumerate(model.blocks):
        cache={}
        for name in ("self_attention","memory_attention","ff","vim","ln1","ln2","ln3"):
            if hasattr(block,name):handles.append(getattr(block,name).register_forward_hook(
                lambda mod,a,y,n=name,d=cache:d.__setitem__(n,y.detach().clone())))
        def finish(mod,args,y,i=i,d=cache):
            with torch.no_grad():
                if getattr(mod,"variant","A0")=="Q3":a,b,c=[d[n] for n in ("ln1","ln2","ln3")]
                else:
                    a=args[0]+d["self_attention"];b=a+d["memory_attention"];c=b+d["ff"]
                for name,value in (("self",a),("memory",b),("ff",c),("vim",c+d["vim"]),("output",y)):
                    stages[f"LD.{i+1}.{name}"]=value.detach().clone()
                if not torch.equal(stages[f"LD.{i+1}.vim"],y):raise AssertionError("capture residual mismatch")
                d.clear()
            outputs.append(y)
        handles.append(block.register_forward_hook(finish))
    try:model(x,memory,visual)
    finally:
        for h in handles:h.remove()
    if len(outputs)!=7 or len(stages)!=38:raise AssertionError("incomplete seven-block capture")
    return outputs,stages


def targets(model,cut,leaves):
    result={}
    for name,p in model.named_parameters():
        index=int(name.split('.')[1])
        if index>=cut:continue
        if any(role in name for role in ("self_attention.q.","self_attention.k.","memory_attention.q.","memory_attention.k.",".vim.",".ln1.",".ln2.",".ln3.")):
            result[name]=p
    result.update(zip(("input.state","input.memory","input.visual"),leaves))
    return result


def check_gradients(named,gradients,scale):
    for name,g in zip(named,gradients):
        if g is None or not g.isfinite().all():raise FloatingPointError("missing/nonfinite gradient: "+name)
        if ".vim.q." in name or ".vim.k." in name:
            if g.count_nonzero():raise AssertionError("singleton structural zero violation: "+name)
        if scale==0 and g.count_nonzero():raise AssertionError("zero cotangent produced nonzero VJP")


def run(j,batch,index,variant,destination,rows):
    source=load_decoder(j,variant);fingerprint=state_hash(source)
    fp32,u_base,v_base=fixture(j,batch,index)
    expected_shapes=[(batch,308,512),(batch,308,512),(batch,1,512)]
    assert [tuple(t.shape) for t in fp32]==expected_shapes
    write(destination/"inputs.json",{"state_hash_FP32":fingerprint,"parameter_count":sum(p.numel() for p in source.parameters()),
          "parameters":{n:stats(p) for n,p in source.named_parameters()},
          "leaves":dict(zip(("state","memory","visual"),map(stats,fp32))),
          "u":stats(u_base),"v":stats(v_base),"multipliers":MULTIPLIERS[index],"cut_schedule":[1,4,7],"scales":SCALES})
    scratch=destination/"temporary_gradients";scratch.mkdir()
    temporary_bytes=0;peak_disk=0
    for precision in ("float32","float64"):
        model=copy.deepcopy(source)
        assert state_hash(model)==fingerprint
        if precision=="float64":model.double()
        before=state_hash(model);identities=[id(p) for p in model.parameters()]
        leaves=[t.to(dtype=getattr(torch,precision)).clone().requires_grad_() for t in fp32]
        started=time.perf_counter();outputs,stages=capture(model,*leaves);cold=time.perf_counter()-started
        np.savez(destination/f"activations_{precision}.npz",**{n:array(v) for n,v in stages.items()})
        for name,value in stages.items():
            rows.add("activation",{"precision":precision,"stage":name},stats(value))
        if precision=="float64":
            with np.load(destination/"activations_float32.npz") as earlier:
                for name,value in stages.items():
                    rows.add("forward_FP32_vs_FP64",{"stage":name},compare(earlier[name],value,"float32"))
        del stages
        for cut in (1,4,7):
            named=targets(model,cut,leaves)
            sizes=[p.numel() for p in named.values()];bounds=np.cumsum([0]+sizes)
            for scale in SCALES:
                u32=u_base*scale;v32=v_base*scale
                u=u32.to(getattr(torch,precision));v=v32.to(getattr(torch,precision))
                cot=u+v
                rows.add("cotangent_sum_rounding",{"precision":precision,"cut":cut,"scale":scale},
                         compare((u32+v32).double(),u32.double()+v32.double(),"float32"))
                started=time.perf_counter()
                gu=torch.autograd.grad(outputs[cut-1],tuple(named.values()),u,retain_graph=True)
                gv=torch.autograd.grad(outputs[cut-1],tuple(named.values()),v,retain_graph=True)
                gj=torch.autograd.grad(outputs[cut-1],tuple(named.values()),cot,retain_graph=True)
                elapsed=time.perf_counter()-started
                for gradients in (gu,gv,gj):check_gradients(named,gradients,scale)
                files={role:scratch/f"cut{cut}_scale{scale}_{role}.npy" for role in ("u","v","joint")}
                if precision=="float32":
                    for role,gradients in zip(files,(gu,gv,gj)):
                        np.save(files[role],np.concatenate([array(g).reshape(-1) for g in gradients]))
                        temporary_bytes+=files[role].stat().st_size
                    peak_disk=max(peak_disk,temporary_bytes)
                    saved={}
                else:saved={role:np.load(path,mmap_mode="r") for role,path in files.items()}
                for i,(name,a,b,c) in enumerate(zip(named,gu,gv,gj)):
                    av=array(a);bv=array(b);cv=array(c)
                    summed=av.astype(np.float64)+bv.astype(np.float64)
                    reference=summed.astype(av.dtype)
                    meta={"precision":precision,"cut":cut,"scale":scale,"parameter":name}
                    r=compare(cv,reference,precision,name,True)
                    denominator=float(np.sqrt(np.sum(summed*summed)))
                    numerator=stats(a)["l2"]+stats(b)["l2"]
                    r.update(sum_individual_l2=numerator,sum_gradient_l2=denominator,
                             cancellation_ratio=numerator/denominator if denominator else None,
                             cancellation_zero_denominator=("infinite" if numerator else "undefined") if not denominator else None,
                             allocated=True,gradient_sha256=tensor_hash(c),VJP_triplet_seconds=elapsed,
                             singleton_QK=(".vim.q." in name or ".vim.k." in name),
                             exact_structural_zero=bool(np.count_nonzero(cv)==0) if ".vim.q." in name or ".vim.k." in name else None)
                    rows.add("VJP_additivity",meta,r)
                    # Separate norm/direction diagnostics retain the unrounded FP64 sum.
                    if precision=="float32":rows.add("VJP_vs_unrounded_sum",meta,compare(cv,summed,precision,name,True))
                    if ".k.bias" in name:
                        rows.add("ordinary_key_bias_vs_zero",meta,compare(cv,np.zeros_like(cv),precision,name,True))
                    if precision=="float64":
                        for role,g in (("u",av),("v",bv),("joint",cv)):
                            earlier=saved[role][bounds[i]:bounds[i+1]].reshape(g.shape)
                            rows.add("VJP_FP32_vs_FP64",{**meta,"role":role},compare(earlier,g,"float32",name,True))
                for mapped in saved.values():mapped._mmap.close()
                if precision=="float64":
                    for path in files.values():
                        assert path.resolve().is_relative_to((OUT).resolve()) and path.parent==scratch
                        temporary_bytes-=path.stat().st_size;path.unlink()
                rows.flush();del gu,gv,gj,saved
        del outputs,leaves;gc.collect()
        # Separate timing protocol: identical state/input reload, one untimed
        # no-grad warm-up, three cut-7 unit-scale no-update VJPs, plus cold time.
        frozen={n:p.detach().clone() for n,p in model.state_dict().items()}
        with torch.no_grad():model(*[t.to(getattr(torch,precision)) for t in fp32])
        timings=[]
        for repeat in range(3):
            model.load_state_dict(frozen,strict=True)
            leaves=[t.to(getattr(torch,precision)).clone().requires_grad_() for t in fp32]
            started=time.perf_counter();y=model(*leaves)
            named=targets(model,7,leaves)
            grad=torch.autograd.grad(y,tuple(named.values()),u_base.to(getattr(torch,precision)))
            check_gradients(named,grad,1.);timings.append(time.perf_counter()-started)
            del y,grad,leaves
        if state_hash(model)!=before or identities!=[id(p) for p in model.parameters()]:raise AssertionError("state/identity mutation")
        rows.add("timing",{"precision":precision},{"cold_forward_seconds":cold,"warmup_no_grad":1,
                 "measured_full_depth_unit_cotangent_seconds":timings,"identical_reloaded_state":True})
        del model,frozen;gc.collect()
    assert state_hash(source)==fingerprint and temporary_bytes==0
    return {"status":"COMPLETED_LOCAL_DIAGNOSTIC","temporary_gradient_peak_bytes":peak_disk,
            "state_and_parameter_identities_unchanged":True,"optimizer_steps":0}
