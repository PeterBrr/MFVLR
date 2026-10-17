"""Registered ordinary-attention primitive and explicit residual/LN compositions.

NATT is the independent unmasked ordinary attention primitive (N3 convention),
not a replacement for the causal attention inside the frozen NLD blocks.
"""
import copy
import math
import time

import numpy as np
import torch

from mfvlr.language_encoder import LanguageAttention,initialize_attention
from mfvlr_research.a11_evidence import state_hash
from tools.review_phase2d2_n3_local import explicit_attention,explicit_gradients
from tests.research_a11.oracles import norm_forward,norm_vjp
from tools.d02a.common import SCALES,write,stats,compare,array,directions,tensor_hash


def forward(model,norm,x,variant):
    z=x+model(norm(x) if variant=="P3" else x)
    return norm(z) if variant=="Q3" else z


def manual(x,weights,gamma,beta,cot,variant):
    if variant=="P3":
        normalized=norm_forward(x,gamma,beta)
        value=explicit_attention(normalized,weights)[0]
        grads=explicit_gradients(normalized,weights,cot)
        dx,dg,db=norm_vjp(x,gamma,grads['input'])
        grads['input']=cot+dx;grads['norm.weight']=dg;grads['norm.bias']=db
        return x+value,grads
    value=explicit_attention(x,weights)[0];z=x+value
    if variant=="Q3":
        dz,dg,db=norm_vjp(z,gamma,cot)
        grads=explicit_gradients(x,weights,dz);grads['input']+=dz
        grads['norm.weight']=dg;grads['norm.bias']=db
        return norm_forward(z,gamma,beta),grads
    grads=explicit_gradients(x,weights,cot);grads['input']+=cot
    return z,grads


def independent_forward(x,weights,gamma,beta,variant):
    if variant=="P3":return x+explicit_attention(norm_forward(x,gamma,beta),weights)[0]
    z=x+explicit_attention(x,weights)[0]
    return norm_forward(z,gamma,beta) if variant=="Q3" else z


def run(j,batch,index,variant,destination,rows):
    with torch.random.fork_rng(devices=[]):source=LanguageAttention()
    initialize_attention(source,torch.Generator().manual_seed(6101+j))
    fingerprint=state_hash(source)
    x32=torch.randn(batch,308,512,generator=torch.Generator().manual_seed(6801+j))*(.02,.2,2)[index]
    c32=torch.randn(batch,308,512,generator=torch.Generator().manual_seed(6901+j))
    write(destination/"inputs.json",{"state_hash_FP32":fingerprint,"x":stats(x32),"cotangent_base":stats(c32),
          "initialization_seed":6101+j,"parameters":{n:stats(p) for n,p in source.named_parameters()},
          "primitive":"independent unmasked ordinary attention","composition":"x+S(x), x+S(N(x)), N(x+S(x)) for A0,P3,Q3",
          "parameter_count":sum(p.numel() for p in source.parameters())+(1024 if variant!="A0" else 0)})
    earlier={}
    for precision in ("float32","float64"):
        dtype=getattr(torch,precision);model=copy.deepcopy(source).to(dtype)
        norm=torch.nn.LayerNorm(512,eps=1e-5,dtype=dtype)
        before=state_hash(model);norm_before=state_hash(norm)
        x=x32.to(dtype).clone().requires_grad_();named={n:p for n,p in model.named_parameters() if n in ('q.weight','k.weight','q.bias','k.bias')}
        if variant!="A0":named.update({'norm.weight':norm.weight,'norm.bias':norm.bias})
        named['input']=x
        started=time.perf_counter();y=forward(model,norm,x,variant);cold=time.perf_counter()-started
        rows.add("activation",{"precision":precision,"stage":"composition"},stats(y))
        np.save(destination/f"activation_{precision}.npy",array(y))
        with torch.no_grad():
            primitive=model(x)
            primitive_ref=explicit_attention(x,{n:p.detach() for n,p in model.named_parameters()})[0]
            rows.add("ordinary_primitive_forward_oracle",{"precision":precision},compare(primitive,primitive_ref,precision))
        if precision=="float64":rows.add("forward_FP32_vs_FP64",{},compare(np.load(destination/"activation_float32.npy"),y,"float32"))
        for scale in SCALES:
            cot=(c32*scale).to(dtype)
            grads=torch.autograd.grad(y,tuple(named.values()),cot,retain_graph=True)
            if any(not g.isfinite().all() for g in grads):raise FloatingPointError("NATT gradients")
            if scale==0 and any(g.count_nonzero() for g in grads):raise AssertionError("zero cotangent")
            weights={n:p.detach() for n,p in model.named_parameters()}
            with torch.no_grad():expected,oracle=manual(x.detach(),weights,norm.weight.detach(),norm.bias.detach(),cot,variant)
            rows.add("composition_forward_oracle",{"precision":precision,"scale":scale},compare(y,expected,precision))
            for name,g in zip(named,grads):
                meta={"precision":precision,"scale":scale,"parameter":name}
                rows.add("ordinary_derivative_oracle",meta,compare(g,oracle[name],precision,name,True))
                rows.add("gradient_norm",meta,stats(g))
                if name=='k.bias':rows.add("ordinary_key_bias_vs_zero",meta,compare(g,torch.zeros_like(g),precision,name,True))
                if precision=="float32":earlier[(scale,name)]=array(g).copy()
                else:rows.add("gradient_FP32_vs_FP64",meta,compare(earlier[(scale,name)],g,"float32",name,True))
                if precision=="float64" and scale==1 and name in ('q.weight','k.weight'):
                    which=0 if name=='q.weight' else 1
                    d=torch.from_numpy(directions(name,g.numel())[which].copy()).to(torch.float64).reshape_as(g)/math.sqrt(g.numel())
                    with torch.no_grad():
                        plus=independent_forward(x.detach(),{**weights,name:weights[name]+1e-6*d},norm.weight,norm.bias,variant)
                        minus=independent_forward(x.detach(),{**weights,name:weights[name]-1e-6*d},norm.weight,norm.bias,variant)
                        numerical=((plus*cot).sum()-(minus*cot).sum())/(2e-6);analytical=(g*d).sum()
                    error=abs(numerical.item()-analytical.item());limit=1e-5+1e-3*abs(analytical.item())
                    rows.add("directional_FD",meta,{"status":"PASS" if error<=limit else "FAIL","epsilon":1e-6,
                             "atol":1e-5,"rtol":1e-3,"numerical":numerical.item(),"analytical":analytical.item(),
                             "absolute_error":error,"direction_seed":6701+which})
            rows.flush()
        # The registered constant / near-constant LN fixtures are checked for
        # each seed and batch, never selected according to their observed error.
        if precision=="float64":
            for kind in ("constant","near_constant"):
                z32=torch.full((batch,308,512),.25)
                if kind=="near_constant":z32+=1e-4*torch.randn(z32.shape,generator=torch.Generator().manual_seed(6951+j))
                z=z32.double().requires_grad_();output=norm(z);cot=c32.double()
                gs=torch.autograd.grad(output,(z,norm.weight,norm.bias),cot)
                with torch.no_grad():expected=norm_forward(z,norm.weight,norm.bias);refs=norm_vjp(z,norm.weight,cot)
                r=compare(output,expected,"float64")
                rows.add("LN_fixture_forward",{"fixture":kind,"input_sha256":tensor_hash(z32)},r)
                if r['status']!='PASS':raise AssertionError("mandatory LN forward oracle failed")
                for name,g,ref in zip(('input','gamma','beta'),gs,refs):
                    r=compare(g,ref,'float64',name,True)
                    rows.add("LN_fixture_VJP",{"fixture":kind,"parameter":name},r)
                    if r['status']!='PASS':raise AssertionError("mandatory LN VJP oracle failed")
        with torch.no_grad():forward(model,norm,x.detach(),variant)
        timings=[]
        for repeat in range(3):
            model.load_state_dict(source.state_dict(),strict=True)
            # Loading the FP32 source into the FP64 model reproduces the exact
            # promoted values. Norms have never been updated.
            started=time.perf_counter();yy=forward(model,norm,x,variant)
            torch.autograd.grad(yy,tuple(named.values()),c32.to(dtype));timings.append(time.perf_counter()-started)
        if state_hash(model)!=before or state_hash(norm)!=norm_before:raise AssertionError("state mutation")
        rows.add("timing",{"precision":precision},{"cold_forward_seconds":cold,"warmup_no_grad":1,"measured_seconds":timings})
    assert state_hash(source)==fingerprint
    return {"status":"COMPLETED_LOCAL_DIAGNOSTIC","state_unchanged":True,"optimizer_steps":0}
