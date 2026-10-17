"""Independent local mathematical references for N3, no model/policy edits."""
from pathlib import Path
from fractions import Fraction
import copy
import argparse
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)]
OUT=ROOT/'artifacts/phase2d2_n3'


def explicit_attention(x,weights):
    """Independent formula: affine maps, exp-normalization, explicit heads."""
    q,k,v=[x@weights[n+'.weight'].T+weights[n+'.bias'] for n in ('q','k','v')]
    qh,kh,vh=[t.reshape(x.shape[0],x.shape[1],8,64).transpose(1,2) for t in (q,k,v)]
    scores=qh@kh.transpose(-1,-2)/8
    exponent=(scores-scores.max(-1,keepdim=True).values).exp()
    p=exponent/exponent.sum(-1,keepdim=True)
    context=(p@vh).transpose(1,2).reshape_as(x)
    output=context@weights['out.weight'].T+weights['out.bias']
    return output,(qh,kh,vh,p,context)


def explicit_gradients(x,weights,cotangent):
    _,(q,k,v,p,context)=explicit_attention(x,weights)
    dc=(cotangent@weights['out.weight']).reshape(x.shape[0],x.shape[1],8,64).transpose(1,2)
    dp=dc@v.transpose(-1,-2)
    ds=p*(dp-(dp*p).sum(-1,keepdim=True))
    dq=ds@k/8;dk=ds.transpose(-1,-2)@q/8;dv=p.transpose(-1,-2)@dc
    dq,dk,dv=[t.transpose(1,2).reshape_as(x) for t in (dq,dk,dv)]
    flat=x.reshape(-1,512)
    return {'q.weight':dq.reshape(-1,512).T@flat,'k.weight':dk.reshape(-1,512).T@flat,
            'q.bias':dq.sum((0,1)),'k.bias':dk.sum((0,1)),
            'input':dq@weights['q.weight']+dk@weights['k.weight']+dv@weights['v.weight']}


def main():
    import torch
    from mfvlr.language_encoder import LanguageAttention,initialize_attention
    from tools.phase2d2_diagnostics import FP32,FP64,stats,state_hash,write
    from tools.review_phase2c2_n2 import compare,backend,tensor_hash
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    # Exact rational softmax-Jacobian calculation, independent of PyTorch.
    p=[Fraction(1,2),Fraction(1,3),Fraction(1,6)];u=[Fraction(2),Fraction(-3),Fraction(5)]
    jacobian=[[p[i]*(int(i==j)-p[j]) for j in range(3)] for i in range(3)]
    exact=[]
    for scale in (Fraction(1,10000),Fraction(1),Fraction(10000),Fraction(100000000)):
        ds=[sum(scale*u[i]*jacobian[i][j] for i in range(3)) for j in range(3)]
        assert sum(ds)==0
        bias=[sum(q*d for d in ds) for q in (Fraction(1,2),Fraction(2,3))]
        assert bias==[0,0]
        exact.append({'scale':str(scale),'dscore':[str(v) for v in ds],'sum_dscore':str(sum(ds)),'key_bias':[str(v) for v in bias]})
    source=LanguageAttention();initialize_attention(source,torch.Generator().manual_seed(0));source_hash=state_hash(source)
    x32=torch.randn(1,308,512,generator=torch.Generator().manual_seed(247))*.2
    c32=torch.randn(1,308,512,generator=torch.Generator().manual_seed(319))
    records=[];oracles=[];differences=[]
    for dtype in (torch.float32,torch.float64):
        model=copy.deepcopy(source)
        assert state_hash(model)==source_hash
        model.to(dtype=dtype)
        for scale in (1e-4,1.,1e4,1e8):
            x=x32.to(dtype).clone().requires_grad_();cotangent=c32.to(dtype)*scale
            y,trace=model.forward_with_trace(x)
            names=('q.weight','k.weight','q.bias','k.bias')
            targets=tuple(dict(model.named_parameters())[n] for n in names)+(x,trace['logits'])
            gradients=torch.autograd.grad(y,targets,grad_outputs=cotangent)
            records.append({'dtype':str(dtype),'scale':scale,'key_bias':stats(gradients[3]),
                'score_row_sum':stats(gradients[-1].sum(-1)),
                'key_bias_vs_exact_real_zero':compare(gradients[3],torch.zeros_like(gradients[3]),**(FP64 if dtype==torch.float64 else FP32)),
                'q_weight':stats(gradients[0]),'k_weight':stats(gradients[1]),
                'all_values_finite':all(bool(g.isfinite().all()) for g in gradients)})
            if scale==1.:
                # Manual derivatives contain no autograd invocation.
                with torch.no_grad():
                    weights={n:p.detach().clone() for n,p in model.named_parameters()}
                    manual=explicit_gradients(x.detach(),weights,cotangent)
                comparisons={n:compare(g,manual[n],**(FP64 if dtype==torch.float64 else FP32)) for n,g in zip((*names,'input'),gradients[:-1])}
                if dtype==torch.float64:
                    assert all(r['allclose'] for r in comparisons.values()),'unit-scale independent FP64 formula failed'
                    for index,name in enumerate(('q.weight','k.weight')):
                        direction=torch.randn(weights[name].shape,generator=torch.Generator().manual_seed(1003+index),dtype=torch.float64)
                        direction=direction/direction.norm();epsilon=1e-6
                        plus={**weights,name:weights[name]+epsilon*direction}
                        minus={**weights,name:weights[name]-epsilon*direction}
                        with torch.no_grad():
                            fd=((explicit_attention(x.detach(),plus)[0]*cotangent).sum()-(explicit_attention(x.detach(),minus)[0]*cotangent).sum())/(2*epsilon)
                            analytical=(gradients[index]*direction).sum()
                        torch.testing.assert_close(fd,analytical,atol=1e-5,rtol=1e-3)
                        differences.append({'parameter':name,'epsilon':epsilon,'atol':1e-5,'rtol':1e-3,'finite_difference':fd.item(),'autograd_directional':analytical.item(),'absolute_error':abs(fd.item()-analytical.item()),'status':'PASS'})
                oracles.append({'dtype':str(dtype),'scale':1.,'comparisons':comparisons})
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT/'local_analytical.json');args=parser.parse_args()
    dest=args.output;assert not dest.exists()
    write(dest,{'status':'DIAGNOSTIC_COMPLETE_PHASE_BLOCKED','exact_rational_Jacobian':exact,
        'backend':backend(),'FP32_initialized_source_sha256':source_hash,
        'input_FP32_sha256':tensor_hash(x32),'cotangent_FP32_sha256':tensor_hash(c32),
        'scales':[1e-4,1.,1e4,1e8],'precision_records':records,'independent_manual_attention_oracles':oracles,
        'Q_K_weight_finite_differences':differences,'parameter_changes':False,
        'note':'Ordinary key-bias residuals are diagnostics, including any FP64 stress-scale failure. Only unit-scale FP64 independent formulas and original finite-difference tolerances are asserted.'})
    print(json.dumps({'exact_rational':'PASS','FP64_manual_Q_K_and_input':'PASS','Q_K_weight_finite_differences':'PASS','phase':'BLOCKED'}))


if __name__=='__main__':main()
