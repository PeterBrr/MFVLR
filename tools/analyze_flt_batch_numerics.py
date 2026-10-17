"""Diagnostic only: retain original T17 tolerances and all failed comparisons."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src')]
import torch
from mfvlr.flt import FineGrainedLanguageTransformer


def comparison(a,b):
    difference=(a-b).abs()
    allowed=1e-6+1e-5*b.abs()
    return {'elements':a.numel(),'max_abs_difference':difference.max().item(),
            'outside_original_tolerance':int((difference>allowed).sum()),
            'allclose_original':bool(torch.allclose(a,b,atol=1e-6,rtol=1e-5)),
            'max_difference_over_allowed':(difference/allowed).max().item()}


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    module=FineGrainedLanguageTransformer()
    fixture=json.loads((ROOT/'tests/foundation/fixtures/golden_tokens.json').read_text())['prompt_cases']
    ids=torch.tensor([fixture[1]['ids'],fixture[5]['ids']],dtype=torch.int64)
    visual=torch.randn(2,1,512,generator=torch.Generator().manual_seed(341))*0.1
    traces=[];hooks=[]
    for name,m in module.named_modules():
        if name in ('vocabulary',) or (name.startswith(('encoder.blocks.','decoder.blocks.')) and name.count('.')==2):
            hooks.append(m.register_forward_hook(lambda m,args,out,n=name:traces.append((n,out[:1].detach().clone()))))
    with torch.no_grad():
        batch=module(ids,visual);batch_trace=list(traces);traces.clear()
        single=module(ids[:1],visual[:1]);single_trace=list(traces);traces.clear()
        changed_ids=ids.clone();changed_ids[1,250]=1234
        changed_visual=visual.clone();changed_visual[1]+=0.5
        changed=module(changed_ids,changed_visual)
    for hook in hooks:hook.remove()
    per_block=[{'module':name,**comparison(a,b)} for (name,a),(other,b) in zip(batch_trace,single_trace)]
    outputs={name:comparison(getattr(batch,name)[:1],getattr(single,name)) for name in
             ('encoder_hidden','global_language','decoder_hidden','logits')}
    fixed={name:bool(torch.equal(getattr(batch,name)[0],getattr(changed,name)[0])) for name in outputs}
    visual=visual.clone().requires_grad_()
    result=module(ids,visual,return_logits=False)
    result.decoder_hidden[0].square().mean().backward()
    report={'dtype':'float32','threads':4,'atol':1e-6,'rtol':1e-5,'N1_applied':False,
            'outputs':outputs,'per_block':per_block,'fixed_batch_exact_output_equality':fixed,
            'fixed_batch_other_sample_gradient_exact_zero':bool(visual.grad[1].count_nonzero()==0),
            'own_sample_visual_gradient_nonzero':bool(visual.grad[0].norm()>0),
            'interpretation':'Fixed-batch outputs/gradients establish sample separation; cross-batch floating computations differ and amplify through residual depth. No tolerance or architecture was changed.'}
    (ROOT/'artifacts/phase2c2/batch_numerics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
