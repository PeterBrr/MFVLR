"""Precision diagnostic only; does not amend the FP32 production test."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src')]
import torch
from mfvlr.flt import FineGrainedLanguageTransformer


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    # The same FP32 initialized parameters/inputs, promoted before arithmetic.
    model=FineGrainedLanguageTransformer().double()
    fixtures=json.loads((ROOT/'tests/foundation/fixtures/golden_tokens.json').read_text())['prompt_cases']
    ids=torch.tensor([fixtures[1]['ids'],fixtures[5]['ids']],dtype=torch.int64)
    visual=(torch.randn(2,1,512,generator=torch.Generator().manual_seed(341))*0.1).double()
    with torch.no_grad():
        batch=model(ids,visual);single=model(ids[:1],visual[:1])
    values={}
    for name in ('encoder_hidden','global_language','decoder_hidden','logits'):
        a=getattr(batch,name)[:1];b=getattr(single,name)
        diff=(a-b).abs()
        values[name]={'max_abs_difference':diff.max().item(),
            'finite':bool(a.isfinite().all() and b.isfinite().all()),
            'outside_fp64_tolerance':int((diff>1e-10+1e-8*b.abs()).sum()),
            'allclose_fp64':bool(torch.allclose(a,b,atol=1e-10,rtol=1e-8))}
    result={'precision_diagnostic_only':True,'production_dtype_changed':False,
            'FP32_test_or_tolerance_changed':False,'fp64_atol':1e-10,'fp64_rtol':1e-8,
            'same_initialized_weights_and_inputs_promoted':True,'outputs':values}
    (ROOT/'artifacts/phase2c2/batch_numerics_fp64.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
