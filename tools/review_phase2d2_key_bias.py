"""Read-only cancellation probe for ordinary attention key-projection bias.

Mathematically q_i^T b_k adds the same scalar to each valid key in a row;
softmax makes the exact-real derivative with respect to b_k zero. FP32
cancellation residuals here are recorded, not excused by a new threshold.
"""
from pathlib import Path
import copy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)]


def main():
    import torch
    from mfvlr.language_encoder import LanguageAttention,initialize_attention
    from tools.phase2d2_diagnostics import stats,write,FP32,FP64,state_hash
    from tools.review_phase2c2_n2 import compare,backend,tensor_hash
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    source=LanguageAttention()
    initialize_attention(source,torch.Generator().manual_seed(0))
    source_hash=state_hash(source)
    query=torch.randn(1,308,512,generator=torch.Generator().manual_seed(247))*0.2
    direction=torch.randn(1,308,512,generator=torch.Generator().manual_seed(319))
    rows=[]
    for dtype in (torch.float32,torch.float64):
        for scale in (1.,10000.):
            module=copy.deepcopy(source)
            assert state_hash(module)==source_hash
            module.to(dtype=dtype)
            output,trace=module.forward_with_trace(query.to(dtype))
            probe=(output*direction.to(dtype)*scale).sum()
            bias_gradient,logit_gradient=torch.autograd.grad(probe,(module.k.bias,trace['logits']))
            expected=torch.zeros_like(bias_gradient)
            rows.append({'dtype':str(dtype),'cotangent_scale':scale,
                'key_bias_gradient':stats(bias_gradient),'logit_gradient_row_sum':stats(logit_gradient.sum(-1)),
                'comparison_to_exact_real_zero':compare(bias_gradient,expected,**(FP32 if dtype==torch.float32 else FP64)),
                'probe_scalar':probe.item(),'parameter_grad_fields_unpopulated':all(p.grad is None for p in module.parameters())})
    dest=ROOT/'artifacts/phase2d2_retry/key_bias_review.json'
    assert not dest.exists()
    write(dest,{'status':'DIAGNOSTIC_COMPLETE_NOT_AN_ACCEPTANCE_WAIVER','backend':backend(),
        'scope':'Single accepted full-width ordinary attention block, length308; not a full-model loss or VIM',
        'derivation':'softmax(z+c*ones)=softmax(z), so key-projection bias has exact-real zero task derivative',
        'FP32_initialized_source_sha256':source_hash,'query_FP32_sha256':tensor_hash(query),
        'cotangent_FP32_sha256':tensor_hash(direction),'deep_copied_same_FP32_weights':True,
        'records':rows,'optimizer_updates':0,'production_changes':False,'threshold_changes':False})
    print([(r['dtype'],r['cotangent_scale'],r['key_bias_gradient']['max_abs'],r['comparison_to_exact_real_zero']['allclose']) for r in rows])


if __name__=='__main__':main()
