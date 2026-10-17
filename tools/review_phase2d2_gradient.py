"""Diagnose the real-fixture FP32 sum failure; no policy/model amendment."""
from pathlib import Path
import gc
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)]


def main():
    import tempfile
    import torch
    from mfvlr.integration import MFVLRIntegration
    from tools.phase2d2_fixtures import fixture
    from tools.phase2d2_diagnostics import GradientStream,loss_only,ORDER,FP32,write
    from tools.review_phase2c2_n2 import compare
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    out=ROOT/'artifacts/phase2d2/gradient_review';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'review.json').exists(),'preserve earlier diagnostic'
    # Verify file offsets and multi-chunk addition independently of all losses.
    toy=torch.nn.Linear(1024,512,bias=True)
    stream=GradientStream(toy,out)
    try:
        for i in range(1,7):
            for p in toy.parameters():p.grad=torch.full_like(p,float(i))
            stream.add(str(i))
        for p in toy.parameters():p.grad=torch.full_like(p,21.)
        checks=stream.compare_joint();assert checks['allclose']
    finally:stream.close()
    del toy,stream;gc.collect()
    root=MFVLRIntegration();f=fixture('real')
    o=root.with_language(f.image,f.targets.token_ids,sample_ids=f.targets.sample_ids)
    endpoints={'fd_logits':o.predictions.detection_logits,'lr_logits':o.language.logits,
               'mask_logits':o.predictions.decoded.mask_logits,'adapted':o.predicted_language,
               'Iv':o.predictions.visual.fused_global,'Ipre':o.predictions.decoded.reconstruction}
    sums={};scalars={}
    for name in ORDER:
        loss=loss_only(root,o,f.targets,name);scalars[name]=loss.item()
        grads=torch.autograd.grad(loss,tuple(endpoints.values()),allow_unused=True,retain_graph=True)
        for (key,tensor),g in zip(endpoints.items(),grads):
            if g is not None:
                if key not in sums:sums[key]=g.detach().double()
                else:sums[key]+=g.detach().double()
        del loss,grads
    total=root.compute_objectives(o,f.targets).total
    actual=torch.autograd.grad(total,tuple(endpoints.values()),allow_unused=True)
    reports={}
    for (key,tensor),g in zip(endpoints.items(),actual):
        reports[key]=compare(g,sums[key].float(),**FP32)
    assert all(reports[k]['allclose'] for k in ('fd_logits','lr_logits','mask_logits','adapted'))
    original=ROOT/'artifacts/phase2d2/g2/real'
    records={name:json.loads((original/f'{name}.json').read_text()) for name in (*ORDER,'total')}
    same_forwards=all(records[n]['activations']==records['total']['activations'] and records[n]['outputs']==records['total']['outputs'] for n in ORDER)
    same_losses=all(records[n]['scalars'][n]==records['total']['scalars'][n] for n in ORDER)
    assert same_forwards and same_losses
    write(out/'review.json',{'status':'DIAGNOSED_POLICY_FAILURE_NOT_ACCEPTED',
        'stream_multi_chunk_offset_and_sum_control':'PASS','stream_control':checks,
        'same_graph_direct_prediction_derivatives_match':'PASS','endpoint_comparisons':reports,
        'all_seven_deterministic_forward_statistic_records_match_exactly':same_forwards,
        'isolated_and_joint_scalar_values_match_exactly':same_losses,
        'same_graph_has_one_retained_graph_only':True,'parameter_grads_not_populated':all(p.grad is None for p in root.parameters()),
        'inference':'Evidence supports FP32 backward accumulation/rounding amplification in shared upstream Jacobians; does not establish harmlessness or training stability.',
        'full_parameter_mandatory_allclose_still_failed':True,'tolerances_unchanged':True,'architecture_unchanged':True,
        'N2_exemption_applied':False,'automatic_amendment_proposed_or_applied':False})
    print(json.dumps({'stream':'PASS','direct_prediction_derivatives':'PASS','upstream_comparisons':{k:v['allclose'] for k,v in reports.items()},'acceptance':'BLOCKED'}))


if __name__=='__main__':main()
