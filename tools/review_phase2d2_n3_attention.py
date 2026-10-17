"""Actual LE8 attention cut: frozen cotangents, same local Jacobian, no updates."""
from pathlib import Path
import argparse
import copy
import gc
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)]


def main():
    import torch
    import psutil
    from mfvlr.integration import MFVLRIntegration
    from tools.phase2d2_fixtures import fixture
    from tools.phase2d2_diagnostics import ORDER,FP32,FP64,stats,state_hash,write
    from tools.n3_gradient_tools import identity,sha_tensor
    from tools.review_phase2c2_n2 import compare,backend
    from tools.review_phase2d2_n3_local import explicit_gradients
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'artifacts/phase2d2_n3/actual_LE08.json');args=parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    start=time.perf_counter();root=MFVLRIntegration();initial=state_hash(root);f=fixture('real')
    attention=root.flt.encoder.blocks[7].attention;captured={}
    def hook(module,arguments,output):captured.update(input=arguments[0],output=output)
    handle=attention.register_forward_hook(hook)
    try:o=root.with_language(f.image,f.targets.token_ids,sample_ids=f.targets.sample_ids)
    finally:handle.remove()
    losses=root.compute_objectives(o,f.targets);components=losses.as_dict()
    anchors={n:t.grad_fn for n,t in components.items()}
    nodes={'input':identity(captured['input']),'output':identity(captured['output']),
           'losses':{n:{**identity(t),'held_grad_fn_python_id':id(anchors[n])} for n,t in components.items()},
           'parameters':{n:identity(p) for n,p in attention.named_parameters()}}
    names=('q.weight','k.weight','k.bias');parameters=dict(attention.named_parameters())
    targets=(captured['output'],)+(tuple(parameters[n] for n in names))
    cotangents={};full_gradients={}
    for i,name in enumerate(('total',*ORDER)):
        gs=torch.autograd.grad(components[name],targets,retain_graph=i<len(ORDER),allow_unused=True)
        cotangents[name]=None if gs[0] is None else gs[0].detach().clone()
        full_gradients[name]={n:None if g is None else g.detach().clone() for n,g in zip(names,gs[1:])}
        assert all(g is None or bool(g.isfinite().all()) for g in gs)
        del gs
    x32=captured['input'].detach().clone();y32=captured['output'].detach().clone()
    source=copy.deepcopy(attention);source_hash=state_hash(source)
    assert state_hash(root)==initial and all(p.grad is None for p in root.parameters())
    del root,o,losses,components,anchors,captured,targets,parameters,attention;gc.collect()
    # No full visual/language graph remains. Promotion is local attention only.
    active=[cotangents[n] for n in ORDER if cotangents[n] is not None]
    summed64=sum(t.double() for t in active)
    left32=sum(active);right32=sum(reversed(active))
    records={'cotangent_presence':{n:t is not None for n,t in cotangents.items()},
             'cotangent_stats':{n:stats(t) for n,t in cotangents.items() if t is not None},
             'joint_vs_isolated_sum':compare(cotangents['total'],summed64.float(),**FP32),
             'left32_vs_FP64_sum':compare(left32,summed64.float(),**FP32),
             'reverse32_vs_FP64_sum':compare(right32,summed64.float(),**FP32),
             'left32_vs_reverse32':compare(left32,right32,**FP32)}
    precision=[]
    for dtype in (torch.float32,torch.float64):
        model=copy.deepcopy(source);assert state_hash(model)==source_hash
        model.to(dtype=dtype);x=x32.to(dtype).clone().requires_grad_();y=model(x)
        tolerance=FP64 if dtype==torch.float64 else FP32
        pp=dict(model.named_parameters());target=tuple(pp[n] for n in names)
        local={}
        upstream={'joint':cotangents['total'].to(dtype),'summed':summed64.to(dtype),
                  **{n:t.to(dtype) for n,t in cotangents.items() if n!='total' and t is not None}}
        for name,cotangent in upstream.items():
            gs=torch.autograd.grad(y,target,grad_outputs=cotangent,retain_graph=True)
            local[name]={n:g.detach().clone() for n,g in zip(names,gs)}
            assert all(bool(g.isfinite().all()) for g in gs)
        comparison={}
        with torch.no_grad():
            manual=explicit_gradients(x.detach(),{n:p.detach().clone() for n,p in pp.items()},upstream['joint'])
        for name in names:
            isolated=sum(local[n][name].double() for n in ORDER if n in local)
            reference_full=sum(full_gradients[n][name].double() for n in ORDER if full_gradients[n][name] is not None)
            comparison[name]={
                'local_joint_vs_full_joint':compare(local['joint'][name],full_gradients['total'][name].to(dtype),**tolerance),
                'local_sum_vs_sum_local_components':compare(local['summed'][name],isolated.to(dtype),**tolerance),
                'joint_cotangent_vs_sum_cotangent_VJP':compare(local['joint'][name],local['summed'][name],**tolerance),
                'full_joint_vs_sum_full_components':compare(full_gradients['total'][name],reference_full.float(),**FP32),
                'independent_manual_vs_autograd':compare(local['joint'][name],manual[name],**tolerance),
                'local_joint_stats':stats(local['joint'][name]),'manual_stats':stats(manual[name])}
        precision.append({'dtype':str(dtype),'same_FP32_source_before_promotion':True,
            'local_forward_vs_captured_FP32':compare(y.detach(),y32.to(dtype),**tolerance),
            'parameters':comparison,'all_values_finite':True})
        del model,x,y,pp,target,local,upstream,manual,gs;gc.collect()
    write(args.output,{'status':'DIAGNOSTIC_COMPLETE_PHASE_BLOCKED','scope':'actual real fixture LE block index 7 ordinary attention; not full-model FP64',
        'backend':backend(),'fixture':f.provenance,'source_root_sha256':initial,'source_attention_sha256':source_hash,
        'graph_identities':nodes,'input_sha256':sha_tensor(x32),'cotangent_sha256':{n:sha_tensor(t) for n,t in cotangents.items() if t is not None},
        'cotangent_comparisons':records,'precision_records':precision,'source_parameters_unchanged':True,
        'full_graph_released_before_local_promotion':True,'zero_optimizer_updates':True,
        'seconds':time.perf_counter()-start,'Windows_peak_working_set_bytes':getattr(psutil.Process().memory_info(),'peak_wset',None),
        'note':'Cross-precision and actual-scale local derivative comparisons are reported, not silently given new tolerances. Independent unit-scale FP64 oracle remains a separate mandatory executed control.'})
    print('Actual LE08 local VJP and FP64 controls recorded; no acceptance amendment',flush=True)


if __name__=='__main__':main()
