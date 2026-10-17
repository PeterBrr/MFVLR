"""Fresh Phase 2D.2 evidence, unchanged numerical gates, no updates.

Run: python tools/retry_phase2d2.py --all
Original artifacts are never used as output paths. Only the two unaccepted
joint test modules accept an evidence-directory override; assertions unchanged.
"""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import json
import os
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_retry'
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)]


def read(path):return json.loads(path.read_text())


def chain(dtype):
    import torch
    from mfvlr.integration import MFVLRIntegration
    from tools.phase2d2_fixtures import fixture
    from tools.phase2d2_diagnostics import FP32,FP64,state_hash,stats,write
    from tools.review_phase2c2_n2 import compare,backend,tensor_hash
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    root=MFVLRIntegration();f=fixture('real_efs')
    with torch.no_grad():iv=root.visual_only(f.image).visual.fused_global.detach().clone()
    source=root.flt;cmc_source=root.objectives.cmc
    source_hash=state_hash(source);iv_hash=tensor_hash(iv)
    # Deep-copy the actual FP32 initialized modules; no FP64 initialization.
    model=copy.deepcopy(source);cmc=copy.deepcopy(cmc_source)
    assert state_hash(model)==source_hash
    del root,source,cmc_source;gc.collect()
    if dtype=='float64':model.double();cmc.double();iv=iv.double()
    v=iv.requires_grad_();text=model.encode(f.targets.token_ids,v).global_language
    parameters=(v,model.encoder.blocks[0].attention.q.weight,
                model.encoder.blocks[-1].ff.fc1.weight)
    names=('fused_visual','LE1_attention_Q','LE12_FFN1')
    direct_loss=cmc(v,text.detach());via_loss=cmc(v.detach(),text);both_loss=cmc(v,text)
    direct=torch.autograd.grad(direct_loss,parameters,allow_unused=True,retain_graph=True)
    via=torch.autograd.grad(via_loss,parameters,allow_unused=True,retain_graph=True)
    both=torch.autograd.grad(both_loss,parameters,allow_unused=True)
    tolerance=FP64 if dtype=='float64' else FP32
    reports={}
    for name,a,d,l in zip(names,both,direct,via):
        assert a is not None and l is not None
        reference=l if d is None else d+l
        reports[name]=compare(a,reference,**tolerance)
        reports[name]['joint_gradient']=stats(a)
        reports[name]['direct_gradient']=None if d is None else stats(d)
        reports[name]['language_gradient']=stats(l)
    result={'status':'PASS' if all(r['allclose'] for r in reports.values()) else 'FAIL_COMPARISON',
        'scope':'CMC chain rule through exact full 12-block LE; LD and visual Jacobian not exercised',
        'dtype':dtype,'backend':backend(),'fixture':f.provenance,
        'FP32_initialized_FLT_source_sha256':source_hash,'FP32_visual_input_sha256':iv_hash,
        'deep_copied_FP32_weights':True,'gradient_comparisons':reports,
        'global_language':stats(text),'CMC_scalar':both_loss.item(),
        'no_optimizer_updates':True,'production_modules_changed':False,
        'acceptance_gate_exempted':False,'all_parameter_grad_fields_unpopulated':all(p.grad is None for p in model.parameters())}
    dest=OUT/'chain';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/f'{dtype}.json').exists()
    write(dest/f'{dtype}.json',result)
    print(json.dumps({'dtype':dtype,'status':result['status'],'comparisons':{k:v['allclose'] for k,v in reports.items()}}),flush=True)
    return 0 if result['status']=='PASS' else 1


def all_runs():
    import psutil
    from tools import run_phase2d2 as runner
    from tools.phase2d2_diagnostics import write
    runner.OUT=OUT
    OUT.mkdir(parents=True,exist_ok=True)
    os.environ['MFVLR_PHASE2D2_EVIDENCE_ROOT']=str(OUT)
    write(OUT/'retry_scope.json',{'architecture_changes':False,'threshold_changes':False,
        'N2_exemption_for_new_losses':False,'optimizer_updates':0,
        'new_test_harness_change':'Evidence-directory override only; all assertions unchanged',
        'previous_276_tests_unchanged':True,'original_evidence_preserved':True})
    for file in ('baseline_manifest.json','resource_availability.json'):
        shutil.copyfile(ROOT/'artifacts/phase2d2'/file,OUT/file)
    resources=read(OUT/'resource_availability.json')
    resources['CPU_B8']['available_RAM_bytes']=psutil.virtual_memory().available
    resources['CPU_B8']['reason']='NOT RUN: safe B8 headroom unestablished from multi-GiB B1/B2 working sets; G2 strict comparison remains blocked. No substituted smaller graph.'
    write(OUT/'resource_availability.json',resources)
    for dtype in ('float32','float64'):
        code=runner.command('chain_'+dtype,[sys.executable,str(Path(__file__)),'--chain',dtype])
        result=read(OUT/'chain'/f'{dtype}.json')
        if result['status'] not in ('PASS','FAIL_COMPARISON'):raise RuntimeError('unexpected numerical failure')
    results=[]
    for case in ('real','efs','local','real_efs','real_local','local_duplicate'):
        code=runner.command('g2_final_'+case,[sys.executable,str(Path(__file__)),'--worker',case])
        summary=read(OUT/'g2'/('final_'+case)/'summary.json')
        assert summary['all_valid_values_and_gradients_finite']
        assert summary['graph_and_hook_lifecycle']=='PASS'
        assert summary['status'] in ('PASS','FAIL_GRADIENT_ADDITIVITY')
        results.append({'case':case,'exit_code':code,'status':summary['status']})
        write(OUT/'g2/index.json',{'acceptance_gate':'BLOCKED' if any(r['exit_code'] for r in results) else 'AWAITING_SUITE','runs':results})
    # I10 reads the new real fixture, not historical failure evidence.
    shutil.copytree(OUT/'g2/final_real',OUT/'g2/real')
    dest=OUT/'n2';dest.mkdir()
    for index in range(3):
        for dtype in ('float32','float64'):
            name=f'fixture{index}_{dtype}_threads4'
            code=runner.command('n2_'+name,[sys.executable,str(ROOT/'tools/review_phase2c2_n2.py'),
                '--worker',str(index),'--dtype',dtype,'--threads','4','--output',str(dest/f'{name}.json')])
            if code:raise RuntimeError('N2 diagnostic failed; halt')
    code=runner.pytest_run('complete_retry_01',['tests'])
    print(json.dumps({'complete_suite_exit_code':code,'G2_runs':results}),flush=True)
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true')
    parser.add_argument('--worker');parser.add_argument('--chain',choices=('float32','float64'));args=parser.parse_args()
    if args.worker:
        from tools import run_phase2d2 as runner
        runner.OUT=OUT
        raise SystemExit(runner.worker(args.worker,'final_'+args.worker))
    if args.chain:raise SystemExit(chain(args.chain))
    if args.all:raise SystemExit(all_runs())
    parser.error('choose --all, --worker or --chain')
