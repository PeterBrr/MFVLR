"""Fresh evidence + unchanged complete suite, never an acceptance amendment."""
from pathlib import Path
import argparse,json,os,shutil,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)]


def worker(out,case,mode):
    import torch
    from tools import run_phase2d2 as runner,phase2d2_diagnostics as diagnostic
    from tools.n31_full_gradients import CompleteGradientStream
    torch.backends.mkldnn.enabled=mode=='mkldnn'
    diagnostic.GradientStream=CompleteGradientStream # Diagnostic class only; original file unchanged.
    runner.OUT=out
    return runner.worker(case,'final_'+case+('_native' if mode=='native' else ''))


def command(out,name,args):
    from tools import run_phase2d2 as runner
    from tools.n31_evidence import sources,write
    runner.OUT=out
    receipt=out/'job_sources';receipt.mkdir(exist_ok=True)
    write(receipt/(name+'.json'),sources())
    return runner.command(name,args)


def all_runs(out):
    from tools.n31_evidence import begin,read,write,seal,negative_tests,validate
    from tools.n31_full_gradients import selftest
    from tools import run_phase2d2 as runner
    import psutil
    runner.OUT=out
    begin(out)
    print('Waiting for separately launched fresh N3 matrix; no concurrent full models',flush=True)
    expected=out/'logs/same_real_efs_native.execution.json'
    while not expected.exists():time.sleep(2)
    assert read(expected)['exit_code']==0
    write(out/'stream_selftest.json',selftest())
    for name in ('attention','LD','visual_mkldnn','visual_native'):
        code=command(out,'heldout_'+name,[sys.executable,str(ROOT/'tools/review_phase2d2_n31_heldout.py'),'--worker',name,'--output',str(out)])
        if code:raise RuntimeError('Held-out worker failed: preserve evidence and diagnose before further work')
    for case,mode in [(c,'mkldnn') for c in ('real','efs','local','real_efs','real_local','local_duplicate')]+[('real_efs','native')]:
        name=case+('_native' if mode=='native' else '')
        code=command(out,'full_G2_'+name,[sys.executable,str(Path(__file__)),'--worker',case,'--mode',mode,'--output',str(out)])
        r=read(out/'g2'/('final_'+name)/'summary.json')
        assert code in (0,1) and r['status'] in ('PASS','FAIL_GRADIENT_ADDITIVITY') and r['all_valid_values_and_gradients_finite']
        assert (out/'g2'/('final_'+name)/'complete_discrepancies.json').exists()
    shutil.copytree(out/'g2/final_real',out/'g2/real')
    for file in ('baseline_manifest.json','resource_availability.json'):
        shutil.copyfile(ROOT/'artifacts/phase2d2'/file,out/file)
    resources=read(out/'resource_availability.json')
    resources['CPU_B8']['available_RAM_bytes']=psutil.virtual_memory().available
    resources['CPU_B8']['reason']='N3.1 bounded CPU B1/B2 only; B8 capacity not established; no substituted smaller architecture.'
    write(out/'resource_availability.json',resources)
    for index in range(3):
        for dtype in ('float32','float64'):
            name=f'fixture{index}_{dtype}_threads4';path=out/'n2'/f'{name}.json'
            code=command(out,'fresh_n2_'+name,[sys.executable,str(ROOT/'tools/review_phase2c2_n2.py'),'--worker',str(index),'--dtype',dtype,'--threads','4','--output',str(path)])
            if code:raise RuntimeError('N2 fresh evidence failed')
    seal(out);negative_tests(out)
    os.environ['MFVLR_N3_REVIEW_ROOT']=str(ROOT);os.environ['MFVLR_N3_EVIDENCE_ROOT']=str(out)
    os.environ['MFVLR_N31_RUN_ID']=read(out/'workflow_start.json')['run_id']
    os.environ['MFVLR_PHASE2D2_EVIDENCE_ROOT']=str(out)
    code=runner.pytest_run('supplemental_01',['tests/joint/test_n3_controls.py','-p','tools.n31_evidence','-p','no:cacheprovider'])
    if code:raise RuntimeError('Supplemental tests failed')
    validate(out,os.environ['MFVLR_N31_RUN_ID'])
    code=runner.pytest_run('complete_01',['tests','-p','tools.n31_evidence','-p','no:cacheprovider'])
    write(out/'suite_exit.json',{'complete_exit_code':code,'supplemental_exit_code':0,'acceptance':'BLOCKED',
        'no_existing_assertions_removed':True,'no_threshold_changes':True})
    print('Final suite executed; inspect actual counts; Phase2D2 BLOCKED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'artifacts/phase2d2_n31');parser.add_argument('--worker');parser.add_argument('--mode',default='mkldnn');args=parser.parse_args()
    if args.worker:raise SystemExit(worker(args.output.resolve(),args.worker,args.mode))
    all_runs(args.output.resolve())
