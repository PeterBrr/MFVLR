"""New evidence only; subprocess CPU runs have raw logs and exit codes."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2'


def command(name,cmd):
    logs=OUT/'logs';logs.mkdir(parents=True,exist_ok=True)
    assert not (logs/f'{name}.execution.json').exists(),'choose a new run name'
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)])
    started=time.perf_counter()
    with (logs/f'{name}.stdout.log').open('w',encoding='utf-8') as out,(logs/f'{name}.stderr.log').open('w',encoding='utf-8') as err:
        result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=out,stderr=err)
    record={'command':cmd,'exit_code':result.returncode,'elapsed_seconds':time.perf_counter()-started,'cwd':str(ROOT),'PYTHONPATH':env['PYTHONPATH']}
    (logs/f'{name}.execution.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    return result.returncode


def worker(name,run_label=None):
    sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)]
    import threading
    import traceback
    import psutil
    import torch
    from mfvlr.integration import MFVLRIntegration,ownership_report
    from tools.phase2d2_fixtures import fixture
    from tools.phase2d2_diagnostics import run_g2,write
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    label=run_label or name
    if not all(c.isalnum() or c=='_' for c in label):raise ValueError('invalid evidence label')
    dest=OUT/'g2'/label;dest.mkdir(parents=True,exist_ok=False)
    process=psutil.Process();baseline=process.memory_info().rss;samples=[];stop=threading.Event()
    def sample():
        while not stop.is_set():samples.append(process.memory_info().rss);stop.wait(.005)
    monitor=threading.Thread(target=sample,daemon=True);monitor.start();started=time.perf_counter()
    try:
        root=MFVLRIntegration();write(dest/'ownership.json',ownership_report(root))
        result=run_g2(root,fixture(name),dest)
        code=0 if result['status']=='PASS' else 1
    except Exception as error:
        result={'status':'FAIL','exception':repr(error),'traceback':traceback.format_exc()};write(dest/'failure.json',result);code=1
    finally:
        stop.set();monitor.join();samples.append(process.memory_info().rss)
        write(dest/'resources.json',{'status':result['status'],'elapsed_seconds':time.perf_counter()-started,
            'rss_baseline_bytes':baseline,'sampled_peak_rss_bytes':max(samples),'samples':len(samples),'nominal_sample_interval_seconds':.005,
            'available_RAM_start_or_end_bytes':psutil.virtual_memory().available,'torch':torch.__version__,
            'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'cuda_available':torch.cuda.is_available(),
            'precision':'full FP32; no optimizer; FP64 statistics/streamed gradient accumulator only'})
    print(json.dumps(result),flush=True);return code


def pytest_run(name,paths):
    junit=OUT/'junit'/f'{name}.xml';junit.parent.mkdir(parents=True,exist_ok=True)
    historical={p:p.read_bytes() for p in (ROOT/'artifacts/phase2c2/gradients').glob('*.json')}
    try:
        return command(name,[sys.executable,'-m','pytest',*paths,'-vv','-s',f'--junitxml={junit}'])
    finally:
        for p,data in historical.items():
            history=OUT/'history/gradients'/p.name;history.parent.mkdir(parents=True,exist_ok=True)
            if not history.exists():history.write_bytes(data)
            fresh=OUT/'regression_gradients'/name/p.name;fresh.parent.mkdir(parents=True,exist_ok=True)
            fresh.write_bytes(p.read_bytes());p.write_bytes(data)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--g2');parser.add_argument('--run-label');parser.add_argument('--pytest',choices=('joint','readiness','regression','complete'));args=parser.parse_args()
    if args.worker:raise SystemExit(worker(args.worker,args.run_label))
    if args.g2:raise SystemExit(command('g2_'+(args.run_label or args.g2),[sys.executable,str(Path(__file__)),'--worker',args.g2,*(['--run-label',args.run_label] if args.run_label else [])]))
    if args.pytest:
        paths={'joint':['tests/joint/test_joint.py'],'readiness':['tests/joint/test_readiness.py'],
               'regression':['tests','--ignore=tests/joint'],'complete':['tests']}[args.pytest]
        raise SystemExit(pytest_run(args.pytest+'_01',paths))
    parser.error('choose --g2 case')
