"""Own only the seven remaining registered workers, with a per-worker RAM gate."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_d02a'
DEST=OUT/'continuation_0407'


def write(p,x):
    with p.open('x',encoding='utf-8') as h:json.dump(x,h,indent=2,allow_nan=False)


def run(label,args):
    cmd=[sys.executable,*args];start=time.perf_counter()
    with (DEST/f'{label}.stdout.log').open('x') as a,(DEST/f'{label}.stderr.log').open('x') as b:
        result=subprocess.run(cmd,cwd=ROOT,stdout=a,stderr=b)
    write(DEST/f'{label}.execution.json',{'command':cmd,'exit_code':result.returncode,'seconds':time.perf_counter()-start})
    if result.returncode:raise RuntimeError(f'{label} exit {result.returncode}; no automatic retry')


def main():
    for p in psutil.process_iter(['pid','cmdline']):
        if p.pid==psutil.Process().pid:continue
        if any(str(a).replace('\\','/').endswith(('tools/d02a/run.py','tools/continue_d02a_407.py','tools/continue_d02a_registered.py','tools/d02a_ram_guard_v2.py')) for a in p.info['cmdline'] or []):
            raise RuntimeError('Existing numerical coordinator/worker/guard; refuse duplicate ownership')
    assert json.loads((OUT/'history/poweroff_0407_20261013/archive_verification.json').read_text())['files']==42
    jobs=json.loads((OUT/'pilot_protocol.json').read_text())['jobs']
    assert len(list((OUT/'jobs').glob('*/summary.json')))==407
    assert all(not (OUT/'jobs'/jobs[i]['id']).exists() for i in range(407,414))
    DEST.mkdir(exist_ok=False)
    write(DEST/'scope.json',{'pid':psutil.Process().pid,'indices':list(range(407,414)),
        'minimum_available_RAM_bytes':3*2**30,'one_worker_at_a_time':True,'numeric_driver_changed':False,
        'replay_gate_before_408':'All1946 original scientific rows and426 FAILs must match; timing fields excluded only',
        'authorization':'2026-10-14 A1-D02A recovery only; no new experiment scope'})
    with (DEST/'events.jsonl').open('x',encoding='utf-8') as handle:
        def event(kind,**values):
            handle.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'event':kind,**values})+'\n');handle.flush()
        for index in range(407,414):
            while True:
                available=psutil.virtual_memory().available
                event('RAM_GATE_PASS' if available>=3*2**30 else 'RAM_GATE_WAIT',index=index,available_RAM=available,minimum_RAM=3*2**30)
                if available>=3*2**30:break
                time.sleep(5)
            run(f'job_{index}',['tools/d02a/run.py','run','--start',str(index),'--stop',str(index+1)])
            summary=json.loads((OUT/'jobs'/jobs[index]['id']/'summary.json').read_text())
            assert summary['status']=='COMPLETED_LOCAL_DIAGNOSTIC' and summary['optimizer_steps']==0
            event('WORKER_COMPLETED',index=index,numerical_FAIL_comparisons=summary['failed_comparisons'])
            if index==407:
                run('replay_prefix',['tools/compare_d02a_replay407.py'])
                assert json.loads((OUT/'history/poweroff_0407_20261013/replay_comparison.json').read_text())['status']=='PASS'
                event('REPLAY_PREFIX_PASS',records=1946,original_FAILs=426)
        event('REGISTERED_SCHEDULE_FINISHED',canonical_jobs=414)
    write(DEST/'workers_complete.json',{'status':'COMPLETE_LOCAL_EXECUTION_NOT_NUMERICAL_ACCEPTANCE','phase2d2':'BLOCKED'})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if DEST.is_dir():write(DEST/'stopped.json',{'error':repr(exc),'automatic_retry':False})
        raise
