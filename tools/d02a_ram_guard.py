"""External RAM gate for an already-running, frozen sequential coordinator.

Only the coordinator is suspended between launches; numerical workers are never
suspended or changed. This changes host scheduling, not the scientific driver.
An unexpected state leaves the coordinator suspended for explicit inspection.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_d02a'
MIN_RAM=3*2**30


def gate(read_available, sleep, record):
    while True:
        available=read_available()
        if available>=MIN_RAM:
            record('RAM_GATE_PASS',available_RAM=available,minimum_RAM=MIN_RAM)
            return available
        record('RAM_GATE_WAIT',available_RAM=available,minimum_RAM=MIN_RAM)
        sleep(5)


def worker_index(command):
    assert 'worker' in command and '--index' in command
    return int(command[command.index('--index')+1])


def main():
    handoff=json.loads((OUT/'resource_guard_handoff.json').read_text())
    coordinator=psutil.Process(handoff['coordinator_pid'])
    assert coordinator.create_time()==handoff['coordinator_create_time']
    assert coordinator.cmdline()[-4:]==['--start','122','--stop','414']
    protocol=json.loads((OUT/'pilot_protocol.json').read_text())
    worker=psutil.Process(handoff['active_worker_pid']) if psutil.pid_exists(handoff['active_worker_pid']) else None
    index=worker_index(handoff['active_worker_command'])
    held=True  # handoff already suspended it exactly once
    destination=OUT/'resource_guard'
    destination.mkdir(exist_ok=False)
    with (destination/'events.jsonl').open('x',encoding='utf-8') as handle:
        def record(event,**values):
            handle.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'event':event,**values},allow_nan=False)+'\n');handle.flush()
        record('ATTACHED',coordinator_pid=coordinator.pid,existing_worker_index=index,
               policy='3 GiB available before releasing coordinator for next worker; current worker unchanged')
        try:
            while True:
                assert held
                record('COORDINATOR_HELD',previous_worker_index=index)
                while worker is not None and worker.is_running():
                    time.sleep(.25)
                summary_path=OUT/'jobs'/protocol['jobs'][index]['id']/'summary.json'
                if not summary_path.exists():
                    raise RuntimeError('Worker ended without summary; do not infer cause or release next launch')
                summary=json.loads(summary_path.read_text())
                record('WORKER_ENDED',index=index,status=summary['status'],failed_comparisons=summary['failed_comparisons'])
                if summary['status']=='HARD_STOP':
                    record('RELEASE_FOR_RECORDED_HARD_STOP_EXIT',index=index)
                    coordinator.resume()  # frozen coordinator records exit and raises before another launch
                    held=False
                    return
                assert summary['status']=='COMPLETED_LOCAL_DIAGNOSTIC'
                assert summary['optimizer_steps']==0
                expected=next((i for i in range(index+1,414) if not (OUT/'jobs'/protocol['jobs'][i]['id']/'summary.json').exists()),None)
                if expected is not None:
                    gate(lambda:psutil.virtual_memory().available,time.sleep,record)
                record('COORDINATOR_RELEASED',expected_next_index=expected)
                coordinator.resume()
                held=False
                while coordinator.is_running():
                    children=coordinator.children()
                    if children:
                        assert len(children)==1, 'Multiple workers under coordinator'
                        candidate=children[0]
                        try:
                            command=candidate.cmdline()
                        except psutil.NoSuchProcess:
                            continue
                        # Wait until Python command line is available immediately after spawn.
                        if not command:
                            time.sleep(.02);continue
                        index=worker_index(command)
                        assert index==expected, (index,expected)
                        worker=candidate
                        coordinator.suspend()
                        held=True
                        record('NEXT_WORKER_OBSERVED',index=index,pid=worker.pid)
                        break
                    time.sleep(.02)
                else:
                    assert expected is None, 'Coordinator disappeared before expected next worker'
                    record('REGISTERED_SCHEDULE_FINISHED')
                    return
        except BaseException as exc:
            if coordinator.is_running() and not held:
                coordinator.suspend()
            record('GUARD_STOP_REQUIRES_INSPECTION',error=repr(exc),coordinator_left_suspended=True)
            raise


if __name__=='__main__':main()
