"""Fail-closed orchestration, with synthetic dependency injection for tests."""
import json
import os
import shutil
import threading
import time
from pathlib import Path
import psutil

CELL = 'D02B-G01-A0-SAME-FORWARD'
GIB = 1024**3
STAGES = ('before_single_forward','before_traversals','after_traversals')

class ResourceWait(RuntimeError): pass

def write_new(path, record):
    data = json.dumps(record,sort_keys=True,allow_nan=False).encode()+b'\n'
    with Path(path).open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())

def resources(path):
    p = psutil.Process()
    return dict(time_ns=time.time_ns(),pid=p.pid,created=p.create_time(),rss=p.memory_info().rss,
                available_ram=psutil.virtual_memory().available,free_disk=shutil.disk_usage(path).free)

def require_resources(row):
    if any(type(row.get(k)) is not int or row[k] < 0 for k in ('rss','available_ram','free_disk','time_ns','pid')):
        raise ValueError('Invalid measured resources')
    if row['available_ram'] < 8*GIB or row['free_disk'] < 8*GIB:
        raise ResourceWait('8 GiB available physical RAM and temporary disk required')

class Lease:
    """Permanent identity-aware claim; no stale-lock deletion or takeover."""
    def __init__(self, root, name, identity=None):
        root = Path(root)
        if not root.is_dir() or type(name) is not str or not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):
            raise ValueError('Existing parent and simple exclusive attempt name required')
        self.identity = identity or (lambda:(os.getpid(),psutil.Process().create_time()))
        pid,created = self.identity()
        self.claim = dict(cell=CELL,pid=pid,created=created,attempt=name)
        self.lock = root/'coordinator.json'
        write_new(self.lock,self.claim)  # across all attempts under this approved root
        self.path = root/name
        self.path.mkdir(exist_ok=False)
        write_new(self.path/'claim.json',self.claim)
    def check(self):
        if tuple(self.identity()) != (self.claim['pid'],self.claim['created']) or json.loads(self.lock.read_text()) != self.claim or json.loads((self.path/'claim.json').read_text()) != self.claim:
            raise RuntimeError('Coordinator identity/claim changed')

class Monitor:
    def __init__(self, path, sample, interval=.5):
        self.path=Path(path);self.sample=sample;self.interval=interval
        self.stop=threading.Event();self.error=None;self.thread=None
        self.samples=0;self.peak_rss=0
    def __enter__(self):
        self.stream=self.path.open('xb')
        self.tick()
        self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start()
        return self
    def tick(self):
        try:
            row=self.sample()
            self.samples+=1;self.peak_rss=max(self.peak_rss,row['rss'])
            self.stream.write(json.dumps(row,allow_nan=False).encode()+b'\n');self.stream.flush()
            require_resources(row)
        except BaseException as error:
            if self.error is None:self.error=error
    def _loop(self):
        while not self.stop.wait(self.interval):
            self.tick()
    def check(self):
        if self.error is not None:raise self.error
    def __exit__(self,*exc):
        self.stop.set()
        if self.thread:self.thread.join()
        self.stream.close()

def execute_real(*args,**kwargs):
    raise PermissionError('D02B10 is static/mock-only; separate real G01 authorization required')

def _orchestrate(root,name,adapter,*,sample=None,identity=None):
    """Internal engine. Public live dispatch is disabled; mocks use small adapters."""
    lease=Lease(root,name,identity);path=lease.path
    start=time.perf_counter();monitor=None
    sample=sample or (lambda:resources(path))
    result=dict(cell=CELL,execution='NOT RUN',structural='NOT RUN',provenance='NOT RUN',
        resource='NOT VERIFIED',numerical='NOT RUN',dependent_work='BLOCKED',B0_D08='DISABLED')
    stage='resources';cursor=0
    try:
        pre=sample();write_new(path/'preconstruction_resources.json',pre);require_resources(pre)
        result['resource']='PASS'
        stage='provenance';lease.check();adapter.verify_static();result['provenance']='PASS'
        with Monitor(path/'resources.jsonl',sample) as monitor:
            monitor.check();lease.check()
            # Construction is never reached on inadequate resources or invalid pins.
            model,fixture=adapter.build()
            initial=adapter.snapshot(model,fixture)
            write_new(path/'initial_seal.json',initial)
            def seal(label):
                nonlocal cursor,stage
                stage='provenance';lease.check();monitor.check();adapter.verify_static()
                if cursor >= len(STAGES) or label != STAGES[cursor]:raise RuntimeError('Extra/missing/out-of-order forward seal')
                row=sample();write_new(path/f'{cursor:02d}_{label}_resources.json',row);require_resources(row)
                now=adapter.snapshot(model,fixture)
                if now != initial:raise RuntimeError('State/input/backend/ownership mutation')
                write_new(path/f'{cursor:02d}_{label}_seal.json',now)
                cursor+=1;stage='structural'
            stage='structural'
            analysis=adapter.run(model,fixture,path/'science',seal,monitor.check)
            monitor.check();lease.check()
            if cursor != 3:raise RuntimeError('Incomplete seal sequence')
            adapter.verify_complete(path/'science',analysis)
            adapter.verify_static()
            if adapter.snapshot(model,fixture) != initial:raise RuntimeError('Final state mutation')
            monitor.check()
            if analysis['status'] not in ('PASS','FAIL'):raise RuntimeError('Missing numerical decision')
            result.update(execution='COMPLETE',structural='PASS',provenance='PASS',resource='PASS',numerical=analysis['status'])
    except BaseException as error:
        result.update(execution='WAIT/NOT RUN' if isinstance(error,ResourceWait) and cursor==0 else 'HARD STOP',
                      error=type(error).__name__,message=str(error),stage=stage)
        if isinstance(error,ResourceWait):result['resource']='INSUFFICIENT'
        elif stage=='provenance':result['provenance']='FAIL'
        else:result['structural']='FAIL'
        # Includes KeyboardInterrupt. Partial files and permanent lease remain.
    result.update(seconds=time.perf_counter()-start,sampled_peak_rss=monitor.peak_rss if monitor else None,
                  resource_samples=monitor.samples if monitor else 0)
    write_new(path/'result.json',result)
    return result

def run_mock(root,name,adapter,**kwargs):
    if getattr(adapter,'scope',None) != 'synthetic':raise PermissionError('Mock adapter required')
    return _orchestrate(root,name,adapter,**kwargs)
