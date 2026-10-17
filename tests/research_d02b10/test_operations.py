import json
import time
from pathlib import Path
import pytest
from mfvlr_research.d02b10.operations import (Lease,Monitor,ResourceWait,require_resources,run_mock,execute_real,GIB)
from mfvlr_research.d02b10.live import execute,FrozenAdapter

def sample():return dict(time_ns=1,pid=123,created=1.,rss=1024,available_ram=10*GIB,free_disk=20*GIB)

class Mock:
    scope='synthetic'
    def __init__(self,problem=None):self.problem=problem;self.built=0;self.state=1;self.pins=0
    def verify_static(self):
        self.pins+=1
        if self.problem=='source':raise RuntimeError('source mismatch')
    def build(self):self.built+=1;return object(),object()
    def snapshot(self,m,f):return dict(state=self.state,roster=508,fixture='original',backend='mock-C4M')
    def run(self,m,f,path,seal,monitor):
        path.mkdir()
        seal('before_single_forward')
        if self.problem=='interrupt':raise KeyboardInterrupt('controlled interruption')
        if self.problem=='mutation':self.state+=1
        seal('before_traversals')
        if self.problem=='duplicate':seal('before_traversals')
        if self.problem!='missing':seal('after_traversals')
        return dict(status='FAIL' if self.problem=='numerical' else 'PASS')
    def verify_complete(self,path,analysis):
        if self.problem=='incomplete':raise RuntimeError('missing raw shard')

def run(tmp_path,m,**kwargs):return run_mock(tmp_path,'attempt_01',m,sample=kwargs.pop('sample',sample),identity=lambda:(123,1.),**kwargs)

def test_completed_numerical_failure_remains_fail(tmp_path):
    r=run(tmp_path,Mock('numerical'))
    assert r['execution']=='COMPLETE' and r['numerical']=='FAIL' and r['dependent_work']=='BLOCKED' and r['B0_D08']=='DISABLED'

@pytest.mark.parametrize('problem',['source','mutation','duplicate','missing','interrupt','incomplete'])
def test_fail_closed(tmp_path,problem):
    m=Mock(problem);r=run(tmp_path,m)
    assert r['execution']=='HARD STOP' and r['numerical']=='NOT RUN'
    if problem=='source':assert m.built==0
    assert (tmp_path/'attempt_01'/'result.json').exists()
    assert (tmp_path/'coordinator.json').exists()

@pytest.mark.parametrize('field',['available_ram','free_disk'])
def test_resource_gate_before_construction(tmp_path,field):
    def low():r=sample();r[field]=8*GIB-1;return r
    m=Mock();result=run(tmp_path,m,sample=low)
    assert m.built==0 and result['execution']=='WAIT/NOT RUN'

def test_second_resource_gate_before_forward(tmp_path):
    class Drop(Mock):
        def build(self):
            result=super().build();values['available_ram']=7*GIB;return result
    values=sample();r=run(tmp_path,Drop(),sample=lambda:dict(values))
    assert r['execution']=='WAIT/NOT RUN' and not (tmp_path/'attempt_01'/'00_before_single_forward_seal.json').exists()

def test_exclusive_claim_no_replay(tmp_path):
    run(tmp_path,Mock())
    original=(tmp_path/'coordinator.json').read_bytes()
    with pytest.raises(FileExistsError):run_mock(tmp_path,'attempt_02',Mock(),sample=sample,identity=lambda:(999,2.))
    assert (tmp_path/'coordinator.json').read_bytes()==original and not (tmp_path/'attempt_02').exists()

def test_identity_change(tmp_path):
    now=[123,1.];lease=Lease(tmp_path,'attempt',lambda:tuple(now));now[1]=2.
    with pytest.raises(RuntimeError):lease.check()

def test_periodic_sampler_and_failure(tmp_path):
    values=sample()
    with Monitor(tmp_path/'resources.jsonl',lambda:dict(values),interval=.01) as monitor:
        values['available_ram']=0
        time.sleep(.04)
        with pytest.raises(ResourceWait):monitor.check()
    rows=[json.loads(s) for s in (tmp_path/'resources.jsonl').read_text().splitlines()]
    assert len(rows)>=2 and any(r['available_ram']==0 for r in rows)

@pytest.mark.parametrize('entry',[execute_real,execute])
def test_public_dispatch_disabled(entry):
    with pytest.raises(PermissionError):entry(approved=True,cell='D02B-G01-A0-SAME-FORWARD')

def test_non_mock_adapter_rejected(tmp_path):
    m=Mock();m.scope='live-unapproved'
    with pytest.raises(PermissionError):run(tmp_path,m)

def test_invalid_resource_record():
    r=sample();r['available_ram']=True
    with pytest.raises(ValueError):require_resources(r)

def test_tampered_pin_file(tmp_path):
    p=tmp_path/'pins.json';p.write_text('{}')
    with pytest.raises(RuntimeError):FrozenAdapter(Path.cwd(),p,'0'*64)

def test_actual_file_verification_without_model(tmp_path,monkeypatch):
    import hashlib
    monkeypatch.chdir(tmp_path)
    source=tmp_path/'mock_source.py';source.write_text('value = 1\n')
    pins=dict(files={'mock_source.py':hashlib.sha256(source.read_bytes()).hexdigest()},
              cell='D02B-G01-A0-SAME-FORWARD',historical_result=dict(failed_tensors=293,outside_coordinates=8772234))
    path=tmp_path/'pins.json';path.write_text(json.dumps(pins))
    adapter=FrozenAdapter(tmp_path,path,hashlib.sha256(path.read_bytes()).hexdigest())
    adapter.verify_static()
    source.write_text('value = 2\n')
    with pytest.raises(ValueError):adapter.verify_static()

def test_attempt_claim_tampering(tmp_path):
    lease=Lease(tmp_path,'attempt',lambda:(123,1.))
    (lease.path/'claim.json').write_text('{}')
    with pytest.raises(RuntimeError):lease.check()

def test_real_tiny_seven_pass_stream(tmp_path):
    import torch
    from mfvlr_research.d02b9.same_forward import ORDER,run_mock as gradient_mock
    class Tiny(Mock):
        def run(self,m,f,path,seal,monitor):
            seal('before_single_forward')
            p=torch.nn.Parameter(torch.tensor([2.]))
            specs={'x':dict(shape=[1],presence={n:'allocated' for n in (*ORDER,'total')})}
            def forward():
                value=p.square().sum();seal('before_traversals')
                return {n:value*(i+1) for i,n in enumerate(ORDER)}
            result=gradient_mock(forward,[('x',p)],specs,path)
            seal('after_traversals');return result['analysis']
    assert run(tmp_path,Tiny())['numerical']=='PASS'
