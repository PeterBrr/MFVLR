import ast
from pathlib import Path
import pytest
from mfvlr_research.d02b7.policy import require_cell,resource_gate,status
ROOT=Path(__file__).resolve().parents[2]

def test_exact_cell():require_cell('D02B-G01-A0')
@pytest.mark.parametrize('cell',('D02B-G01-P3','D02B-G01-Q3','D02B-G02-A0','D02B-N01-A0','D02B-G06-A0',None))
def test_scope_reject(cell):
    with pytest.raises(PermissionError):require_cell(cell)
@pytest.mark.parametrize('numerical',('PASS','FAIL'))
def test_complete_independent_of_numerical(numerical):
    r=status(completed=True,numerical=numerical,structural='PASS',provenance='PASS',resource='PASS')
    assert r['execution']=='COMPLETE' and r['numerical']==numerical and r['dependent_work']=='BLOCKED'
def test_wait_and_hard_stop():
    assert status(resource='INSUFFICIENT')['execution']=='WAIT/NOT RUN'
    assert status(error=True,structural='FAIL')['execution']=='HARD STOP'
    with pytest.raises(ValueError):status(completed=True,numerical='FAIL')
@pytest.mark.parametrize('ram,disk,ok',((5,8,True),(4,8,False),(5,7,False),(6,9,True)))
def test_resource(ram,disk,ok):assert resource_gate(ram*1024**3,disk*1024**3)==ok
def test_worker_static_scope():
    source=(ROOT/'tools/d02b7_worker.py').read_text();tree=ast.parse(source)
    assert source.index('if not resource_gate')<source.index('import torch')
    assert "fixture('real')" in source and "build_variant(reference,'A0')" in source
    assert 'for active in PASSES:' in source and 'objective.backward()' in source
    assert 'GradientStore' in source and 'store.analyze()' in source
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):assert n.func.attr not in ('step','train')
    runner=(ROOT/'tools/d02b7_run.py').read_text()
    assert 'WAIT/NOT RUN: resource gate failed; single attempt stopped' in runner
    assert 'while True' not in runner and 'P3' not in runner and 'Q3' not in runner

def test_live_arithmetic_uniform_independent_fixture():
    import math,numpy as np,torch
    from types import SimpleNamespace as NS
    from mfvlr_research.d02b7.arithmetic import audit
    # Eight tokens only; no FLT/MVE/VD instance or full graph.
    ids=torch.tensor([[49406,49407,0,0,0,0,7,8]])
    linked=torch.zeros(1,1,4,requires_grad=True)*1
    language=NS(logits=torch.zeros(1,8,49408),global_language=linked)
    decoded=NS(mask_logits=torch.zeros(1,2,2,2),reconstruction=torch.full((1,3,2,2),-.5))
    pred=NS(detection_logits=torch.zeros(1,2),decoded=decoded,visual=NS(fused_global=linked))
    out=NS(predictions=pred,language=language,predicted_language=linked,image=torch.full((1,3,2,2),.25))
    targets=NS(token_ids=ids,labels=torch.zeros(1,dtype=torch.int64),masks=torch.zeros(1,2,2,dtype=torch.int64))
    root=NS(objectives=NS(cmc=NS(log_temperature=torch.tensor(math.log(.07)))))
    expected={'fd':np.float32(math.log(2)),'lr':np.float32(math.log(49408))*np.float32(8),'cmc':np.float32(0),'fl':np.float32(math.log(2)),'ar':np.float32(.75**2),'kl':np.float32(0)}
    total=expected['fd']
    for n in ('lr','cmc','fl','ar','kl'):total=np.float32(total+expected[n])
    observed={n:torch.tensor(v) for n,v in dict(expected,total=total).items()}
    r=audit(out,targets,root,observed)
    assert r['all_exact'] and r['lr']['categories']['pad']['count']==4
    assert r['lr']['categories']['bos']['count']==1 and r['lr']['categories']['content']['count']==2
    assert r['ar']['reconstruction_min']==-.5 and r['cmc']['scores']==[[0.]]
    observed['lr']=observed['lr']+1
    assert not audit(out,targets,root,observed)['all_exact']

def test_small_stream_failure_retained(tmp_path):
    import numpy as np
    from mfvlr_research.d02b1.gradients import GradientStore
    from mfvlr_research.d02b1.schema import PASSES
    store=GradientStore(tmp_path/'stream',{'x':{'shape':[2],'presence':{p:'allocated' for p in PASSES}}})
    for name in PASSES:store.put(name,'x',np.array([0,0] if name=='total' else [1,1e-20],dtype=np.float32))
    r=store.analyze()
    assert r['status']=='FAIL' and r['parameters']['x']['outside_tolerance_count']==1
    assert len(r['parameters']['x']['loss_pairs'])==15 and len(r['parameters']['x']['direction_actual'])==4
    assert status(completed=True,numerical=r['status'],structural='PASS',provenance='PASS',resource='PASS')['execution']=='COMPLETE'

def test_capture_actual_boundaries_and_cleanup():
    import torch
    from types import SimpleNamespace as NS
    from mfvlr_research.d02b7.capture import Capture
    class Shift:
        def shift_low(self,x):return x+7
    flt=Shift();flt.encoder=torch.nn.Identity();flt.encoder.blocks=[];flt.decoder=torch.nn.Identity()
    block=NS(memory_attention=torch.nn.Identity(),ff=torch.nn.Identity(),vim=torch.nn.Identity())
    # The block itself must support an output hook.
    b=torch.nn.Identity();b.memory_attention=block.memory_attention;b.ff=block.ff;b.vim=block.vim
    flt.decoder.blocks=[b]
    root=NS(flt=flt,mve=NS(shared_encoder=NS(cnn=torch.nn.Identity())))
    def stats(x):return dict(shape=list(x.shape),finite=True,rms=float(x),max_abs=float(x),l2=float(x))
    with Capture(root,stats) as cap:
        y=flt.shift_low(torch.tensor(1.));assert float(y)==8
        b.memory_attention(torch.tensor(2.));b.ff(torch.tensor(3.));b.vim(torch.tensor(4.));b(torch.tensor(5.))
        assert cap.values['LD.0.self']['l2']==2 and cap.values['LD.0.memory']['l2']==3
        assert cap.values['LD.0.ff']['l2']==4 and cap.values['LD.0.vim']['l2']==5
        assert cap.values['LD.shifted_lookup']['l2']==8
    assert not cap.handles and 'shift_low' not in flt.__dict__
