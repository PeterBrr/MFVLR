import json
from pathlib import Path
import pytest
import torch
from mfvlr_research.d02b9.same_forward import ORDER, run_mock, dispatch_real, historical_decisions
from mfvlr_research.d02b9.live_preparation import validate_plan, execute, resource_readiness

def setup_graph():
    p = torch.nn.Parameter(torch.tensor([.5, 2.]))
    zero = torch.nn.Parameter(torch.tensor([1.]))
    unused = torch.nn.Parameter(torch.tensor([3.]))
    named = [('weight',p),('zero',zero),('unused',unused)]
    specs = {n:dict(shape=list(t.shape),presence={k:('none' if n=='unused' else 'zero' if n=='zero' else 'allocated')
             for k in (*ORDER,'total')}) for n,t in named}
    calls = []
    def forward():
        calls.append(1)
        hidden = p.square().sum() + zero.sum()*0
        return {name:hidden*(i+1) for i,name in enumerate(ORDER)}
    return named,specs,forward,calls

def test_seven_real_traversals(tmp_path, monkeypatch):
    named,specs,forward,calls = setup_graph()
    observed = []
    original = torch.autograd.grad
    def spy(*a,**kw):
        observed.append(kw['retain_graph'])
        return original(*a,**kw)
    monkeypatch.setattr(torch.autograd,'grad',spy)
    result = run_mock(forward,named,specs,tmp_path/'stream')
    assert calls == [1] and observed == [True]*6+[False]
    assert result['analysis']['status']=='PASS'
    row = result['analysis']['parameters']['weight']
    assert row['gradient_l2'] == pytest.approx((21**2+84**2)**.5)
    assert len(row['loss_pairs'])==15 and len(row['direction_actual'])==4
    assert all(p.grad is None for _,p in named)
    for name in ORDER:
        assert result['analysis']['parameters']['unused']['passes'][name]['allocated'] is False
        assert result['analysis']['parameters']['zero']['passes'][name]['nonzero']==0

@pytest.mark.parametrize('entry',[dispatch_real,execute])
def test_live_disabled(entry):
    with pytest.raises(PermissionError): entry(approved=True,cell='D02B-G01-A0')

@pytest.mark.parametrize('kind',['order','nonfinite','disconnect','duplicate','large','dirty'])
def test_negative_graphs(tmp_path,kind):
    named,specs,forward,_=setup_graph()
    if kind=='order':
        base=forward
        forward=lambda:dict(reversed(list(base().items())))
    elif kind=='nonfinite':
        base=forward
        forward=lambda:{k:v*float('inf') for k,v in base().items()}
    elif kind=='disconnect': specs['unused']['presence']['fd']='allocated'
    elif kind=='duplicate': named[1]=('zero',named[0][1])
    elif kind=='large': named[0]=('weight',torch.nn.Parameter(torch.zeros(4097)))
    else: named[0][1].grad=torch.ones_like(named[0][1])
    with pytest.raises((ValueError,PermissionError,RuntimeError)):
        run_mock(forward,named,specs,tmp_path/'stream')

def test_original_records_immutable():
    old={'a':'FAIL','b':'PASS'}
    got=historical_decisions({'a':'PASS','b':'FAIL'},old)
    assert old=={'a':'FAIL','b':'PASS'} and got['a']['historical']=='FAIL'
    with pytest.raises(ValueError):historical_decisions({},old)

def test_plan():
    plan=json.loads(Path('configs/research_d02b9/same_forward.json').read_text())
    assert validate_plan(plan).startswith('PREPARED')
    plan['execution_enabled']=True
    with pytest.raises(ValueError):validate_plan(plan)

@pytest.mark.parametrize('ram,disk,status',[(5,8,'WAIT'),(8,7,'WAIT'),(8,8,'SCREEN_MET')])
def test_resource_screen_never_authorizes(ram,disk,status):
    result=resource_readiness(ram*1024**3,disk*1024**3)
    assert result['resource']==status and result['execution']=='DISABLED' and result['capacity']=='NOT VERIFIED'

def test_singleton_qk_connected_zero(tmp_path):
    q=torch.nn.Parameter(torch.tensor([[.2,.7]]))
    k=torch.nn.Parameter(torch.tensor([[.4,.8]]))
    v=torch.nn.Parameter(torch.tensor([[.3,.9]]))
    named=[('q',q),('k',k),('v',v)]
    specs={n:dict(shape=list(p.shape),presence={s:'zero' if n in ('q','k') else 'allocated' for s in (*ORDER,'total')}) for n,p in named}
    def forward():
        value=((q@k.T).softmax(-1)@v).sum()
        return {n:value*(i+1) for i,n in enumerate(ORDER)}
    result=run_mock(forward,named,specs,tmp_path/'stream')['analysis']['parameters']
    assert result['q']['passes']['total']['allocated'] and result['q']['passes']['total']['nonzero']==0
    assert result['k']['passes']['total']['nonzero']==0 and result['v']['passes']['total']['nonzero']==2
