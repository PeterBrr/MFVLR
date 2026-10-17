import ast
import copy
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pytest
from mfvlr_research.d02b6 import adapter as a
from mfvlr_research.d02b1 import schema as s
from mfvlr_research.d02b1.gradients import DIRECTION_SEEDS

ROOT=Path(__file__).resolve().parents[2]
ROSTERS=json.loads((ROOT/'tests/research_d02b1/fixtures/historical_rosters.json').read_text())['rosters']

@pytest.mark.parametrize('cell',[c['experiment_id'] for c in a.primary_matrix()])
def test_primary(cell):
    p=a.cell_plan(cell)
    assert p['execution']=='NOT RUN' and p['passes']==['fd','lr','cmc','fl','ar','kl','total']
    assert p['parameter_inventory'][0] in (508,550)
    assert len(p['required_stages'])==63
    with pytest.raises(a.ExecutionNotAuthorized):a.dispatch(cell,approved=True)

@pytest.mark.parametrize('cell',['D02B-C01-A0','D02B-N01-P3','D02B-G07-Q3','D02B-G01-I1','G01',None])
def test_outside_scope(cell):
    with pytest.raises(s.InvalidRecord):a.cell_plan(cell)

def test_bindings_exact_identity():
    fields={n:object() for n in a.FIELDS};b=a.bind_arguments(fields)
    expected={'fd':('detection_logits','labels'),'lr':('language_logits','token_ids'),
              'cmc':('visual','global_language'),'fl':('mask_logits','masks'),
              'ar':('image','reconstruction'),'kl':('global_language','predicted_language')}
    for loss,names in expected.items():assert all(x is fields[n] for x,n in zip(b[loss],names))
    with pytest.raises(s.InvalidRecord):a.bind_arguments(dict(fields,ignore_padding=True))

def test_no_execution_switch_or_torch():
    with pytest.raises(a.ExecutionNotAuthorized):a.continue_after_failure(approved=True)
    tree=ast.parse((ROOT/'src/mfvlr_research/d02b6/adapter.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):assert all(not x.name.startswith(('torch','subprocess')) for x in node.names)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert node.func.attr not in ('backward','step','Popen')

@pytest.mark.parametrize('variant',('A0','P3','Q3'))
def test_all_roster_names_small_scalars(tmp_path,variant):
    specs={r['name']:{'shape':[],'presence':s.participation(r['name'])} for r in ROSTERS[variant]}
    state={n:np.array(0,dtype=np.float32) for n in specs};adapter=a.MockEvidenceAdapter(tmp_path,specs,state)
    for loss in s.PASSES:
        gradients={}
        for n,spec in specs.items():
            presence=spec['presence'][loss]
            value=sum(spec['presence'][k]=='allocated' for k in s.ORDER) if loss=='total' else 1
            gradients[n]=None if presence=='none' else np.array(0 if presence=='zero' else value,dtype=np.float32)
        adapter.consume(loss,gradients,6. if loss=='total' else 1.)
    r=adapter.finish()
    assert r['status']=='PASS' and r['execution']=='NOT RUN' and len(r['parameters'])==s.COUNTS[variant][0]
    for n,p in r['parameters'].items():
        assert len(p['loss_pairs'])==15 and len(p['direction_actual'])==4
        if '.vim.q.' in n or '.vim.k.' in n:assert p['gradient_l2']==0

def vector_run(tmp_path,joint=None,name='heads.detector.weight'):
    xs=[np.array([i,-i,i/8,1e-10*i],dtype=np.float32) for i in range(1,7)]
    spec={name:{'shape':[4],'presence':{p:'allocated' for p in s.PASSES}}}
    adapter=a.MockEvidenceAdapter(tmp_path,spec,{name:np.zeros(4,dtype=np.float32)})
    summed=sum((x.astype(np.float64) for x in xs),np.zeros(4))
    for p,x in zip(s.PASSES,xs+[summed.astype(np.float32) if joint is None else joint]):adapter.consume(p,{name:x},1.)
    return adapter.finish(),xs,summed

def test_independent_math_pairs_and_directions(tmp_path):
    r,xs,ref=vector_run(tmp_path);name='heads.detector.weight';p=r['parameters'][name]
    assert r['status']=='PASS'
    for i,j in itertools.combinations(range(6),2):
        assert p['loss_pairs'][s.ORDER[i]+':'+s.ORDER[j]]['dot']==float(xs[i].astype(float)@xs[j].astype(float))
    salt=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')
    for k,seed in enumerate(DIRECTION_SEEDS):
        d=np.random.default_rng(seed+salt).integers(0,2,4,dtype=np.int8)*2-1
        assert p['direction_reference'][k]==float(ref@d)/2
    stored=np.fromfile(tmp_path/'gradients'/p['sum_path'],dtype='<f8')
    np.testing.assert_array_equal(stored,ref)

def test_finite_fail_retained_blocks_continuation(tmp_path):
    r,_,_=vector_run(tmp_path/'stream',joint=np.zeros(4,dtype=np.float32))
    assert r['status']=='FAIL' and r['parameters']['heads.detector.weight']['outside_tolerance_count']==3
    gate=a.failure_record(tmp_path/'failure.json',provenance=True,structural=True,finite=True,complete=True,numerical=r['status'],resource='PASS')
    assert gate['numerical']=='FAIL' and gate['dependent_work']=='BLOCKED'
    with pytest.raises(a.ExecutionNotAuthorized):a.continue_after_failure(gate)
    assert json.loads((tmp_path/'failure.json').read_text())['numerical']=='FAIL'

@pytest.mark.parametrize('bad',('provenance','structural','finite','complete','resource'))
def test_mandatory_stop(tmp_path,bad):
    fields=dict(provenance=True,structural=True,finite=True,complete=True,numerical='PASS',resource='PASS')
    fields[bad]='FAIL/OOM' if bad=='resource' else False
    assert a.failure_record(tmp_path/'status.json',**fields)['execution']=='HARD STOP'

@pytest.mark.parametrize('mode',('none','nonfinite','wrong_shape','order','mutation','alias','too_large','vim_nonzero','incomplete'))
def test_stream_negative(tmp_path,mode):
    name='flt.encoder.blocks.0.vim.q.weight' if mode=='vim_nonzero' else 'p'
    specs={name:{'shape':[2],'presence':{p:'allocated' for p in s.PASSES}}};state={name:np.zeros(2,dtype=np.float32)}
    with pytest.raises(s.InvalidRecord):
        if mode=='alias':
            specs['q']=copy.deepcopy(specs[name]);state['q']=state[name]
        if mode=='too_large':specs[name]['shape']=[17];state[name]=np.zeros(17,dtype=np.float32)
        adapter=a.MockEvidenceAdapter(tmp_path,specs,state)
        if mode=='mutation':state[name][0]=1
        if mode=='incomplete':adapter.finish();return
        value=None if mode=='none' else np.array([np.nan,1] if mode=='nonfinite' else [1] if mode=='wrong_shape' else [1,2],dtype=np.float32)
        adapter.consume('lr' if mode=='order' else 'fd',{name:value},1.)

@pytest.mark.parametrize('batch,ram,disk,expected',[(1,5,8,'PASS'),(2,6,8,'PASS'),(1,4,8,'INSUFFICIENT'),(2,5,8,'INSUFFICIENT'),(2,8,7,'INSUFFICIENT')])
def test_resource_floors(batch,ram,disk,expected):
    assert a.Reservations().check(batch,ram*1024**3,disk*1024**3)['resource']==expected

def test_exclusive_owner_and_interruption(tmp_path):
    live={7:1.};owner={'pid':7,'created':1.};c=a.Coordinator(tmp_path/'run',owner,live.get);c.claim()
    path=c.reserve('D02B-G01-A0','a'*64)
    with pytest.raises(FileExistsError):c.reserve('D02B-G01-A0','a'*64)
    with pytest.raises(s.InvalidRecord):a.archive_attempt(path,tmp_path/'archive',owner=owner,liveness=live.get)
    live.clear();archive=a.archive_attempt(path,tmp_path/'archive',owner=owner,liveness=live.get)
    assert archive['source_preserved'] and path.exists()

def test_seal_rejects_changes_and_duplicates(tmp_path):
    record=dict(cell='D02B-G01-A0',pass_='fd')
    record['pass']=record.pop('pass_')
    record.update({k:{'mock':'a'*64} for k in ('state','ownership','inputs','source')})
    record.update(backend={'mock':True},resource={'mock':True})
    a.seal_pass(record,copy.deepcopy(record),tmp_path/'seal.json')
    with pytest.raises(FileExistsError):a.seal_pass(record,record,tmp_path/'seal.json')
    changed=copy.deepcopy(record);changed['inputs']['mock']='b'*64
    with pytest.raises(s.InvalidRecord):a.seal_pass(changed,record,tmp_path/'other.json')
