import ast,copy,json
from pathlib import Path
import numpy as np
import pytest
from mfvlr_research.d02b6 import adapter as a
from mfvlr_research.d02b1 import schema as s
from mfvlr_research.d02b1.safety import scientific_prefix
ROOT=Path(__file__).resolve().parents[2]
ROSTERS=json.loads((ROOT/'tests/research_d02b1/fixtures/historical_rosters.json').read_text())['rosters']

def record():
    inv=[dict(r,dtype='float32',sha256='a'*64,object_id=f'o{i}',storage_id=f's{i}') for i,r in enumerate(ROSTERS['P3'])]
    r=dict(identity=next(x for x in s.schedule() if x['experiment_id']=='D02B-G01-P3'),
        hashes=dict(source='1'*64,config='2'*64,protocol='3'*64,driver='4'*64,
                    inputs={k:'5'*64 for k in ('image','tokens','mask','labels','sample_order')},state='6'*64,inventory=s.fingerprint(inv)),
        backend=dict(device='cpu',dtype='float32',intra=4,inter=4,mkldnn=True,deterministic=True,matmul='highest',attention='explicit',python='3.14.3',torch='2.12.1+cpu',platform='Windows'),
        seeds=dict(mve=0,vd=0,heads=0,flt=0),ownership=inv,vocabulary_tied=True,loss_order=list(s.ORDER),
        hash_algorithms={'files':'sha256-file-bytes','inputs':'sha256-contiguous-little-endian-bytes','state':'sha256-ordered-named-parameter-name-plus-bytes','inventory':'sha256-canonical-json'})
    a0_names={x['name'] for x in ROSTERS['A0']}
    reference=[dict(p,object_id='r'+p['object_id'],storage_id='r'+p['storage_id']) for p in inv if p['name'] in a0_names]
    assert len(reference)==508
    return r,reference

@pytest.mark.parametrize('mutation',(None,'hash','source','backend','alias','inventory','unknown','seed'))
def test_sealed_preflight(tmp_path,mutation):
    r,ref=record();pin=s.fingerprint(r);p=tmp_path/'source.txt';p.write_text('mock')
    hashes={'source.txt':s.file_hash(p)}
    if mutation=='hash':r['hashes']['inputs']['image']='b'*64
    if mutation=='source':p.write_text('modified')
    if mutation=='backend':r['backend']['intra']=1
    if mutation=='alias':ref[0]['object_id']=r['ownership'][0]['object_id']
    if mutation=='inventory':r['ownership'].pop()
    if mutation=='unknown':r['execute']=True
    if mutation=='seed':r['seeds']['flt']=1
    kwargs=dict(pinned=pin,roster=ROSTERS['P3'],source_root=tmp_path,source_hashes=hashes,reference_inventory=ref,ram=5*1024**3,disk=8*1024**3)
    if mutation:
        with pytest.raises(s.InvalidRecord):a.validate_preflight(r,**kwargs)
    else:
        result=a.validate_preflight(r,**kwargs)
        assert result['execution']=='NOT RUN' and result['parity']['common_parameters']==508

@pytest.mark.parametrize('mode',(None,'missing','nonfinite','shape'))
def test_full_stage_record_coverage(mode):
    records={n:dict(shape=shape,finite=True,rms=0.,max_abs=0.,l2=0.) for n,shape in a.stage_shapes(2).items()}
    if mode=='missing':del records['LD.6.vim']
    if mode=='nonfinite':records['Iv']['finite']=False
    if mode=='shape':records['logits_lr']['shape']=[2,1,4]
    if mode:
        with pytest.raises(s.InvalidRecord):a.validate_stages(records,2)
    else:a.validate_stages(records,2)

def test_non_lr_invariants_keep_failure():
    identity=dict(variant='A0',fixture='real',backend='C4M',batch=1,inputs_hash='a'*64,common_state_hash='b'*64,protocol_hash='c'*64)
    other=dict(identity,variant='P3');losses=('fd','cmc','fl','ar','kl')
    values={k:np.array(1,dtype=np.float32) for k in losses};grads={k:{'x':np.array([1e-30],dtype=np.float32)} for k in losses}
    r=a.invariant_pair(identity,other,values,values,grads,grads,['x']);assert r['status']=='PASS'
    changed=copy.deepcopy(grads);changed['cmc']['x'][0]=0
    assert a.invariant_pair(identity,other,values,values,grads,changed,['x'])['status']=='FAIL'

def test_replay_preserves_failed_science():
    row=dict(comparison_id='p',scientific={'status':'FAIL','weak':1e-30},resources={'seconds':1,'rss_bytes':1})
    replay=copy.deepcopy(row);replay['resources']['seconds']=2
    assert scientific_prefix([row],[replay])['status']=='PASS'
    replay['scientific']['weak']=0
    assert scientific_prefix([row],[replay])['status']=='FAIL'

def test_original_objective_boundaries_static():
    tree=ast.parse((ROOT/'src/mfvlr/objectives.py').read_text())
    functions={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
    for target,_ in a.LOSS_BINDINGS.values():
        if target.startswith('mfvlr.objectives.'):assert target.split('.')[-1] in functions
    total=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='unit_weight_sum')
    assignment=next(n for n in total.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Tuple))
    assert [x.id for x in assignment.value.elts]==['fd','lr','cmc','fl','ar','kl']
    source=(ROOT/'src/mfvlr/objectives.py').read_text()
    assert 'cmc = self.cmc(visual, global_language)' in source
    assert 'kl = knowledge_distillation_loss(global_language, predicted_language)' in source
