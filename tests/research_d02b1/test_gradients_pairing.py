import copy
import hashlib
import itertools
import json
import math
import numpy as np
import pytest
from mfvlr_research.d02b1 import gradients as g
from mfvlr_research.d02b1 import pairing as p
from mfvlr_research.d02b1.schema import InvalidRecord, ORDER, PASSES
from mfvlr_research.d02b1.runner import require_complete

def specification(shape=(4,), presence=None):
    return {'shape':list(shape),'presence':presence or {k:'allocated' for k in PASSES}}

def store_values(tmp_path, values, name='ordinary.q.weight', presence=None):
    first=next(x for x in values if x is not None)
    store=g.GradientStore(tmp_path/'gradients',{name:specification(first.shape,presence)})
    before=[None if x is None else x.copy() for x in values]
    for key,x in zip(PASSES,values): store.put(key,name,x)
    for a,b in zip(before,values):
        if a is not None: assert np.array_equal(a,b)
    return store

def test_independent_closed_form_six_linear_objectives(tmp_path):
    # f_i(w)=c_i dot w: gradients analytically equal c_i, independent of production.
    coefficients=np.array([[1,-2,3,4],[-1,5,0,-4],[2,0,1,1],[0,0,0,0],[3,-1,-2,0],[1,2,1,0]],dtype=np.float32)
    weights=np.array([.2,-.5,.3,1.],dtype=np.float64)
    eps=1e-6
    for c in coefficients.astype(np.float64):
        fd=np.array([((weights+eps*np.eye(4)[i])@c-(weights-eps*np.eye(4)[i])@c)/(2*eps) for i in range(4)])
        np.testing.assert_allclose(fd,c,atol=1e-5,rtol=1e-3)
    exact=np.array([sum(float(c[j]) for c in coefficients) for j in range(4)],dtype=np.float64)
    store=store_values(tmp_path,[*coefficients,exact.astype(np.float32)])
    report=store.analyze(); r=report['parameters']['ordinary.q.weight']
    assert report['status']=='PASS' and r['error_l2']==0
    assert r['reference_l2']==math.sqrt(sum(float(x*x) for x in exact))
    assert len(r['loss_pairs'])==15
    for i,j in itertools.combinations(range(6),2):
        pair=r['loss_pairs'][ORDER[i]+':'+ORDER[j]]
        a,b=coefficients[i].astype(float),coefficients[j].astype(float)
        assert pair['dot']==sum(float(x*y) for x,y in zip(a,b))
        if np.any(a) and np.any(b):
            np.testing.assert_allclose(pair['cosine'],float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),atol=1e-10,rtol=1e-8)
        else: assert pair['undefined_reason']=='allocated zero norm'
    assert np.array_equal(np.fromfile(store.root/r['sum_path'],dtype='<f8'),exact)
    json.dumps(report,allow_nan=False)

def test_fp64_order_and_rounded_reference_are_separate(tmp_path):
    values=[np.array([v],dtype=np.float32) for v in (1e20,1.,-1e20,0.,0.,0.,0.)]
    r=store_values(tmp_path,values).analyze()['parameters']['ordinary.q.weight']
    assert r['reference_l2']==0  # Ordered FP64 sum intentionally loses one at 1e20.
    assert r['cancellation_numerator']>1e20
    assert r['cancellation_ratio'] is None

def test_rounding_acceptance_versus_unrounded_error(tmp_path):
    values=[np.array([v],dtype=np.float32) for v in (1.,2**-25,0.,0.,0.,0.,1.)]
    r=store_values(tmp_path,values).analyze()['parameters']['ordinary.q.weight']
    assert r['outside_tolerance_count']==0
    assert r['max_abs_error']==2**-25

def test_failure_and_weak_gradients_are_not_suppressed(tmp_path):
    values=[np.array([1e-30,0.,1.],dtype=np.float32) for _ in range(6)]
    actual=np.array([-6e-30,2e-6,7.],dtype=np.float32)
    r=store_values(tmp_path,[*values,actual],name='flt.decoder.blocks.0.self_attention.q.weight').analyze()
    assert r['status']=='FAIL'
    d=next(iter(r['parameters'].values()))
    assert d['outside_tolerance_count']==2 and d['reference_zero_actual_nonzero']==1
    assert d['sign_disagreements']==1 and d['max_relative_nonzero_reference']>=1.99
    assert d['mathematical_class']=='ordinary_qk_weight'

def test_ordinary_key_bias_reports_residual_without_structural_reclassification(tmp_path):
    x=np.full(3,1e-12,dtype=np.float32)
    r=store_values(tmp_path,[x]*6+[6*x],name='flt.decoder.blocks.0.self_attention.k.bias').analyze()
    d=next(iter(r['parameters'].values()))
    assert d['analytical_zero_residuals']['fd']['max_abs']>0
    assert d['mathematical_class']=='ordinary_key_bias_cancellation'

def test_structural_zero_presence_and_unused_distinction(tmp_path):
    presence={k:'zero' if k in ('lr','total') else 'none' for k in PASSES}
    values=[np.zeros(3,dtype=np.float32) if presence[k]=='zero' else None for k in PASSES]
    r=store_values(tmp_path,values,name='flt.decoder.blocks.0.vim.q.weight',presence=presence).analyze()
    d=next(iter(r['parameters'].values()))
    assert d['passes']['fd']['allocated'] is False and d['passes']['lr']['nonzero']==0
    assert d['loss_pairs']['fd:lr']['undefined_reason']=='unused gradient'
    assert d['relative_l2'] is None and d['cosine'] is None

@pytest.mark.parametrize('kind',('disconnected','nonzero','nonfinite','dtype','shape','order','duplicate','corrupt','incomplete'))
def test_gradient_negative_controls(tmp_path,kind):
    spec=specification((2,),{k:'zero' for k in PASSES} if kind=='nonzero' else None)
    st=g.GradientStore(tmp_path/'store',{'p':spec})
    x=np.zeros(2,dtype=np.float32)
    with pytest.raises((InvalidRecord,FileExistsError)):
        if kind=='disconnected': st.put('fd','p',None)
        elif kind=='nonzero': st.put('fd','p',x+1)
        elif kind=='nonfinite': st.put('fd','p',np.array([0.,np.nan],dtype=np.float32))
        elif kind=='dtype': st.put('fd','p',x.astype(np.float64))
        elif kind=='shape': st.put('fd','p',x.reshape(1,2))
        elif kind=='order': st.put('lr','p',x)
        elif kind=='duplicate':
            st.put('fd','p',x); st.put('fd','p',x)
        elif kind=='incomplete': st.analyze()
        else:
            st.put('fd','p',x)
            (st.root/'00000.f32').write_bytes(b'bad')
            st.open_gradient('fd','p')

def test_four_registered_directions_across_real_chunk_boundary(tmp_path):
    # ~1 MiB arrays, still a synthetic vector, not a model or full gradient graph.
    x=(np.arange(g.CHUNK+9,dtype=np.float32)%11)-5
    r=store_values(tmp_path,[x]*6+[x*6],name='boundary').analyze()['parameters']['boundary']
    salt=int.from_bytes(hashlib.sha256(b'boundary').digest()[:4],'little')
    for i,seed in enumerate((5101,5102,5103,5104)):
        rng=np.random.default_rng(seed+salt)
        dot=0.
        # Preserve original generation chunk boundaries (int8 generator buffering).
        for start in range(0,len(x),262144):
            a=x[start:start+262144].astype(float)*6
            d=rng.integers(0,2,len(a),dtype=np.int8)*2-1
            dot+=float(a@d)
        assert r['direction_reference'][i]==dot/math.sqrt(len(x))
    assert r['direction_absolute_error']==[0.]*4

def test_state_identity_and_mutation(tmp_path):
    arrays={'w':np.ones(4,dtype=np.float32),'b':np.zeros(2,dtype=np.float32)}
    before=g.state_snapshot(arrays)
    g.require_state_unchanged(before,arrays)
    arrays['w'][0]+=1
    with pytest.raises(InvalidRecord): g.require_state_unchanged(before,arrays)
    arrays['w'][0]-=1
    arrays['w']=arrays['w'].copy()
    with pytest.raises(InvalidRecord): g.require_state_unchanged(before,arrays)
    with pytest.raises(InvalidRecord): g.state_snapshot({'a':arrays['w'],'b':arrays['w'][:2]})

def identity(variant):
    return dict(variant=variant,fixture='real',backend='C4M',batch=1,inputs_hash='1'*64,common_state_hash='2'*64,protocol_hash='3'*64)

@pytest.mark.parametrize('variants',(('A0','P3'),('A0','Q3'),('P3','Q3')))
def test_invariant_non_lr_outputs_and_complete_common_gradients(variants):
    outputs={k:np.array(2.,dtype=np.float32) for k in p.INVARIANT_LOSSES}
    grads={k:{'p':np.ones(3,dtype=np.float32),'unused':None} for k in p.INVARIANT_LOSSES}
    report=p.invariant_pair(identity(variants[0]),identity(variants[1]),outputs,outputs,grads,grads,['p','unused'])
    assert report['status']=='PASS' and report['variant_dependent_excluded']==['lr','total']
    bad=copy.deepcopy(grads); bad['fd']['p'][0]+=1
    assert p.invariant_pair(identity(variants[0]),identity(variants[1]),outputs,outputs,grads,bad,['p','unused'])['status']=='FAIL'
    bad['fd'].pop('unused')
    with pytest.raises(InvalidRecord): p.invariant_pair(identity(variants[0]),identity(variants[1]),outputs,outputs,grads,bad,['p','unused'])

def test_pairing_identity_and_batch_dependent_cmc():
    b=identity('P3'); b['fixture']='efs'
    with pytest.raises(InvalidRecord): p.pair_identity(identity('A0'),b)
    x=np.array([1.,2.],dtype=np.float32)
    assert p.same_sample_pair(x,x,kind='fixed_batch_derivative')['status']=='PASS'
    assert p.same_sample_pair(x,x+1,kind='fixed_batch_output')['status']=='FAIL'
    cross=p.same_sample_pair(x,x+1,kind='cross_batch_output')
    assert cross['outside_count']==2 and cross['status']=='FAIL'
    with pytest.raises(InvalidRecord): p.same_sample_pair(x,x,kind='cmc_objective')
    with pytest.raises(InvalidRecord): p.same_sample_pair(None,None,kind='fixed_batch_derivative')

def test_incomplete_outputs_and_state_changes():
    entry=dict(allocated=True,shape=[1],expectation='allocated',path='p.f32',sha256='a'*64,elements=1,bytes=4,finite=True,nonzero=0,l2=0.,rms=0.,max_abs=0.)
    gradient=dict(passes={p:copy.deepcopy(entry) for p in PASSES},loss_pairs={a+':'+b:{} for a,b in itertools.combinations(ORDER,2)},
                  direction_actual=[0.]*4,direction_reference=[0.]*4)
    record=dict(gradients={'p':gradient},stages={'LE':dict(shape=[1,2],finite=True,rms=0.,max_abs=0.,l2=0.)},losses={k:0. for k in PASSES},
                state_before={'p':'1'*64},state_after={'p':'1'*64},ownership_before={'p':'owner'},ownership_after={'p':'owner'})
    specs={'p':specification((1,))}
    require_complete(record,specs,['LE'])
    record['state_after']['p']='2'*64
    with pytest.raises(InvalidRecord): require_complete(record,specs,['LE'])
    record['state_after']['p']='1'*64
    record['losses']['total']=float('inf')
    with pytest.raises(InvalidRecord): require_complete(record,specs,['LE'])
    record['losses']['total']=0.
    record['gradients']['p']['passes']['fd']={}
    with pytest.raises(InvalidRecord): require_complete(record,specs,['LE'])
