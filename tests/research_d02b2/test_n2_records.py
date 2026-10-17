import copy
import math
import numpy as np
import pytest
from mfvlr_research.d02b2 import contract as c, verification as v

def gradient_records(variant):
    result={}
    for p in c.bundle()['rosters'][variant]:
        zero=c.gradient_role(p['name'])=='allocated_exact_zero'
        # Analytical mock records, not claims about any MFVLR gradient.
        value=0. if zero else 1e-30 if c.gradient_role(p['name']) in ('ordinary_qk_weight','analytical_key_bias_residual') else 1.
        result[p['name']]=dict(allocated=True,shape=p['shape'],finite=True,nonzero=0 if zero else 1,l2_norm=value,max_abs=value)
    return result

@pytest.mark.parametrize('variant',c.VARIANTS)
def test_complete_gradient_participation_and_weak_categories(variant):
    rows=gradient_records(variant)
    r=v.validate_gradients(rows,variant)
    assert r['vim_allocated_exact_zero']==38
    assert len(r['ordinary_qk'])==52  # 12 LE pairs + 7 LD self/memory pairs.
    assert len(r['ordinary_key_bias_residuals'])==26
    assert all(x['l2_norm']==1e-30 for x in r['ordinary_qk'].values())
    assert all(x['max_abs']==1e-30 for x in r['ordinary_key_bias_residuals'].values())

@pytest.mark.parametrize('failure',('missing','none','vim_nonzero','nonfinite','wrong_shape','positive_disconnected','zero_inconsistent'))
def test_gradient_negative_controls(failure):
    rows=gradient_records('P3'); name='decoder.blocks.0.vim.q.weight'
    if failure=='missing': rows.pop(name)
    elif failure=='none': rows[name]['allocated']=False
    elif failure=='vim_nonzero': rows[name].update(nonzero=1,l2_norm=1e-30,max_abs=1e-30)
    elif failure=='nonfinite': rows['vocabulary.weight']['l2_norm']=float('inf')
    elif failure=='wrong_shape': rows[name]['shape']=[1]
    elif failure=='positive_disconnected': rows['vocabulary.weight'].update(nonzero=0,l2_norm=0.,max_abs=0.)
    else: rows[name]['l2_norm']=1.
    with pytest.raises(c.InvalidEvidence): v.validate_gradients(rows,'P3')

def cross_records():
    return [dict(probe_sample=i,other_sample=1-i,lookup_boundary='actual_output.low_EmbeddingBackward',
                 visual_connected=True,lookup_connected=True,other_visual_nonzero=0,other_lookup_nonzero=0,
                 own_visual_l2=1.,own_lookup_l2=2.,finite=True) for i in (0,1)]

def test_bidirectional_analytic_jacobian_and_connected_positive_controls():
    # f_b(low,v) = sum((low_b + 2*v_b)^2), independent of every other row.
    low=np.array([[1.,2.,3.],[4.,5.,6.]],dtype=np.float64)
    visual=np.array([[.1,.2,.3],[.4,.5,.6]],dtype=np.float64)
    records=[]
    for b in (0,1):
        gl=np.zeros_like(low); gv=np.zeros_like(visual)
        gl[b]=2*(low[b]+2*visual[b]); gv[b]=4*(low[b]+2*visual[b])
        r=cross_records()[b]; r.update(own_visual_l2=float(np.linalg.norm(gv[b])),own_lookup_l2=float(np.linalg.norm(gl[b])),
                                      other_visual_nonzero=int(np.count_nonzero(gv[1-b])),other_lookup_nonzero=int(np.count_nonzero(gl[1-b])))
        records.append(r)
    v.validate_cross_sample(records)
    before=(low+2*visual)**2
    changed=low.copy(); changed[1]+=10
    assert v.compare_arrays(before[:1],((changed+2*visual)**2)[:1],category='fixed_batch')['numerical']=='PASS'
    assert v.compare_arrays(before[::-1],(low[::-1]+2*visual[::-1])**2,category='permutation')['numerical']=='PASS'

@pytest.mark.parametrize('failure',('direction_missing','direction_repeated','lookup_detached','visual_detached','other_nonzero','own_zero','nan'))
def test_cross_sample_negative_controls(failure):
    rows=cross_records()
    if failure=='direction_missing': rows.pop()
    elif failure=='direction_repeated': rows[1]=copy.deepcopy(rows[0])
    elif failure=='lookup_detached': rows[0]['lookup_boundary']='new_detached_leaf'
    elif failure=='visual_detached': rows[0]['visual_connected']=False
    elif failure=='other_nonzero': rows[1]['other_lookup_nonzero']=1
    elif failure=='own_zero': rows[0]['own_visual_l2']=0.
    else: rows[1]['own_lookup_l2']=float('nan')
    with pytest.raises(c.InvalidEvidence): v.validate_cross_sample(rows)

@pytest.mark.parametrize('category,dtype,role',(('fp32_cross_batch',np.float32,'diagnostic_only'),('fp64_cross_batch',np.float64,'mandatory'),('fixed_batch',np.float32,'mandatory_exact'),('permutation',np.float64,'mandatory_exact')))
def test_original_comparison_policy_and_failures_remain_failures(category,dtype,role):
    a=np.array([0.,2.,1e-20],dtype=dtype); b=np.array([1.,2.,-1e-20],dtype=dtype)
    r=v.compare_arrays(a,b,category=category)
    assert r['numerical']=='FAIL' and r['acceptance']=='NOT GRANTED' and r['role']==role
    assert r['max_absolute_difference']==1.
    assert r['max_relative_difference_nonzero_reference']==2.
    expected=np.sqrt(np.sum((a.astype(float)-b.astype(float))**2)/np.sum(b.astype(float)**2))
    np.testing.assert_allclose(r['relative_l2_difference'],expected,atol=1e-10,rtol=1e-8)
    assert v.compare_arrays(a,a,category=category)['numerical']=='PASS'

def test_zero_reference_and_no_new_category_exception():
    r=v.compare_arrays(np.array([1.,0.],dtype=np.float32),np.zeros(2,dtype=np.float32),category='fp32_cross_batch')
    assert r['relative_l2_difference'] is None and r['zero_reference_nonzero_difference_count']==1
    assert r['max_relative_difference_is_infinite']
    for category in ('gradient_additivity','ordinary_qk','N1','fd','full_training'):
        with pytest.raises(c.InvalidEvidence): v.compare_arrays(np.ones(1),np.ones(1),category=category)

@pytest.mark.parametrize('failure',('nan','inf','shape','dtype','empty'))
def test_comparison_negative_controls(failure):
    a=np.ones(2,dtype=np.float32); b=a.copy()
    if failure=='nan': a[0]=np.nan
    elif failure=='inf': a[0]=np.inf
    elif failure=='shape': b=b[:1]
    elif failure=='dtype': b=b.astype(np.float64)
    else: a=a[:0]; b=b[:0]
    with pytest.raises(c.InvalidEvidence): v.compare_arrays(a,b,category='fp32_cross_batch')

@pytest.mark.parametrize('fixture',('F0','F1','F2'))
def test_no_capture_omission_or_graph_retention(fixture):
    rows={n:dict(shape=shape,dtype='float32',finite=True,l2_norm=0.,rms=0.,max_abs=0.,graph_owned=False) for n,shape in c.stages(fixture,2).items()}
    v.validate_stages(rows,fixture,2,'float32')
    bad=copy.deepcopy(rows); bad['decoder.blocks.6']['graph_owned']=True
    with pytest.raises(c.InvalidEvidence): v.validate_stages(bad,fixture,2,'float32')
    bad=copy.deepcopy(rows); bad.pop('output.logits')
    with pytest.raises(c.InvalidEvidence): v.validate_stages(bad,fixture,2,'float32')
    bad=copy.deepcopy(rows); bad['encoder.blocks.11']['max_abs']=float('nan')
    with pytest.raises(c.InvalidEvidence): v.validate_stages(bad,fixture,2,'float32')

def test_original_chunk_boundary_on_small_independent_vectors():
    a=np.ones(262149,dtype=np.float32); b=a.copy(); a[-1]=2.
    r=v.compare_arrays(a,b,category='fp32_cross_batch')
    assert r['outside_tolerance_count']==1 and r['elements']==262149
    assert r['outside_proportion']==1/262149
    np.testing.assert_allclose(r['rms_absolute_difference'],1/math.sqrt(262149),atol=1e-10,rtol=1e-8)
