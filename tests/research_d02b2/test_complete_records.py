"""Explicitly mocked model-sized METADATA, never model-sized arrays."""
import copy
import hashlib
import math
import struct
import numpy as np
import pytest
from mfvlr_research.d02b2 import contract as c, verification as v
from test_n2_preparation import observation,request,promotion
from test_n2_records import gradient_records,cross_records

def record_for(req):
    f,d,variant=req['fixture'],req['precision'],req['variant']
    owner=observation(variant)
    one=np.array([1.],dtype=d)
    exemplar=v.compare_arrays(one,one,category='fp32_cross_batch' if d=='float32' else 'fp64_cross_batch')
    comparisons={}
    for name,shape in c.stages(f,1).items():
        comparisons[name]=dict(exemplar,shape=shape,elements=math.prod(shape))
    return dict(request_sha256=c.seal(req),fp32_state_before_sha256=req['candidate_fp32_state_sha256'],
                fp32_state_after_sha256=req['candidate_fp32_state_sha256'],fp32_before=owner,fp32_after=copy.deepcopy(owner),
                promoted_before=promotion(owner) if d=='float64' else None,promoted_after=promotion(owner) if d=='float64' else None,
                activations={n:dict(shape=shape,dtype=d,finite=True,l2_norm=0.,rms=0.,max_abs=0.,graph_owned=False) for n,shape in c.stages(f,2).items()},
                cross_batch=[dict(sample=i,stages=copy.deepcopy(comparisons)) for i in (0,1)],
                fixed_batch=[dict(changed_sample=i,unaffected_sample=1-i,exact_equality={n:True for n in comparisons}) for i in (0,1)],
                permutation={n:True for n in comparisons},
                full_backward=dict(probe='mean(logits^2)+mean(LD_hidden^2)+mean(LE_global^2)',gradients=gradient_records(variant)) if d=='float32' else None,
                cross_derivatives=cross_records() if d=='float32' else None)

@pytest.mark.parametrize('variant',c.VARIANTS)
@pytest.mark.parametrize('precision',c.PRECISIONS)
def test_complete_mock_records_and_preserved_failures(variant,precision):
    req=request(variant=variant,precision=precision); rec=record_for(req)
    result=v.validate_complete_evidence(rec,req,c.seal(req))
    assert result['record_validation']=='COMPLETE' and result['acceptance']=='NOT GRANTED'
    row=rec['cross_batch'][1]['stages']['output.logits']
    row.update(outside_tolerance_count=1,outside_proportion=1/row['elements'],exact_equal=False,numerical='FAIL',max_absolute_difference=1.)
    result=v.validate_complete_evidence(rec,req,c.seal(req))
    assert result['numerical']=='FAIL' and result['outside_tolerance_count']==1
    assert result['mandatory_numerical_failure']==(precision=='float64')

@pytest.mark.parametrize('failure',('missing_sample','missing_stage','false_pass','policy','nonfinite','state_hash','permutation','fixed','probe','missing_gradient','unknown'))
def test_complete_record_negative_controls(failure):
    req=request('P3'); rec=record_for(req)
    if failure=='missing_sample': rec['cross_batch'].pop()
    elif failure=='missing_stage': del rec['cross_batch'][0]['stages']['output.logits']
    elif failure=='false_pass': rec['cross_batch'][0]['stages']['output.logits']['outside_tolerance_count']=1
    elif failure=='policy': rec['cross_batch'][0]['stages']['output.logits']['atol']=5e-6
    elif failure=='nonfinite': rec['cross_batch'][0]['stages']['output.logits']['max_absolute_difference']=float('nan')
    elif failure=='state_hash': rec['fp32_state_after_sha256']='f'*64
    elif failure=='permutation': rec['permutation']['decoder.blocks.6']=False
    elif failure=='fixed': rec['fixed_batch'][1]['exact_equality']['output.logits']=False
    elif failure=='probe': rec['full_backward']['probe']='production_loss'
    elif failure=='missing_gradient': rec['full_backward']['gradients'].pop('decoder.blocks.6.vim.k.weight')
    else: rec['execute']=True
    with pytest.raises(c.InvalidEvidence): v.validate_complete_evidence(rec,req,c.seal(req))

def test_n2_hash_convention_is_not_d02b1_or_shape_free_hash():
    arrays=[('x',np.array([[1.,2.]],dtype=np.float32)),('s',np.array(3.,dtype=np.float32))]
    raw=b'x'+b'(1, 2)'+struct.pack('<ff',1.,2.)+b's'+b'()'+struct.pack('<f',3.)
    assert v.n2_state_hash(arrays)==hashlib.sha256(raw).hexdigest()
    assert v.n2_state_hash(arrays)!=v.n2_state_hash(list(reversed(arrays)))
    with pytest.raises(c.InvalidEvidence): v.n2_state_hash(arrays+arrays)
