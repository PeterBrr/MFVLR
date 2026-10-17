import copy
import itertools
import json
from pathlib import Path
import numpy as np
import pytest
from mfvlr_research.d02b1 import schema as s, safety, pairing, gradients
from mfvlr_research.d02b1.runner import require_complete
from mfvlr_research.d02b1.stages import validate_stages, stage_shapes

def test_parity_all_common_parameter_metadata():
    rosters=json.loads((Path(__file__).parent/'fixtures/historical_rosters.json').read_text())['rosters']
    records={v:[dict(x,dtype='float32',sha256='a'*64,object_id=f'{v}-o{i}',storage_id=f'{v}-s{i}') for i,x in enumerate(rosters[v])] for v in s.VARIANTS}
    for a,b in itertools.combinations(s.VARIANTS,2):
        assert pairing.common_state_parity(records[a],records[b])['common_parameters']==(550 if a=='P3' else 508)
    bad=copy.deepcopy(records['P3']); bad[0]['sha256']='b'*64
    with pytest.raises(s.InvalidRecord): pairing.common_state_parity(records['A0'],bad)
    bad=copy.deepcopy(records['P3']); bad[0]['storage_id']=records['A0'][0]['storage_id']
    with pytest.raises(s.InvalidRecord): pairing.common_state_parity(records['A0'],bad)

@pytest.mark.parametrize('variant',s.VARIANTS)
def test_all_parameter_record_coverage_with_scalar_synthetic_gradients(tmp_path,variant):
    # Actual roster names; deliberately SCALAR synthetic controls, never model-shaped tensors.
    rosters=json.loads((Path(__file__).parent/'fixtures/historical_rosters.json').read_text())['rosters']
    specs={x['name']:{'shape':[],'presence':s.participation(x['name'])} for x in rosters[variant]}
    store=gradients.GradientStore(tmp_path/'scalar_controls',specs)
    for loss in s.PASSES:
        for name, spec in specs.items():
            expected=spec['presence'][loss]
            value=0. if expected=='zero' else float(sum(spec['presence'][k]!='none' for k in s.ORDER)) if loss=='total' else 1.
            store.put(loss,name,None if expected=='none' else np.array(value,dtype=np.float32))
    report=store.analyze()
    assert report['status']=='PASS' and len(report['parameters'])==s.COUNTS[variant][0]
    assert sum(r['mathematical_class']=='singleton_vim_qk' for r in report['parameters'].values())==38
    assert all(len(r['loss_pairs'])==15 for r in report['parameters'].values())

def test_replay_preserves_failure_and_records_mismatch_before_stop(tmp_path):
    a=tmp_path/'old.jsonl'; b=tmp_path/'replay.jsonl'
    row=dict(comparison_id='weak-qk',scientific=dict(status='FAIL',outside=426,error=1e-19),resources=dict(seconds=1.,rss_bytes=20))
    a.write_bytes(s.canonical(row)+b'\n{"partial":')
    row['resources']['seconds']=2.
    b.write_bytes(s.canonical(row)+b'\n')
    good=safety.verify_replay(a,b,tmp_path/'match.json')
    assert good['status']=='PASS' and good['original_fragment']['incomplete_fragment_bytes']>0
    row['scientific']['outside']=425
    b.write_bytes(s.canonical(row)+b'\n')
    with pytest.raises(s.InvalidRecord): safety.verify_replay(a,b,tmp_path/'mismatch.json')
    assert json.loads((tmp_path/'mismatch.json').read_text())['mismatch_indices']==[0]
    a.write_bytes(b'{bad complete record}\n')
    with pytest.raises(s.InvalidRecord): safety.read_complete_prefix(a)

@pytest.mark.parametrize('failure',('missing','nan','shape','unknown'))
def test_stage_negative_controls(failure):
    records={n:dict(shape=sh,finite=True,rms=0.,max_abs=0.,l2=0.) for n,sh in stage_shapes(2).items()}
    if failure=='missing': del records['LD.6.vim']
    elif failure=='nan': records['LE.0.output']['rms']=float('nan')
    elif failure=='shape': records['LD.1.ff']['shape']=[2,1,512]
    else: records['Ipre']['skipped']=True
    with pytest.raises(s.InvalidRecord): validate_stages(records,2)

def test_both_direction_cross_sample_and_permutation_mathematical_control():
    # Independent polynomial map has analytic block-diagonal Jacobian.
    x=np.array([[1.,2.],[3.,4.]],dtype=np.float64)
    y=x*x
    jacobian=np.zeros((2,2,2,2),dtype=np.float64)
    for b in range(2):
        jacobian[b,:,b,:]=np.diag(2*x[b])
    assert np.array_equal(jacobian[0,:,1,:],np.zeros((2,2)))
    assert np.array_equal(jacobian[1,:,0,:],np.zeros((2,2)))
    assert np.count_nonzero(jacobian[0,:,0,:])==2
    assert pairing.exact_arrays(y[::-1],x[::-1]*x[::-1])
    # This tests a mathematical fixture, not a new MFVLR independence experiment.

def test_resource_probe_errors_fail_closed(tmp_path):
    def inaccessible(pid): raise OSError('mock process access failure')
    with pytest.raises(OSError): safety.Coordinator(tmp_path,dict(pid=2,created=10.),inaccessible).claim()
    assert not (tmp_path/'coordinator.lock.json').exists()

def test_inventory_mutation_cannot_change_stream_contract(tmp_path):
    spec={'x':{'shape':[1],'presence':{p:'allocated' for p in s.PASSES}}}
    st=gradients.GradientStore(tmp_path/'store',spec)
    spec['x']['presence']['fd']='none'
    with pytest.raises(s.InvalidRecord): st.put('fd','x',None)

def test_unexpected_vim_gradient_cannot_be_reclassified(tmp_path):
    name='flt.decoder.blocks.0.vim.q.weight'
    st=gradients.GradientStore(tmp_path/'store',{name:{'shape':[1],'presence':{p:'allocated' for p in s.PASSES}}})
    with pytest.raises(s.InvalidRecord): st.put('fd',name,np.ones(1,dtype=np.float32))
