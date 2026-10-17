import ast
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from mfvlr_research.d02b2 import contract as c, verification as v
from mfvlr_research.d02b2.adapter import FrozenN2Adapter,dispatch

ROOT=Path(__file__).resolve().parents[2]

def observation(variant,owner='candidate'):
    rows=[]
    for i,p in enumerate(c.bundle()['rosters'][variant]):
        digest='a'*64
        if '.ln' in p['name'] and p['name'].startswith('decoder.'):
            digest=hashlib.sha256(np.full(512,1 if p['name'].endswith('weight') else 0,dtype='<f4').tobytes()).hexdigest()
        rows.append(dict(p,dtype='float32',sha256=digest,object_id=f'{owner}-object-{i}',storage_id=f'{owner}-storage-{i}'))
    vocab=next(p for p in rows if p['name']=='vocabulary.weight')['object_id']
    return dict(parameters=rows,norms=c.norm_inventory(variant),encoder_blocks=[f'encoder.blocks.{i}' for i in range(12)],
                decoder_blocks=[f'decoder.blocks.{i}' for i in range(7)],vocabulary_lookup_id=vocab,vocabulary_output_id=vocab,
                vim_instances=[f'encoder.blocks.{i}.vim' for i in range(12)]+[f'decoder.blocks.{i}.vim' for i in range(7)])

def request(variant='A0',fixture='F0',precision='float32'):
    b=c.bundle(); f=b['fixtures'][fixture]; d=f['fixture_description']
    return dict(experiment_id=f'D02B2-N2-{fixture}-{variant}-{precision}',variant=variant,fixture=fixture,precision=precision,
        backend=f['backend'],source_hashes=b['sources'],input_hashes={'ids':d['ids_sha256'],'visual_fp32':d['fp32_visual_sha256']},
        base_fp32_state_sha256=f['a0_fp32_state_sha256'],candidate_fp32_state_sha256=f['a0_fp32_state_sha256'] if variant=='A0' else 'b'*64,
        state_hash_algorithm=b['state_hash_algorithm'],initialization_seed=0)

@pytest.mark.parametrize('fixture,indices',(('F0',(1,5)),('F1',(0,4)),('F2',(8,9))))
def test_frozen_literal_fixtures_and_golden_token_order(fixture,indices):
    ids,visual=c.load_fixture(fixture)
    golden=json.loads((ROOT/'tests/foundation/fixtures/golden_tokens.json').read_text())['prompt_cases']
    expected=np.array([golden[i]['ids'] for i in indices],dtype='<i8')
    assert np.array_equal(ids,expected)
    assert ids.shape==(2,308) and visual.shape==(2,1,512) and visual.dtype==np.float32
    assert not ids.flags.writeable and not visual.flags.writeable
    for row in ids:
        for segment in row.reshape(4,77):
            assert segment[0]==49406
            eos=np.flatnonzero(segment==49407)
            assert len(eos)==1 and np.all(segment[eos[0]+1:]==0)
    desc=c.bundle()['fixtures'][fixture]['fixture_description']
    historical=json.loads((ROOT/c.bundle()['fixtures'][fixture]['historical_source']).read_text())
    assert historical['fixture_description']==desc
    assert c.tensor_hash(visual)==desc['fp32_visual_sha256']
    views=c.fixture_views(fixture)
    for sample in (0,1):
        a,b=views[f'B1_sample{sample}']
        assert np.array_equal(a,ids[sample:sample+1]) and np.array_equal(b,visual[sample:sample+1])
        ci,cv=c.fixed_batch_change(ids,visual,sample)
        assert ci[sample,250]==1234 and np.array_equal(ci[1-sample],ids[1-sample])
        assert np.array_equal(cv[1-sample],visual[1-sample])
        assert np.array_equal(cv[sample],visual[sample]+np.float32(.5))

@pytest.mark.parametrize('fixture',('F0','F1','F2'))
def test_stage_schema_matches_existing_observations(fixture):
    old=json.loads((ROOT/c.bundle()['fixtures'][fixture]['historical_source']).read_text())['activations']
    expected=c.stages(fixture,2)
    assert len(expected)==(39 if fixture=='F0' else 27)
    assert set(expected)==set(old)
    assert all(old[n]['shape']==shape for n,shape in expected.items())
    for batch in (1,2):
        assert c.stages(fixture,batch)['output.logits']==[batch,308,49408]

@pytest.mark.parametrize('variant',c.VARIANTS)
def test_full_language_inventory_and_independent_common_copy(variant):
    ref=observation('A0','reference'); candidate=observation(variant)
    result=c.validate_ownership(candidate,variant)
    assert (result['parameter_tensors'],result['unique_elements'])==c.COUNTS[variant]
    v.validate_common_copy(ref,candidate,variant)
    v.validate_state_preservation(candidate,copy.deepcopy(candidate),variant)
    assert len(candidate['norms'])==(12 if variant=='A0' else 33)
    assert len([p for p in candidate['parameters'] if c.gradient_role(p['name'])=='allocated_exact_zero'])==38

@pytest.mark.parametrize('failure',('alias','tie','shape','missing','norm_position','norm_eps','block_order','vim_role','unknown','dtype'))
def test_ownership_negative_controls(failure):
    x=observation('P3')
    if failure=='alias': x['parameters'][1]['storage_id']=x['parameters'][0]['storage_id']
    elif failure=='tie': x['vocabulary_output_id']='copy'
    elif failure=='shape': x['parameters'][0]['shape']=[512]
    elif failure=='missing': x['parameters'].pop()
    elif failure=='norm_position': x['norms']['decoder.blocks.0.ln1']['placement']='post_residual_self_attention'
    elif failure=='norm_eps': x['norms']['decoder.blocks.0.ln1']['eps']=1e-6
    elif failure=='block_order': x['decoder_blocks'].reverse()
    elif failure=='vim_role': x['vim_instances'].pop()
    elif failure=='unknown': x['execute']=True
    else: x['parameters'][0]['dtype']='float64'
    with pytest.raises(c.InvalidEvidence): c.validate_ownership(x,'P3')

def test_source_preservation_and_exact_disabled_builder_binding():
    assert c.verify_sources(ROOT)['status']=='PASS'
    tree=ast.parse((ROOT/'src/mfvlr_research/decoder_a11.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NormalizedDecoderBlock')
    f=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    branch=f.body[0]
    expected_pre=['state + self.self_attention(self.ln1(state), causal=True)',
                  'a + self.memory_attention(self.ln2(a), memory)','b + self.ff(self.ln3(b))']
    expected_post=['self.ln1(state + self.self_attention(state, causal=True))',
                   'self.ln2(a + self.memory_attention(a, memory))','self.ln3(b + self.ff(b))']
    for body,expected in ((branch.body,expected_pre),(branch.orelse,expected_post)):
        assert len(body)==3
        for stmt,expr in zip(body,expected):
            assert ast.dump(stmt.value.args[0])==ast.dump(ast.parse(expr,mode='eval').body)
    assert ast.dump(f.body[-1].value.args[0])==ast.dump(ast.parse('c + self.vim(c, visual)',mode='eval').body)
    adapter_tree=ast.parse((ROOT/'src/mfvlr_research/d02b2/adapter.py').read_text())
    adapter=next(n for n in adapter_tree.body if isinstance(n,ast.ClassDef))
    for name in ('build','promoted_copy','execute'):
        method=next(n for n in adapter.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert isinstance(method.body[0],ast.Raise)
    build=next(n for n in adapter.body if isinstance(n,ast.FunctionDef) and n.name=='build')
    assert any(isinstance(n,ast.ImportFrom) and n.module=='mfvlr_research.decoder_a11' for n in build.body)
    assert ast.unparse(build.body[-1].value)=='build_variant(reference, self.variant)'

@pytest.mark.parametrize('variant',c.VARIANTS)
def test_adapter_metadata_and_all_execution_methods_refuse(variant):
    adapter=FrozenN2Adapter(variant)
    r=request(variant)
    assert adapter.prepare(r,c.seal(r),ROOT)['execution']=='NOT RUN'
    for method in (adapter.build,adapter.promoted_copy):
        with pytest.raises(c.ExecutionDisabled): method(object())
    with pytest.raises(c.ExecutionDisabled): adapter.execute(approve=True,execute=True)
    with pytest.raises(c.ExecutionDisabled): dispatch(approved=True,variant=variant)

@pytest.mark.parametrize('field',('unknown','source_missing','source_changed','input_missing','input_changed','backend','base_state','candidate_state','algorithm','seed','identity','seal'))
def test_provenance_negative_controls(field):
    r=request(); seal=c.seal(r)
    if field=='unknown': r['execute']=True
    elif field=='source_missing': r['source_hashes'].pop(next(iter(r['source_hashes'])))
    elif field=='source_changed': r['source_hashes'][next(iter(r['source_hashes']))]='f'*64
    elif field=='input_missing': del r['input_hashes']['ids']
    elif field=='input_changed': r['input_hashes']['visual_fp32']='e'*64
    elif field=='backend': r['backend']['num_threads']=1
    elif field=='base_state': r['base_fp32_state_sha256']='b'*64
    elif field=='candidate_state': r['candidate_fp32_state_sha256']=''
    elif field=='algorithm': r['state_hash_algorithm']='D02B1'
    elif field=='seed': r['initialization_seed']=False
    elif field=='identity': r['experiment_id']='G01'
    else: seal='0'*64
    with pytest.raises(c.InvalidEvidence): c.validate_request(r,seal)

def test_all_prepared_cells_stay_not_run():
    cells=c.schedule()
    assert len(cells)==len({r['experiment_id'] for r in cells})==18
    for row in cells:
        r=request(row['variant'],row['fixture'],row['precision'])
        assert c.validate_request(r,c.seal(r))['execution']=='NOT RUN'
    for bad in ('I1','R2','a0',None):
        with pytest.raises(c.InvalidEvidence): FrozenN2Adapter(bad)

def promotion(x):
    rows=[dict(p,dtype='float64',sha256='d'*64,expected_fp64_sha256='d'*64,roundtrip_fp32_sha256=p['sha256'],
               object_id='fp64-'+p['object_id'],storage_id='fp64-'+p['storage_id']) for p in x['parameters']]
    vocab=next(p for p in rows if p['name']=='vocabulary.weight')['object_id']
    return dict(parameters=rows,vocabulary_lookup_id=vocab,vocabulary_output_id=vocab)

@pytest.mark.parametrize('variant',c.VARIANTS)
def test_promoted_weight_receipts_and_small_array_mathematical_control(variant):
    x=observation(variant); y=promotion(x)
    v.validate_promotion(x,y,variant)
    original=np.array([.1,-.7,2**-20,20.],dtype=np.float32)
    before=original.copy(); promoted=copy.deepcopy(original).astype(np.float64)
    assert not np.shares_memory(original,promoted)
    assert np.array_equal(promoted.astype(np.float32),before)
    assert np.array_equal(original,before)
    receipt=v.exact_promotion_arrays(original,promoted)
    assert receipt['expected_fp64_sha256']==hashlib.sha256(promoted.tobytes()).hexdigest()
    promoted[0]+=1e-12
    assert np.array_equal(promoted.astype(np.float32),original)  # Insufficient weaker check.
    with pytest.raises(c.InvalidEvidence): v.exact_promotion_arrays(original,promoted)

@pytest.mark.parametrize('error',('reinitialized','alias','missing','tie','inexact64'))
def test_promotion_negative_controls(error):
    x=observation('Q3'); y=promotion(x)
    if error=='reinitialized': y['parameters'][0]['roundtrip_fp32_sha256']='e'*64
    elif error=='alias': y['parameters'][0]['object_id']=x['parameters'][0]['object_id']
    elif error=='missing': y['parameters'].pop()
    elif error=='tie': y['vocabulary_output_id']='independent-output'
    else: y['parameters'][0]['sha256']='e'*64
    with pytest.raises(c.InvalidEvidence): v.validate_promotion(x,y,'Q3')

@pytest.mark.parametrize('failure',('common_weights','norm_initialization','source_alias','state_mutation'))
def test_common_initialization_and_state_negative_controls(failure):
    ref=observation('A0','source'); candidate=observation('P3')
    if failure=='common_weights': candidate['parameters'][0]['sha256']='f'*64
    elif failure=='norm_initialization': next(p for p in candidate['parameters'] if p['name']=='decoder.blocks.0.ln1.weight')['sha256']='f'*64
    elif failure=='source_alias': candidate['parameters'][0]['storage_id']=ref['parameters'][0]['storage_id']
    else:
        before=copy.deepcopy(candidate); candidate['parameters'][0]['sha256']='f'*64
        with pytest.raises(c.InvalidEvidence): v.validate_state_preservation(before,candidate,'P3')
        return
    with pytest.raises(c.InvalidEvidence): v.validate_common_copy(ref,candidate,'P3')
