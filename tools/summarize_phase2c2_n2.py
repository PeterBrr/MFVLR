"""Summarize immutable N2 runs and generate an UNAPPLIED test-only proposal."""
from pathlib import Path
import difflib
import hashlib
import json
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2c2_n2'


def write(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


REPLACEMENT='''def test_flt_t17_cross_batch_numerical_diagnostic(flt,language_inputs,root):
    # N2 PROPOSED: apply only after explicit approval. FP32 constants unchanged.
    import tempfile
    from tools.review_phase2c2_n2 import forward_capture,compare,backend,write
    ids,v=language_inputs
    before=[id(p) for p in flt.parameters()]
    batch=forward_capture(flt,ids,v)
    expected={'vocabulary','encoder.input','decoder.input'}
    expected.update(f'encoder.blocks.{i}' for i in range(12))
    expected.update(f'decoder.blocks.{i}' for i in range(7))
    expected.update('output.'+name for name in
                    ('low','encoder_hidden','global_language','decoder_hidden','logits'))
    assert set(batch)==expected
    assert batch['output.logits'].shape==(2,308,49408)
    records=[]
    for sample in (0,1):
        single=forward_capture(flt,ids[sample:sample+1],v[sample:sample+1])
        assert set(single)==expected
        # compare raises on nonfinite inputs; an outside-tolerance count is evidence.
        records.append({'sample':sample,'stages':{name:compare(value[sample:sample+1],
                       single[name],**FP32) for name,value in batch.items()}})
        del single
    directory=root/'artifacts/phase2c2_n2_acceptance'
    directory.mkdir(parents=True,exist_ok=True)
    destination=Path(tempfile.mkdtemp(prefix='t17_',dir=directory))/'cross_batch.json'
    evidence={'policy':'N2','purpose':'mandatory FP32 diagnostic, not equality assertion',
              'backend':backend(),'atol':FP32['atol'],'rtol':FP32['rtol'],
              'N1_applied':False,'cross_batch':records}
    write(destination,evidence)
    assert json.loads(destination.read_text())==evidence
    assert [id(p) for p in flt.parameters()]==before
    assert flt.vocabulary_output_weight is flt.vocabulary.weight
    print('N2_DIAGNOSTIC:',destination)


def test_flt_t17_fixed_batch_permutation(flt,language_inputs):
    from tools.review_phase2c2_n2 import forward_capture
    ids,v=language_inputs
    permutation=torch.tensor([1,0])
    baseline=forward_capture(flt,ids,v)
    reordered=forward_capture(flt,ids[permutation],v[permutation])
    assert set(baseline)==set(reordered)
    for name,value in baseline.items():
        assert torch.equal(value,reordered[name][permutation]),name


def test_flt_t17_bidirectional_cross_sample_gradients(flt,language_inputs):
    from tools.review_phase2c2_n2 import cross_sample_gradients
    ids,v=language_inputs
    records=cross_sample_gradients(flt,ids,v)
    assert len(records)==2
    for record in records:
        assert record['other_visual_gradient_exact_zero']
        assert record['other_lookup_gradient_exact_zero']
        assert record['own_visual_gradient']['finite']
        assert record['own_lookup_gradient']['finite']
        assert record['own_visual_gradient']['l2_norm']>0
        assert record['own_lookup_gradient']['l2_norm']>0


def test_flt_t17_promoted_weight_fp64_cross_batch(flt,language_inputs):
    from tools.review_phase2c2_n2 import forward_capture
    ids,v=language_inputs
    # Deep copy preserves source state/parameter identity; no new initialization.
    reference=copy.deepcopy(flt).double()
    v64=v.double()
    for actual,original in zip(reference.parameters(),flt.parameters()):
        assert torch.equal(actual.float(),original)
    batch=forward_capture(reference,ids,v64)
    for sample in (0,1):
        single=forward_capture(reference,ids[sample:sample+1],v64[sample:sample+1])
        assert set(single)==set(batch)
        for name,value in batch.items():
            torch.testing.assert_close(value[sample:sample+1],single[name],atol=1e-10,rtol=1e-8)
        del single
    del batch,reference
    gc.collect()
'''


def main():
    cases=[json.loads(p.read_text()) for p in sorted((OUT/'run_01').glob('fixture*.json'))]
    assert len(cases)==7 and all(c['status']=='COMPLETED' for c in cases)
    hashes={c['fp32_initialized_state_sha256'] for c in cases}
    assert len(hashes)==1
    for i in range(3):
        pair=[c for c in cases if c['fixture']==i and c['backend']['num_threads']==4]
        assert len(pair)==2
        for key in ('ids_sha256','fp32_visual_sha256'):
            assert pair[0]['fixture_description'][key]==pair[1]['fixture_description'][key]
    preservation=json.loads((OUT/'baseline_preservation.json').read_text())
    mismatches=[name for name,expected in preservation['existing_files_sha256'].items()
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected]
    assert not mismatches,mismatches
    write('preservation_after.json',{'prior_phase2c2_payloads_verified':242,
        'existing_files_verified':len(preservation['existing_files_sha256']),'mismatches':mismatches})
    results=[]
    for c in cases:
        row={'fixture':c['fixture'],'dtype':c['dtype'],'threads':c['backend']['num_threads'],
             'same_state_hash':c['fp32_initialized_state_sha256'],
             'parameter_count':c['parameter_report']['total_unique'],
             'parameter_objects_unchanged':c['parameter_objects_unchanged'],
             'state_unchanged':c['state_unchanged_after_probes'],'vocabulary_tie':c['vocabulary_tie_identity'],
             'fixed_independence_exact':all(all(r['exact_equality'].values()) for r in c['fixed_batch_independence']),
             'permutation_exact':all(r['exact_equal'] for r in c['permutation'].values()),
             'all_stages_cross_batch_within_existing_tolerance':all(r['allclose'] for s in c['cross_batch'] for r in s['stages'].values()),
             'logit_comparisons':[s['stages']['output.logits'] for s in c['cross_batch']],
             'LE_last':c['activations']['output.encoder_hidden'],
             'LD_last':c['activations']['output.decoder_hidden']}
        if c['dtype']=='float32':
            b=c['full_backward']
            row.update(all_parameter_gradients_finite=not b['missing_gradients'] and all(v['finite'] for v in b['parameter_gradients'].values()),
                all_38_vim_qk_allocated_zero=len(b['vim_qk_allocated_exact_zero'])==38 and all(b['vim_qk_allocated_exact_zero'].values()),
                cross_sample_gradients_exact_zero=all(g['other_visual_gradient_exact_zero'] and g['other_lookup_gradient_exact_zero'] for g in c['cross_sample_gradients']),
                full_parameter_gradient_l2=b['parameter_gradient_global_l2'])
        else:row['LD_directional_gains']=[e['stages'][-1]['rms_gain_from_actual_input_perturbation'] for e in c['decoder_perturbation']['experiments']]
        results.append(row)
    write('numerical_summary.json',{'status':'REVIEW COMPLETE, NOT ACCEPTANCE','N2_status':'PROPOSED',
        'N2_implementation':'BLOCKED pending explicit user approval','policy_applied':False,
        'fixtures':3,'workers':7,'cases':results,'same_fp32_initialized_weights_in_all_workers':True,
        'nonfinite_valid_probe_outputs_or_gradients':False,'GPU':'NOT RUN; CPU-only environment',
        'independent_linux_evidence':'User-reported corroboration; raw Linux logs not supplied or merged into local measurements'})
    source=ROOT/'tests/language/test_flt.py'
    original=source.read_text(encoding='utf-8')
    start=original.index('def test_flt_t17_cross_batch_original_tolerance(')
    end=original.index('\n\ndef test_flt_t18_serialization(',start)
    proposed=original[:start]+REPLACEMENT+original[end:]
    patch=''.join(difflib.unified_diff(original.splitlines(True),proposed.splitlines(True),
        fromfile='a/tests/language/test_flt.py',tofile='b/tests/language/test_flt.py'))
    (OUT/'proposed_t17.patch').write_text(patch,encoding='utf-8')
    # This compiles text for syntax only. It is neither written into tests nor executed.
    compile(proposed,'UNAPPLIED_N2_TEST_PROPOSAL','exec')
    write('proposal_status.json',{'proposal':'N2','status':'PROPOSED','implementation_status':'BLOCKED',
        'approval_required':'Explicit user approval','applied':False,'full_phase2c2_acceptance':'BLOCKED',
        'existing_test_file_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'proposed_patch':'artifacts/phase2c2_n2/proposed_t17.patch','patch_applied':False,
        'proposal_syntax_checked':True,'proposed_tests_executed':False,
        'before':{'complete_cases':161,'passed':160,'failed':1,'source':'artifacts/phase2c2/junit/complete_03.xml'},
        'after_if_approved':{'expected_collection_cases':164,'actual_execution':'NOT RUN',
            'required_full_suite_result':'164 passed, zero failed/errors/skips/xfails/xpasses',
            'test_replaced':'FP32 universal cross-batch allclose -> mandatory recorded diagnostic',
            'test_additions':['exact fixed-shape batch permutation','both-direction zero cross-sample gradients',
                              'full promoted-weight FP64 cross-batch check at existing FP64 tolerances']},
        'unchanged':['all production source and architecture','all 119 previously accepted regression cases',
            'existing fixed-batch independence test','FP32 tolerance constants','FP64 tolerance constants',
            'MVE N1 scope','all other tests and thresholds','runtime/scoped configurations'],
        'future_configuration_policy':'Record approved N2 separately as a verification-policy amendment; no runtime topology/tolerance schema edit.'})
    tree=ET.parse(OUT/'verification_01/pytest.xml').getroot()
    suite=tree.find('testsuite')
    write('targeted_test_summary.json',dict(suite.attrib,proposal_applied=False,
        explanation='Unchanged T17 intentionally remains a real failure; no skips or expected-failure overrides.'))
    print(json.dumps({'preserved_files':len(preservation['existing_files_sha256']),
        'review_cases':len(cases),'proposal':'PROPOSED / BLOCKED','patch_applied':False},indent=2))


if __name__=='__main__':main()
