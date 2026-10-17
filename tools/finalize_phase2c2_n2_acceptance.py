"""Validate actual N2 results, compare immutable references, and package review evidence."""
from pathlib import Path
import csv
import difflib
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2c2_n2_acceptance'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def junit(path):
    tree=ET.parse(path)
    cases=list(tree.iter('testcase'))
    return {'cases':len(cases),'passed':sum(not any(c.tag in ('failure','error','skipped') for c in x) for x in cases),
            **{name:len(list(tree.iter(tag))) for name,tag in [('failed','failure'),('errors','error'),('skipped','skipped')]},
            'seconds':sum(float(x.attrib['time']) for x in tree.iter('testsuite')),
            'case_ids':[[x.attrib.get('classname'),x.attrib['name']] for x in cases]}


def summarize():
    full=junit(OUT/'junit/complete.xml');targeted=junit(OUT/'junit/new_t17.xml')
    assert full['cases']==full['passed']==164 and full['failed']==full['errors']==full['skipped']==0
    assert targeted['cases']==targeted['passed']==5
    old=junit(ROOT/'artifacts/phase2c1/junit/complete_01.xml')
    assert old['cases']==119 and all(x in full['case_ids'] for x in old['case_ids'])
    stdout=(OUT/'logs/complete.stdout.log').read_text()
    assert '164 passed' in stdout and 'xfail' not in stdout.lower() and 'xpass' not in stdout.lower()
    assert 'deselected' not in stdout.lower()
    audit=read(OUT/'helper_audit.json');assert audit['status']=='PASS' and all(audit['checks'].values())
    original=(OUT/'history/test_flt_before_n2.py.txt').read_text().splitlines(keepends=True)
    current=(ROOT/'tests/language/test_flt.py').read_text().splitlines(keepends=True)
    actual=''.join(difflib.unified_diff(original,current,fromfile='a/tests/language/test_flt.py',tofile='b/tests/language/test_flt.py'))
    assert actual==(OUT/'history/proposed_t17.patch').read_text()
    baseline=read(OUT/'baseline_manifest.json')
    changed=[name for name,digest in baseline.items() if not (ROOT/name).exists() or sha(ROOT/name)!=digest]
    assert changed==['tests/language/test_flt.py'],changed
    assert sha(OUT/'history/test_flt_before_n2.py.txt')==baseline['tests/language/test_flt.py']
    prior_manifest=read(ROOT/'artifacts/phase2c1/manifest.json')['files_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in prior_manifest.items())
    preservation={'existing_files_checked':len(baseline),'only_authorized_modified_file':changed,
        'original_test_archived_hash_verified':True,'approved_patch_exact':True,
        'all_existing_source_configs_helpers_logs_manifests_preserved':True,
        'phase2c1_payloads_unchanged':len(prior_manifest),'accepted_119_case_ids_present_and_passing':True,
        'historical_integration_logs_restored_byte_for_byte':True,
        'fresh_integration_measurements':{p.name:{'new_sha256':sha(p),
            'historical_sha256':sha(OUT/'history/gradients'/p.name),
            'identical':p.read_bytes()==(OUT/'history/gradients'/p.name).read_bytes()}
            for p in (OUT/'integration_gradients').glob('*.json')}}
    write(OUT/'preservation.json',preservation)
    rows=[];comparisons=[];changes=[];gradients=[];traces=[]
    numeric_keys=('fixture_description','fp32_initialized_state_sha256','parameter_report','activations','cross_batch',
        'permutation','fixed_batch_independence','projection_only_cross_batch','full_backward','cross_sample_gradients',
        'decoder_perturbation','parameter_objects_unchanged','vocabulary_tie_identity','state_unchanged_after_probes')
    for fixture in range(3):
        for dtype in ('float32','float64'):
            filename=f'fixture{fixture}_{dtype}_threads4.json'
            fresh=read(OUT/'diagnostics'/filename)
            prior=read(ROOT/'artifacts/phase2c2_n2/run_01'/filename)
            assert fresh['status']=='COMPLETED'
            assert fresh['parameter_report']['total_unique']==112758272
            assert fresh['parameter_report']['unique_storage_elements']==112758272
            assert fresh['parameter_report']['vim_instances']==19
            for key in ('parameter_objects_unchanged','vocabulary_tie_identity','state_unchanged_after_probes'):assert fresh[key]
            assert all(m['exact_equal'] for m in fresh['permutation'].values())
            assert all(all(item['exact_equality'].values()) for item in fresh['fixed_batch_independence'])
            differences=[key for key in numeric_keys if fresh.get(key)!=prior.get(key)]
            backend_equal=fresh['backend']==prior['backend']
            comparison={'fixture':fixture,'dtype':dtype,'same_scientific_measurements':not differences,
                        'changed_measurement_sections':differences,'backend_identical':backend_equal,
                        'reference_sha256':sha(ROOT/'artifacts/phase2c2_n2/run_01'/filename),
                        'fresh_sha256':sha(OUT/'diagnostics'/filename),'elapsed_seconds':fresh['elapsed_seconds']}
            comparisons.append(comparison)
            if differences or not backend_equal:changes.append(comparison)
            for sample in fresh['cross_batch']:
                for name,metric in sample['stages'].items():
                    if dtype=='float64':assert metric['allclose'],(fixture,name)
                    rows.append({'fixture':fixture,'dtype':dtype,'sample':sample['sample'],'stage':name,
                        **metric,'outside_tolerance_proportion':metric['outside_tolerance_count']/metric['elements'],
                        'reference_measurements_identical':metric==prior['cross_batch'][sample['sample']]['stages'][name]})
            if dtype=='float32':
                backward=fresh['full_backward']
                assert not backward['missing_gradients']
                assert len(backward['parameter_gradients'])==387
                assert all(x['finite'] for x in backward['parameter_gradients'].values())
                assert len(backward['vim_qk_allocated_exact_zero'])==38 and all(backward['vim_qk_allocated_exact_zero'].values())
                assert all(x['other_lookup_gradient_exact_zero'] and x['other_visual_gradient_exact_zero'] for x in fresh['cross_sample_gradients'])
                gradients.append({'fixture':fixture,'parameter_report':fresh['parameter_report'],
                    'full_backward':backward,'cross_sample_gradients':fresh['cross_sample_gradients'],
                    'parameter_identities_unchanged':True,'vocabulary_tie_identity':True,
                    'initialization_state_sha256':fresh['fp32_initialized_state_sha256'],
                    'activation_maxima_changed_from_reference':fresh['activations']!=prior['activations'],
                    'gradient_norms_changed_from_reference':backward!=prior['full_backward']})
            traces.append({'fixture':fixture,'dtype':dtype,'activations':fresh['activations']})
    write(OUT/'numerical_comparison.json',{'reference':'artifacts/phase2c2_n2/run_01',
        'fresh':'artifacts/phase2c2_n2_acceptance/diagnostics','comparisons':comparisons,'changes_requiring_review':changes,
        'all_scientific_measurements_identical':not changes,'stages':rows})
    write(OUT/'parameter_gradient_preservation.json',{'FLT':112758272,'MVE':32078912,'VD':9055365,
        'visual':41134277,'visual_plus_FLT':153892549,'paper_reported_approximate':141000000,
        'discrepancy_not_corrected':12892549,'production_source_unchanged':True,'fixtures':gradients,
        'visual_counts_and_sharing_verified_by_complete_suite':True})
    write(OUT/'activation_traces.json',traces)
    with (OUT/'stage_differences.csv').open('w',newline='',encoding='utf-8') as stream:
        columns=['fixture','dtype','sample','stage','elements','max_absolute_difference',
                 'max_relative_difference_nonzero_reference','zero_reference_nonzero_difference_count',
                 'outside_tolerance_count','outside_tolerance_proportion','relative_l2_difference',
                 'rms_absolute_difference','reference_measurements_identical']
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    write(OUT/'verification.json',{'status':'PASS' if not changes else 'BLOCKED: numerical evidence changed; user review required',
        'complete_suite':full,'targeted_t17':targeted,'helper_audit':'PASS',
        'regression_cases_unchanged_and_passing':119,'all_scientific_measurements_identical':not changes,
        'N2_approved_and_applied':True,'final_user_acceptance':'PENDING','GPU':'NOT RUN: CPU-only torch',
        'batch8':'NOT RUN: outside this acceptance run','no_skips_xfails_xpasses':True})
    assert not changes,'STOP: fresh numerical evidence changed; review required before acceptance reporting'
    # Verify mandatory pytest F0 evidence agrees with the independent fresh worker.
    reference=read(OUT/'diagnostics/fixture0_float32_threads4.json')
    test_diagnostics=[]
    for path in OUT.glob('t17_*/cross_batch.json'):
        evidence=read(path)
        assert evidence['backend']==reference['backend']
        for sample in evidence['cross_batch']:
            for name,metric in sample['stages'].items():
                assert metric==reference['cross_batch'][sample['sample']]['stages'][name]
        test_diagnostics.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'matches_fresh_F0':True})
    assert len(test_diagnostics)==2
    write(OUT/'pytest_diagnostic_consistency.json',test_diagnostics)
    render_report(full,targeted,preservation,rows,gradients,comparisons)
    print(json.dumps({'suite':full['passed'],'all_measurements_match_reference':True,
                      'preserved_files':len(baseline)-1,'final_user_acceptance':'PENDING'}),flush=True)


def render_report(full,targeted,preservation,rows,gradients,comparisons):
    tables=['# Fresh N2 stage measurements','',
            'B=2 sample versus the identical sample at B=1. All values match the preserved F0/F1/F2 evidence exactly. '
            'FP32 classification uses unchanged atol=1e-6, rtol=1e-5; it is diagnostic. '
            'FP64 uses mandatory unchanged atol=1e-10, rtol=1e-8. Complete maximum-relative errors, '
            'zero-reference exceptions and all stages/precisions are in stage_differences.csv and numerical_comparison.json.','']
    for fixture in range(3):
        for sample in range(2):
            tables += [f'## F{fixture}, sample {sample}, FP32','',
                '| Stage | Max absolute difference | Outside / elements | Outside % | Relative L2 |',
                '|---|---:|---:|---:|---:|']
            for row in rows:
                if (row['fixture'],row['sample'],row['dtype'])!=(fixture,sample,'float32'):continue
                if row['stage'].startswith('encoder.blocks.0.') :continue
                tables.append(f"| {row['stage']} | {row['max_absolute_difference']:.10g} | "
                    f"{row['outside_tolerance_count']:,} / {row['elements']:,} | {row['outside_tolerance_proportion']*100:.6f} | "
                    f"{row['relative_l2_difference']:.10g} |")
            tables.append('')
    (OUT/'stage_comparison.md').write_text('\n'.join(tables)+'\n',encoding='utf-8')
    logit_lines=[]
    for row in rows:
        if row['dtype']=='float32' and row['stage']=='output.logits':
            logit_lines.append(f"| F{row['fixture']} / {row['sample']} | {row['max_absolute_difference']:.10g} | "
                f"{row['outside_tolerance_count']:,} ({100*row['outside_tolerance_proportion']:.4f}%) | "
                f"{row['relative_l2_difference']:.10g} |")
    act_lines=[]
    for item in gradients:
        fixture=item['fixture'];d=read(OUT/f'diagnostics/fixture{fixture}_float32_threads4.json')
        act_lines.append(f"| F{fixture} | {d['activations']['output.encoder_hidden']['max_abs']:.9g} | "
            f"{d['activations']['output.decoder_hidden']['max_abs']:.9g} | {d['activations']['output.logits']['max_abs']:.9g} | "
            f"{item['full_backward']['parameter_gradient_global_l2']:.10g} | {item['full_backward']['visual_gradient']['l2_norm']:.10g} |")
    report=f'''# Phase 2C.2 N2 acceptance verification

Date: 2026-10-11. N2 verification is complete: **164/164 mandatory cases pass**, with zero failures, errors, skips, xfails, xpasses or deselections. All **119 accepted regression cases** remain present, byte-preserved and passing. Final user acceptance remains pending this review. No subsequent component has started.

## Approval and exact change

The user approved only FLT-T17's universal FP32 cross-batch equality policy. The exact reviewed patch was applied to `tests/language/test_flt.py` after verifying baseline SHA-256 `484efd03bb95725e353c7c2257147f8ee2fe8914d1d5d98abcbf9f2a99c8e6d8`. New SHA-256: `e2d91f5431e64dc7754983bf473294916837ddcd13bd37ade1a8d4d7e9c40184`. The reconstructed unified diff equals the approved patch exactly. Its literal historical “N2 PROPOSED” comment is retained; the [scoped approval record](../artifacts/phase2c2_n2_acceptance/approval.json) supersedes that comment. No architecture or full-model configuration approval was added.

The old failing test is archived verbatim at [test_flt_before_n2.py.txt](../artifacts/phase2c2_n2_acceptance/history/test_flt_before_n2.py.txt). Original failed-run logs, numerical review, proposal, manifests and ZIPs remain unchanged. {preservation['existing_files_checked']-1} pre-existing files match their captured hashes; only the authorized test file changed. All {preservation['phase2c1_payloads_unchanged']} Phase 2C.1 manifest payloads match. Integration tests write historical log paths: originals were archived before execution, fresh results copied to the new evidence directory, and original bytes restored. [Preservation audit](../artifacts/phase2c2_n2_acceptance/preservation.json).

## Before and after acceptance policy

| Requirement | Before N2 | After N2 |
|---|---|---|
| Complete suite | 160 passed / 161, T17 failed | 164 passed / 164 |
| FP32 cross-batch allclose | Mandatory; failed | Mandatory complete diagnostic; allclose is recorded, not asserted |
| FP32 diagnostic constants | atol=1e-6, rtol=1e-5 | Unchanged |
| Fixed-batch independence | Exact equality and zero cross-sample visual gradient | Preserved unchanged |
| Batch permutation | Numerical review evidence | Mandatory exact equality for all 27 stages/outputs |
| Cross-sample gradients | Numerical review evidence | Mandatory exact zero in both directions for visual and differentiable lookup tensors |
| Full promoted-weight FP64 | Numerical review evidence | Mandatory deep copy, all stages and both samples; atol=1e-10, rtol=1e-8 |
| Finiteness, full dimensions, ownership and ties | Mandatory | Mandatory, unchanged |
| MVE N1 | MVE-only | MVE-only; never applied to FLT |

The targeted T17 run passed 5/5 cases (JUnit duration {targeted['seconds']:.2f}s; pytest stdout 26.53s). The complete suite passed (JUnit duration {full['seconds']:.2f}s; pytest stdout 125.70s; subprocess wall time 130.11s). These timers cover different execution boundaries. All VIM and FLT tests remain active. Full FP32 forward and backward at batches 1 and 2, serialization, configuration consistency, reconstruction-gradient integration, parameter counts and ownership passed. Exact commands and subprocess durations are in [execution records](../artifacts/phase2c2_n2_acceptance/logs/complete.execution.json); raw [stdout](../artifacts/phase2c2_n2_acceptance/logs/complete.stdout.log), [stderr](../artifacts/phase2c2_n2_acceptance/logs/complete.stderr.log) and [JUnit XML](../artifacts/phase2c2_n2_acceptance/junit/complete.xml) are preserved. No test threshold was raised. The targeted run's 24 deselections are limited to that preliminary run; the complete run has none.

## Diagnostic helper audit

The unchanged `tools/review_phase2c2_n2.py` was inspected and exercised, including negative controls; [audit evidence](../artifacts/phase2c2_n2_acceptance/helper_audit.json) and stdout/stderr are included.

- `forward_capture`: exactly 27 required captures: lookup, LE/LD inputs, 12 LE blocks, seven LD blocks and all five output fields. Shapes and finiteness are checked. A full forward runs under `torch.no_grad()`; stored tensors have no `grad_fn` or `requires_grad`. Hooks are removed. F0 detailed workers additionally capture 12 first-block primitives (39 total).
- `compare`: asserts shape/dtype equality; rejects NaN/Inf; visits all elements in bounded chunks. Original-dtype `torch.isclose` supplies mismatch counts. FP64 accumulation supplies max absolute error, relative errors without epsilon clamping, zero-reference exceptions, RMS and relative L2. Outside proportions are explicitly added in the comparison JSON/CSV. Known analytical values and shape/dtype/nonfinite rejection controls passed.
- `cross_sample_gradients`: `output.low` is the actual differentiable vocabulary lookup (verified `EmbeddingBackward0`), not detached IDs or a fabricated leaf. Both sample directions are tested. Own lookup/visual gradients are finite and nonzero, other-sample gradients exactly zero. Graph retention is only needed between the two backward probes; no parameter gradients are accumulated by this helper.
- `backend`: actual CPU, thread counts, deterministic flags, dtype policy, Torch commit/build, MKL/MKLDNN and environment are recorded and checked against runtime.
- `write`: strict JSON (`allow_nan=False`) round-trips exactly and rejects NaN. No required model output is silently omitted: missing keys/fields, wrong shapes, nonfinite tensors or broken identities fail mandatory checks.
- FP64 pytest deep-copies the live FP32 model and asserts every promoted parameter equals its original FP32 value; no independent FP64 initialization. All captured outputs for both samples pass unchanged FP64 tolerances. The six numerical workers instead promote identically hashed seed-0 FP32 values, matching the earlier review protocol. Their immutable historical `PROPOSED` metadata describes that review routine; current policy is recorded separately in [diagnostic provenance](../artifacts/phase2c2_n2_acceptance/diagnostics/provenance.json).

## Fresh numerical evidence against F0/F1/F2

Six full diagnostic workers ran **after** the complete suite, one FP32/FP64 pair per fixture. Every scientific measurement in each fresh worker matches the corresponding preserved four-thread reference exactly, including all per-stage differences, activation statistics, gradient norms, perturbation results and backend metadata. Only elapsed times differ. Both pytest diagnostic JSONs also match the fresh F0 worker for all 27 stages. No changed numerical evidence requires escalation.

Fixture provenance: F0 uses golden prompt indices 1/5 and FP32 normal visual seed 341, std 0.1; F1 uses indices 0/4 and seed 642, std 0.1; F2 uses indices 8/9 and the accepted untrained MVE+VD fused visual feature from random image seed 412. All use the same seed-0 FLT FP32 state hash `e4233731564ba462fc32423bb4832e996f96331d2bf21509addad8bd0cffaed2`. Literal input IDs, visual values and hashes are preserved per worker. FP64 promotes those identical FP32 fixture values. Backend: Windows 11 AMD64, Python 3.14.3, Torch 2.12.1+cpu, commit `7269437d655783a26cba32aa88195b741ff496aa`, intra/inter-op threads 4/4, deterministic algorithms enabled, highest FP32 matmul precision, MKL 2026.0 / MKLDNN 3.11.2. Exact build strings are in every worker JSON. GPU **NOT RUN** (CPU-only Torch); batch 8 **NOT RUN** in this acceptance run.

Per-stage tables for all LE/LD blocks, embeddings, global feature and vocabulary outputs for **both samples and all fixtures**: [stage_comparison.md](../artifacts/phase2c2_n2_acceptance/stage_comparison.md). Complete FP32/FP64 max absolute/relative differences, outside counts/proportions, relative L2 and zero-reference handling: [numerical_comparison.json](../artifacts/phase2c2_n2_acceptance/numerical_comparison.json) and [CSV](../artifacts/phase2c2_n2_acceptance/stage_differences.csv).

Each logit comparison contains 15,217,664 elements:

| Fixture / sample | FP32 max absolute difference | Outside unchanged tolerance | Relative L2 |
|---|---:|---:|---:|
{chr(10).join(logit_lines)}

The embedding lookup and initial position sums remain exact across batch sizes. Earlier evidence of first-block singleton VIM K/V projection rounding is reproduced; divergence accumulates through subsequent LE/LD blocks. Vocabulary projection with an identical hidden tensor remains exact in the isolated batch-shape control, so observed logit drift includes upstream hidden-state drift. Code inspection finds no feature-producing cross-sample reduction or attention across samples. Input-validity and finiteness checks do reduce across the input to accept or reject a call; they do not mix feature values. Exact independence, permutation and zero cross-sample gradients confirm isolation on these valid fixtures. These findings support a batch-shape-dependent FP32 computation explanation, not sample leakage. They do not establish universal numerical safety.

## Stability and parameter preservation

| Fixture | Final LE max abs | Final LD max abs | Logit max abs | All-parameter gradient L2 | Visual gradient L2 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(act_lines)}

All 387 FLT parameter tensors receive finite gradients for each controlled full-graph probe. All 38 VIM Q/K projection gradients are allocated and exactly zero, as required by singleton attention. Stage/parameter gradient norms, activation norms/maxima and both-direction cross-sample probes are in [parameter_gradient_preservation.json](../artifacts/phase2c2_n2_acceptance/parameter_gradient_preservation.json) and [activation_traces.json](../artifacts/phase2c2_n2_acceptance/activation_traces.json). No activation maximum or gradient norm changed from preserved evidence. No nonfinite activation or gradient was observed in valid full-graph probes; deliberate NaN/Inf helper rejection controls passed.

Counts remain MVE **32,078,912**, VD **9,055,365**, visual **41,134,277**, FLT **112,758,272**, combined **153,892,549**. Shared vocabulary identity, 19 independent VIM instances, unique storage and parameter objects remain intact. The approximately **12.89M** excess over the paper's approximately 141M is still an architectural reproduction uncertainty; dimensions were not adjusted.

The no-LayerNorm LD baseline still amplifies perturbations. Preserved and fresh FP64 fixed-batch probes give final LD RMS gains about 46.79 (F0), 36.53 (F1), and 804.53–840.84 (F2), with memory/visual inputs fixed. This measures the implemented LD's sensitivity; no normalization counterfactual was run, so it does not isolate normalization as the sole cause. FP32 F2 logit drift reaches 0.1786499 and approximately 55.82% outside the old elementwise tolerance for sample 0. Diagnostic success is **not proof that this drift is harmless**. Training, task accuracy, optimization stability and generalization were not tested or authorized. No normalization, clipping, AMP, scaling, attention change or architecture adjustment was introduced.

## Reproduction and review package

See [REPRODUCE.md](../artifacts/phase2c2_n2_acceptance/REPRODUCE.md), [SHA-256 manifest](../artifacts/phase2c2_n2_acceptance/manifest.json), and [review ZIP](../artifacts/phase2c2_n2_acceptance/phase2c2_n2_review_package.zip). The package includes actual production/test source, all scoped configurations, tokenizer artifacts/dependency pins, preserved historical failing evidence, the exact patch and archived failing source, fresh raw logs/JUnit, helper audit and full diagnostic evidence. No pretrained weights are included. Existing source/configuration/tests outside this one approved patch were preserved. Verification is complete; final acceptance awaits the user's review.
'''
    (ROOT/'docs/phase2c2_n2_acceptance_report.md').write_text(report,encoding='utf-8')


def package():
    assert read(OUT/'verification.json')['status']=='PASS'
    # Historical manifest payloads already include tokenizer/BPE assets, pins,
    # golden fixtures, old failed runs and source. Add review and acceptance data.
    files=set(read(ROOT/'artifacts/phase2c2/manifest.json')['files_sha256'])
    files.update(read(ROOT/'artifacts/phase2c2_n2/manifest.json')['files_sha256'])
    files.update(['artifacts/phase2c2/manifest.json','artifacts/phase2c2_n2/manifest.json',
                  'artifacts/phase2c2_n2/verification.json','docs/phase2c2_n2_acceptance_report.md',
                  'tools/accept_phase2c2_n2.py','tools/finalize_phase2c2_n2_acceptance.py'])
    files.update(p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file()
                 and p.name not in ('manifest.json','package_verification.json','phase2c2_n2_review_package.zip')
                 and '__pycache__' not in p.parts)
    # Nested earlier ZIPs are preserved on disk, not redundantly embedded.
    files={name for name in files if not name.endswith('.zip')}
    manifest={'phase':'2C.2 N2 acceptance verification','N2_approved_and_applied':True,
              'suite':'164/164 PASS','final_user_acceptance':'PENDING','files_sha256':{name:sha(ROOT/name) for name in sorted(files)},
              'excluded_self_referential_outputs':['manifest.json','package_verification.json','phase2c2_n2_review_package.zip'],
              'historical_manifests':'Preserved as historical records; original T17 source is archived under history, not the current test path.'}
    write(OUT/'manifest.json',manifest)
    files.add('artifacts/phase2c2_n2_acceptance/manifest.json')
    path=OUT/'phase2c2_n2_review_package.zip'
    assert not path.exists(),'Never overwrite a review ZIP'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in sorted(files):
            info=zipfile.ZipInfo(name,(2026,10,11,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            archive.writestr(info,(ROOT/name).read_bytes())
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        for name,digest in manifest['files_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest,name
        assert hashlib.sha256(archive.read('artifacts/phase2c2_n2_acceptance/manifest.json')).hexdigest()==sha(OUT/'manifest.json')
    result={'status':'PASS','archive_sha256':sha(path),'archive_bytes':path.stat().st_size,
            'payloads':len(manifest['files_sha256']),'entries':len(files),'CRC_and_all_payload_hashes_verified':True,
            'manifest_sha256':sha(OUT/'manifest.json')}
    write(OUT/'package_verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('summarize','package'))
    if parser.parse_args().stage=='summarize':summarize()
    else:package()
