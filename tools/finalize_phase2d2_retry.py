"""Publish fresh retry outcomes honestly and preserve the previous review."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_retry'
OLD=ROOT/'artifacts/phase2d2'
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)]
from tools.finalize_phase2d1_review import suite,sha,read,write


def scientific(value):
    if isinstance(value,dict):
        result={k:scientific(v) for k,v in value.items() if k not in ('elapsed_seconds','seconds')}
        if 'selected_loss_gradient_pair_cosines' in result:
            # This diagnostic is a set of parameter/loss-pair entries emitted
            # from a Python set; its serialization order has no tensor meaning.
            result['selected_loss_gradient_pair_cosines']=sorted(result['selected_loss_gradient_pair_cosines'],key=lambda r:(r['parameter'],tuple(r['losses'])))
        return result
    if isinstance(value,list):return [scientific(v) for v in value]
    return value


def main():
    baseline=read(OLD/'baseline_manifest.json')
    assert all(sha(ROOT/p)==h for p,h in baseline.items())
    before=read(OUT/'history/pre_retry_manifest.json')
    changes=[p for p,h in before.items() if sha(ROOT/p)!=h]
    expected={'tests/joint/test_joint.py','tests/joint/test_readiness.py'}
    assert set(changes)==expected,changes
    original_payload=read(OLD/'manifest.json')['files_sha256']
    assert all(sha(ROOT/p)==h for p,h in original_payload.items() if p not in expected)
    for name in ('test_joint','test_readiness'):
        original=ast.parse((OUT/'history'/f'{name}.before_retry.py.txt').read_text())
        current=ast.parse((ROOT/'tests/joint'/f'{name}.py').read_text())
        assertions=lambda tree:[ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Assert)]
        assert assertions(original)==assertions(current)
        # All function definitions unchanged except I10's evidence path assignment.
        old_functions={n.name:n for n in original.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        new_functions={n.name:n for n in current.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        assert old_functions.keys()==new_functions.keys()
        for key in old_functions:
            if key=='test_joint_i10_fp64_oracle_and_full_fp32_gradient_additivity':
                for function in (old_functions[key],new_functions[key]):
                    function.body=[n for n in function.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='evidence' for t in n.targets))]
            assert ast.dump(old_functions[key],include_attributes=False)==ast.dump(new_functions[key],include_attributes=False),key
    write(OUT/'preservation.json',{'status':'PASS','accepted_baseline_files':len(baseline),
        'accepted_baseline_SHA256_all_match':True,'original_phase2d2_artifacts_byte_identical':True,
        'original_review_package_payload_except_documented_two_test_edits_matches':True,
        'pre_retry_files':len(before),'changed_files':changes,
        'changes':'Only unaccepted new joint tests: optional evidence-directory override and I10 evidence path. Assertions/function bodies otherwise unchanged.',
        'original_changed_test_sources_archived':True,'architecture_and_configuration_changes':False})
    complete=suite(OUT/'junit/complete_retry_01.xml')
    previous=suite(OLD/'junit/complete_01.xml')
    accepted=suite(ROOT/'artifacts/phase2d1/junit/complete_01.xml')
    assert set(map(tuple,complete['case_ids']))==set(map(tuple,previous['case_ids']))
    old_ids=set(map(tuple,accepted['case_ids']));old_passes=0;failures=[]
    for case in ET.parse(OUT/'junit/complete_retry_01.xml').getroot().findall('.//testcase'):
        key=(case.attrib['classname'],case.attrib['name'])
        node=case.find('failure')
        if node is None:node=case.find('error')
        if node is not None:failures.append({'id':list(key),'message':node.attrib.get('message',''),'traceback':node.text})
        elif key in old_ids:old_passes+=1
    assert old_passes==276 and complete['skipped']==0
    stdout=(OUT/'logs/complete_retry_01.stdout.log').read_text()
    assert not any(label in stdout for label in ('XFAIL','XPASS','SKIPPED','deselected'))
    status='BLOCKED' if failures else 'PASS_AWAITING_REVIEW'
    write(OUT/'verification.json',{'status':status,'complete_suite':complete,'old_complete_suite':previous,
        'accepted_regression_cases_passed':old_passes,'new_cases':complete['cases']-old_passes,
        'failed_cases':failures,'test_case_identities_unchanged':True,'no_skips_xfails_or_deselections':True,
        'tolerances_unchanged':True,'new_numerical_amendment':'NONE','N2_exempts_new_gradient_tests':False})
    comparison=[]
    rows=[]
    for case in ('real','efs','local','real_efs','real_local','local_duplicate'):
        old=OLD/'g2'/('final_'+case);new=OUT/'g2'/('final_'+case)
        summary=read(new/'summary.json');gradient=read(new/'gradient_additivity.json')
        assert summary['parameter_bytes_unchanged'] and summary['parameter_identities_unchanged']
        assert summary['all_valid_values_and_gradients_finite']
        same={}
        for loss in ('fd','lr','cmc','fl','ar','kl','total'):
            fresh=read(new/f'{loss}.json');prior=read(old/f'{loss}.json')
            same[loss]=scientific(fresh)==scientific(prior)
        same_gradient=scientific(gradient)==scientific(read(old/'gradient_additivity.json'))
        outside=sum(r.get('outside_tolerance_count',0) for r in gradient['parameters'].values())
        resources=read(new/'resources.json')
        total=read(new/'total.json')
        comparison.append({'case':case,'status':summary['status'],'parameter_bytes_unchanged':True,
            'parameter_identities_unchanged':True,'isolated_and_joint_records_scientifically_exact_equal':same,
            'gradient_additivity_scientifically_exact_equal':same_gradient,
            'outside_tolerance_count':outside,'relative_l2_difference':gradient['relative_l2_difference'],
            'LE_final_max':total['activations']['LE11']['max_abs'],
            'LD_final_max':total['activations']['LD6']['max_abs'],
            'sampled_peak_RSS_MiB':resources['sampled_peak_rss_bytes']/2**20,'seconds':resources['elapsed_seconds']})
        rows.append(f"| {case} | {outside:,} | {gradient['relative_l2_difference']:.9g} | {resources['sampled_peak_rss_bytes']/2**20:.2f} | {resources['elapsed_seconds']:.2f} |")
    write(OUT/'g2_comparison.json',{'runs':comparison,'comparison_policy':'Exact scientific values; timing excluded; only unordered selected cosine-entry list canonicalized by parameter/loss pair. Raw files unchanged.'})
    n2=[]
    for i in range(3):
        for dtype in ('float32','float64'):
            file=f'fixture{i}_{dtype}_threads4.json'
            fresh=read(OUT/'n2'/file);prior=read(OLD/'n2'/file)
            same=scientific(fresh)==scientific(prior)
            n2.append({'file':file,'status':fresh['status'],'all_scientific_fields_exact_equal':same})
    write(OUT/'n2_comparison.json',n2)
    ownership=read(OUT/'joint/I12.json')
    assert ownership['full']['total_unique']==154156232
    assert ownership['deployment']['total_unique']==41135303
    write(OUT/'parameter_ownership_serialization.json',ownership)
    c32=read(OUT/'chain/float32.json');c64=read(OUT/'chain/float64.json')
    assert c32['FP32_initialized_FLT_source_sha256']==c64['FP32_initialized_FLT_source_sha256']
    assert c32['FP32_visual_input_sha256']==c64['FP32_visual_input_sha256']
    with (OUT/'module_gradients.csv').open('w',newline='',encoding='utf-8') as output:
        writer=csv.writer(output);writer.writerow(['case','loss','module','allocated','l2','max_abs','nonzero'])
        for case in comparison:
            for loss in ('fd','lr','cmc','fl','ar','kl','total'):
                for name,g in read(OUT/'g2'/('final_'+case['case'])/f'{loss}.json')['gradients']['modules'].items():
                    writer.writerow([case['case'],loss,name,g['allocated_tensors'],g['l2'],g['max_abs'],g['nonzero_elements']])
    write(OUT/'approval.json',{'phase':'2D.2 retry','implementation_scope':'Existing approved full synthetic integration and G2 no-update only',
        'verification':status,'P2D15':'UNAPPROVED','P2D19':'UNAPPROVED','optimizer_updates':0,
        'new_numerical_policy_approved':False,'architecture_change_approved_or_applied':False})
    exact_g2=all(all(c['isolated_and_joint_records_scientifically_exact_equal'].values()) and c['gradient_additivity_scientifically_exact_equal'] for c in comparison)
    exact_n2=all(c['all_scientific_fields_exact_equal'] for c in n2)
    chain32=c32['gradient_comparisons']['fused_visual'];chain64=c64['gradient_comparisons']['fused_visual']
    bias=read(OUT/'key_bias_review.json')
    bias_rows='\n'.join(f"| {r['dtype']} | {r['cotangent_scale']:g} | {r['key_bias_gradient']['max_abs']:.12g} | {r['comparison_to_exact_real_zero']['outside_tolerance_count']} |" for r in bias['records'])
    report=f'''# Phase 2D.2 retry verification

**Status: {status}.** The fresh, unfiltered suite executed **{complete['cases']} cases: {complete['passed']} passed, {complete['failed']} failed, {complete['errors']} errors, {complete['skipped']} skips**. All **276 accepted cases pass unchanged**. No xfails, deselections, relaxed thresholds, architecture changes or optimizer updates. The original [report](phase2d2_implementation_report.md), failed logs, evidence, manifests and ZIP remain byte-identical.

The retry fixes a verification-harness limitation: the new joint tests can now read a separately selected evidence directory. Previously a pytest rerun alone would read the old G2 records. `MFVLR_PHASE2D2_EVIDENCE_ROOT` redirects only the two unaccepted Phase 2D.2 test modules; I10 and READY-N02 now consume the freshly recomputed evidence. Default paths remain the original ones. Every assertion AST and every test function apart from I10's file-path assignment is unchanged. Original test sources are archived, not discarded. Accepted production source and configurations are untouched.

## Fresh computation and preserved comparisons

Each of the six full-size CPU FP32 fixtures ran in a separate process, with six isolated fresh forward/backward graphs and one literal joint graph. All outputs, captured 12 LE/7 LD stages, loss scalars and gradients remain finite. Parameters and identities do not update; lifecycle checks pass. G2 status remains determined by the unchanged elementwise FP32 gradient-additivity assertion, not by finiteness alone.

| Fixture | Outside-tolerance elements | Relative L2 difference | Peak sampled RSS MiB | Wall seconds |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

All scientific values in the per-loss/joint records and gradient comparisons exactly match the earlier six-fixture evidence: **{exact_g2}**. Timing fields are excluded; only the unordered selected cosine-entry list is canonicalized by parameter/loss pair because its serialization order comes from a Python set. Tensor/stage/token arrays retain their original order. Raw evidence is unchanged. Original N2 F0/F1/F2 were also recomputed in FP32 and promoted-weight FP64; all scientific/backend fields exactly match: **{exact_n2}**, excluding elapsed timing only. Neither N1 nor N2 was extended to the new gradient criteria. Measurements include model construction, seven graphs, statistics, hashing and disk streaming; RSS is sampled, not an exact allocator peak. CPU B8/GPU are **NOT RUN** with explicit resource reasons, not successful capacity checks.

## Additional CMC numerical diagnosis

The independent retry probe uses the exact real/EFS fused visual input, 308-token prompts and all 12 accepted LE blocks. Both workers deep-copy the same FP32-initialized full FLT and CMC before optional FP64 promotion. Source-weight and FP32 visual-input hashes agree between workers. No alternate initialization, normalization, rescaling, clipping, detachment of production paths or pretrained weights is introduced.

At fused Iv, direct-plus-language CMC partial gradients versus the joint derivative have:

| Precision | Outside tolerance | Maximum absolute difference | Relative L2 difference | Existing tolerance |
|---|---:|---:|---:|---|
| FP32 | {chain32['outside_tolerance_count']} / 1,024 | {chain32['max_absolute_difference']:.12g} | {chain32['relative_l2_difference']:.12g} | atol 1e-6, rtol 1e-5 |
| Promoted FP64 | {chain64['outside_tolerance_count']} / 1,024 | {chain64['max_absolute_difference']:.12g} | {chain64['relative_l2_difference']:.12g} | atol 1e-10, rtol 1e-8 |

Selected ordinary LE1 attention-Q and LE12 FFN1 parameter derivatives equal their relevant language partials exactly in both precisions. These are ordinary attention projections, distinct from singleton VIM Q/K. This supports floating-point accumulation as the CMC chain-rule mismatch mechanism. It **does not prove full visual/LD gradient additivity in FP64, harmlessness of small-coordinate drift, repeated-step stability, convergence or accuracy**. The VD public contract permits FP32 only; no hidden FP64 visual wrapper or changed production API was used. Large LR gradients and unnormalized LD activation magnitudes remain recorded risks.

The CMC scalar itself changes from {c32['CMC_scalar']:.12g} in FP32 to {c64['CMC_scalar']:.12g} in promoted FP64. FP64 promotion re-evaluates forward arithmetic too; passing a high-precision chain-rule comparison is not a claim that the two precision modes give identical forward results.

## Ordinary attention key-bias cancellation probe

**DERIVED mathematics, not recovered paper architecture:** for the accepted biased ordinary attention, adding key bias adds `q_i^T b_k / sqrt(64)` to every key score in row i. Softmax is invariant to that common offset, so the exact-real key-bias task derivative cancels. This differs from the accepted singleton VIM whose Q/K derivatives are exact zero in the executed graph. Ordinary biased attention can leave floating-point cancellation residuals.

A separate full-width 512-dimensional, eight-head, 308-token attention block probe deep-copies identical FP32-initialized weights and input/cotangent fixtures, with scales 1 and 10,000. It exercises the accepted block, not a full-model loss or training step. No bias is removed or gradient overwritten.

| Precision | Cotangent scale | Maximum absolute key-bias gradient | Outside existing zero-comparison tolerance |
|---|---:|---:|---:|
{bias_rows}

[Machine-readable probe](../artifacts/phase2d2_retry/key_bias_review.json) includes score-gradient row sums, hashes and backend metadata. For the full real G2 fixture, ordinary LE3 key-bias joint gradient L2 is about 0.003992 and the isolated-sum discrepancy has relative L2 about 1.531; MVE block2 key-bias relative discrepancy is about 1.677. These near-cancelling derivatives explain why global relative-L2 agreement alone can obscure large coordinate-wise ratios. Other ordinary Q/weight discrepancies also remain; this probe does not explain away every failing parameter or grant a numerical exemption.

## Acceptance boundary and review files

Full root: **154,156,232** unique parameters. Physical visual deployment: **41,135,303**. Weight tying, ownership and serialization are reverified in the fresh suite. [Verification](../artifacts/phase2d2_retry/verification.json), [preservation](../artifacts/phase2d2_retry/preservation.json), [G2 comparison](../artifacts/phase2d2_retry/g2_comparison.json), [raw pytest log](../artifacts/phase2d2_retry/logs/complete_retry_01.stdout.log), [JUnit](../artifacts/phase2d2_retry/junit/complete_retry_01.xml), [FP32 CMC](../artifacts/phase2d2_retry/chain/float32.json), [FP64 CMC](../artifacts/phase2d2_retry/chain/float64.json), [review ZIP](../artifacts/phase2d2_retry/phase2d2_retry_review.zip).

The failed comparisons remain mandatory failures. No numerical policy amendment is inferred from “retry.” Phase 2D.2 remains blocked under the current criteria if these failures persist. P2D15/P2D19, dataset processing, optimizer/training execution and benchmark evaluation remain unapproved and unimplemented. Stop here for review.
'''
    (ROOT/'docs/phase2d2_retry_report.md').write_text(report,encoding='utf-8')
    (OUT/'REPRODUCE.md').write_text('''# Retry reproduction

Use the pinned runtime/dependencies recorded in artifacts/phase2d2/runtime_versions.json and the original REPRODUCE.md. No dependency substitution is authorized. Source checkout, Windows CPU environment used here.

Preserve existing retry evidence; for a further run choose a new OUT directory in tools/retry_phase2d2.py and archive that script version before running. Then execute `python tools/retry_phase2d2.py --all`. Each worker runs with four CPU threads and deterministic/highest FP32 settings. Driver exits with the actual complete-suite pytest code; known strict failures remain failures. Full run takes minutes and uses multi-GiB RAM plus a temporary 1.23GB gradient-sum file per worker. No optimizer is present.

Run the additional single-block cancellation probe with `python tools/review_phase2d2_key_bias.py`, selecting a fresh destination in that script for further retries. Its command, stdout/stderr and exit code are recorded here as logs/key_bias_review.*. It is a diagnostic, not an extra waiver or production loss.

For pytest alone set `MFVLR_PHASE2D2_EVIDENCE_ROOT` to the intended already computed G2/N2 directory and run all `tests` with raw logs and JUnit. The runner restores historical accepted gradient-file bytes after capturing fresh data. Standalone pytest may overwrite those historical outputs; use tools.run_phase2d2.pytest_run with its OUT set to a new directory to preserve them.

Archive includes full source, tests, configs, original paper, tokenizer assets/wheels and historical/raw evidence. Historical review ZIPs required by unchanged preservation tests are included byte-exact, together with the original Phase2D.2 ZIP. Installed dependency folders and caches are omitted. Validate manifest payload SHA-256 and ZIP CRC. Full checkpoints are temporary test artifacts, not supplied trained weights. B8 CPU and GPU are NOT RUN, not capacity successes.

The larger archive intentionally includes historical ZIPs so READY-N01's original preservation manifest can be checked after extraction without omitted historical inputs. Do not rewrite that baseline. Use the current package manifest to validate all actual payload files. Install only the original pinned dependencies for the target platform; unavailable platform wheels remain an explicit limitation.
''',encoding='utf-8')
    excluded={OUT/'manifest.json',OUT/'package_verification.json',OUT/'phase2d2_retry_review.zip',OUT/'phase2d2_retry_review.zip.sha256'}
    required_history_archives={ROOT/p for p in baseline if p.endswith('.zip')} | {OLD/'phase2d2_review_package.zip'}
    paths=[]
    for directory in ('src','tests','configs','docs','tools','artifacts','papers'):
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and p not in excluded and '__pycache__' not in p.parts and p.suffix!='.pyc' and (p.suffix!='.zip' or p in required_history_archives))
    paths.extend(ROOT/p for p in ('README.md','pyproject.toml') if (ROOT/p).is_file())
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}
    write(OUT/'manifest.json',{'status':status,'files_sha256':hashes,'historical_review_archives_included_for_preservation_tests':True,'excluded':'cache/installed dependencies and current package self-reference'})
    archive=OUT/'phase2d2_retry_review.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as package:
        for path in hashes:package.write(ROOT/path,path)
        package.write(OUT/'manifest.json','artifacts/phase2d2_retry/manifest.json')
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        assert all(hashlib.sha256(package.read(p)).hexdigest()==h for p,h in hashes.items())
        count=len(package.namelist())
    digest=sha(archive)
    (OUT/'phase2d2_retry_review.zip.sha256').write_text(f'{digest}  {archive.name}\n')
    write(OUT/'package_verification.json',{'status':'PASS','phase_verification':status,'bytes':archive.stat().st_size,'entries':count,'sha256':digest,'all_payload_SHA256_verified':True,'CRC_errors':0})
    assert all(sha(ROOT/p)==h for p,h in baseline.items())
    assert all(sha(ROOT/p)==h for p,h in before.items() if p not in expected)
    print(json.dumps({'status':status,'cases':complete['cases'],'passed':complete['passed'],'failed':complete['failed'],'accepted_regressions':old_passes,'G2_scientific_exact_match':exact_g2,'N2_scientific_exact_match':exact_n2,'zip':str(archive)}))


if __name__=='__main__':main()
