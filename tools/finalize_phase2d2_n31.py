"""Publish actual N3.1 outcomes, full records and preservation; no policy waiver."""
from pathlib import Path
import csv,hashlib,json,math,xml.etree.ElementTree as ET,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/phase2d2_n31'


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def text(p,v):Path(p).write_text(v,encoding='utf-8')
def csvout(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def suite(p):
    rows=[]
    for c in ET.parse(p).getroot().findall('.//testcase'):
        status='PASS';detail=None
        for tag in ('failure','error','skipped'):
            node=c.find(tag)
            if node is not None:status=tag.upper();detail=node.attrib.get('message');break
        rows.append({'id':[c.attrib['classname'],c.attrib['name']],'status':status,'detail':detail})
    return {'cases':len(rows),'passed':sum(r['status']=='PASS' for r in rows),'failed':sum(r['status']=='FAILURE' for r in rows),
        'errors':sum(r['status']=='ERROR' for r in rows),'skips':sum(r['status']=='SKIPPED' for r in rows),'records':rows}


def main():
    patches=read(OUT/'applied_patches.json');baseline=read(OUT/'history/baseline_manifest.json')
    allowed={r['target']:r['after_sha256'] for r in patches['applied'] if r['before_sha256'] is not None}
    changes=[p for p,h in baseline.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    assert set(changes)==set(allowed),changes
    assert all(sha(ROOT/p)==h for p,h in allowed.items())
    for r in patches['applied']:assert sha(ROOT/r['target'])==r['after_sha256']
    full=suite(OUT/'junit/complete_01.xml');new=suite(OUT/'junit/supplemental_01.xml')
    old=suite(ROOT/'artifacts/phase2d2_retry/junit/complete_retry_01.xml');accepted=suite(ROOT/'artifacts/phase2d1/junit/complete_01.xml')
    oldids={tuple(r['id']) for r in old['records']};now={tuple(r['id']):r for r in full['records']};acceptedids={tuple(r['id']) for r in accepted['records']}
    assert oldids<=set(now) and len(set(now)-oldids)==7
    acceptedpass=sum(now[i]['status']=='PASS' for i in acceptedids)
    oldfails={tuple(r['id']) for r in old['records'] if r['status']=='FAILURE'};fails={tuple(r['id']) for r in full['records'] if r['status']=='FAILURE'}
    verify={'Phase2D2':'BLOCKED','complete':full,'supplemental':new,'accepted_276_passed':acceptedpass,
        'original_321_case_ids_present':True,'original_eight_failures_remain':oldfails<=fails,
        'new_or_changed_failures':[list(i) for i in fails-oldfails],'exit_codes':read(OUT/'suite_exit.json'),
        'existing_assertions_unchanged':True,'numerical_tolerances_unchanged':True}
    write(OUT/'verification.json',verify)
    write(OUT/'preservation.json',{'status':'PASS','baseline_files':len(baseline),'unchanged_files':len(baseline)-len(changes),
        'approved_preexisting_changes_only':changes,'new_test_patch_exact':True,'production_config_276_tests_history_unchanged':True,
        'old_321_test_sources_unchanged':True,'old_N3_helper_before_patch_archived':'history/review_phase2d2_n3.py'})
    params=[];groups=[];g2=[]
    cases=('real','efs','local','real_efs','real_local','local_duplicate','real_efs_native')
    for case in cases:
        p=OUT/'g2'/('final_'+case);r=read(p/'complete_discrepancies.json');g=read(p/'gradient_additivity.json');s=read(p/'summary.json');resource=read(p/'resources.json')
        for n,v in r['parameters'].items():params.append({'case':case,'parameter':n,**v})
        for n,v in r['groups'].items():groups.append({'case':case,'group':n,**v})
        g2.append({'case':case,'status':s['status'],'outside':sum(v['outside_tolerance_count'] for v in r['parameters'].values()),
            'global_relative_l2':g['relative_l2_difference'],'all_finite':s['all_valid_values_and_gradients_finite'],
            'seconds':resource['elapsed_seconds'],'peak_RSS_MiB':resource['sampled_peak_rss_bytes']/2**20,
            'LE08_Q_relative_l2':r['parameters']['flt.encoder.blocks.7.attention.q.weight']['relative_l2']})
    csvout(OUT/'complete_parameter_discrepancies.csv',params);csvout(OUT/'complete_group_discrepancies.csv',groups)
    write(OUT/'complete_G2_summary.json',g2)
    attention=read(OUT/'heldout_attention.json');ld=read(OUT/'heldout_LD.json')
    bounds=[r for r in attention['conditional_local_bounds'] if 'outside_conditional_bound' in r]
    localrows=[]
    for r in attention['records']:
        for n,c in r['manual_derivative_comparisons'].items():localrows.append({'family':'attention','magnitude':r['input_magnitude'],'dtype':r['dtype'],'scale':r['scale'],'target':n,**c})
        localrows.append({'family':'key_bias_zero','magnitude':r['input_magnitude'],'dtype':r['dtype'],'scale':r['scale'],'target':'k.bias',**r['key_bias_vs_zero']})
    for r in ld['records']:
        for n,v in r['parameters'].items():
            if v['allocated']:localrows.append({'family':'LD','magnitude':r['multipliers'][0],'dtype':r['dtype'],'scale':r['scale'],'target':n,'cut':r['cut'],**v['comparison']})
    for mode in ('mkldnn','native'):
        visual=read(OUT/f'heldout_visual_{mode}.json')
        for r in visual['records']:
            for n,v in r['parameters'].items():
                if v['allocated']:localrows.append({'family':'visual_'+mode,'dtype':'torch.float32','scale':r['scale'],'target':n,**v['comparison']})
    csvout(OUT/'heldout_comparisons.csv',localrows)
    write(OUT/'heldout_nonpassing_comparisons.json',{'scope':'All recorded local manual/VJP/zero-reference comparisons outside unchanged coordinate tolerances; not pytest failures unless asserted by a test.',
        'records':[r for r in localrows if not r['allclose']]})
    details={'attention_records':len(attention['records']),'LD_records':len(ld['records']),
        'conditional_bounds_checked':len(bounds),'outside_conditional_bound_total':sum(r['outside_conditional_bound'] for r in bounds),
        'finite_difference_passes':sum(r['pass'] for r in attention['finite_differences']),'finite_difference_cases':len(attention['finite_differences']),
        'local_comparisons':len(localrows),'local_comparisons_outside_unchanged_tolerance':sum(not r['allclose'] for r in localrows)}
    write(OUT/'heldout_summary.json',details)
    history=read(OUT/'historical_scientific_comparison.json')
    assert history['completion']=={'G2_cases_available':6,'N2_cases_available':6}
    repeat_exact=all(all(v.values()) for v in history['same_forward'].values()) and all(history['CMC'].values()) and all(all(v.values()) for v in history['G2'].values()) and all(history['N2'].values())
    failure_lines='\n'.join('- `'+'.'.join(r['id'])+'`' for r in full['records'] if r['status']!='PASS')
    gtable='\n'.join(f"| {r['case']} | {r['outside']:,} | {r['global_relative_l2']:.7g} | {r['LE08_Q_relative_l2']:.7g} | {r['peak_RSS_MiB']:.1f} | {r['seconds']:.2f} |" for r in g2)
    attable='\n'.join(f"| {r['input_magnitude']} | {r['parameter']} | {r['absolute_error']:.6g} | {'PASS' if r['pass'] else 'FAIL'} |" for r in attention['finite_differences'])
    ldtable=[]
    for r in ld['records']:
        if r['scale']!=1:continue
        memory=r['parameters']['memory'];state=r['parameters']['state'];q=r['parameters']['blocks.0.self_attention.q.weight']
        def rel(v):return f"{v['comparison']['relative_l2_difference']:.7g}" if v['allocated'] and v['comparison']['relative_l2_difference'] is not None else 'undefined/disconnected'
        ldtable.append(f"| {r['multipliers'][0]} | {r['dtype']} | {r['cut']} | {rel(state)} | {rel(memory)} | {rel(q)} |")
    ldtable='\n'.join(ldtable)
    text(ROOT/'docs/phase2d2_n31_heldout_validation.md',f'''# Phase 2D.2 N3.1 held-out validation

**Phase 2D.2 remains BLOCKED.** The final mandatory suite ran **{full['cases']} cases: {full['passed']} passed, {full['failed']} failed, {full['errors']} errors, {full['skips']} skips**. The seven supplemental cases separately ran **{new['passed']}/{new['cases']} passing**. All **{acceptedpass}/276 accepted regression cases pass**; all original 321 identities remain present. Original eight failure identities preserved: **{oldfails<=fails}**. New failure identities: `{verify['new_or_changed_failures']}`. See [verification](../artifacts/phase2d2_n31/verification.json), [full stdout](../artifacts/phase2d2_n31/logs/complete_01.stdout.log), [stderr](../artifacts/phase2d2_n31/logs/complete_01.stderr.log), [JUnit](../artifacts/phase2d2_n31/junit/complete_01.xml).

Actual nonpassing mandatory cases (pytest exit code {verify['exit_codes']['complete_exit_code']}):

{failure_lines}

## Locked protocol and approved patches

[Registration](../artifacts/phase2d2_n31/heldout_preregistration.json) and its [SHA lock](../artifacts/phase2d2_n31/preregistration_lock.json) precede experiments. No fixture definition was altered after observing results. Attention initialization/input/cotangent/direction seeds are 2101/2102/2103/2104–2105; `[1,308,512]`, magnitudes 0.02/0.2/2, cotangent scales 0/1e-4/1/1e4. LD uses copied actual real/EFS accepted decoder inputs at multipliers 0.25/1/2, cotangent seeds 3101/3102, cuts 1/4/7 and scales 1e-4/1/1024. Visual controls use new image seed 4101 and cotangent seeds 4102/4103 at the same three nonzero scales, with MKLDNN on/off. All remain finite no-update diagnostics. Zero, analytically canceling and generally nonzero derivatives are explicitly distinct.

The exact approved N3D01 new-file test patch and N3D02 metadata patch were applied only after verifying expected source hashes. [Applied-patch record](../artifacts/phase2d2_n31/applied_patches.json) contains before/after hashes and the original patch hashes. Original helper bytes and both diffs are archived. Existing 321 tests/assertions, 276 accepted regressions, production sources and configs are unchanged. Fresh primary records retain seven distinct loss-grad-function wrappers and live tensor/storage identities; the old defective metadata remains historical.

## Fresh evidence integrity

The final workflow seals the required record inventory, source/test/config hashes, locked protocol and raw execution logs. A pytest session plugin verifies the fresh run nonce, exact sealed record hashes, all required filenames, regenerated synthetic image/token/mask provenance, initialized parameter hashes, ownership/counts, repaired graph metadata and no-update/finiteness flags **before tests run**. The external-evidence checks within the seven supplemental cases consume the newly generated N3.1 directory via environment variables; the other supplemental cases execute live mathematical controls. No N3 historical JSON is substituted.

[Negative provenance tests](../artifacts/phase2d2_n31/provenance_negative_tests.json) reject wrong-run nonce, missing records, incomplete inventory, altered source hashes and altered evidence. Integrity is not a numerical proof: the seal explicitly grants no acceptance, and numerical failures remain recorded and failing. A local SHA manifest is an integrity/reproducibility mechanism, not a signed adversarial attestation.

The workflow refreshed four same-forward N3 cases, CMC FP32/FP64, all six G2 fixtures plus one alternate backend, six N2 F0/F1/F2 precision cases and the newly held-out local experiments. No production FP64 MVE/VD interface was introduced. Local FP64 promotion copies FP32-initialized parameters and stored inputs; it is not full-model FP64 validation.

The declared repeated scientific fields match the preserved evidence exactly: **{repeat_exact}**. This compares selected A/B gradient statistics and comparisons for all four N3 runs, both CMC records, all six original-backend G2 records and six N2 records. Timing/process identities are excluded; the unordered legacy cosine-entry list alone is canonicalized. The corrected seven distinct loss-wrapper identities are separately checked. See [exact comparison audit](../artifacts/phase2d2_n31/historical_scientific_comparison.json). Held-out seeds/magnitudes and the newly complete alternate-backend G2 are new experiments, not expected to match old fixtures.

## Independent attention results

There are **{details['attention_records']}** precision/magnitude/scale attention records. Independent FP64 formula comparisons, key-bias zero-reference comparisons and every failed FP32/local comparison remain in [heldout_attention.json](../artifacts/phase2d2_n31/heldout_attention.json) and [flat comparison CSV](../artifacts/phase2d2_n31/heldout_comparisons.csv). No failing weak or nonzero Q/K derivative is omitted.

Conditional local affine/backward-contraction/bias-reduction envelopes: **{len(bounds)}** checked, **{details['outside_conditional_bound_total']}** coordinates outside. These use dimensions/roundoff models fixed before observations, not maxima-fitted thresholds. The envelopes cover frozen operands only, are computed in FP64 without interval rounding, and do not certify optimized kernels or a full network. Stored probability defects and their Jacobian row-sum consequences are separately recorded.

Central finite differences use epsilon 1e-6 and unchanged atol 1e-5 / rtol 1e-3. They compare independent functional perturbations, with no in-place Parameter edit:

| Input magnitude | Parameter | Absolute directional error | Existing check |
|---|---|---:|---|
{attable}

Finite differences are approximate references, subject to truncation and subtractive rounding. Exact rational mathematics verifies ordinary key-bias invariance separately. Allocated singleton VIM Q/K zeros retain their exact requirement when connected; ordinary attention biases do not receive a structural-zero reinterpretation.

## Seven-block LD held-out controls

There are **{len(ld['records'])}** predeclared LD precision/magnitude/cut/scale records. The table reports scale-one VJP additivity relative L2, without changing the original elementwise criterion. Full absolute/count/zero-reference/norm/cosine evidence at all scales remains in [heldout_LD.json](../artifacts/phase2d2_n31/heldout_LD.json).

| Input multiplier | Precision | Cut | State relative L2 | Memory relative L2 | LD1 Q-weight relative L2 |
|---|---|---|---:|---:|---:|
{ldtable}

These are fixed local Jacobians with independent state/memory/visual leaves, not full LE/visual gradients. Each cut's u, v and sum VJPs use one forward graph. The FP32 u+v includes input-sum rounding; promoted FP64 sums promoted operands. Different precisions recompute forward intermediates, so precision improvement does not isolate every error source. No threshold is inferred from these measurements. A count of completed diagnostics must not be presented as passing numerical comparisons.

Across all local manual/VJP/zero-reference comparisons, **{details['local_comparisons_outside_unchanged_tolerance']} of {details['local_comparisons']}** are outside their unchanged coordinate tolerances. They are listed in [nonpassing comparisons](../artifacts/phase2d2_n31/heldout_nonpassing_comparisons.json). They are exploratory comparison outcomes, not extra pytest case counts, and include near-zero cancellation as well as nonzero derivative disagreement.

## Complete parameter and module coverage

Each G2 worker streams six isolated FP32 gradients to the existing FP64 accumulator and runs the literal joint backward. An added diagnostic pass reads the same stream without modifying the legacy comparison or its result. All **508 parameter tensors / 154,156,232 elements** are covered per run. Exact original coordinate mismatch counts are cross-checked against the unchanged comparator. Counts, zero-reference coordinates, absolute/relative/norm errors, signs, cosine and four seeded directions are reported per parameter and module. Module directions concatenate parameter-specific Rademacher vectors and normalize by module size. This is full module coverage, not N3's selected subset.

The original elementwise gate casts the accumulated reference back to FP32 before comparing with the joint FP32 gradient. The new norm, absolute-error, zero-reference and directional diagnostics instead compare the joint values promoted to FP64 against the stored FP64 sum. These are explicitly different arithmetic views of the same stream; no change to the legacy reference or tolerance is implied. The FP64 sum contains already-rounded isolated FP32 gradients and is not a full-model FP64 derivative oracle.

| Fixture/backend | Outside elements | Global relative L2 (diagnostic only) | LE8 Q-weight relative L2 | Peak RSS MiB | Worker seconds |
|---|---:|---:|---:|---:|---:|
{gtable}

[Complete parameter CSV](../artifacts/phase2d2_n31/complete_parameter_discrepancies.csv), [complete group CSV](../artifacts/phase2d2_n31/complete_group_discrepancies.csv), [G2 summaries](../artifacts/phase2d2_n31/complete_G2_summary.json). Every full JSON retains vector sufficient statistics and directional values. Whole-model norms never replace weak-parameter evidence. Undefined zero-reference relative L2/cosine remains null. Classification distinguishes analytical ordinary key-bias zeros, structural VIM zeros and other parameters. The independent stream control detects a deliberately injected +1 gradient error across a chunk boundary and checks zero-reference handling.

The MKLDNN-disabled run is the same CPU/platform with a backend flag changed, not an independent platform. Full G2 uses actual unit-weight losses and original scale; cotangent-scale sensitivity is independently measured in the held-out LD and visual controls. No synthetic scaling was applied to production objectives or gradients.

## Preservation and resource limits

All {len(baseline)} pre-existing files were audited. The only changed pre-existing file is the explicitly approved diagnostic metadata helper; its original bytes are archived. Every other pre-existing source/config/test/historical artifact remains byte-identical. The new test file exactly matches its approved patch. [Preservation](../artifacts/phase2d2_n31/preservation.json).

CPU batch 1 and 2 were actually exercised. GPU and CPU batch 8 are NOT RUN; no capacity is inferred. Full workers have sampled RSS/runtime records including disk streaming/statistics, not pure model throughput or exact allocator peaks. No simultaneous seven full graphs are retained. All parameters and their identities remain unchanged within runs. No optimizer, update, training, benchmark, dataset, clipping, normalization, AMP or parameter freezing was implemented.

The mandated unchanged regression suite includes the previously accepted **MVE-T10** test: it performs one controlled SGD update on three MVE parameters, checks the exact result and restores their original bytes in `finally`. That pre-existing unit test was retained unchanged. The no-update claim applies to every N3.1 experiment and full G2 worker; it does not erase this historical unit-test behavior. No new optimizer, full-graph optimization, G3/G4 execution or training step was introduced.

The original orchestration stopped when the first fresh N2 worker could not write its result into a missing output directory. Its original stderr/exit record is preserved. The recovery created that directory, verified the original pinned source snapshot and reran all six N2 workers under distinct retry log names; no numerical source or fixture changed. See [recovery record](../artifacts/phase2d2_n31/resume_record.json). During recovery the system had only 565,764,096 bytes of available physical memory (96.7% use); the first FP32 worker took about 1,705 seconds. The user then freed RAM. This is a resource-confounded wall time, not representative model throughput. See [live observation](../artifacts/phase2d2_n31/resume_resource_observation.json) and actual per-worker execution logs. The reproduction guide explicitly creates the output directory before starting.

The [numerical budget](phase2d2_n31_numerical_budget.md) explains conditional bound limitations; the [proposal](phase2d2_n31_acceptance_proposal.md) gives the stopping decision. Review ZIP and manifests are under [artifacts/phase2d2_n31](../artifacts/phase2d2_n31).
''')
    print(json.dumps({'full_suite':{k:v for k,v in full.items() if k!='records'},'heldout':details,'Phase2D2':'BLOCKED'}))


if __name__=='__main__':main()
