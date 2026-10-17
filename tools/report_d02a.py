"""Generate the pilot's transparent report from immutable measured records."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.d02a.common import OUT, sha, write
from tools.summarize_d02a import load_rows


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])+'\n'


def number(v):return f'{v:.6g}' if isinstance(v,(float,int)) else str(v)


def main(analysis, report_path):
    coverage=json.loads((analysis/'coverage.json').read_text())
    resources=json.loads((analysis/'backend_and_resources.json').read_text())
    aggregate=json.loads((analysis/'aggregation_record.json').read_text())
    registry=json.loads((OUT/'pilot_protocol.json').read_text())
    expected_parameters={'A0':(36746752,112758272,154156232),
                         'P3':(36768256,112779776,154177736),'Q3':(36768256,112779776,154177736)}
    inventory=[]
    for j in range(3):
        reference=json.loads((OUT/f'states/j{j}_A0.json').read_text())
        for variant in ('A0','P3','Q3'):
            p=OUT/f'states/j{j}_{variant}.json';state=json.loads(p.read_text());parity=state['report']
            assert sha(p.with_suffix('.pt'))==state['checkpoint_sha256']
            assert state['source_unchanged'] and state['caller_RNG_unchanged']
            assert state['FLT_FP32_source_hash']==reference['FLT_FP32_source_hash']
            assert all(r['byte_equal'] and r['independent_parameter_and_storage'] for r in parity['common'].values())
            assert parity['parameters']==expected_parameters[variant][1]
            assert state['full_model_count_derived_with_frozen_visual_heads_objectives']==expected_parameters[variant][2]
            assert parity['vocabulary_tied'] and len(parity['vim_roles'])==19
            assert parity['norm_modules']==(0 if variant=='A0' else 21)
            inventory.append({'j':j,'variant':variant,'LD_parameters':expected_parameters[variant][0],
                'FLT_parameters_freshly_instantiated':parity['parameters'],
                'full_model_parameters_derived_not_freshly_instantiated':expected_parameters[variant][2],
                'source_FP32_hash':state['FLT_FP32_source_hash'],'LD_state_hash':state['LD_state_hash'],
                'checkpoint_SHA256':state['checkpoint_sha256'],'common_tensors':len(parity['common']),
                'new_tensors':len(parity['new_names']),'common_bytes_and_disjoint_storage_verified':True})
    write(analysis/'parameter_and_state_inventory.json',inventory)
    activation=defaultdict(list); derivatives=defaultdict(Counter); zero=defaultdict(Counter)
    forward=defaultdict(list); stage_csv=[]; examples=[]; counts=Counter(); primitive_states=defaultdict(set)
    natt_prefix=json.loads((OUT/'NATT_batch_prefix_verification.json').read_text())['rows']
    for job in coverage['jobs']:
        if job['execution']=='NOT RUN':continue
        directory=OUT/'jobs'/job['id'];inputs=json.loads((directory/'inputs.json').read_text())
        if job['family']=='NLD':
            state=json.loads((OUT/f"states/j{job['j']}_{job['variant']}.json").read_text())
            assert inputs['state_hash_FP32']==state['LD_state_hash']
            assert inputs['parameter_count']==expected_parameters[job['variant']][0]
        else:
            primitive_states[job['j']].add(inputs['state_hash_FP32'])
            for role,multiplier in (('x',(.02,.2,2)[job['index']]),('cotangent_base',1.)):
                expected=next(r for r in natt_prefix if r['j']==job['j'] and r['role']==role and r['multiplier']==multiplier)
                assert inputs[role]['sha256']==expected[f"B{job['batch']}_SHA256"]
        for row in load_rows(directory/'comparisons.jsonl'):
            counts[(job['family'],job['variant'],row.get('status','MEASUREMENT'))]+=1
            if row['kind']=='activation' and row['precision']=='float32' and row['stage'].endswith('.output'):
                activation[(job['variant'],row['stage'])].append(row)
            if row['kind']=='activation':stage_csv.append({'job':job['id'],**row})
            if row['kind']=='forward_FP32_vs_FP64' and row.get('stage','').endswith('.output'):
                forward[(job['variant'],row['stage'])].append(row)
            if row['kind'] in ('VJP_additivity','ordinary_derivative_oracle','directional_FD'):
                name=row.get('parameter','')
                category=('singleton_QK' if '.vim.q.' in name or '.vim.k.' in name else
                          'ordinary_key_bias' if name.endswith('k.bias') else
                          'ordinary_QK_weight' if ('.q.weight' in name or '.k.weight' in name or name in ('q.weight','k.weight')) else
                          'new_LN' if '.ln' in name or name.startswith('norm.') else 'other_selected')
                group=(job['family'],job['variant'],row['kind'],row['precision'],row.get('cut','primitive'),category)
                derivatives[group][row['status']]+=1
                if row['status']=='FAIL' and category in ('ordinary_QK_weight','ordinary_key_bias'):
                    examples.append({'job':job['id'],'category':category,**row})
            if row.get('singleton_QK'):zero[(job['variant'],'VIM_QK_allocated_exact_zero')]['PASS']+=1
            if row['kind']=='ordinary_key_bias_vs_zero':zero[(job['variant'],'ordinary_key_bias_within_original_tolerance')][row['status']]+=1
    assert all(len(s)==1 for s in primitive_states.values())
    write(analysis/'derivative_category_summary.json',[
        dict(zip(('family','variant','kind','precision','cut','category'),g))|dict(c) for g,c in sorted(derivatives.items(),key=lambda x:str(x[0]))])
    with (analysis/'weak_and_cancellation_failures.jsonl').open('x',encoding='utf-8') as handle:
        for row in examples:handle.write(json.dumps(row,allow_nan=False)+'\n')
    write(analysis/'activation_table.json',stage_csv)
    import csv
    with (analysis/'activation_table.csv').open('x',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(stage_csv[0]) if stage_csv else ['job'])
        writer.writeheader();writer.writerows(stage_csv)
    tests=[]
    for label in ('oracles','driver_tests','summary_tests','ram_guard_tests','ram_guard_v2_tests'):
        execution=json.loads((OUT/label/'execution.json').read_text())
        root=ET.parse(OUT/label/'junit.xml').getroot();cases=root.findall('.//testcase')
        failed=sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
        skipped=sum(c.find('skipped') is not None for c in cases)
        assert execution['exit_code']==0 and failed==0 and skipped==0
        tests.append((label,len(cases),failed,skipped,execution['exit_code']))
    write(analysis/'executed_test_summary.json',{'fresh_tests':tests,'regression_suite_rerun':False,
        'MVE_T10':'NOT RUN / prohibited','VD_T12':'NOT RUN / prohibited','full_A1_D02B':'NOT RUN / unapproved',
        'historical_eight_A0_failures':'unchanged genuine failures; not rerun or waived'})
    backend_variants={json.dumps(b,sort_keys=True) for b in resources['backends'].values()}
    r=resources['resources'];elapsed=sum(x['elapsed_seconds'] for x in r)
    failed_jobs=[j for j in coverage['jobs'] if j['execution']=='FAIL']
    complete_by=Counter((j['family'],j['variant']) for j in coverage['jobs'] if j['execution']=='PASS')
    numerical_by=Counter((j['family'],j['variant']) for j in coverage['jobs'] if j['numerical']=='FAIL')
    paired=json.loads((analysis/'paired_change_summary.json').read_text())['groups']
    paired_table=[(g['variant'],g['metric'],g.get('LOWER',0),g.get('HIGHER',0),g.get('EQUAL',0),g.get('UNDEFINED',0))
                  for g in paired if g['family']=='NLD' and g['kind']=='VJP_FP32_vs_FP64'
                  and g['metric'] in ('outside_count','max_abs_error','relative_l2')]
    recovery_path=OUT/'history/poweroff_20261013/recovery_record.json'
    recovery=json.loads(recovery_path.read_text()) if recovery_path.exists() else None
    recovery_text=(f"Windows recorded a user-initiated power-off at 00:09 on 2026-10-13 (+07:00), interrupting job 84 after 102 other jobs had completed. The interrupted attempt has no completion record or process exit code; it is not counted as a passing job. Its {recovery['complete_partial_rows']} complete rows, including {recovery['partial_attempt_failed_comparisons']} FAIL comparisons, and all 19 files were archived byte-for-byte in `history/poweroff_20261013/`. All 1,594 frozen baseline files were rechecked unchanged. Only the interrupted cell is restarted from its original state/input recipe; completed cells are not rerun. The historical partial JSONL/CSV is additional evidence outside the canonical one-attempt-per-cell tables. Any prefix differences in the restart are reported separately in `replay_comparison.json`; they do not erase original failures. The host-resource RAM gate affects launch timing only, not numerical thresholds." if recovery else 'No interrupted-attempt recovery record exists.')
    second_recovery_path=OUT/'history/interruption_0121_20261013/recovery_record.json'
    if second_recovery_path.exists():
        second=json.loads(second_recovery_path.read_text())
        replay_path=second_recovery_path.parent/'replay_comparison.json'
        replay=json.loads(replay_path.read_text()) if replay_path.exists() else None
        recovery_text+=f" A later interruption left job 121 without a summary or exit code after 139 jobs had completed. Its cause is UNKNOWN; the available event queries did not establish power-off, OOM or a numerical hard stop. All 28 files and {second['complete_partial_rows']} partial rows, including {second['partial_attempt_failed_comparisons']} FAILs, remain preserved under `history/interruption_0121_20261013/`. The 1,594 frozen files and locked driver were rechecked before replay. Replay prefix verification status: {replay['status'] if replay else 'NOT RUN'}. This replay compares every mathematical field and ignores timing fields only. Neither interrupted attempt is counted as a completed job, nor are its failures discarded."
    third_recovery_path=OUT/'history/poweroff_0407_20261013/recovery_record.json'
    if third_recovery_path.exists():
        third=json.loads(third_recovery_path.read_text())
        replay=json.loads((third_recovery_path.parent/'replay_comparison.json').read_text())
        recovery_text+=f" A third user-initiated power-off at 23:09 on 2026-10-13 interrupted job407 after 407 canonical jobs completed. The old coordinator recorded exit1073807364; the worker has no original exit/completion record. All40 partial files, including temporary gradients, plus its two stdout/stderr files were archived with byte-for-byte SHA verification. Replay matched all{replay['compared_prefix_records']} complete scientific records, retaining all{replay['old_prefix_FAILs']} original FAIL comparisons; only timings are excluded. Recovery status: {replay['status']}. The subsequent single recovery coordinator ran only407–413, with a recorded 3 GiB available-RAM gate before every launch. It stopped between407 and408 until the exact prefix comparison passed. Earlier completed jobs were not rerun. `continuation_0407/` and `history/poweroff_0407_20261013/` retain the original and resumed evidence."
    lines=['# Phase 2D.2-A1-D02A: controlled decoder numerical pilot',
        '', '**Phase 2D.2 remains BLOCKED. No variant is promoted. No numerical-policy amendment is made.**',
        '',f"Local execution completed {coverage['complete']}/{coverage['registered']} registered jobs; {coverage['not_run']} are NOT RUN and {len(failed_jobs)} had an execution hard stop. A completed job may contain failed numerical comparisons. The raw records retain {aggregate['comparison_failures']} such failures, plus {aggregate['cross_batch_failed_comparisons']} failed same-variant cross-batch forward comparisons.",
        '', 'This is a no-update local CPU study, not a six-loss full-model experiment, a training-stability result, or an accuracy/fidelity claim. A0/P3/Q3 remain approved research engineering variants. The historical unauthorized VD-T12 SGD execution is not retroactively approved.',
        '', '## Scope and provenance',
        '',f"Corrected preregistration SHA-256: `{registry['preregistration_sha256']}`. Pilot protocol SHA-256: `{sha(OUT/'pilot_protocol.json')}`. Per-job source/config/backend/input hashes precede execution in `jobs/*/provenance.json` and `inputs.json`; paired initialization/parity inventories are in `states/*.json`. All results use each variant's own promoted FP32-weight FP64 graph, not another architecture as an equality reference.",
        '', 'The complete local registration contains 288 NLD four-thread jobs, 108 NATT four-thread jobs and 18 nominal NLD one-thread controls. The four-thread grid includes MKLDNN enabled and disabled. `coverage.json` lists every cell, including NOT RUN entries. Any explicit runtime-bound handoff is separately recorded; no unexecuted cell is described as passing.',
        '', 'NLD uses the registered j=0/1/2 seeds, eight multiplier triples, B=1/2, all seven actual LD blocks, all four residual substages, cuts 1/4/7 and cotangent scales 0/1e-4/1/1024. NATT uses the registered independent ordinary-attention primitive and magnitude/scale fixtures. The primitive is unmasked (the prior N3 analytical convention); the frozen NLD self-attention remains causal. NATT includes the explicit A0/P3/Q3 residual/LN compositions. This operational definition was locked before numerical runs.',
        '', 'FP32 comparisons retain atol=1e-6, rtol=1e-5; FP64 retains atol=1e-10, rtol=1e-8. Finite differences retain eps=1e-6, atol=1e-5, rtol=1e-3. VIM structural zeros remain exact. Undefined zero-reference ratios remain null; no denominator floor, gradient waiver, N1 transfer or N2 expansion is applied.',
        '', '## Execution and independent verification', '', table(['Family','Variant','Completed jobs','Jobs with numerical FAILs'],[(f,v,complete_by[(f,v)],numerical_by[(f,v)]) for f in ('NLD','NATT') for v in ('A0','P3','Q3')]),
        '',table(['Fresh pytest group','Cases','Failures','Skips','Exit code'],tests),
        '', 'These are selected controlled primitive, driver and reporting tests. The 276 accepted regression cases were **not** freshly rerun. Neither MVE-T10 nor VD-T12 executed. A pilot sentinel rejects their test bodies and patches optimizer step methods to raise. The original eight A0 numerical assertion failures remain genuine historical failures, unchanged and unwaived.',
        '', 'The table counts 34 unique helper/primitive cases in their final recorded per-group runs. An earlier four-case v1 RAM-guard preflight also passed (`ram_guard_tests_junit.xml`) and was repeated after correcting host suspension bookkeeping; it is not four additional unique cases. Thus these recorded pytest invocations contain 38 case executions, including that repeat. The host-guard tests do not replace actual worker/process or numerical evidence.',
        '', 'The first launch failed during driver-hash preflight because Windows path separators differed from manifest separators, before any model or input was created. Its exit-1 log and exact v1 source remain in `logs/`, `history/driver_v1/` and `preflight_v1_issue.json`. Canonical POSIX-relative manifest keys fixed only that path check; the v2 driver manifest and `logs_v2/` distinguish subsequent executions. Fixtures, numerical operations and thresholds were not changed.',
        '', recovery_text,
        '', 'The original frozen coordinator had no automatic per-worker RAM gate; earlier 3 GiB checks occurred at manual continuation boundaries only. On the renewed continuation request, its scheduling process was suspended after worker 276 started; that worker ran unchanged. An external guard holds only the coordinator between launches and requires at least 3 GiB available RAM before releasing the next worker (starting at job 277). This preserves one coordinator and the frozen scientific driver. The first guard held the coordinator during Windows child creation before the child thread was resumed; job 277 had zero CPU time and no job directory. This host-scheduling startup race is preserved in `history/ram_guard_v1/incident.json`. The same job-277 process was allowed to finish startup after another RAM check; it was not duplicated or numerically replayed. Guard v2 waits for child CPU activity before holding the coordinator. `resource_guard_v2/events.jsonl` and both handoff records retain releases, waits and process identities. Historical launches are not retroactively described as gated. Coordinator wall times after this handoff may include host waiting; worker resource/timing records exclude that scheduling wait. Eight host-guard unit cases (four for each version) are separate from the original 26 numerical/helper tests. No optimizer-containing regression case was run.',
        '', '## Parameters and unchanged initial state', '',table(['Variant','LD measured inventory','FLT instantiated inventory','Full model derived count'],[(v,*expected_parameters[v]) for v in ('A0','P3','Q3')]),
        '', 'Full FLT common tensors were copied by name/shape from the same FP32 initialization for each j, checked byte-equal and separately owned; new norms use constant gamma=1/beta=0 without disturbing the caller RNG. P3/Q3 add 21 independent LayerNorms, 42 parameter tensors and 21,504 elements. The full-model counts add the unchanged visual/head/objective subtotal of 41,397,960 to FLT; no full six-loss model was constructed for these local jobs. Each completed local worker checks unchanged state, and NLD checks unchanged parameter identities. No gradients are applied.',
        '', '## All seven decoder outputs',
        '', 'The following table summarizes measured FP32 activation ranges over executed fixtures only. `activation_table.csv/json` retains every input/substage/precision; no stage is selected for being favorable. Lower RMS is not by itself better derivative behavior.',
        '',table(['Variant','Block output','RMS min','RMS max','Max |activation|'],[(v,s,number(min(x['rms'] for x in rows)),number(max(x['rms'] for x in rows)),number(max(x['max_abs'] for x in rows))) for (v,s),rows in sorted(activation.items())]),
        '',table(['Variant','Block output','Max FP32/FP64 abs error','Max outside proportion','Max relative L2'],[(v,s,number(max(x['max_abs_error'] for x in rows)),number(max(x['outside_proportion'] for x in rows)),number(max(x['relative_l2'] for x in rows if x['relative_l2'] is not None))) for (v,s),rows in sorted(forward.items())]),
        '', '## Derivative findings and limits',
        '', 'The complete derivative table is `derivative_category_summary.json`; `all_local_records.csv` and original per-job JSONL contain every cut, cotangent scale, precision and parameter. `weak_and_cancellation_failures.jsonl` preserves every failed ordinary Q/K weight and key-bias comparison. `paired_variant_metrics.csv` compares common parameters within identical fixtures; new LN parameters remain explicitly unpaired. LOWER/HIGHER describe one metric, not universal improvement or regression.',
        '', table(['Variant vs A0','FP32/FP64 derivative metric','Lower','Higher','Equal','Undefined/unpaired'],paired_table),
        '', 'Lower and higher counts are reported together: they are paired numerical metric changes under identical local inputs, not a model-quality ranking. Undefined includes zero reference norms and extra LN parameters without an A0 counterpart. Different architectures have different derivatives and activation scales, so a smaller absolute error can coexist with worse coordinate-wise or relative accuracy. The full paired table retains every such regression.',
        '', table(['Family','Variant','Check','Precision','Cut','Parameter class','PASS','FAIL'],
                  [(*g,c['PASS'],c['FAIL']) for g,c in sorted(derivatives.items(),key=lambda x:str(x[0]))
                   if g[-1] in ('ordinary_QK_weight','ordinary_key_bias')]),
        '', 'Singleton VIM Q/K gradients are allocated exact zeros in the completed NLD runs. Ordinary key-bias derivatives cancel analytically but floating arithmetic can leave nonzero residues; these are reported independently against zero and, for NATT, against the analytical derivative. Large relative errors on tiny reference norms are retained rather than dismissed. Local ordinary Q/K derivatives and LN derivatives use independent formulas; directional finite differences are an additional check, not a substitute for coordinate-wise failures.',
        '',table(['Variant','Zero/cancellation class','PASS records','FAIL records'],[(v,c,n['PASS'],n['FAIL']) for (v,c),n in sorted(zero.items())]),
        '', 'Fixed-cotangent VJP additivity uses one forward graph per precision. The original comparison reference is the FP64 sum of individual gradients cast back to the native precision. A separate record compares to the unrounded FP64 sum. Cotangent-sum rounding, cancellation numerator/denominator, weak coordinates, sign disagreements and four fixed directional errors remain visible. Native FP64 forms u+v after promotion; all individual weights/inputs originate from FP32.',
        '', 'For `*_FP32_vs_FP64` rows, the precision metadata identifies the FP64 reference pass; the stored atol/rtol are the unchanged FP32 comparison thresholds. Temporary FP32 gradient vectors are removed only after their corresponding FP64 comparisons are recorded; all comparison statistics, gradient hashes and failures remain. Reproducing full vectors requires replaying the frozen recipe. The interrupted attempt retains its still-present temporary vectors and their SHA-256 inventory.',
        '', 'A numerical FAIL does not become PASS because its worker exited zero. The exit code confirms that mandatory finiteness, ownership, state and structural-zero checks allowed the finite diagnostic schedule to continue. The pilot supplies no automatic acceptance boundary and cannot justify production promotion or explain away the eight existing failed assertions.',
        '', '## Resources and backend',
        '',f"Measured worker time summed across executed jobs: {elapsed:.3f} s. Maximum sampled process RSS: {max(x['sampled_peak_rss_bytes'] for x in r)/2**30:.3f} GiB (50 ms sampling, not a guaranteed instantaneous peak). Minimum available RAM at a worker boundary: {min(min(x['available_RAM_start'],x['available_RAM_end']) for x in r)/2**30:.3f} GiB. Individual cold forwards, warm-up policy and three state-reloaded no-update timing repetitions are retained per precision in raw records. No concurrent full model was run.",
        '', 'These are shared-workstation measurements, not isolated performance benchmarks. Read-only interim aggregation, checkpoint hashing and small reporting tests overlapped some numerical workers; `concurrent_analysis_record.json` records this timing limitation. No variant speed ranking is inferred. A follow-on controller wait found that the prior PID had already exited successfully; that exit-1 handoff attempt performed no numerical work, and `controller_handoff_record.json` records the inspection and continuation without repeating jobs.',
        '', 'NLD cold-forward timing includes stage-capture hooks and detached clones. The three warmed repetitions measure an unhooked full-depth forward plus the registered selected-parameter/input VJP, not a full six-loss backward. Process RSS includes reference/copy states, captured tensors and diagnostic temporaries; it is not an isolated production-module footprint or a full-model training-capacity measurement.',
        '', 'Actual backend records:', '', '```json',json.dumps([json.loads(b) for b in sorted(backend_variants)],indent=2),'```',
        '', 'CPU B1/B2 results are listed cell by cell. GPU and B8: NOT RUN. N2 full-language sweeps, G2/NG2 six-loss sweeps, A1-D02B, G3/G4, I1/R2, optimizers, schedulers, training and datasets: NOT RUN / outside this authorization.',
        '', '## Evidence and reproduction',
        '', 'Raw logs/JUnit, immutable driver versions, paired state/input hashes, JSONL records, complete CSV tables, parameter inventories and backend/resource records are under `artifacts/phase2d2_d02a/`. The final preservation manifest verifies the frozen baseline and original historical evidence. The review package manifest identifies included files and any regenerated caches excluded from the ZIP. Follow its reproduction instructions in a fresh directory; existing evidence is never overwritten.',
        '', '**Stop for review. Phase 2D.2 remains BLOCKED; full A1-D02B, G3/G4 and training require separate authorization.**','']
    report_path.parent.mkdir(exist_ok=True,parents=True)
    with report_path.open('x',encoding='utf-8') as handle:handle.write('\n'.join(lines))
    print(json.dumps({'report':str(report_path),'fresh_pytest_cases':sum(t[1] for t in tests),'complete_jobs':coverage['complete']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    main(args.analysis,args.report)
