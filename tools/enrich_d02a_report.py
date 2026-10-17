"""Add measured interpretation and final evidence pointers after raw aggregation."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_d02a'


def main():
    report=ROOT/'docs/phase2d2_d02a_numerical_pilot_report.md'
    original=report.read_text(encoding='utf-8')
    assert '## Interpretation of the completed pilot' not in original
    stats=json.loads((OUT/'verification_final/raw_record_audit.json').read_text())
    aggregation=json.loads((OUT/'analysis/aggregation_record.json').read_text())
    failures=[]
    with (OUT/'analysis/weak_and_cancellation_failures.jsonl').open() as h:
        for line in h:
            row=json.loads(line)
            if '_Q3_' in row['job'] and row['kind']=='VJP_additivity' and row['precision']=='float64':failures.append(row)
    assert len(failures)==2
    with (OUT/'analysis/Q3_FP64_additivity_failures.json').open('x') as h:json.dump(failures,h,indent=2,allow_nan=False)
    minimum={}
    for r in stats['groups']:
        if r['family']=='NLD' and r['kind']=='VJP_additivity' and r['parameter_category']=='ordinary_QK_weight' and r['smallest_nonzero_reference_l2'] is not None:
            key=(r['variant'],r['precision'])
            minimum[key]=min(minimum.get(key,float('inf')),r['smallest_nonzero_reference_l2'])
    lines=['## Interpretation of the completed pilot','',
        f"The record audit validates all {stats['counts']['rows']:,} raw rows, the unchanged comparison thresholds and consistency of every recorded PASS/FAIL with its outside-tolerance count (or finite-difference rule). All {stats['counts']['singleton_zero_checks']:,} allocated singleton-VIM Q/K checks are exact zero. No nonzero reference vector was found with a computed zero L2 norm in this full-grid audit; this narrow bookkeeping finding does not establish broad numerical stability.",'',
        'Both normalized variants show mixed outcomes on matched common parameters. For NLD FP32-versus-promoted-FP64 VJPs, P3 lowers maximum absolute discrepancy in 97,856 comparisons and raises it in 14,140; Q3 lowers it in 102,357 and raises it in 9,639. Relative L2 worsens in 23,290 P3 comparisons and 25,325 Q3 comparisons, while improving in 88,706 and 86,671 respectively. Equal and undefined/unpaired cases remain in the full tables. These are parameter/fixture-level metric counts, not independent statistical trials or architecture accuracy scores. Each variant is compared with its own mathematical FP64 reference, and normalization changes the function being differentiated.', '',
        'All 216 ordinary Q/K directional FP64 finite-difference checks and all 2,736 native FP64 ordinary-attention derivative-oracle comparisons satisfy their original tolerances. That does not imply every mathematical-zero derivative is accurately zero: A0 still has two native FP64 NATT key-bias cancellation failures. All constant/near-constant controlled LN forward and VJP oracle checks passed.', '',
        'Native FP64 NLD VJP additivity still fails in A0, P3 and Q3. Q3 has two such failures, both at cut7 and cotangent scale1024 on ordinary Q/K weights; their complete values and fixture provenance are in `analysis/Q3_FP64_additivity_failures.json`. They remain FAIL, alongside all weak-gradient and key-bias residual failures. Lower activation amplitudes and fewer global FAILs do not justify choosing Q3 or promoting any variant.', '',
        'The following smallest strictly positive reference L2 norms describe ordinary Q/K weights across all registered NLD additivity fixtures. Exact-zero references are tabulated separately in `verification_final/raw_record_audit.json`. No near-zero cutoff, denominator floor, or extra acceptance tolerance is introduced.', '',
        '| Variant | Precision | Smallest positive reference L2 |', '|---|---|---:|']
    lines += [f'| {v} | {p} | {value:.9g} |' for (v,p),value in sorted(minimum.items())]
    lines += ['',f"The final aggregate also records {aggregation['cross_batch_failed_comparisons']:,} failed same-variant cross-batch forward comparisons out of {aggregation['cross_batch_comparisons']:,}; these are additional diagnostics, not replacements for the {aggregation['comparison_failures']:,} failures in raw local jobs.",'',
        'All414 unique canonical jobs, required output/log files, registered backend/thread controls, source/driver/config/protocol hashes, serialized checkpoints and independently reconstructed seeded FP32 inputs were verified in `verification_final/identity_input_provenance_audit.json`. The full job coverage CSV is `verification_final/coverage.csv`. `verification_final/RAM_gate_audit.json` records the original gate chain and the separate407–413 recovery chain. The original40 interrupted407 files plus both logs remain byte-preserved, with all1946 complete scientific rows and426 FAILs exactly reproduced.', '',
        'This report remains a local no-update numerical study. It does not verify full-model six-loss stability, optimizer updates, convergence, detection performance or author architecture fidelity. Phase2D.2 stays BLOCKED, with no tolerance amendment and no production promotion.', '',
        'The review ZIP includes all interrupted-attempt files (including temporary gradients) and all nine frozen decoder checkpoints. Canonical generated activation arrays remain in the workspace and are individually hashed in `cache_manifest.json`; their exclusion from the ZIP is explicit and applies uniformly. Raw numerical records, full paired CSV/JSON, logs, JUnit, source/configs and preservation records are included.', '',]
    marker='## Scope and provenance'
    assert marker in original
    report.write_text(original.replace(marker,'\n'.join(lines)+'\n'+marker,1),encoding='utf-8')
    print(json.dumps({'report_enriched':True,'Q3_FP64_failures_preserved':len(failures),'raw_rows_audited':stats['counts']['rows']}))


if __name__=='__main__':main()
