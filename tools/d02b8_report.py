"""Render read-only forensics findings; no model computation."""
import json,math,collections,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/phase2d2_d02b8'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(name,x):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
s=read(OUT/'summary.json');params=read(OUT/'parameter_forensics.json');reconcile=read(OUT/'reconciliation.json')
assert s['failed_parameters']==293 and s['outside']==8772234 and s['sum_shards_exact']
old=read(ROOT/'artifacts/phase2d2_d02b7_r2/attempt_01/additivity.json')['parameters']
assert all(v['outside_tolerance_count']==old[n]['outside_tolerance_count'] for n,v in params.items())
modules=read(OUT/'module_rankings.json');losses=read(OUT/'loss_rankings.json')
categories={}
for n,r in params.items():
    c=categories.setdefault(r['category'],dict(parameters=0,failed=0,outside=0,sign_disagreements=0,zero_reference_nonzero=0,near_nonzero_failed=0))
    c['parameters']+=1;c['failed']+=r['outside_tolerance_count']>0;c['outside']+=r['outside_tolerance_count'];c['sign_disagreements']+=r['sign_disagreements'];c['zero_reference_nonzero']+=r['reference_zero_actual_nonzero'];c['near_nonzero_failed']+=r['near_nonzero_failed']
write('category_rankings.json',categories)
deltas=[]
for r in reconcile:
    if not r['metrics_exact']:deltas.append(r)
write('secondary_metric_differences.json',deltas)
audit=dict(all_parameter_failure_counts_exact=True,all_junit_decisions_exact=True,ordered_fp64_sum_shards_exact=True,
    original_numerical_result='FAIL',secondary_exact_parameters=s['metrics_exact'],secondary_difference_parameters=len(deltas),
    secondary_difference_fields=dict(collections.Counter(k for r in deltas for k in r['mismatches'])))
write('independent_recomputation_report.json',audit)
lines=['# D02B8 G01-A0 gradient failure forensics','',
'2026-10-17, Asia/Saigon. **Read-only raw-evidence analysis; Phase 2D.2 remains BLOCKED.** No MFVLR forward/backward, optimizer step, new fixture or backend experiment was executed. B0-D08 remains disabled.','',
'## Integrity and independent recomputation','',
'All **2,392 binary shards**, totaling **3,790,126,768 bytes**, match the preserved SHA-256 inventory. Each original FP32 gradient was read independently; missing gradients retained their original None status while contributing mathematical zero to the reference sum. The six allocated FP32 gradients were converted to FP64 and added in FD/LR/CMC/FL/AR/KL order. Every coordinate of all 508 recomputed sums equals the saved FP64 sum shards exactly.','',
'The independent tool imports neither the production GradientStore nor torch. It independently evaluates the original FP32 elementwise test against the rounded FP64 sum at **atol=1e-6, rtol=1e-5**. All 508 parameter failure counts and 508 JUnit PASS/FAIL decisions match: **293 failed parameters and 8,772,234 outside-tolerance coordinates**. This reconciliation passes as evidence verification; the numerical comparisons remain FAIL.','',
f"Secondary metrics match exactly for {s['metrics_exact']}/508 parameters. The complete exact-difference ledger is `secondary_metric_differences.json`; no tolerance was used to suppress differences. The original chunk size and direction seeds/name salt were retained because they define directional probe identities. Independent implementations can share those scientific definitions without importing the original calculator.",
'', '## Module attribution','',
'| Module | Failed / parameters | Outside coordinates | Absolute error L2 | Relative L2 |','| --- | ---: | ---: | ---: | ---: |']
for r in modules:lines.append(f"| {r['module']} | {r['failed']}/{r['parameters']} | {r['outside']} | {r['error_l2']:.9g} | {r['relative_l2'] if r['relative_l2'] is not None else 'undefined'} |")
lines+=['','Module L2 values combine per-parameter squared norms. They do not replace coordinate checks. A low global relative error can coexist with many failed weak coordinates. All failed parameters, including absolute/relative L2, zero-reference behavior, signs, six contributions, cancellation, cosine, all 15 pair metrics and four directional probes, appear in `failed_parameter_rankings.json`; `parameter_forensics.json` also retains passing parameters.','',
'## Gradient categories','', '| Category | Failed / parameters | Outside | Sign disagreements | Zero-reference/nonzero actual | Near-nonzero failed |','| --- | ---: | ---: | ---: | ---: | ---: |']
for name,c in categories.items():lines.append(f"| {name} | {c['failed']}/{c['parameters']} | {c['outside']} | {c['sign_disagreements']} | {c['zero_reference_nonzero']} | {c['near_nonzero_failed']} |")
lines+=['','The near-nonzero reporting bin is **0 < abs(FP64 reference) <= 1e-6**. It uses the existing absolute-tolerance scale solely to describe coordinates, not to change any PASS/FAIL criterion. Exact reference zero is recorded separately. Undefined cosine/relative error/cancellation for zero denominators remains null rather than zero or an invented finite value.','',
'Ordinary attention key biases have analytically cancelling derivatives, but numerical residuals are not the singleton VIM structural-zero category. Ordinary Q/K weights are not expected to be zero. Their weak-gradient errors must remain visible. All singleton VIM Q/K gradient shards remain exactly zero wherever allocated; unconnected paths remain None. Key-bias residuals cannot be waived because the analytical result is zero.','',
'## Objective scale and B1 limitations','', '| Loss | Global isolated gradient L2 |','| --- | ---: |']
for r in losses:lines.append(f"| {r['loss']} | {r['global_gradient_l2']:.12g} |")
lr=next(r['global_gradient_l2'] for r in losses if r['loss']=='lr');kl=next(r['global_gradient_l2'] for r in losses if r['loss']=='kl');fd=next(r['global_gradient_l2'] for r in losses if r['loss']=='fd')
lines += ['',f'LR gradient norm is about {lr/kl:.3f} times KL and {lr/fd:.3f} times FD. This is an observed scale disparity, not proof that LR alone caused joint-versus-isolated discrepancies. Per-loss norms and pair cosines quantify contributions and alignment; they cannot assign a unique causal share of total numerical error to one loss.',
'','The existing LR scalar is 248989.578125 and total is 249028.046875. All 308 positions are included: 276 pad, 24 content, four BOS and four EOS. Recorded FP64 sums of actual FP32 token NLL values are pad 187149.6273803711, content 49556.35498046875, BOS 6816.90380859375, EOS 5466.690673828125. Category regrouping differs from production FP32 LR by -0.00128173828125; it does not replace the approved reduction. Padding contributes most of this fixture’s LR value, but token counts/scalars do not by themselves establish category-wise parameter-gradient dominance. No PAD masking, normalization, weighting or reduction change is proposed for the reproduction baseline.','',
'The B1 CMC objective is zero and all its allocated gradients are zero. It supplies no evidence about nontrivial in-batch contrastive coupling, duplicate-prompt effects or B2 temperature gradients. G01 is Real with all-background localization targets and no manipulated foreground; it cannot establish fake detection/localization behavior, foreground gradient conditioning, dataset generalization, convergence or accuracy. Finite gradients do not establish finite optimizer updates or repeated-step stability.','',
'## Observed facts versus hypotheses','',
'| Hypothesis | Supporting observations | Limitation / falsification test (not executed) |',
'| --- | --- | --- |',
'| FP32 backward accumulation/reduction order contributes | Actual isolated and joint gradients disagree while source/state are preserved and values remain finite; error is not a disk-sum bookkeeping problem | Does not identify a kernel or operation. A separately approved same-forward full-roster control can limit separate-forward explanations; operator/backend controls need their own approval |',
'| Cancellation amplifies relative error | Per-parameter cancellation ratios, weak references, signs and key-bias residuals are explicitly measured | Many errors may occur at cancellation ratio near one; cancellation is not a universal explanation. Examine full ratio/error distributions and controlled local derivatives |',
'| Separate graph paths/forward rounding contribute | R2 used seven fresh graphs with identical scalar results | Equal scalar losses do not prove all intermediate/gradient values equal. Historical N3 same-forward failures already show fresh graphs are not the only possible mechanism; a new controlled full-roster comparison would test this case |',
'| Backend sensitivity contributes | FP32 arithmetic is operation/order-sensitive in principle | This phase has only one recorded C4M cohort; no causal backend comparison is available here. A matched backend-control experiment would be required and is not authorized |',
'| LR scale increases numerical risk | LR scalar and gradient norms dominate this fixture | Correlation does not locate the source of error. Loss rescaling would change the baseline and is neither performed nor authorized |','',
'No single root cause has been established. Summing isolated gradients in FP64 is a comparison reference, not a mathematically exact ground-truth derivative or full FP64 autodiff. All original failures remain intact.','',
'## Decision and preparation','',
'Further read-only attribution is supported. The recommended next numerical step is one separately approved, preregistered G01-A0 same-forward no-update control with full-roster coverage; see `phase2d2_d02b8_next_step_decision.md`. A bounded optimizer smoke test is not recommended as the immediate step and remains separately unauthorized. No automatic continuation or tolerance amendment follows from this analysis.','',
'Separate proposed configuration/dataset/checkpoint contracts are in `phase2d2_d02b8_preparation_contract.md` and `src/mfvlr_research/d02b8/contracts.py`. They validate metadata only and cannot run or load a model. Training/inference modes remain disabled; no dataset preprocessing, optimizer or checkpoint execution is implemented.','',
'## Evidence limits and reproducibility','',
'The compact archive contains independent analysis source, per-parameter/per-loss/module results, exact mismatch ledger, full shard index, original summary/JUnit records, mock tests and preservation hashes. Original 3.79 GB binary shards remain untouched on disk and are omitted from the compact archive; reproducing the full arithmetic independently requires those indexed files. Inspecting summaries does not substitute for those raw bytes. `shard_integrity.json` records the direct on-machine verification performed in this phase.','',
'No original failed-run record, source, model, fixture or threshold was rewritten. Stop for independent review. Phase 2D.2 remains BLOCKED.']
with (ROOT/'docs/phase2d2_d02b8_gradient_forensics.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(audit))
