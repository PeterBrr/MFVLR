# Phase 2D.2-A1-D02A: controlled decoder numerical pilot

**Phase 2D.2 remains BLOCKED. No variant is promoted. No numerical-policy amendment is made.**

Local execution completed 414/414 registered jobs; 0 are NOT RUN and 0 had an execution hard stop. A completed job may contain failed numerical comparisons. The raw records retain 399522 such failures, plus 4907 failed same-variant cross-batch forward comparisons.

This is a no-update local CPU study, not a six-loss full-model experiment, a training-stability result, or an accuracy/fidelity claim. A0/P3/Q3 remain approved research engineering variants. The historical unauthorized VD-T12 SGD execution is not retroactively approved.

## Interpretation of the completed pilot

The record audit validates all 1,594,980 raw rows, the unchanged comparison thresholds and consistency of every recorded PASS/FAIL with its outside-tolerance count (or finite-difference rule). All 58,752 allocated singleton-VIM Q/K checks are exact zero. No nonzero reference vector was found with a computed zero L2 norm in this full-grid audit; this narrow bookkeeping finding does not establish broad numerical stability.

Both normalized variants show mixed outcomes on matched common parameters. For NLD FP32-versus-promoted-FP64 VJPs, P3 lowers maximum absolute discrepancy in 97,856 comparisons and raises it in 14,140; Q3 lowers it in 102,357 and raises it in 9,639. Relative L2 worsens in 23,290 P3 comparisons and 25,325 Q3 comparisons, while improving in 88,706 and 86,671 respectively. Equal and undefined/unpaired cases remain in the full tables. These are parameter/fixture-level metric counts, not independent statistical trials or architecture accuracy scores. Each variant is compared with its own mathematical FP64 reference, and normalization changes the function being differentiated.

All 216 ordinary Q/K directional FP64 finite-difference checks and all 2,736 native FP64 ordinary-attention derivative-oracle comparisons satisfy their original tolerances. That does not imply every mathematical-zero derivative is accurately zero: A0 still has two native FP64 NATT key-bias cancellation failures. All constant/near-constant controlled LN forward and VJP oracle checks passed.

Native FP64 NLD VJP additivity still fails in A0, P3 and Q3. Q3 has two such failures, both at cut 7 and cotangent scale 1024 on ordinary Q/K weights; their complete values and fixture provenance are in `analysis/Q3_FP64_additivity_failures.json`. They remain FAIL, alongside all weak-gradient and key-bias residual failures. Lower activation amplitudes and fewer global FAILs do not justify choosing Q3 or promoting any variant.

The following smallest strictly positive reference L2 norms describe ordinary Q/K weights across all registered NLD additivity fixtures. Exact-zero references are tabulated separately in `verification_final/raw_record_audit.json`. No near-zero cutoff, denominator floor, or extra acceptance tolerance is introduced.

| Variant | Precision | Smallest positive reference L2 |
|---|---|---:|
| A0 | float32 | 3.6346631e-09 |
| A0 | float64 | 2.27569307e-17 |
| P3 | float32 | 4.33227327e-07 |
| P3 | float64 | 4.33234377e-07 |
| Q3 | float32 | 3.85112643e-11 |
| Q3 | float64 | 1.99641383e-14 |

The final aggregate also records 4,907 failed same-variant cross-batch forward comparisons out of 11,736; these are additional diagnostics, not replacements for the 399,522 failures in raw local jobs.

All 414 unique canonical jobs, required output/log files, registered backend/thread controls, source/driver/config/protocol hashes, serialized checkpoints and independently reconstructed seeded FP32 inputs were verified in `verification_final/identity_input_provenance_audit.json`. The full job coverage CSV is `verification_final/coverage.csv`. `verification_final/RAM_gate_audit.json` records the original gate chain and the separate 407–413 recovery chain. The original 40 interrupted 407 files plus both logs remain byte-preserved, with all 1,946 complete scientific rows and 426 FAILs exactly reproduced.

This report remains a local no-update numerical study. It does not verify full-model six-loss stability, optimizer updates, convergence, detection performance or author architecture fidelity. Phase 2D.2 stays BLOCKED, with no tolerance amendment and no production promotion.

The review ZIP includes all interrupted-attempt files (including temporary gradients) and all nine frozen decoder checkpoints. Canonical generated activation arrays remain in the workspace and are individually hashed in `cache_manifest.json`; their exclusion from the ZIP is explicit and applies uniformly. Raw numerical records, full paired CSV/JSON, logs, JUnit, source/configs and preservation records are included.

## Scope and provenance

Corrected preregistration SHA-256: `3577e592a3a4013d136c840093d1d843ac92481b2f4088e4caf66fed7b3f92a3`. Pilot protocol SHA-256: `cdc6bd1c9bd940e61b6f5fbbff7d8b7e2cedbb66a52e5ff1b5a7a3a5b3ae8908`. Per-job source/config/backend/input hashes precede execution in `jobs/*/provenance.json` and `inputs.json`; paired initialization/parity inventories are in `states/*.json`. All results use each variant's own promoted FP32-weight FP64 graph, not another architecture as an equality reference.

The complete local registration contains 288 NLD four-thread jobs, 108 NATT four-thread jobs and 18 nominal NLD one-thread controls. The four-thread grid includes MKLDNN enabled and disabled. `coverage.json` lists every cell, including NOT RUN entries. Any explicit runtime-bound handoff is separately recorded; no unexecuted cell is described as passing.

NLD uses the registered j=0/1/2 seeds, eight multiplier triples, B=1/2, all seven actual LD blocks, all four residual substages, cuts 1/4/7 and cotangent scales 0/1e-4/1/1024. NATT uses the registered independent ordinary-attention primitive and magnitude/scale fixtures. The primitive is unmasked (the prior N3 analytical convention); the frozen NLD self-attention remains causal. NATT includes the explicit A0/P3/Q3 residual/LN compositions. This operational definition was locked before numerical runs.

FP32 comparisons retain atol=1e-6, rtol=1e-5; FP64 retains atol=1e-10, rtol=1e-8. Finite differences retain eps=1e-6, atol=1e-5, rtol=1e-3. VIM structural zeros remain exact. Undefined zero-reference ratios remain null; no denominator floor, gradient waiver, N1 transfer or N2 expansion is applied.

## Execution and independent verification

| Family | Variant | Completed jobs | Jobs with numerical FAILs |
| --- | --- | --- | --- |
| NLD | A0 | 102 | 102 |
| NLD | P3 | 102 | 102 |
| NLD | Q3 | 102 | 102 |
| NATT | A0 | 36 | 36 |
| NATT | P3 | 36 | 36 |
| NATT | Q3 | 36 | 36 |


| Fresh pytest group | Cases | Failures | Skips | Exit code |
| --- | --- | --- | --- | --- |
| oracles | 12 | 0 | 0 | 0 |
| driver_tests | 7 | 0 | 0 | 0 |
| summary_tests | 7 | 0 | 0 | 0 |
| ram_guard_tests | 4 | 0 | 0 | 0 |
| ram_guard_v2_tests | 4 | 0 | 0 | 0 |


These are selected controlled primitive, driver and reporting tests. The 276 accepted regression cases were **not** freshly rerun. Neither MVE-T10 nor VD-T12 executed. A pilot sentinel rejects their test bodies and patches optimizer step methods to raise. The original eight A0 numerical assertion failures remain genuine historical failures, unchanged and unwaived.

The table counts 34 unique helper/primitive cases in their final recorded per-group runs. An earlier four-case v1 RAM-guard preflight also passed (`ram_guard_tests_junit.xml`) and was repeated after correcting host suspension bookkeeping; it is not four additional unique cases. Thus these recorded pytest invocations contain 38 case executions, including that repeat. The host-guard tests do not replace actual worker/process or numerical evidence.

The first launch failed during driver-hash preflight because Windows path separators differed from manifest separators, before any model or input was created. Its exit-1 log and exact v1 source remain in `logs/`, `history/driver_v1/` and `preflight_v1_issue.json`. Canonical POSIX-relative manifest keys fixed only that path check; the v2 driver manifest and `logs_v2/` distinguish subsequent executions. Fixtures, numerical operations and thresholds were not changed.

Windows recorded a user-initiated power-off at 00:09 on 2026-10-13 (+07:00), interrupting job 84 after 102 other jobs had completed. The interrupted attempt has no completion record or process exit code; it is not counted as a passing job. Its 2551 complete rows, including 1164 FAIL comparisons, and all 19 files were archived byte-for-byte in `history/poweroff_20261013/`. All 1,594 frozen baseline files were rechecked unchanged. Only the interrupted cell is restarted from its original state/input recipe; completed cells are not rerun. The historical partial JSONL/CSV is additional evidence outside the canonical one-attempt-per-cell tables. Any prefix differences in the restart are reported separately in `replay_comparison.json`; they do not erase original failures. The host-resource RAM gate affects launch timing only, not numerical thresholds. A later interruption left job 121 without a summary or exit code after 139 jobs had completed. Its cause is UNKNOWN; the available event queries did not establish power-off, OOM or a numerical hard stop. All 28 files and 2804 partial rows, including 536 FAILs, remain preserved under `history/interruption_0121_20261013/`. The 1,594 frozen files and locked driver were rechecked before replay. Replay prefix verification status: PASS. This replay compares every mathematical field and ignores timing fields only. Neither interrupted attempt is counted as a completed job, nor are its failures discarded. A third user-initiated power-off at 23:09 on 2026-10-13 interrupted job 407 after 407 canonical jobs completed. The old coordinator recorded exit 1073807364; the worker has no original exit/completion record. All 40 partial files, including temporary gradients, plus its two stdout/stderr files were archived with byte-for-byte SHA verification. Replay matched all 1,946 complete scientific records, retaining all 426 original FAIL comparisons; only timings are excluded. Recovery status: PASS. The subsequent single recovery coordinator ran only 407–413, with a recorded 3 GiB available-RAM gate before every launch. It stopped between 407 and 408 until the exact prefix comparison passed. Earlier completed jobs were not rerun. `continuation_0407/` and `history/poweroff_0407_20261013/` retain the original and resumed evidence.

The original frozen coordinator had no automatic per-worker RAM gate; earlier 3 GiB checks occurred at manual continuation boundaries only. On the renewed continuation request, its scheduling process was suspended after worker 276 started; that worker ran unchanged. An external guard holds only the coordinator between launches and requires at least 3 GiB available RAM before releasing the next worker (starting at job 277). This preserves one coordinator and the frozen scientific driver. The first guard held the coordinator during Windows child creation before the child thread was resumed; job 277 had zero CPU time and no job directory. This host-scheduling startup race is preserved in `history/ram_guard_v1/incident.json`. The same job-277 process was allowed to finish startup after another RAM check; it was not duplicated or numerically replayed. Guard v2 waits for child CPU activity before holding the coordinator. `resource_guard_v2/events.jsonl` and both handoff records retain releases, waits and process identities. Historical launches are not retroactively described as gated. Coordinator wall times after this handoff may include host waiting; worker resource/timing records exclude that scheduling wait. Eight host-guard unit cases (four for each version) are separate from the original 26 numerical/helper tests. No optimizer-containing regression case was run.

## Parameters and unchanged initial state

| Variant | LD measured inventory | FLT instantiated inventory | Full model derived count |
| --- | --- | --- | --- |
| A0 | 36746752 | 112758272 | 154156232 |
| P3 | 36768256 | 112779776 | 154177736 |
| Q3 | 36768256 | 112779776 | 154177736 |


Full FLT common tensors were copied by name/shape from the same FP32 initialization for each j, checked byte-equal and separately owned; new norms use constant gamma=1/beta=0 without disturbing the caller RNG. P3/Q3 add 21 independent LayerNorms, 42 parameter tensors and 21,504 elements. The full-model counts add the unchanged visual/head/objective subtotal of 41,397,960 to FLT; no full six-loss model was constructed for these local jobs. Each completed local worker checks unchanged state, and NLD checks unchanged parameter identities. No gradients are applied.

## All seven decoder outputs

The following table summarizes measured FP32 activation ranges over executed fixtures only. `activation_table.csv/json` retains every input/substage/precision; no stage is selected for being favorable. Lower RMS is not by itself better derivative behavior.

| Variant | Block output | RMS min | RMS max | Max |activation| |
| --- | --- | --- | --- | --- |
| A0 | LD.1.output | 0.276455 | 25.0726 | 128.71 |
| A0 | LD.2.output | 0.401838 | 49.0063 | 262.825 |
| A0 | LD.3.output | 0.61709 | 85.6763 | 404.32 |
| A0 | LD.4.output | 1.0645 | 143.979 | 699.735 |
| A0 | LD.5.output | 1.90505 | 231.68 | 1126.85 |
| A0 | LD.6.output | 3.79227 | 384.659 | 1882.76 |
| A0 | LD.7.output | 6.72925 | 632.407 | 2989.19 |
| P3 | LD.1.output | 0.776451 | 21.7041 | 107.829 |
| P3 | LD.2.output | 1.1618 | 30.4743 | 141.434 |
| P3 | LD.3.output | 1.49726 | 37.5282 | 177.811 |
| P3 | LD.4.output | 1.802 | 43.4006 | 231.735 |
| P3 | LD.5.output | 2.06026 | 48.3833 | 238.406 |
| P3 | LD.6.output | 2.3023 | 52.7254 | 252.47 |
| P3 | LD.7.output | 2.52474 | 57.0128 | 276.78 |
| Q3 | LD.1.output | 1.00405 | 10.8008 | 43.5104 |
| Q3 | LD.2.output | 1.0031 | 10.7661 | 44.5009 |
| Q3 | LD.3.output | 1.0017 | 11.7001 | 43.63 |
| Q3 | LD.4.output | 1.00247 | 10.5903 | 40.2287 |
| Q3 | LD.5.output | 1.00013 | 10.1033 | 37.9862 |
| Q3 | LD.6.output | 1.00333 | 10.5103 | 38.6246 |
| Q3 | LD.7.output | 1.0009 | 10.3051 | 38.0151 |


| Variant | Block output | Max FP32/FP64 abs error | Max outside proportion | Max relative L2 |
| --- | --- | --- | --- | --- |
| A0 | LD.1.output | 0.00219031 | 0.205354 | 5.19349e-06 |
| A0 | LD.2.output | 3.87728 | 0.393171 | 0.00133403 |
| A0 | LD.3.output | 109.133 | 0.780299 | 0.040123 |
| A0 | LD.4.output | 315.922 | 0.973925 | 0.100117 |
| A0 | LD.5.output | 646.698 | 0.993196 | 0.283554 |
| A0 | LD.6.output | 1191.5 | 0.997039 | 0.515945 |
| A0 | LD.7.output | 2100.17 | 0.997299 | 0.672046 |
| P3 | LD.1.output | 0.000759221 | 0.150708 | 2.92091e-06 |
| P3 | LD.2.output | 0.00465398 | 0.351033 | 9.98675e-06 |
| P3 | LD.3.output | 0.0225327 | 0.540851 | 2.74705e-05 |
| P3 | LD.4.output | 0.0780462 | 0.688052 | 6.86872e-05 |
| P3 | LD.5.output | 0.521956 | 0.785822 | 0.000198741 |
| P3 | LD.6.output | 0.541189 | 0.849638 | 0.000354402 |
| P3 | LD.7.output | 1.06427 | 0.895305 | 0.0005962 |
| Q3 | LD.1.output | 5.19053e-05 | 0.0575982 | 1.85449e-06 |
| Q3 | LD.2.output | 0.000163682 | 0.197462 | 5.43391e-06 |
| Q3 | LD.3.output | 0.000312246 | 0.39677 | 1.55409e-05 |
| Q3 | LD.4.output | 0.00120921 | 0.625444 | 4.19295e-05 |
| Q3 | LD.5.output | 0.0041779 | 0.776488 | 0.000105105 |
| Q3 | LD.6.output | 0.0122148 | 0.903878 | 0.000311235 |
| Q3 | LD.7.output | 0.0723247 | 0.961248 | 0.00107883 |


## Derivative findings and limits

The complete derivative table is `derivative_category_summary.json`; `all_local_records.csv` and original per-job JSONL contain every cut, cotangent scale, precision and parameter. `weak_and_cancellation_failures.jsonl` preserves every failed ordinary Q/K weight and key-bias comparison. `paired_variant_metrics.csv` compares common parameters within identical fixtures; new LN parameters remain explicitly unpaired. LOWER/HIGHER describe one metric, not universal improvement or regression.

| Variant vs A0 | FP32/FP64 derivative metric | Lower | Higher | Equal | Undefined/unpaired |
| --- | --- | --- | --- | --- | --- |
| P3 | max_abs_error | 97856 | 14140 | 75276 | 88128 |
| P3 | outside_count | 74267 | 6731 | 106274 | 88128 |
| P3 | relative_l2 | 88706 | 23290 | 6426 | 156978 |
| Q3 | max_abs_error | 102357 | 9639 | 75276 | 88128 |
| Q3 | outside_count | 75603 | 4749 | 106920 | 88128 |
| Q3 | relative_l2 | 86671 | 25325 | 6426 | 156978 |


Lower and higher counts are reported together: they are paired numerical metric changes under identical local inputs, not a model-quality ranking. Undefined includes zero reference norms and extra LN parameters without an A0 counterpart. Different architectures have different derivatives and activation scales, so a smaller absolute error can coexist with worse coordinate-wise or relative accuracy. The full paired table retains every such regression.

| Family | Variant | Check | Precision | Cut | Parameter class | PASS | FAIL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NATT | A0 | directional_FD | float64 | primitive | ordinary_QK_weight | 72 | 0 |
| NATT | A0 | ordinary_derivative_oracle | float32 | primitive | ordinary_QK_weight | 216 | 72 |
| NATT | A0 | ordinary_derivative_oracle | float32 | primitive | ordinary_key_bias | 108 | 36 |
| NATT | A0 | ordinary_derivative_oracle | float64 | primitive | ordinary_QK_weight | 288 | 0 |
| NATT | A0 | ordinary_derivative_oracle | float64 | primitive | ordinary_key_bias | 144 | 0 |
| NATT | P3 | directional_FD | float64 | primitive | ordinary_QK_weight | 72 | 0 |
| NATT | P3 | ordinary_derivative_oracle | float32 | primitive | ordinary_QK_weight | 144 | 144 |
| NATT | P3 | ordinary_derivative_oracle | float32 | primitive | ordinary_key_bias | 72 | 72 |
| NATT | P3 | ordinary_derivative_oracle | float64 | primitive | ordinary_QK_weight | 288 | 0 |
| NATT | P3 | ordinary_derivative_oracle | float64 | primitive | ordinary_key_bias | 144 | 0 |
| NATT | Q3 | directional_FD | float64 | primitive | ordinary_QK_weight | 72 | 0 |
| NATT | Q3 | ordinary_derivative_oracle | float32 | primitive | ordinary_QK_weight | 204 | 84 |
| NATT | Q3 | ordinary_derivative_oracle | float32 | primitive | ordinary_key_bias | 96 | 48 |
| NATT | Q3 | ordinary_derivative_oracle | float64 | primitive | ordinary_QK_weight | 288 | 0 |
| NATT | Q3 | ordinary_derivative_oracle | float64 | primitive | ordinary_key_bias | 144 | 0 |
| NLD | A0 | VJP_additivity | float32 | 1 | ordinary_QK_weight | 1285 | 347 |
| NLD | A0 | VJP_additivity | float32 | 1 | ordinary_key_bias | 690 | 126 |
| NLD | A0 | VJP_additivity | float32 | 4 | ordinary_QK_weight | 3540 | 2988 |
| NLD | A0 | VJP_additivity | float32 | 4 | ordinary_key_bias | 1996 | 1268 |
| NLD | A0 | VJP_additivity | float32 | 7 | ordinary_QK_weight | 4826 | 6598 |
| NLD | A0 | VJP_additivity | float32 | 7 | ordinary_key_bias | 2828 | 2884 |
| NLD | A0 | VJP_additivity | float64 | 1 | ordinary_QK_weight | 1632 | 0 |
| NLD | A0 | VJP_additivity | float64 | 1 | ordinary_key_bias | 816 | 0 |
| NLD | A0 | VJP_additivity | float64 | 4 | ordinary_QK_weight | 6118 | 410 |
| NLD | A0 | VJP_additivity | float64 | 4 | ordinary_key_bias | 2988 | 276 |
| NLD | A0 | VJP_additivity | float64 | 7 | ordinary_QK_weight | 9799 | 1625 |
| NLD | A0 | VJP_additivity | float64 | 7 | ordinary_key_bias | 4841 | 871 |
| NLD | P3 | VJP_additivity | float32 | 1 | ordinary_QK_weight | 1128 | 504 |
| NLD | P3 | VJP_additivity | float32 | 1 | ordinary_key_bias | 598 | 218 |
| NLD | P3 | VJP_additivity | float32 | 4 | ordinary_QK_weight | 4636 | 1892 |
| NLD | P3 | VJP_additivity | float32 | 4 | ordinary_key_bias | 2372 | 892 |
| NLD | P3 | VJP_additivity | float32 | 7 | ordinary_QK_weight | 8021 | 3403 |
| NLD | P3 | VJP_additivity | float32 | 7 | ordinary_key_bias | 4092 | 1620 |
| NLD | P3 | VJP_additivity | float64 | 1 | ordinary_QK_weight | 1632 | 0 |
| NLD | P3 | VJP_additivity | float64 | 1 | ordinary_key_bias | 816 | 0 |
| NLD | P3 | VJP_additivity | float64 | 4 | ordinary_QK_weight | 6526 | 2 |
| NLD | P3 | VJP_additivity | float64 | 4 | ordinary_key_bias | 3252 | 12 |
| NLD | P3 | VJP_additivity | float64 | 7 | ordinary_QK_weight | 11410 | 14 |
| NLD | P3 | VJP_additivity | float64 | 7 | ordinary_key_bias | 5658 | 54 |
| NLD | Q3 | VJP_additivity | float32 | 1 | ordinary_QK_weight | 1247 | 385 |
| NLD | Q3 | VJP_additivity | float32 | 1 | ordinary_key_bias | 624 | 192 |
| NLD | Q3 | VJP_additivity | float32 | 4 | ordinary_QK_weight | 4726 | 1802 |
| NLD | Q3 | VJP_additivity | float32 | 4 | ordinary_key_bias | 2521 | 743 |
| NLD | Q3 | VJP_additivity | float32 | 7 | ordinary_QK_weight | 8446 | 2978 |
| NLD | Q3 | VJP_additivity | float32 | 7 | ordinary_key_bias | 4426 | 1286 |
| NLD | Q3 | VJP_additivity | float64 | 1 | ordinary_QK_weight | 1632 | 0 |
| NLD | Q3 | VJP_additivity | float64 | 1 | ordinary_key_bias | 816 | 0 |
| NLD | Q3 | VJP_additivity | float64 | 4 | ordinary_QK_weight | 6528 | 0 |
| NLD | Q3 | VJP_additivity | float64 | 4 | ordinary_key_bias | 3264 | 0 |
| NLD | Q3 | VJP_additivity | float64 | 7 | ordinary_QK_weight | 11422 | 2 |
| NLD | Q3 | VJP_additivity | float64 | 7 | ordinary_key_bias | 5712 | 0 |


Singleton VIM Q/K gradients are allocated exact zeros in the completed NLD runs. Ordinary key-bias derivatives cancel analytically but floating arithmetic can leave nonzero residues; these are reported independently against zero and, for NATT, against the analytical derivative. Large relative errors on tiny reference norms are retained rather than dismissed. Local ordinary Q/K derivatives and LN derivatives use independent formulas; directional finite differences are an additional check, not a substitute for coordinate-wise failures.

| Variant | Zero/cancellation class | PASS records | FAIL records |
| --- | --- | --- | --- |
| A0 | VIM_QK_allocated_exact_zero | 19584 | 0 |
| A0 | ordinary_key_bias_within_original_tolerance | 14459 | 5413 |
| P3 | VIM_QK_allocated_exact_zero | 19584 | 0 |
| P3 | ordinary_key_bias_within_original_tolerance | 17011 | 2861 |
| Q3 | VIM_QK_allocated_exact_zero | 19584 | 0 |
| Q3 | ordinary_key_bias_within_original_tolerance | 17651 | 2221 |


Fixed-cotangent VJP additivity uses one forward graph per precision. The original comparison reference is the FP64 sum of individual gradients cast back to the native precision. A separate record compares to the unrounded FP64 sum. Cotangent-sum rounding, cancellation numerator/denominator, weak coordinates, sign disagreements and four fixed directional errors remain visible. Native FP64 forms u+v after promotion; all individual weights/inputs originate from FP32.

For `*_FP32_vs_FP64` rows, the precision metadata identifies the FP64 reference pass; the stored atol/rtol are the unchanged FP32 comparison thresholds. Temporary FP32 gradient vectors are removed only after their corresponding FP64 comparisons are recorded; all comparison statistics, gradient hashes and failures remain. Reproducing full vectors requires replaying the frozen recipe. The interrupted attempt retains its still-present temporary vectors and their SHA-256 inventory.

A numerical FAIL does not become PASS because its worker exited zero. The exit code confirms that mandatory finiteness, ownership, state and structural-zero checks allowed the finite diagnostic schedule to continue. The pilot supplies no automatic acceptance boundary and cannot justify production promotion or explain away the eight existing failed assertions.

## Resources and backend

Measured worker time summed across executed jobs: 46187.552 s. Maximum sampled process RSS: 2.183 GiB (50 ms sampling, not a guaranteed instantaneous peak). Minimum available RAM at a worker boundary: 0.935 GiB. Individual cold forwards, warm-up policy and three state-reloaded no-update timing repetitions are retained per precision in raw records. No concurrent full model was run.

These are shared-workstation measurements, not isolated performance benchmarks. Read-only interim aggregation, checkpoint hashing and small reporting tests overlapped some numerical workers; `concurrent_analysis_record.json` records this timing limitation. No variant speed ranking is inferred. A follow-on controller wait found that the prior PID had already exited successfully; that exit-1 handoff attempt performed no numerical work, and `controller_handoff_record.json` records the inspection and continuation without repeating jobs.

NLD cold-forward timing includes stage-capture hooks and detached clones. The three warmed repetitions measure an unhooked full-depth forward plus the registered selected-parameter/input VJP, not a full six-loss backward. Process RSS includes reference/copy states, captured tensors and diagnostic temporaries; it is not an isolated production-module footprint or a full-model training-capacity measurement.

Actual backend records:

```json
[
  {
    "cuda_available": false,
    "deterministic": true,
    "interop_threads": 4,
    "matmul_precision": "highest",
    "mkldnn": false,
    "platform": "Windows-11-10.0.26200-SP0",
    "python": "3.14.3",
    "threads": 4,
    "torch": "2.12.1+cpu"
  },
  {
    "cuda_available": false,
    "deterministic": true,
    "interop_threads": 4,
    "matmul_precision": "highest",
    "mkldnn": true,
    "platform": "Windows-11-10.0.26200-SP0",
    "python": "3.14.3",
    "threads": 1,
    "torch": "2.12.1+cpu"
  },
  {
    "cuda_available": false,
    "deterministic": true,
    "interop_threads": 4,
    "matmul_precision": "highest",
    "mkldnn": true,
    "platform": "Windows-11-10.0.26200-SP0",
    "python": "3.14.3",
    "threads": 4,
    "torch": "2.12.1+cpu"
  }
]
```

CPU B1/B2 results are listed cell by cell. GPU and B8: NOT RUN. N2 full-language sweeps, G2/NG2 six-loss sweeps, A1-D02B, G3/G4, I1/R2, optimizers, schedulers, training and datasets: NOT RUN / outside this authorization.

## Evidence and reproduction

Raw logs/JUnit, immutable driver versions, paired state/input hashes, JSONL records, complete CSV tables, parameter inventories and backend/resource records are under `artifacts/phase2d2_d02a/`. The final preservation manifest verifies the frozen baseline and original historical evidence. The review package manifest identifies included files and any regenerated caches excluded from the ZIP. Follow its reproduction instructions in a fresh directory; existing evidence is never overwritten.

**Stop for review. Phase 2D.2 remains BLOCKED; full A1-D02B, G3/G4 and training require separate authorization.**
