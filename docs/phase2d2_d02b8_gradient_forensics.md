# D02B8 G01-A0 gradient failure forensics

2026-10-17, Asia/Saigon. **Read-only raw-evidence analysis; Phase 2D.2 remains BLOCKED.** No MFVLR forward/backward, optimizer step, new fixture or backend experiment was executed. B0-D08 remains disabled.

## Integrity and independent recomputation

All **2,392 binary shards**, totaling **3,790,126,768 bytes**, match the preserved SHA-256 inventory. Each original FP32 gradient was read independently; missing gradients retained their original None status while contributing mathematical zero to the reference sum. The six allocated FP32 gradients were converted to FP64 and added in FD/LR/CMC/FL/AR/KL order. Every coordinate of all 508 recomputed sums equals the saved FP64 sum shards exactly.

The independent tool imports neither the production GradientStore nor torch. It independently evaluates the original FP32 elementwise test against the rounded FP64 sum at **atol=1e-6, rtol=1e-5**. All 508 parameter failure counts and 508 JUnit PASS/FAIL decisions match: **293 failed parameters and 8,772,234 outside-tolerance coordinates**. This reconciliation passes as evidence verification; the numerical comparisons remain FAIL.

Secondary metrics match exactly for 486/508 parameters. The complete exact-difference ledger is `secondary_metric_differences.json`; no tolerance was used to suppress differences. The original chunk size and direction seeds/name salt were retained because they define directional probe identities. Independent implementations can share those scientific definitions without importing the original calculator.

## Module attribution

| Module | Failed / parameters | Outside coordinates | Absolute error L2 | Relative L2 |
| --- | ---: | ---: | ---: | ---: |
| FLT | 179/387 | 5640256 | 12.7850533 | 4.7210901061906855e-07 |
| MVE | 88/88 | 2231781 | 31.5530807 | 1.2413416266993764e-06 |
| VD | 26/28 | 900197 | 0.959540585 | 3.4970902910467295e-06 |
| heads | 0/4 | 0 | 0 | 0.0 |
| objectives | 0/1 | 0 | 0 | undefined |

Module L2 values combine per-parameter squared norms. They do not replace coordinate checks. A low global relative error can coexist with many failed weak coordinates. All failed parameters, including absolute/relative L2, zero-reference behavior, signs, six contributions, cancellation, cosine, all 15 pair metrics and four directional probes, appear in `failed_parameter_rankings.json`; `parameter_forensics.json` also retains passing parameters.

## Gradient categories

| Category | Failed / parameters | Outside | Sign disagreements | Zero-reference/nonzero actual | Near-nonzero failed |
| --- | ---: | ---: | ---: | ---: | ---: |
| other | 246/380 | 4030708 | 15 | 0 | 9 |
| ordinary_key_bias_cancellation | 15/30 | 7183 | 2091 | 1 | 227 |
| ordinary_qk_weight | 32/60 | 4734343 | 6815 | 0 | 915 |
| singleton_vim_qk | 0/38 | 0 | 0 | 0 | 0 |

The near-nonzero reporting bin is **0 < abs(FP64 reference) <= 1e-6**. It uses the existing absolute-tolerance scale solely to describe coordinates, not to change any PASS/FAIL criterion. Exact reference zero is recorded separately. Undefined cosine/relative error/cancellation for zero denominators remains null rather than zero or an invented finite value.

Ordinary attention key biases have analytically cancelling derivatives, but numerical residuals are not the singleton VIM structural-zero category. Ordinary Q/K weights are not expected to be zero. Their weak-gradient errors must remain visible. All singleton VIM Q/K gradient shards remain exactly zero wherever allocated; unconnected paths remain None. Key-bias residuals cannot be waived because the analytical result is zero.

## Objective scale and B1 limitations

| Loss | Global isolated gradient L2 |
| --- | ---: |
| lr | 37142368.6591 |
| total | 37142195.4807 |
| kl | 1444.86740729 |
| fd | 1046.97116527 |
| fl | 1.60907806457 |
| ar | 1.57327659994 |
| cmc | 0 |

LR gradient norm is about 25706.420 times KL and 35476.019 times FD. This is an observed scale disparity, not proof that LR alone caused joint-versus-isolated discrepancies. Per-loss norms and pair cosines quantify contributions and alignment; they cannot assign a unique causal share of total numerical error to one loss.

The existing LR scalar is 248989.578125 and total is 249028.046875. All 308 positions are included: 276 pad, 24 content, four BOS and four EOS. Recorded FP64 sums of actual FP32 token NLL values are pad 187149.6273803711, content 49556.35498046875, BOS 6816.90380859375, EOS 5466.690673828125. Category regrouping differs from production FP32 LR by -0.00128173828125; it does not replace the approved reduction. Padding contributes most of this fixture’s LR value, but token counts/scalars do not by themselves establish category-wise parameter-gradient dominance. No PAD masking, normalization, weighting or reduction change is proposed for the reproduction baseline.

The B1 CMC objective is zero and all its allocated gradients are zero. It supplies no evidence about nontrivial in-batch contrastive coupling, duplicate-prompt effects or B2 temperature gradients. G01 is Real with all-background localization targets and no manipulated foreground; it cannot establish fake detection/localization behavior, foreground gradient conditioning, dataset generalization, convergence or accuracy. Finite gradients do not establish finite optimizer updates or repeated-step stability.

## Observed facts versus hypotheses

| Hypothesis | Supporting observations | Limitation / falsification test (not executed) |
| --- | --- | --- |
| FP32 backward accumulation/reduction order contributes | Actual isolated and joint gradients disagree while source/state are preserved and values remain finite; error is not a disk-sum bookkeeping problem | Does not identify a kernel or operation. A separately approved same-forward full-roster control can limit separate-forward explanations; operator/backend controls need their own approval |
| Cancellation amplifies relative error | Per-parameter cancellation ratios, weak references, signs and key-bias residuals are explicitly measured | Many errors may occur at cancellation ratio near one; cancellation is not a universal explanation. Examine full ratio/error distributions and controlled local derivatives |
| Separate graph paths/forward rounding contribute | R2 used seven fresh graphs with identical scalar results | Equal scalar losses do not prove all intermediate/gradient values equal. Historical N3 same-forward failures already show fresh graphs are not the only possible mechanism; a new controlled full-roster comparison would test this case |
| Backend sensitivity contributes | FP32 arithmetic is operation/order-sensitive in principle | This phase has only one recorded C4M cohort; no causal backend comparison is available here. A matched backend-control experiment would be required and is not authorized |
| LR scale increases numerical risk | LR scalar and gradient norms dominate this fixture | Correlation does not locate the source of error. Loss rescaling would change the baseline and is neither performed nor authorized |

No single root cause has been established. Summing isolated gradients in FP64 is a comparison reference, not a mathematically exact ground-truth derivative or full FP64 autodiff. All original failures remain intact.

## Decision and preparation

Further read-only attribution is supported. The recommended next numerical step is one separately approved, preregistered G01-A0 same-forward no-update control with full-roster coverage; see `phase2d2_d02b8_next_step_decision.md`. A bounded optimizer smoke test is not recommended as the immediate step and remains separately unauthorized. No automatic continuation or tolerance amendment follows from this analysis.

Separate proposed configuration/dataset/checkpoint contracts are in `phase2d2_d02b8_preparation_contract.md` and `src/mfvlr_research/d02b8/contracts.py`. They validate metadata only and cannot run or load a model. Training/inference modes remain disabled; no dataset preprocessing, optimizer or checkpoint execution is implemented.

## Evidence limits and reproducibility

The compact archive contains independent analysis source, per-parameter/per-loss/module results, exact mismatch ledger, full shard index, original summary/JUnit records, mock tests and preservation hashes. Original 3.79 GB binary shards remain untouched on disk and are omitted from the compact archive; reproducing the full arithmetic independently requires those indexed files. Inspecting summaries does not substitute for those raw bytes. `shard_integrity.json` records the direct on-machine verification performed in this phase.

No original failed-run record, source, model, fixture or threshold was rewritten. Stop for independent review. Phase 2D.2 remains BLOCKED.

## Secondary-metric reconciliation

The 22 cancellation-ratio differences were isolated to NumPy summation of six recomputed norms versus the original sequential Python sum. Replaying the original sequential sum on independently recomputed norms exactly reproduces all 508 original ratios. Both values and every difference remain preserved in `cancellation_ratio_reconciliation.json`. This establishes the cause of that reporting-only difference, **not** the cause of joint-gradient numerical failures. All other checked secondary metrics, including pair cosines and directional probes, matched exactly.

## Per-loss attribution boundary

`per_loss_discrepancy_associations.json` ranks each loss's nonzero parameter contributions and associates them with observed joint-additivity errors. These are overlapping associations, not causal shares of an error; adding their failure counts would double-count shared parameters. Ordinary Q/K weights account for 4,734,343 failed coordinates and key biases for 7,183. All 38 singleton VIM Q/K parameters have zero failed coordinates. The maximum reporting-only cancellation-ratio discrepancy was 2.220446049250313e-16; replay of the original scalar order resolves it exactly without changing any gradient comparison.

## Executed verification

The isolated D02B8 suite passed 11/11 cases, exit code 0, with zero skips/xfails and no torch import. The independent recomputation process exited 0. Its 508-case reconciliation JUnit describes exact agreement with the original decisions, not 508 numerical passes: the included original live-additivity JUnit still has 293 failures. Source and historical preservation results are in `before.json` and `after.json`; compact-package CRC and SHA-256 checks are in `package_verification.json`.
