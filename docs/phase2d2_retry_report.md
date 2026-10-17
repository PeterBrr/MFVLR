# Phase 2D.2 retry verification

**Status: BLOCKED.** The fresh, unfiltered suite executed **321 cases: 313 passed, 8 failed, 0 errors, 0 skips**. All **276 accepted cases pass unchanged**. No xfails, deselections, relaxed thresholds, architecture changes or optimizer updates. The original [report](phase2d2_implementation_report.md), failed logs, evidence, manifests and ZIP remain byte-identical.

The retry fixes a verification-harness limitation: the new joint tests can now read a separately selected evidence directory. Previously a pytest rerun alone would read the old G2 records. `MFVLR_PHASE2D2_EVIDENCE_ROOT` redirects only the two unaccepted Phase 2D.2 test modules; I10 and READY-N02 now consume the freshly recomputed evidence. Default paths remain the original ones. Every assertion AST and every test function apart from I10's file-path assignment is unchanged. Original test sources are archived, not discarded. Accepted production source and configurations are untouched.

## Fresh computation and preserved comparisons

Each of the six full-size CPU FP32 fixtures ran in a separate process, with six isolated fresh forward/backward graphs and one literal joint graph. All outputs, captured 12 LE/7 LD stages, loss scalars and gradients remain finite. Parameters and identities do not update; lifecycle checks pass. G2 status remains determined by the unchanged elementwise FP32 gradient-additivity assertion, not by finiteness alone.

| Fixture | Outside-tolerance elements | Relative L2 difference | Peak sampled RSS MiB | Wall seconds |
|---|---:|---:|---:|---:|
| real | 8,772,234 | 9.16973535e-07 | 2266.16 | 202.78 |
| efs | 13,636,267 | 1.95149713e-06 | 2276.80 | 197.39 |
| local | 11,558,833 | 1.53557234e-06 | 2277.40 | 207.72 |
| real_efs | 14,149,076 | 1.93785674e-06 | 2855.69 | 272.25 |
| real_local | 12,728,291 | 1.67403775e-06 | 2847.75 | 239.14 |
| local_duplicate | 14,611,405 | 2.05079921e-06 | 2882.44 | 271.62 |

All scientific values in the per-loss/joint records and gradient comparisons exactly match the earlier six-fixture evidence: **True**. Timing fields are excluded; only the unordered selected cosine-entry list is canonicalized by parameter/loss pair because its serialization order comes from a Python set. Tensor/stage/token arrays retain their original order. Raw evidence is unchanged. Original N2 F0/F1/F2 were also recomputed in FP32 and promoted-weight FP64; all scientific/backend fields exactly match: **True**, excluding elapsed timing only. Neither N1 nor N2 was extended to the new gradient criteria. Measurements include model construction, seven graphs, statistics, hashing and disk streaming; RSS is sampled, not an exact allocator peak. CPU B8/GPU are **NOT RUN** with explicit resource reasons, not successful capacity checks.

## Additional CMC numerical diagnosis

The independent retry probe uses the exact real/EFS fused visual input, 308-token prompts and all 12 accepted LE blocks. Both workers deep-copy the same FP32-initialized full FLT and CMC before optional FP64 promotion. Source-weight and FP32 visual-input hashes agree between workers. No alternate initialization, normalization, rescaling, clipping, detachment of production paths or pretrained weights is introduced.

At fused Iv, direct-plus-language CMC partial gradients versus the joint derivative have:

| Precision | Outside tolerance | Maximum absolute difference | Relative L2 difference | Existing tolerance |
|---|---:|---:|---:|---|
| FP32 | 3 / 1,024 | 7.62939453125e-05 | 8.56569771995e-08 | atol 1e-6, rtol 1e-5 |
| Promoted FP64 | 0 / 1,024 | 1.70530256582e-13 | 1.50757812404e-16 | atol 1e-10, rtol 1e-8 |

Selected ordinary LE1 attention-Q and LE12 FFN1 parameter derivatives equal their relevant language partials exactly in both precisions. These are ordinary attention projections, distinct from singleton VIM Q/K. This supports floating-point accumulation as the CMC chain-rule mismatch mechanism. It **does not prove full visual/LD gradient additivity in FP64, harmlessness of small-coordinate drift, repeated-step stability, convergence or accuracy**. The VD public contract permits FP32 only; no hidden FP64 visual wrapper or changed production API was used. Large LR gradients and unnormalized LD activation magnitudes remain recorded risks.

The CMC scalar itself changes from 206.232910156 in FP32 to 206.21038359 in promoted FP64. FP64 promotion re-evaluates forward arithmetic too; passing a high-precision chain-rule comparison is not a claim that the two precision modes give identical forward results.

## Ordinary attention key-bias cancellation probe

**DERIVED mathematics, not recovered paper architecture:** for the accepted biased ordinary attention, adding key bias adds `q_i^T b_k / sqrt(64)` to every key score in row i. Softmax is invariant to that common offset, so the exact-real key-bias task derivative cancels. This differs from the accepted singleton VIM whose Q/K derivatives are exact zero in the executed graph. Ordinary biased attention can leave floating-point cancellation residuals.

A separate full-width 512-dimensional, eight-head, 308-token attention block probe deep-copies identical FP32-initialized weights and input/cotangent fixtures, with scales 1 and 10,000. It exercises the accepted block, not a full-model loss or training step. No bias is removed or gradient overwritten.

| Precision | Cotangent scale | Maximum absolute key-bias gradient | Outside existing zero-comparison tolerance |
|---|---:|---:|---:|
| torch.float32 | 1 | 4.74974513054e-08 | 0 |
| torch.float32 | 10000 | 0.000411987304688 | 494 |
| torch.float64 | 1 | 9.71445146547e-17 | 0 |
| torch.float64 | 10000 | 9.52127265919e-13 | 0 |

[Machine-readable probe](../artifacts/phase2d2_retry/key_bias_review.json) includes score-gradient row sums, hashes and backend metadata. For the full real G2 fixture, ordinary LE3 key-bias joint gradient L2 is about 0.003992 and the isolated-sum discrepancy has relative L2 about 1.531; MVE block2 key-bias relative discrepancy is about 1.677. These near-cancelling derivatives explain why global relative-L2 agreement alone can obscure large coordinate-wise ratios. Other ordinary Q/weight discrepancies also remain; this probe does not explain away every failing parameter or grant a numerical exemption.

## Acceptance boundary and review files

Full root: **154,156,232** unique parameters. Physical visual deployment: **41,135,303**. Weight tying, ownership and serialization are reverified in the fresh suite. [Verification](../artifacts/phase2d2_retry/verification.json), [preservation](../artifacts/phase2d2_retry/preservation.json), [G2 comparison](../artifacts/phase2d2_retry/g2_comparison.json), [raw pytest log](../artifacts/phase2d2_retry/logs/complete_retry_01.stdout.log), [JUnit](../artifacts/phase2d2_retry/junit/complete_retry_01.xml), [FP32 CMC](../artifacts/phase2d2_retry/chain/float32.json), [FP64 CMC](../artifacts/phase2d2_retry/chain/float64.json), [review ZIP](../artifacts/phase2d2_retry/phase2d2_retry_review.zip).

The failed comparisons remain mandatory failures. No numerical policy amendment is inferred from “retry.” Phase 2D.2 remains blocked under the current criteria if these failures persist. P2D15/P2D19, dataset processing, optimizer/training execution and benchmark evaluation remain unapproved and unimplemented. Stop here for review.
