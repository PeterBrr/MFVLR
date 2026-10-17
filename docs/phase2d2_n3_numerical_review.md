# Phase 2D.2 N3 joint-gradient numerical review

**Diagnosis completed; Phase 2D.2 remains BLOCKED. No numerical-policy amendment is applied or recommended for immediate acceptance.** Same-forward controls reproduce the gradient discrepancy. Separate forwards, stale gradients and incorrect ownership do not explain the selected results. Floating-point backward accumulation and cancellation are supported mechanisms, but additional fixed-cotangent controls reveal percent-scale sensitivity that a small global gradient error conceals.

This is a review of the approved engineering baseline, not a recovery of the authors' undocumented implementation. No production source, existing test, configuration, loss coefficient, initialization or historical artifact was changed. No optimizer, update, dataset processing, evaluation, G3 or G4 was executed. The authorization is preserved in [user_request.txt](../artifacts/phase2d2_n3/history/user_request.txt).

## Evidence and scope

The preserved [initial report](phase2d2_implementation_report.md), [retry report](phase2d2_retry_report.md), diagnostic implementation and two joint test modules were reviewed. The audit reads **all 14 preserved G2 directories**, including the original real pilot, six final original fixtures, six retry fixtures and the retry real alias. Each contains 508 parameter discrepancy records and seven per-loss/joint gradient records. The original versus retry final scientific records agree exactly, excluding timing and canonicalizing only the unordered cosine-entry list. See [historical audit](../artifacts/phase2d2_n3/historical_audit.json).

The historical mandatory result remains **321 collected, 313 passed, eight failed**, with all 276 accepted cases passing unchanged. This review did **not** rerun that full suite or convert its failures into diagnostic successes. Seven proposed supplemental controls were executed only in an isolated artifact directory: **7 passed**, exit 0, no skips or xfails. They are not installed under `tests/`, do not replace an assertion and do not grant acceptance. [Actual log](../artifacts/phase2d2_n3/logs/proposed_controls_dryrun.stdout.log), [JUnit](../artifacts/phase2d2_n3/junit/proposed_controls_dryrun.xml).

The independent evidence is in [same_forward/](../artifacts/phase2d2_n3/same_forward), [actual_LE08.json](../artifacts/phase2d2_n3/actual_LE08.json), [decoder_vjp.json](../artifacts/phase2d2_n3/decoder_vjp.json), [local_analytical.json](../artifacts/phase2d2_n3/local_analytical.json) and [chain/](../artifacts/phase2d2_n3/chain). Raw commands, stdout, stderr and exit codes are in [logs/](../artifacts/phase2d2_n3/logs).

**EXPLICIT accepted engineering requirements:** full dimensions, singleton VIM formulation, shared parameters, six unit coefficients and original tolerances. **DERIVED:** exact-real gradient linearity and ordinary key-bias cancellation. **ASSUMED diagnostic choices:** selected tensors, seeds 801–804 for directional projections, 901/902 for cotangents, controlled scales and alternative CPU backend. None is a published MFVLR requirement. Paper/equation interpretations remain in the unchanged [loss contract](phase2d_losses_contract.md); no published equation is rewritten here.

## Same-forward experiment and identities

For each run, initialize one full root with the accepted seed and inputs. Its 154,156,232 parameters have state SHA-256 `1865527facba9adc8a14d443c112d492698d275ce4e8beaa784aee61a514895a` in all four runs.

1. **A:** `autograd.grad` of the literal FD/LR/CMC/FL/AR/KL sum on one forward graph G0.
2. **B:** six isolated `autograd.grad` calls on those exact same six scalar tensor objects and selected input tuple, retaining G0 until its final use. Sum the rounded gradients in a streamed FP64 accumulator.
3. **C:** six fresh deterministic forwards with the original full `.backward()` execution path, plus one fresh joint backward. Only one full graph is live at a time. Capture the selected vectors and compare them with A/B.

There are 24 selected parameter tensors, 11 intermediate tensors and six scalar loss nodes: 41 selected inputs. Coverage includes early CNN; shared MVE ordinary Q/K weights/key bias; VD shared trunk, AD and MD weights; detector/adapter/temperature; entire tied vocabulary; early/late LE/LD ordinary attention; four selected singleton VIM Q/K parameters; detector/mask/adapter/vocabulary outputs; Iv, Ipre, Tl; LE01/LE12 and LD01/LD07. Parameter names are explicit in the tool and JSON. This is selected raw-vector coverage, not a claim that all 508 parameter vectors were recaptured.

Records include Python tensor/Parameter IDs, storage/data pointers, offsets, shape, stride, dtype, leaf status, grad-function type and forward-value hashes. Same-forward calls reuse live objects. Across separate forwards, output hashes and parameter identity dictionaries match G0 exactly within each run. Graph labels are experiment labels, not a PyTorch global graph identifier. Weakrefs verify old outputs are released; hooks are removed; parameter `.grad` fields are empty before C. Full parameter gradients are checked finite during C; all 38 connected singleton VIM Q/K gradients are checked allocated exact zero. Root state and parameter identities do not change.

**Metadata limitation and proposed repair:** `loss_grad_fn_python_ids` in the four primary G0 records sampled transient Python wrappers before retaining them. CPython reused their addresses, so those optional IDs cannot identify distinct loss graph nodes. The live loss **Tensor** IDs/storage/type records remain valid; forward intermediate wrappers were held correctly. The actual LE8 probe holds named loss wrappers before recording their IDs. This is a new diagnostic-metadata bug, not an arithmetic, ownership or production-model defect. It cannot cause the measured gradients: calculations use live tensors, never sampled integers. The exact cause, source hash and separate [unapplied repair](../artifacts/phase2d2_n3/proposal/proposed_metadata_repair.patch) are preserved in [metadata_issue.json](../artifacts/phase2d2_n3/proposal/metadata_issue.json). No record was silently corrected.

## Results: separate forwards are not the explanation

In **all four runs**, every selected B_i and C_i agrees exactly, including gradient presence; B-sum equals C-sum exactly; A equals the fresh C-joint exactly. Thus separate-forward execution and `autograd.grad` versus full `.backward()` do not account for the selected discrepancy. The three standard-backend runs also reproduce all 24 selected historical parameter norm/statistic records exactly for every isolated loss and the total. The alternative backend changes forward arithmetic and some historical statistics, as explicitly recorded; its within-run equalities still hold.

The comparison below is A versus B. Relative L2 is `||A-B||2 / ||B||2`; the denominator is the unrounded FP64 sum of FP32 gradients. Elementwise counts use the old single cast to FP32 and **atol 1e-6, rtol 1e-5**, unchanged.

| Fixture/backend | Iv relative L2 | Iv outside count | Ipre relative L2 | Ipre outside count | LE8 Q-weight relative L2 |
|---|---:|---:|---:|---:|---:|
| real / MKLDNN | 1.23830e-6 | 37 | 1.54376e-6 | 14,355 | 0.00570339 |
| real_efs / MKLDNN | 3.39871e-6 | 186 | 3.58359e-6 | 59,624 | 0.00480802 |
| local_duplicate / MKLDNN | 3.66108e-6 | 223 | 3.47619e-6 | 62,671 | 0.00351698 |
| real_efs / MKLDNN disabled | 3.46958e-6 | see CSV | 3.70361e-6 | see CSV | 0.0113030 |

Detector, mask, adapter and vocabulary-logit gradient comparisons are exact on the tested fixtures, as are the selected LD01/LD07 derivatives. Discrepancies appear upstream where losses share nonlinear Jacobians and parameter owners. For the real fixture, LE01 has 4,747 outside coordinates; the vocabulary has 89. LE12 has no outside coordinates although its unrounded sum difference is nonzero. Complete counts, maxima, proportions, signs, cosines and four directional projections are retained; no result is reduced to a global norm.

The historical full-gradient relative L2 ranges from 9.16974e-7 to 2.05080e-6, despite 8,772,234–14,611,405 outside coordinates. Real joint gradient L2 is about 37.14 million, almost entirely LR; FD is about 1,047, FL 1.61, AR 1.57, KL 1,445 and B=1 CMC is zero. A norm dominated by LR cannot certify small, weak or canceling parameter groups. Ordinary LE3 key-bias relative discrepancy is about 1.531; real LE8 Q-weight discrepancy is about 0.5703%. These are different mathematical classes.

## Fixed-cotangent VJPs: a material numerical risk

On the same G0, use fixed seeded full-shape cotangents u/v at Iv and LD07, scales 1 and 1024. Compare the VJP of the FP32 sum with the FP64 sum of the two individually computed FP32 VJPs. This includes rounding in `u+v` and in backward arithmetic; it is not an exact-real cotangent sum. Scales are diagnostic fixtures, not production rescaling.

| Cut/backend, scale 1 | Target | Relative L2 discrepancy |
|---|---|---:|
| Iv / MKLDNN | first CNN weight | 1.46855e-6 |
| Iv / MKLDNN | AD weight | 4.88130e-7 |
| Iv / MKLDNN | Ipre | 1.11955e-6 |
| LD07 / MKLDNN | Iv | 0.0559506 |
| LD07 / MKLDNN | AD weight | 0.0670823 |
| LD07 / MKLDNN | LE12 | 0.0332795 |
| LD07 / MKLDNN disabled | Iv | 0.0921837 |
| LD07 / MKLDNN disabled | first CNN weight | 0.136858 |
| LD07 / MKLDNN disabled | LE12 | 0.165449 |

Multiplying cotangents by 1024 preserves these relative discrepancies in the recorded controls; absolute differences scale. This does not justify a 16.6% tolerance. It demonstrates that the universal elementwise gate and a global-norm-only replacement are both inadequate descriptions of correctness and numerical risk.

The second backend is the same Windows/PyTorch/MKL environment with MKLDNN disabled, not a new platform or independent BLAS implementation. Batch size remains fixed in these VJPs; this is not the earlier N2 cross-batch test.

### Actual seven-block LD FP64 control

Capture the exact full-model real/EFS decoder inputs under `no_grad`, copy the same FP32-initialized seven-block LD, release the visual/full root, then run FP32 and promoted FP64 copies on those same stored values. State, LE memory and visual feature are independent differentiable leaves for this local diagnostic. No production path is detached or changed. The probe includes all seven accepted unnormalized LD blocks, but excludes the LE/visual Jacobians and vocabulary projection.

At scale 1, the FP32 relative VJP additivity discrepancies are **3.59431%** for decoder input state, **3.32795%** for LE memory, **5.12954%** for the visual leaf and **2.87173%** for LD1 self-attention Q weights. The local FP32 forward exactly reproduces the captured LD output.

Promoted FP64 reduces the same relative discrepancies to approximately **2.08e-11, 1.30e-10, 1.30e-10 and 1.65e-11**, respectively. Nevertheless, the unchanged FP64 elementwise criterion **fails at 2 state, 12 memory and one visual coordinate** at scale 1. At scale 1024 it fails at 1,099 state, 1,660 memory, three visual and 39 final memory-key-bias coordinates. Maximum state/memory/visual differences at scale 1 are 5.30e-10 / 8.40e-10 / 4.18e-10. These are reported failures, not discarded as diagnostic success. Selected VIM Q/K remain exactly zero in both precisions/scales.

Separately, using the rounded FP32 cotangent sum in the FP64 graph rather than its high-precision sum changes nonzero target VJPs by about 2.0–2.7e-8 relative L2. The input cotangent rounding itself is 2.75e-8 relative L2. For this local high-precision Jacobian, input-sum rounding alone is much smaller than the FP32 percent-scale discrepancy. Promotion also recomputes forward intermediates, so this does not give a full mathematical decomposition of each FP32 kernel error.

### Actual LE8 ordinary Q/K weights

The real-fixture probe captures LE8 normalized attention input, actual accepted attention weights and the joint/isolated loss cotangents on one full graph, then releases the full graph. Local FP32 recomputation exactly matches its full-graph Q/K/key-bias gradients with the joint cotangent.

The joint versus isolated-sum cotangents differ by only **2.12936e-7** relative L2. Forward/reverse FP32 addition of the three connected isolated cotangents gives the same rounded result here; simply reversing the final sum does not explain the discrepancy. At that exact FP32 attention input, replacing the joint cotangent with the summed cotangent changes Q/K-weight VJPs by **0.005722 / 0.005809** relative L2. In the promoted FP64 local graph, the corresponding effect is only **3.354e-7 / 3.304e-7**. FP64 local VJP additivity passes the unchanged tolerance for Q/K weights; its relative errors are about 9.64e-12 / 9.26e-12.

Independent explicit attention/softmax derivatives agree with the actual-cotangent FP64 Q/K reference at the unchanged tolerance (maximum differences 4.55e-12 / 3.21e-12). The FP32 explicit formula versus production arithmetic differs by about 0.778% / 0.729%, although both express the same real-arithmetic derivative. The corresponding FP32/FP64 Q/K comparison also differs by about 0.535% / 0.517%. This supports cancellation and arithmetic sensitivity in nonzero ordinary Q/K derivatives, not merely key-bias cancellation. It is not evidence that those percent differences are harmless to optimization.

## Ordinary attention key bias and structural zeros

**DERIVED exact mathematics:** with `k_j = W_k x_j + b_k`, each score row receives the same offset `q_i^T b_k / sqrt(64)`. Softmax is invariant to that offset. For row probabilities p and upstream g, `ds_j = p_j (g_j - sum_l p_l g_l)` and `sum_j ds_j = 0` in exact arithmetic. Therefore the task derivative of ordinary key bias vanishes. The same reasoning applies to each nonempty allowed-key set in causal attention and to ordinary memory attention. It does not make Q/K **weight** derivatives vanish, since token-dependent keys are not a common row shift.

The independent rational-number softmax Jacobian verifies exact cancellation without PyTorch at four scales. A full-width, eight-head, 308-token accepted attention primitive uses identical copied FP32 initialization/input/cotangent values in both precisions. Its initialization is a controlled local fixture, not a claim to be LE1's full-root initializer draw.

| Precision | Cotangent scale | Maximum absolute ordinary key-bias gradient | Coordinates outside existing zero tolerance |
|---|---:|---:|---:|
| FP32 | 1e-4 | 4.718e-12 | 0 |
| FP32 | 1 | 4.74975e-8 | 0 |
| FP32 | 1e4 | 4.11987e-4 | 494 |
| FP32 | 1e8 | 5.0625 | 502 |
| FP64 | 1e-4 | 9.31736e-21 | 0 |
| FP64 | 1 | 9.71445e-17 | 0 |
| FP64 | 1e4 | 9.52127e-13 | 0 |
| FP64 | 1e8 | 9.31323e-9 | 493 |

The last row is an explicit FP64 stress-scale zero-comparison failure. It illustrates why a scale-independent absolute-zero bound cannot be assumed for a canceling reduction. Unit-scale independent FP64 formulas and Q/K directional finite differences pass unchanged tolerances. Finite-difference absolute errors are 2.96e-9 and 2.03e-9, with epsilon 1e-6 and the already established atol 1e-5 / rtol 1e-3.

Singleton VIM is distinct: key dimension one, computed softmax exactly one, and connected Q/K task gradients are allocated exact zero in these executions. Ordinary Q/K weights are generally nonzero and receive their own independent derivative checks. No parameter or gradient was deleted, frozen, zeroed, rewritten or detached in production. Undefined zero-norm relative errors/cosines are JSON null, not a fabricated zero or an epsilon-floored success.

## CMC repeat

Both CMC workers repeat the preserved probe unchanged: all 12 accepted LE blocks, exact real/EFS FP32 visual input and deep-copied FP32-initialized FLT/CMC values, followed by optional language-only FP64 promotion. The two new scientific records match the retry records exactly.

| Precision | Outside coordinates at Iv | Maximum absolute difference | Relative L2 | Exit code |
|---|---:|---:|---:|---:|
| FP32 | 3 / 1,024 | 7.62939453125e-5 | 8.56570e-8 | 1, comparison failure |
| FP64 | 0 / 1,024 | 1.70530256582e-13 | 1.50758e-16 | 0 |

FP32 retains atol 1e-6 / rtol 1e-5; FP64 retains atol 1e-10 / rtol 1e-8. The scalar changes from 206.232910156 to 206.210383590 with promotion. This is not full-model FP64 validation: neither the VD/MVE Jacobian nor the LD path is part of this CMC probe. Accepted FP32-only visual interfaces remain unchanged.

## Root-cause assessment and limits

| Hypothesis | Evidence | Assessment |
|---|---|---|
| Separate-forward drift | Within-run forward hashes exact; every selected B_i/C_i and A/C-joint exact across four runs | Not the cause for selected comparisons |
| Stale graph or `.grad` accumulation | Live tensor reuse, empty `.grad` before C, fresh-graph zeroing, weakref/hook lifecycle checks | No such defect found in executed paths |
| Wrong parameter ownership or initialization | Same state hash, unique count, parameter identities and exact historical norm reproduction | No such defect found; prior ownership tests preserved |
| Final isolated-sum order only | FP64 streamed accumulation; actual LE8 forward/reverse rounded sums equal | Insufficient explanation |
| Shared backward paths and finite arithmetic | A/B disagree even on G0; fixed-cotangent VJPs disagree; FP64 reduces discrepancy strongly | Supported mechanism, not a blanket exemption |
| Cancellation in ordinary key bias | Exact rational proof and scale-dependent FP32/FP64 residuals | Demonstrated; unlike structural VIM zeros |
| Nonzero Q/K derivative accuracy | Independent local FP64 formulas pass; actual FP32 attention is sensitive | Separate numerical risk remains |
| Loss/Jacobian magnitude | LR dominates global norms; LD-cut controls reach percent-scale residuals | Material risk; global relative L2 cannot certify it |
| Unnormalized LD is sole causal cause | No architecture ablation performed or authorized | Not established; measured sensitivity is in the accepted LD path |

Floating-point results need not be invariant under algebraic rearrangement; PyTorch's versioned [numerical accuracy documentation](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html) explains finite precision and backend-dependent arithmetic. That general fact is not a proof of correctness, a tolerance derivation or a claim about training stability here.

No NaN/Inf was found in executed forward, loss or gradient diagnostics. Finiteness, local analytical correctness, numerical robustness of the complete graph, optimizer-step finiteness, repeated training stability, convergence and detection accuracy remain separate claims. Only the first and bounded local checks are evidenced. No diagnosis authorizes normalization, clipping, AMP, balancing, alternate attention, reduced dimensions or a new optimizer.

## Reports, memory and preservation

[Historical per-parameter discrepancies](../artifacts/phase2d2_n3/historical_parameter_discrepancies.csv) include every preserved record; [historical module report](../artifacts/phase2d2_n3/historical_module_discrepancies.csv) aggregates counts/maxima and joint gradient norms; [module loss norms](../artifacts/phase2d2_n3/historical_module_loss_norms.csv) retain all objectives. Archived records lack per-parameter error/reference squared norms and direction vectors, so exact **full-module** error norms/directions cannot be reconstructed from them. They are explicitly marked unavailable, not estimated from global norms.

New [selected tensor/parameter](../artifacts/phase2d2_n3/selected_parameter_tensor_discrepancies.csv), [selected module](../artifacts/phase2d2_n3/selected_module_discrepancies.csv), [fixed-cotangent](../artifacts/phase2d2_n3/fixed_cotangent_vjp.csv) reports provide raw-vector error norms and directional evidence for the declared subset. Per-module aggregates do not double-count intermediate tensors as parameters. Raw JSON contains four seeded normalized Rademacher projections, errors, sign disagreements, maximum absolute/relative differences, outside proportions and zero-reference counts. Four projections are diagnostics, not exhaustive proof of directional agreement.

| Main run | Internal elapsed seconds | Sampled peak RSS MiB |
|---|---:|---:|
| real / MKLDNN | 178.29 | 2446.36 |
| real_efs / MKLDNN | 271.07 | 3005.32 |
| local_duplicate / MKLDNN | 260.91 | 2988.46 |
| real_efs / MKLDNN disabled | 369.98 | 3191.66 |

Times include diagnostics, disk streaming and hashing, not pure model throughput. Main RSS samples use a nominal 10 ms interval, not an allocator maximum. Raw `resources.json` and process wall times are authoritative. Local seven-block LD takes about 51.57 internal seconds; Windows peak working set is 1,577,230,336 bytes. Temporary gradient vectors are removed after recording statistics/hashes. At most one full 154M graph is retained; local copies are promoted only after releasing the full graph.

Environment: Windows 11, Python 3.14.3, Torch 2.12.1+cpu, four threads, deterministic algorithms and highest FP32 matmul precision. Both backend modes retain MKL. GPU and CPU batch 8 are **NOT RUN**, with no inferred capacity result. Main fixtures cover CPU batch 1 and 2. Full build and environment metadata are in each summary.

The [preservation report](../artifacts/phase2d2_n3/preservation.json) verifies all 851 pre-existing files by SHA-256, including accepted source/tests/configs and historical logs/manifests/ZIPs. Full root count remains **154,156,232**, and all production initialization/parameter identities are preserved. No old test threshold, N1 or N2 scope changes. Reproduction and package details are in [REPRODUCE.md](../artifacts/phase2d2_n3/REPRODUCE.md), [manifest](../artifacts/phase2d2_n3/manifest.json) and [package verification](../artifacts/phase2d2_n3/package_verification.json). The proposed next decision is in [phase2d2_n3_proposal.md](phase2d2_n3_proposal.md).
