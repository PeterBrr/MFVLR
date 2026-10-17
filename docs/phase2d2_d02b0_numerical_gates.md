# A1-D02B0 correctness gates and failure taxonomy

2026-10-15. **DESIGN ONLY; all new execution checks NOT RUN. Phase 2D.2 BLOCKED.** No tolerance amendment is proposed. Gate requirements derive from [A1.1](phase2d2_a1_ablation_preregistration_v1_1.md), [N3.1 review](phase2d2_n31_heldout_validation.md), the accepted six-loss contract and unchanged test assertions. Additional paired bookkeeping is PROPOSED engineering verification, not paper requirements.

## Separate status axes

| Axis | Required evidence | Failure meaning / response |
| --- | --- | --- |
| Execution/provenance | Approved cell identity; complete source/config/state/fixture hashes; actual backend; every required record; cold/timing/resource logs; unchanged state | Missing/mismatched identity, state mutation, duplicate execution or incomplete output is a **mandatory hard stop**. Exit 0 alone cannot pass this axis |
| Structure/ownership | Exact dimensions/counts/parameter names, tied vocabulary, one owner per shared module, correct graph participation, connected VIM Q/K zeros, no hook/graph leak | Unexpected None, illegal alias, wrong ordering/residual, missing stage or nonzero structural VIM derivative is a **mandatory hard stop** |
| Mandatory numerical agreement | Original FP32 joint-vs-isolated gradient assertion on every participating parameter; original controlled oracle/chain tests where separately invoked | Any outside-tolerance coordinate is a genuine **FAIL**, never a diagnostic PASS or waiver. A complete finite research cell may still have correct execution/structure |
| Full-parameter comparisons | Every common/new parameter and every loss, joint sum, zero-reference coordinates, errors/directions/cancellation, shape/presence map | Coverage is mandatory; omitting weak gradients or new LN parameters is a hard stop. Recorded numerical discrepancies retain FAIL status |
| Activation/objective scales | All stages, scalars, entropy/score/gradient scales and categories; both improvements and regressions | Finite magnitudes are diagnostics, not an automatic acceptance/ranking threshold. Any nonfinite is a **hard stop** |
| Capacity/resource | Real available RAM, sampled RSS, disk usage, time, interruptions, device/B status | Insufficient launch resources means **NOT RUN**, observed OOM means **FAIL/OOM**; unexpected termination without evidence has UNKNOWN cause, not inferred OOM |

The proposed finite-failure continuation rule must be expressly approved with any execution handoff: record `execution=COMPLETE`, `structure=PASS`, `numerical=FAIL_GRADIENT_ADDITIVITY`, keep a failing assertion/JUnit outcome and raw worker exit code, and finish only the already authorized finite diagnostic matrix. This does not soften the original mandatory numerical gate. Mandatory controlled primitive failures, nonfinite values or structural/provenance failures halt dependent work. There is no broad `code in {0,1}` success rule: an unexplained code 1 is a failed worker needing diagnosis.

## Original numerical conventions, unchanged

| Comparison | Absolute tolerance | Relative tolerance | Scope |
| --- | ---: | ---: | --- |
| FP32 elementwise | 1e-6 | 1e-5 | Original new-head/loss/full-gradient criteria |
| FP64 elementwise | 1e-10 | 1e-8 | Existing analytical/local/promoted-weight tests only; no full visual FP64 claim |
| Central finite differences | 1e-5 | 1e-3 | Existing declared fixtures with epsilon=1e-6; no new sweep here |
| Fixed-batch per-sample independence/permutation/cross-sample derivatives | exact | exact | Relevant model-output paths and genuine differentiable leaves; excludes intentionally batch-coupled CMC objective |
| Connected singleton VIM Q/K gradients | exact zero and allocated | exact | Never replace None by zero to pass |

The full gradient elementwise reference remains `float32(sum_in_FP64(g_fd,...,g_kl))`, where each isolated g was produced by FP32 autodiff, accumulated in the literal loss order. Report the FP64 unrounded sum separately for error norms, directions and cancellation. Neither reference is a full-model FP64 derivative oracle. Do not mark a row as “FP64 tolerance” merely because diagnostic accumulation uses double precision. Key fields must identify actual dtype, reference dtype, reference rounding and tolerance policy.

For each parameter, preserve: shape/numel; allocated versus None per loss; tensor hash; finite/count-nonzero; isolated and joint L2/RMS/max; FP64 sum norm; per-coordinate outside count/proportion; max absolute and nonzero-reference relative error; error L2/relative L2; actual/reference dot and cosine; sign disagreements; exact-zero reference coordinates and nonzero-actual-on-zero-reference count. Never floor a small denominator. Undefined relative L2/cosine are tagged null with a reason, not zero or PASS. Report smallest positive reference norms and weak nonzero cases separately from structural zeros.

Retain existing full-gradient directions: four normalized Rademacher directions with seeds **5101–5104**, salt = first four SHA-256 bytes of parameter name, little-endian; chunk size **262,144** from `n31_full_gradients.py`. D02A's 6701–6704 local directions are a different experiment and must not silently replace these. Report each difference/direction, not a selected favorable subset. Extra parameters have their own names/seed salts and no invented A0 counterpart.

Per-parameter cancellation numerator is `sum_loss ||g_loss,p||2`; denominator is `||sum_loss g_loss,p||2`, using the unrounded FP64 sum. Record both and undefined denominator cases. The older global `joint_to_sum_component_norms` is a different ratio, not an interchangeable column. Include all 15 isolated-loss pairs per common parameter with zero/unused reasons; the former selected-layer cosine table is supplementary only. Additional complete interaction diagnostics require future research code review and storage gating.

## Expected graph participation and zeros

| Loss | Connected primary owners | VIM Q/K expectation |
| --- | --- | --- |
| FD | detector, fused MVE appearance/residual paths, VD trunk/AD via absolute residual | All FLT VIM Q/K unused/None |
| LR | visual/residual/AD path, shared vocabulary, positions, LE, LD and all VIMs | 38 tensors allocated exactly zero (24 LE +14 LD) |
| CMC | unadapted Iv, LE/global language, shared vocabulary/encoder positions, theta, visual/residual/AD path | 24 LE tensors allocated zero; 14 LD unused/None |
| FL | MVE appearance local encoder, VD trunk/MD | All FLT VIM Q/K unused/None |
| AR | MVE appearance local encoder, VD trunk/AD | All FLT VIM Q/K unused/None |
| KL | adapter, LE/global language and both visual/language routes | 24 LE tensors allocated zero; 14 LD unused/None |
| Total | union of all six, including new LN parameters | All 38 allocated exactly zero |

These are connectivity requirements, not unconditional nonzero claims. B=1 CMC has allocated exact-zero derivatives through its connected graph; saturation, identical values and abs residual at zero can produce other mathematically valid zeros. Added LD LN/decoder-position parameters are unused for every isolated loss except LR. FL/AR do not require gradients in the unused visual Transformer branch just because the full forward evaluated it. Parameter presence must be checked against actual graph ownership, not blindly requiring every parameter per loss.

Ordinary Q/K weights are **not** singleton structural zeros: retain their weak and nonzero derivatives, failures and directional errors. Ordinary attention key bias cancels analytically for this softmax geometry, but floating residuals must be reported against mathematical zero and, where available from existing primitive evidence, the independent derivative. Do not force exact zero for these biases or suppress their residuals with an epsilon. No new local oracle sweep is included in D02B0/D02B matrix; historical D02A/N3 oracles remain evidence, and any future mandatory pre-execution oracle run needs an explicitly scoped handoff.

Model sample independence uses genuine floating lookup or visual leaves, a connected same-sample positive control, and both cross-sample directions. Integer IDs or a disconnected surrogate do not prove isolation. CMC intentionally couples rows; no exact B1/B2 CMC equality or zero CMC cross-sample gradient is imposed. New head/loss comparisons are not covered by N2's FLT-T17 FP32 cross-batch diagnostic amendment.

Existing isolated detached-control evidence remains preserved. New full-model controls are not silently run: if needed to establish a route, enumerate them in a separate approved list with expected absence/presence before executing. Never detach Ipre, Iv, Tl or adapter inputs in the scientific six-loss path. Shared module gradients alone cannot distinguish reconstruction-mediated and direct routes; use intermediate gradients and the separately scoped controls.

## Stage and objective coverage

Capture CNN/visual local endpoints, appearance/residual global tokens and fused Iv; Ipre, mask/detection logits, residual image; vocabulary lookup, encoder input, all twelve LE outputs, Tl at position 307; shifted decoder input; every LD self/memory/FF/VIM post-residual and final output; adapter output and full logits. Use a frozen stage-name/shape inventory, including all seven blocks. Q3 post-LN residual capture must observe actual norm outputs; do not reconstruct it as the A0 arithmetic. Statistics must be finite, graph-free after release, and their collection must not alter the executed forward.

LR categories are pad 0, BOS 49406, EOS 49407 and remaining content; their counts sum to B*308 and their token-sum/batch-mean contributions recover LR. The embedding remains one Parameter with encoder-input, shifted decoder-input and tied-output contributions; no copied output matrix. CMC records raw and temperature-scaled BxB scores, row/column entropy, duplicate-prompt count, theta and effective tau. KL records feature probabilities/entropy at 0.5 and both-side gradients. AR records reconstruction outside [0,1] and absolute residual sign/zero counts. Equal weights do not imply equal scalar or gradient magnitudes. Do not introduce balancing when LR dominates.

## Eight mandatory historical A0 failures remain failures

The original assertions, node identities and historical JUnits remain untouched:

1. `tests.joint.test_joint.test_joint_i08_cmc_chain_rule_and_batch_coupling`
2. `tests.joint.test_joint.test_joint_i10_fp64_oracle_and_full_fp32_gradient_additivity`
3. `tests.joint.test_readiness.test_ready_n02_full_g2_no_update[real]`
4. `tests.joint.test_readiness.test_ready_n02_full_g2_no_update[efs]`
5. `tests.joint.test_readiness.test_ready_n02_full_g2_no_update[local]`
6. `tests.joint.test_readiness.test_ready_n02_full_g2_no_update[real_efs]`
7. `tests.joint.test_readiness.test_ready_n02_full_g2_no_update[real_local]`
8. `tests.joint.test_readiness.test_ready_n02_full_g2_no_update[local_duplicate]`

N1 stays MVE-T12 cross-batch only; N2 stays FLT-T17 cross-batch FP32 verification only. N3/N3.1 conditional local error budgets do not exempt full gradients. No xfail/skip, changed tolerance, norm-only criterion or historical-approval inference is allowed. Neither MVE-T10 nor VD-T12 may execute under this design or an unqualified no-update pilot. Existing regression source is preserved; D02B0 runs **zero** tests, not 276.

D02A's 399,522 local FAILs, 4,907 cross-batch FAILs and both Q3 FP64 additivity failures remain genuine. Its 58,752 structural checks and 2,736 FP64 ordinary-attention oracle records support their recorded local categories, not the future full-model gradient claim. A successful no-update graph would establish neither finite optimizer updates, repeated-step stability, convergence nor detection/localization accuracy.
