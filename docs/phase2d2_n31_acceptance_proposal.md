# Phase 2D.2 N3.1 stopping decision and future acceptance proposal

**Phase 2D.2 remains BLOCKED. No defensible full-graph quantitative error budget or replacement acceptance criterion has been established by this study. Do not replace the eight existing failing assertions or increase a numerical tolerance.**

N3.1 applies the approved seven supplemental tests and diagnostic metadata repair, locks held-out fixtures before execution, refreshes external evidence, adds complete parameter-group diagnostics and investigates conditional local rounding bounds. These improve diagnosis and reproducibility. They do not convert finite execution, local derivative correctness or a bounded individual contraction into a guarantee for the full six-loss graph. Actual test and held-out outcomes are reported in [held-out validation](phase2d2_n31_heldout_validation.md); the reasoning and reference limitations are in [numerical budget](phase2d2_n31_numerical_budget.md).

## Scope and before/after policy

| Item | Before N3.1 | After N3.1 |
|---|---|---|
| Mandatory test tree | Original 321 cases | Original 321 unchanged, plus exactly seven approved cases |
| Original eight failing assertions | Genuine mandatory failures | Preserved; no skip, xfail, exemption or threshold increase |
| N3 graph metadata | Optional transient loss-wrapper addresses could be reused | Approved repair holds wrappers before recording identities; original files archived |
| External evidence | N3 included recorded diagnostic checks | Final workflow regenerates evidence and validates sealed inventory/source/config/input/parameter hashes |
| Gradient reports | Full historical counts plus selected raw-vector norms/directions | Fresh complete 508-parameter and full-group streamed norm/direction/zero-reference reports |
| Quantitative policy | No approved replacement budget | No new budget adopted; conditional local envelopes remain diagnostic |
| N1/N2 | Original accepted narrow scopes | Unchanged; no new-loss or gradient exemption |
| Production model/configs | Accepted engineering baseline | Unchanged |
| Phase 2D.2 | BLOCKED | BLOCKED |

The seven additional tests are not seven resolved original failures. The full suite's actual pass/fail counts must always accompany that distinction. Successful integrity validation means that the intended fresh records were used; it does not make failed numerical comparisons pass.

## Why a replacement gate is not ready

The conditional `gamma_n` envelopes address ordinary affine/contraction/reduction arithmetic on stored operands. They are not proved bounds for the backend's complete optimized kernels, upstream softmax/FFN derivatives, all saved activation errors or products of Jacobians through seven unnormalized LD blocks and shared visual routes. Computing the envelopes in FP64 is not outward-rounded interval certification. Their usefulness must therefore not be overstated even when observed errors lie inside them.

The independent mathematical and promoted-weight FP64 controls validate important local formulas, while central finite differences provide separate approximate derivative evidence. Near-zero key-bias cancellation and nonzero Q/K-weight disagreements remain distinct. The held-out scale, magnitude and intermediate-cut results are empirical sensitivity observations; choosing a number just above their maximum would fit the policy to the validation set.

No global gradient relative L2, aggregate cosine, selected directional projection, finite loss or finite gradient alone can replace per-parameter assessment. Strong LR gradients can conceal weak-parameter errors. Exact-zero references need absolute treatment justified by their mathematical class; they cannot be removed or assigned a convenient denominator floor. Passing four seeded directions is not an exhaustive directional guarantee.

## Retained design requirements for any later composite gate

N3D03 remains a **design direction**, not executable acceptance authorization. Any later candidate must preserve all of the following together:

1. Existing independent FP64 loss/attention/VIM correctness checks and finite-difference criteria at their original tolerances.
2. Full six-loss, activation and gradient finiteness; expected connectivity/presence; ownership, parameter identity and initialization checks.
3. Allocated exact singleton VIM Q/K zeros when connected, with ordinary key-bias cancellation assessed separately and ordinary Q/K weights independently checked.
4. Original FP32 comparison outcomes retained as complete diagnostics even if their future gate role changes by explicit approval.
5. Complete per-parameter and per-module absolute errors, norms, mismatch counts, zero-reference behavior, cosine/directional metrics, backend and scale provenance.
6. Full six-fixture G2 coverage, independently preregistered held-out validation and explicit reporting of unsupported platforms/capacities.
7. Controls that detect deliberately injected errors, including weak/near-zero cases; evidence integrity checks cannot substitute for these numerical controls.
8. Independently justified quantitative bounds or explicitly authorized manual adjudication, with new failures and unsupported bounds preventing automatic acceptance.

This document does **not** supply a new numeric threshold. The original FP32 atol 1e-6 / rtol 1e-5, FP64 atol 1e-10 / rtol 1e-8, and existing finite-difference tolerance remain unchanged. Their diagnostic failures are retained, not silently relaxed.

## Recommended next architectural-research decision

Request a **separate design review of the language decoder numerical assumptions**, before more acceptance-policy fitting. The accepted baseline intentionally follows seven unnormalized LD blocks under L06, ordinary attention/FFN choices under L08 and scratch initialization under L11. These are approved engineering assumptions, not confirmed author implementation details. The preserved [language contract](phase2c_language_contract.md) traces the decoder ordering to MFVLR Section III-D, page 7, Eqs. (12)–(16), and distinguishes omissions and assumptions.

That future review should reassess the evidence for normalization omission, residual placement and initialization, and formulate controlled research ablations only if separately authorized. Keep the present implementation as the frozen reference and distinguish any alternative as a new engineering approximation. An ablation would need a predeclared question, identical fixtures, full parameter/activation/gradient reporting and paper traceability. Reduced discrepancies would be evidence about sensitivity, not proof that an alternative matches the authors.

No normalization, clipping, AMP, loss balancing, activation scaling, altered attention, parameter freezing or weight change is implemented or approved here. The numerical evidence does not establish that normalization is the unique cause or that adding it is the correct remedy. An author clarification or stronger architectural evidence may be more valuable than repeated attempts to fit a tolerance to the current baseline.

## Decision table

| ID / action | Status | Exact boundary |
|---|---|---|
| N3D01 supplemental tests | APPROVED / APPLIED | Exact approved new-file patch; original 321 cases unchanged |
| N3D02 metadata repair | APPROVED / APPLIED | Diagnostic wrapper identity lifetime only; original source/evidence preserved |
| N3D03 composite verification | APPROVED AS DESIGN DIRECTION ONLY | No executable replacement policy granted |
| N3D06 bounded investigation | APPROVED / COMPLETED FOR N3.1 | Held-out, local FP64 and complete streamed no-update diagnostics |
| N31-P01 accept local envelopes as a full-model guarantee | NOT JUSTIFIED | Frozen-operand conditional bounds are insufficient |
| N31-P02 replace the existing eight failing assertions | BLOCKED / NOT APPROVED | N3D04 remains unapproved |
| N31-P03 select higher tolerances or fit a threshold to held-out maxima | NOT PROPOSED / NOT APPROVED | N3D05 remains unapproved |
| N31-P04 language-decoder architectural-research design review | PROPOSED | Requires separate authorization; no implementation or ablation execution yet |
| N3D07 extend N1/N2 | NOT APPROVED | Existing scopes preserved |
| N3D08 change architecture/numerical behavior | NOT APPROVED | No stabilization or parameter modification |
| N3D09 accept Phase 2D.2 | BLOCKED / NOT APPROVED | A separate justified remedy/policy and complete new verification would be required |
| N3D10 G3/G4, optimizer, datasets, training/evaluation | NOT APPROVED | No work in those scopes performed |

N3.1 stops at this review. See [applied patches](../artifacts/phase2d2_n31/applied_patches.json), [actual verification](../artifacts/phase2d2_n31/verification.json), [preservation](../artifacts/phase2d2_n31/preservation.json), [reproduction guide](../artifacts/phase2d2_n31/REPRODUCE.md) and the [review ZIP](../artifacts/phase2d2_n31/phase2d2_n31_review.zip).
