# Phase 2D.2 N3.1 numerical error-budget study

This document separates conditional local arithmetic bounds from measured whole-graph discrepancy. **Phase 2D.2 remains BLOCKED.** N3D03 authorizes a composite verification design direction only; no replacement assertion, numerical tolerance, architecture change or optimizer is approved.

## What is being bounded

For exact real arithmetic, differentiation is linear in the output cotangent at a fixed differentiable forward state. The experiment compares differently rounded realizations of that identity, not two exact derivatives. A streamed FP64 sum of six already-rounded FP32 gradients is a more accurate sum of those values; it is not an independent FP64 derivative of the full network.

Use three distinct reference levels:

1. **Frozen-operand arithmetic:** hold the stored input operands fixed and bound one contraction or reduction. This isolates arithmetic in that operation and excludes errors that produced its operands.
2. **Independent local derivative:** explicitly evaluate attention and its derivative in FP64 from copies of the same initialized FP32 values. This includes local forward recomputation; it is not the derivative of every stored FP32 intermediate.
3. **Whole execution:** compare joint and isolated gradients, retaining per-parameter error norms and directions. These measurements have no proved network-wide error bound in this study.

“Backward” also has two different meanings here. An autograd backward pass computes derivatives; a numerical-analysis backward error asks for an input perturbation that makes a computed result exact. The measured envelopes below are absolute forward-error envelopes for arithmetic **inside a derivative computation**. The standard dot-product model can also express one computed dot product using componentwise perturbed operands, bounded by the associated rounding factors. That entrywise interpretation does not establish one common perturbed set of parameters/activations for the entire network. This study does not claim a certified minimal backward error for the full derivative graph.

**DERIVED** formulas below are consequences of arithmetic/attention mathematics. Local fixture selection and use of particular diagnostic projections are **ASSUMED verification choices**. The architecture and six executable objectives remain the approved engineering assumptions recorded in the existing contracts. Neither these calculations nor parameter counts recover author implementation details.

## Conditional floating-point model

For round-to-nearest elementary operations without overflow or underflow, use `fl(a op b) = (a op b)(1 + delta)`, `|delta| <= u`. If `n*u < 1`, products of rounding factors admit `gamma_n = n*u/(1-n*u)`. A conventional length-n dot product then has an absolute error bounded by `gamma_n * sum_i |a_i*b_i|` under its corresponding accumulation model. This is a standard local error-analysis model, not a backend or full-network certification. See Higham, [The Mathematics of Floating-Point Arithmetic](https://nhigham.com/wp-content/uploads/2021/04/high21m.pdf), Lemma 1 and the discussion of rounding/FMA.

Here FP32 unit roundoff is `2^-24` and FP64 is `2^-53`. They are unit roundoffs, not machine epsilon values `2^-23` and `2^-52`. A bound on a dot product relative to its **result** may be arbitrarily poor if positive and negative contributions cancel. Dividing an absolute bound by a near-zero result does not create a useful universal relative tolerance.

The measured envelope is computed in FP64 and is **not outward-rounded interval arithmetic**. The code adds a separate FP64 reference-contraction allowance. Subnormal behavior, optimized kernel accumulation order and all implementation-specific intermediates are not formally certified. Thus an observed coordinate within the envelope is conditional supporting evidence, not proof that every assumption of the bound holds on the backend.

## Actual dimensions and accumulation paths

| Operation | Accepted dimensions | Local reduction dimension | Conditional scope |
|---|---|---:|---|
| Ordinary Q/K affine projection | `[B,308,512] @ [512,512] + [512]` | 512 products plus bias | `gamma_513 (|X| |W|^T + |b|)`; frozen X/W/b |
| Per-head score QK transpose | eight heads, `[B,8,308,64]` | 64 | `gamma_64 |Q| |K|^T`; `/8` exact power-of-two scaling only when representable |
| Softmax denominator/row-dot | up to 308 allowed keys | at most 308 | exp accuracy and forward input perturbations require separate analysis |
| Q/K weight derivative | `[512,B*308] @ [B*308,512]` | `B*308` | frozen computed dQ/dK and X; held-out B1 bound uses 308 |
| Ordinary key-bias reduction | `[B,308,512] -> [512]` | `B*308` | frozen computed dK; not a bound on exact task derivative |
| Head VJP matrix products | full per-head token/head dimensions | 64 or 308 | each contraction has its own operands and upstream error |
| Six isolated-gradient sum | six same-shaped gradient vectors | at most six terms | bound on summing computed gradients only |
| Shared visual parameter backward | CNN/transposed-conv/attention reused along several paths | varies by channel/kernel/spatial support and backend | cannot use six-term loss-sum bound for internal backward |

The held-out tool records actual operand absolute-sum maxima, absolute error norms/maxima, envelope norms/maxima, outside-envelope counts and maximum error-to-nonzero-envelope ratios. No observed failure maximum sets an envelope coefficient. The dimensional `gamma_n` expression and fixtures were fixed before numerical execution.

For a computed upstream derivative `Dhat`, the error in a weight gradient includes both the contraction error and `|(Dhat-D)^T X|`. A bound on `fl(Dhat^T X)` relative to `Dhat^T X` says nothing by itself about `Dhat-D`. This is why small local matmul errors cannot be substituted for a full derivative-accuracy guarantee.

## Softmax Jacobian and ordinary key-bias cancellation

With ideal normalized probabilities p, exact reverse mode gives `ds_j = p_j (g_j - mu)`, `mu = sum_k p_k g_k`. Its row sum is zero. An ordinary key bias adds the same score shift to every key of a row, so its exact task derivative cancels; ordinary Q/K **weights** generally do not.

For the **stored rounded probabilities** `phat`, even an exact evaluation of the saved-probability formula instead gives:

`sum_j phat_j (g_j - sum_k phat_k*g_k) = (phat^T g) * (1 - sum_j phat_j)`.

This identity is derived for the actual stored operands. A nonzero saved-probability sum defect can therefore produce a row-sum residual before considering rounding in the backward dot product, subtraction or final reduction. A large cotangent increases `phat^T g`; small output residuals can coexist with large relative differences in analytically zero or weak parameter gradients.

For a conventional backward row-dot on stored `phat`/g, a conditional dot-product envelope is `E_mu = gamma_n sum |phat*g|`. If the subsequent subtraction and multiplication meet the elementary-operation model, their errors must also be added, along with any error in phat relative to the ideal probability. A bound only on the final bias sum covers none of those upstream terms. The tool measures the stored-probability defect and FP64 evaluation of the identity; it does not label that identity a complete softmax-gradient bound.

Stable shifted softmax avoids some overflow hazards but does not imply exact normalization, exact cancellation or exact derivatives at finite precision. The independent reference is the explicit exponential/division formula, not a silent replacement of production attention. Forward conditioning and arithmetic of softmax have been analyzed separately by Blanchard, Higham and Higham, [Accurate Computation of the Log-Sum-Exp and Softmax Functions](https://eprints.maths.manchester.ac.uk/2728/1/paper.pdf); their forward analysis is not a seven-block backward-network bound.

The exact rational cancellation fixture, FP64 independent formulas, central finite differences and promoted-weight VJPs test different claims. Rational fixtures prove a small exact identity, finite differences include truncation/subtractive-rounding error, and FP64 remains finite precision. None is called an exact full-model oracle.

## Structural zero versus cancellation versus nonzero derivatives

Singleton VIM key-axis softmax is exactly one in the accepted executed graph. Connected Q/K gradients must remain allocated exact zero; no near-zero exemption applies. Ordinary key-bias derivatives are analytically zero but computed through canceling multi-token reductions. Ordinary Q/K-weight derivatives are generally nonzero and require independent correctness and sensitivity evidence. A small or zero reference denominator is explicitly reported, never silently replaced by an epsilon or removed from an aggregate.

Zero cotangents test exact graph zeros independently of key-bias cancellation. Small cotangents test weak but nonzero derivatives; large cotangents test finite scaling and cancellation. The absolute scales are diagnostic inputs only. Production gradients, feature magnitudes, objectives and parameters are unchanged.

## Seven-block LD and shared visual paths

The accepted LD has no added normalization. Its backward contains causal attention, LE memory attention, two-layer FFNs, VIM and residual additions across seven blocks. Bounding one matrix product does not bound products/sums of all these Jacobians or the discrepancy induced by rounded saved activations. N3.1 measures cuts at LD01, LD04 and LD07, with independent leaf state/memory/visual values copied from accepted execution and predeclared magnitude changes. Those local FP64 controls do not include an FP64 MVE/VD graph.

Only LR directly depends on LD in the six approved objectives. Consequently, joint-versus-isolated LD parameter derivatives can agree exactly simply because they compare the same single-loss contribution. That exact additivity outcome does not test LD VJP linearity for general cotangents. The held-out LD-cut probes address this different question. In the held-out scale-one controls, LD1 Q-weight relative VJP discrepancy measured through LD07 grows from approximately 0.0039647 at input multiplier 0.25 to 0.0836588 at multiplier 2. Promoted FP64 reduces it strongly; state/memory coordinates can still exceed their unchanged elementwise tolerances. These are actual empirical results, not bounds or a proposed threshold.

Shared visual controls use the accepted public FP32 interfaces and a new deterministic image. The appearance and reconstruction-residual routes share parameter owners and reconverge. The experiment tests fixed-cotangent VJPs without changing that topology. It does not infer a whole-path error bound from a convolution's nominal kernel size, because backend accumulation and upstream derivatives are additional factors.

Even a hypothetical bound `||delta g|| <= bound` is useful as an acceptance criterion only if it is independently justified, implemented with adequate numerical conservatism and tight enough to detect meaningful defects. A very loose envelope that contains both correct and deliberately corrupted gradients is not sufficient.

## Approval boundary

The held-out attention execution tested 33 frozen-operand affine/contraction/reduction envelopes with zero coordinates outside them. The maximum observed error-to-envelope ratio was approximately 0.02417; this is an observed ratio, not a selected threshold. All six directional Q/K finite differences passed the unchanged criterion. All held-out FP64 independent Q/K-weight formula comparisons passed, but the ordinary key-bias manual-formula comparison at input magnitude 2 and cotangent scale 10,000 failed at 257 coordinates. At that setting the FP64 key-bias comparison to exact zero failed at 306 coordinates, with maximum residual approximately 5.23869e-10. Thus passing final-reduction envelopes does not establish the exact task-gradient cancellation requirement: stored probabilities and computed upstream dK already contain error. See the raw [attention evidence](../artifacts/phase2d2_n31/heldout_attention.json).

No new numerical threshold is selected. Original FP32 coordinate comparisons remain genuine mandatory failures where they fail; their diagnostics remain recorded. Existing FP64 and finite-difference tolerances remain unchanged. Full loss/gradient finiteness, connectivity, singleton structural zeros, ownership and no-update checks remain mandatory.

Actual measured outcomes, including every failed local comparison and complete parameter-group evidence, are reported in [held-out validation](phase2d2_n31_heldout_validation.md). The final recommendation is in [acceptance proposal](phase2d2_n31_acceptance_proposal.md). Passing any conditional local envelope does not imply optimizer readiness, repeated-step stability, convergence, model accuracy or Phase 2D.2 acceptance.
