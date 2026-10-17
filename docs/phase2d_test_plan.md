# Phase 2D head, loss and readiness test plan

Status: **PROPOSED TEST DESIGN, NOT EXECUTED**. This file does not supply implementation or claim passes. New decisions require explicit approval through [P2D01-P2D20](phase2d_heads_contract.md). Reference equations and operational formulas are in the [loss contract](phase2d_losses_contract.md); stages G0-G5 and capacity limits are in [numerical readiness](phase2d_numerical_readiness.md). Scope is one detector, one alignment adapter, six separately verified objectives and future synthetic integration. Dataset loading and benchmark evaluation are excluded.

Paper traceability: detection/adapter p. 8 Section III-E Eqs. (19)/(26); AR p. 7 Eq. (17); FL p. 8 Eq. (18); CMC p. 8 Eqs. (20)-(23); LR p. 8 Eqs. (24)-(25); unit-weight total p. 8 Eq. (27). Accepted fusion/residual/shared visual graph: pp. 5-6 Fig. 4 and Sections III-B/C. VIM/FLT: pp. 6-7 Eqs. (3)-(16). Paper tests do not specify the following acceptance thresholds or fixtures; these are **ASSUMED engineering controls**.

## Test discipline and numerical policy

1. Preserve the full **164-case accepted suite** and all its original source/configurations, golden fixtures and historical artifacts. Run it unfiltered after new implementation, with no skips, xfails, xpasses, threshold changes or hidden deselections. Reconfirm all 119 earlier cases are present and pass. Record actual discovered new/total case counts rather than infer pytest cases from IDs below.
2. New deterministic mathematical comparisons: FP64 `atol=1e-10, rtol=1e-8`; FP32 `atol=1e-6, rtol=1e-5`. These are PROPOSED thresholds for independent head/loss oracles, not a blanket N2 extension. No failed threshold is raised automatically. Scalars with analytically nonnegative values may be >= -1e-10 (FP64) or -1e-6 (FP32) solely to detect roundoff; do not clamp a returned loss. Fixed-batch feature independence, parameter identity/count, required integer labels and intentional exact-zero gradients use exact checks.
3. Finite differences, where used, are a separate proposed numerical test: central difference eps=1e-6 on selected FP64 input/weight directions, derivative comparison `atol=1e-5, rtol=1e-3`; compare analytical derivatives at the stricter FP64 tolerance too. This finite-difference threshold is declared before execution and does not weaken existing analytical tests. Avoid abs equality/ReLU kinks for derivative checks except the dedicated accepted abs-subgradient control.
4. Independent oracles use explicit logsumexp/one-hot mathematics or hand-derived gradients, not the same production loss helper called twice. Tiny toy arrays test scalar algebra. Fixed-shape production APIs remain `[B,1,512]`, 224x224 and 308x49,408; public-shape tests embed structured values into those full shapes. A smaller toy oracle is never counted as full-model acceptance.
5. FP64 full-FLT checks promote deep copies of FP32-initialized weights; no fresh independent FP64 initialization. Accepted VD stays FP32. Non-backward probes use no-grad and detached captures, release hooks, and must not retain graphs in logs. Do not insert production detaches to make an isolation test pass.
6. Invalid inputs deliberately test failure paths; no-valid-input NaN/Inf is tolerated. On failure preserve raw values or hashes, stage, test ID, backend, configuration and traceback, stop the affected acceptance stage, and diagnose. No implicit `nan_to_num`, extra norm, altered attention, tau clamp, loss balancing or gradient clipping.
7. Keep accepted N2 exact fixed-batch independence/permutation/cross-sample model-gradient tests and full FP64 cross-batch checks. FP32 cross-batch stage diagnostics retain original tolerances/statistics. Contrastive **loss** cross-sample coupling is intentional; do not reuse the model-isolation assertion on CMC gradients. New total-loss permutation comparisons allow the declared arithmetic tolerance because reductions can reorder additions.

## Head unit tests

| ID | Controlled test and acceptance criterion | Decision / evidence |
|---|---|---|
| HEAD-T01 | B=1 and B=2 Iv `[B,1,512]` -> logits `[B,2]`; adapter output `[B,1,512]`. Optional isolated B=8 head-only test is not a full B=8 graph capacity pass | P2D01/02; p. 8 Eqs. (19)/(26) |
| HEAD-T02 | Compare both affine outputs to independent sum-of-products FP64 oracle; signed values remain signed. No hidden activation/norm/dropout and exactly one affine each | P2D01/02; one-FC prose |
| HEAD-T03 | Detector class order 0=Real/1=Fake, probability helper normalizes only class axis, fake score selects column 1, ties select Real. Loss receives raw logits | P2D01/06; assumed encoding |
| HEAD-T04 | Exact weights/bias shapes, detector 1,026 / adapter 262,656 / bundle 263,682; independent Parameter/storage objects; no second MD head | P2D01-04; derived counts |
| HEAD-T05 | Dedicated seeded initializer matches exact documented draw order; bias exactly zero; repeated seed identical; distinct seed changes values. Caller RNG and every accepted parameter bit preserved | P2D03; initialization assumed |
| HEAD-T06 | Hand-derived affine input/weight/bias gradients, finite controlled nonzero entries; adapter and detector isolation. Full dimension FP64 promoted copies | P2D01-03 |
| HEAD-T07 | Fixed B=2 change only sample 1 -> sample 0 outputs exactly equal; swap/undo batch exact; per-sample head probe has exactly zero other-sample input gradient | P2D17; model isolation |
| HEAD-T08 | Save/load state round trip, outputs within same-runtime declared tolerance, no new aliases/ties; all accepted MVE/VD/FLT identities/counts intact in proposed root | P2D04; sharing preserved |
| HEAD-T09 | Reject B=0, wrong token axis/dim, integer hidden, mixed devices/dtypes, NaN/Inf and invalid seeds; FP64 allowed only on explicit promoted analytical copies | P2D03/04 |
| HEAD-T10 | New strict reference/debug validation: unknown/duplicate/missing keys, bool-as-int, altered class order/topology, self-granted approval and objective-altering overlays rejected; accepted loaders/files untouched | P2D05 |

## Independent loss tests

All tests must verify rank-0 finite returns and their expected gradients where specified. Full wrapper shapes remain fixed; only independent algebra oracles may use toy dimensions. Pre-register test tensors and expected formulas in fixtures separate from production source.

### FD: P2D06, Eq. (26), p. 8

| ID | Fixture and acceptance |
|---|---|
| LOSS-FD01 | B=1/2 logits include `[0,0]`, `[log(2),0]`, targets both classes. Compare CE to exact probabilities, uniform ln(2), explicit one-hot sum then batch mean |
| LOSS-FD02 | Analytic `(p-y)/B`; gradients sum to zero over class axis; both classes represented; no 1/2 class mean |
| LOSS-FD03 | Add a constant to both logits: CE unchanged within tolerance. Extreme finite scores `[10000,-10000]` with wrong label yield finite expected large loss/gradient, not log(0) |
| LOSS-FD04 | Reject non-int targets, -1/2/out-of-range labels, wrong length/rank, NaN/Inf logits; output accepts negative finite logits and does not mistake them for invalid probabilities |

### LR: P2D07, Eqs. (24)-(25), p. 8

| ID | Fixture and acceptance |
|---|---|
| LOSS-LR01 | Independent toy vocabulary oracle, then full `[B,308,49408]` uniform logits for B=1/2: loss `308 ln(49408)`; every target position contributes. Integer targets equivalent to mathematical one-hot |
| LOSS-LR02 | Set selected full-shape token rows to known nonuniform scores; isolate pad ID 0, BOS 49406 and EOS 49407. Each contributes its actual NLL with no ignore mask; report pad/special/content sums recover raw total |
| LOSS-LR03 | Verify loss target IDs are unshifted original IDs. Hook accepted `shift_low` ownership, explicitly test positions 0, 76/77, 153/154, 230/231 and 307. No second target shift/segment reset |
| LOSS-LR04 | Analytic `(p-onehot)/B` on selected coordinates, not divided by 308; duplicating the same example preserves batch-mean loss, but preserves two per-sample derivative slices each scaled by 1/B |
| LOSS-LR05 | Tied Wvoc gradient equals sum of prompt-lookup, directly prepended BOS-embedding and output-projection contributions in an independent controlled graph, and exact same Parameter identity survives full-FLT integration. Detaching `output.low` alone does not remove the direct BOS access. Tests cannot replace accepted Wvoc with independent production weights |
| LOSS-LR06 | Reject vocabulary size/sequence mismatch, wrong dtype, IDs -1/49408 and NaN/Inf logits. Extreme finite wrong-target scores remain finite under stable log-softmax. Full vocabulary output is required, not `return_logits=False` |

### CMC: P2D08-P2D10, Eqs. (20)-(23), p. 8

| ID | Fixture and acceptance |
|---|---|
| LOSS-CMC01 | Embed V=`[[1,0],[0,2]]`, L=`[[3,4],[5,6]]` in first two of 512 coordinates, remaining zero. Raw scores `[[3,5],[8,12]]/tau` are nonsymmetric. Independent row and transpose-row CE oracle verifies both directions/diagonal positives/one-half factor |
| LOSS-CMC02 | Compare hand-derived G, dV, dL and dtheta from loss contract; use moderate rescaled inputs to prevent gradient saturation masking errors. Both operands and theta receive controlled nonzero finite gradients |
| LOSS-CMC03 | Effective initial tau matches 0.07 within declared FP32/FP64 precision; theta scalar registered once; scaling V by 2 doubles raw A at fixed tau, demonstrating raw dot rather than cosine. No CMC adapter gradient |
| LOSS-CMC04 | B=1 complete formula gives exact zero loss and allocated zero task gradients for V/L/theta; no detached constant. B=2 identical V/L rows gives ln(2); duplicate-prompt metadata does not deduplicate or create multiple positive labels |
| LOSS-CMC05 | Paired batch permutation preserves total within tolerance; changing another sample changes a selected sample's CMC loss/gradient in a nonsymmetric fixture. This is expected objective coupling, while standalone encoder forward isolation remains exact |
| LOSS-CMC06 | Finite extreme scaled scores exercise stable logsumexp; tau overflow/underflow, nonfinite theta, nonfinite dot product/division and malformed B/dim/device fail explicitly. No clamping or N1 tolerance |
| LOSS-CMC07 | Fixed local B defines negatives: compare B=2 true matrix with two B=1 computations and establish they differ. Log unique ID/prompt counts; no queue, distributed all-gather, category positives or microbatch equivalence claim |

### FL: P2D11/P2D14, Eq. (18), p. 8

| ID | Fixture and acceptance |
|---|---|
| LOSS-FL01 | Constant full 224x224 logits yield exact analytical CE for all-zero real masks and all-one EFS masks. Wrong foreground predictions on real masks incur positive loss |
| LOSS-FL02 | Structured rectangle/checkerboard masks; independent per-pixel two-class one-hot sum divided by BHW, compared to integer-target result; uniform logits yield ln(2), not 50,176 ln(2) |
| LOSS-FL03 | Analytic per-pixel `(p-onehot)/(BHW)`, both channels differentiated, class-gradient sum zero; spatial replication/permutation invariance on controlled reduction oracle |
| LOSS-FL04 | Reject fractional masks, ignore sentinels, 255-valued masks passed without explicit conversion, bad class/channel axis, resolution mismatch and nonfinite logits; do not silently round/resample/cast |
| LOSS-FL05 | Target semantics: real zero, EFS one, local mixed/empty/full allowed with fake label. Missing local target fails rather than broadcasting image label; empty local mask is logged and does not turn detection target into Real |

### AR: P2D12, Eq. (17), p. 7

| ID | Fixture and acceptance |
|---|---|
| LOSS-AR01 | Exact reconstruction -> zero. Constant delta -> delta^2 after B,C,H,W mean; independent exact element-count denominator 3BHW |
| LOSS-AR02 | Gradient `2(Ipre-I)/(3BHW)`; when an independent I requires grad its direct target derivative has opposite sign. No hidden target detach |
| LOSS-AR03 | I=0/Ipre=2 -> loss 4; I=1/Ipre=-1 -> loss 4; both finite and differentiated. No sigmoid/tanh/clipping; same unrestricted Ipre reaches residual encoder |
| LOSS-AR04 | Real/fake samples receive equal batch weighting; target is actual input, not pristine source. Reject invalid I range or mismatched shapes; deliberate finite-but-overflowing reconstruction square must fail finiteness |

### KL: P2D13, Eq. (19), p. 8

| ID | Fixture and acceptance |
|---|---|
| LOSS-KL01 | Independent probability oracle using repeated 256 pairs P=(0.8,0.2)/256, Q=(0.3,0.7)/256 across 512 features; l=0.5 logP, a=0.5 logQ. Expected value equals two-category KL(0.8/0.2 || 0.3/0.7), not its reverse or 1/512 fraction |
| LOSS-KL02 | P=Q -> zero loss and gradients within analytic tolerance. Asymmetric P/Q -> unequal reverse loss; no T^2 factor, feature sum followed by B mean |
| LOSS-KL03 | Check full teacher and student derivatives given in contract, including derivative of P logP. Test-only detach(l) removes teacher gradients and preserves student; production path must differentiate both |
| LOSS-KL04 | Feature softmax has 512 entries and sum one; singleton token-axis softmax must be caught as incorrect. Temperature division is 0.5, not multiplication/inverse-direction scaling |
| LOSS-KL05 | Extreme but finite feature scores with tiny/underflowed probabilities remain finite via logP/logQ formulation; no P/Q division, log(underflowed P), epsilon addition or clipping |
| LOSS-KL06 | Reject dimension/batch/device/dtype mismatch and nonfinite features. Verify no LD/Pd-only gradient and no shared parameters between adapter and detector/VIM |

## Joint gradient and interface tests

Run synthetic full graphs only after independent loss tests pass. Three full-shape fixture pairs and B=1 sample-kind coverage are specified in G2; do not substitute random labels without recording their semantic role. Fresh forward graphs for each isolated loss avoid retained-graph confusion.

| ID | Acceptance criterion | Decisions / paper anchor |
|---|---|---|
| JOINT-I01 | Appearance encoded once, VD once, reconstruction residual encoded once with same MVE weights. Detector/FLT/adapter see the same raw fused Iv. FL consumes accepted MD logits; no skip-connected decoder, duplicate MD or CMC adapter input | P2D04; Fig. 4 pp. 5-6 |
| JOINT-I02 | Correct six rank-0 components and total exactly equals the chosen left-to-right unit-coefficient arithmetic on those returned scalars. No class/token/spatial normalization beyond contracts | P2D16; Eq. (27) |
| JOINT-I03 | Isolated loss-versus-module gradient presence matrix matches loss contract; controlled nonzero norms on required paths. Distinguish `None` for an unused path from allocated zeros for connected singleton VIM Q/K or B=1 CMC | P2D17; Eqs. (17)-(26) |
| JOINT-I04 | FD/LR/CMC/KL reach Ipre and AD through residual branch in nonsaturated fixtures; detach Ipre **only in the test residual path**, keep its value, and these AD gradients disappear. MVE appearance path still differentiates | P2D17; Section III-B absolute residual |
| JOINT-I05 | AR+residual-mediated objective: Ipre gradient equals direct AR plus residual-mediated gradient. Away from equality, abs derivative sign is correct; exact equality follows accepted zero subgradient. Do not infer branch isolation solely from shared parameter norms | P2D12/17; Eq. (17) and accepted MVE |
| JOINT-I06 | FL alone reaches MD/trunk/local appearance encoder, not AD head/global visual Transformer/FLT. AR alone reaches AD/trunk/local encoder, not MD. Same trunk and local parameters are shared and accumulate both contributions | P2D11/12; Section III-C |
| JOINT-I07 | KL separate partial probes: hold adapter feature constant to expose Tl path, hold Tl constant to expose adapter path, then full graph combines both into Iv. Detach Iv supplied to LE in test to distinguish visual-conditioning path from adapter direct path. No LD dependency | P2D02/13; Eq. (19) |
| JOINT-I08 | CMC independent V/L derivatives match analytical G, and full Iv derivative adds the chain through LE(V). LE global is physical position 307; no EOS/mean substitution. Batch coupling belongs to loss, not encoder | P2D08/09; Eqs. (20)-(23) |
| JOINT-I09 | All 19 VIM instances preserve one key, published scale and delta-only output. 38 Q/K matrix task gradients are allocated exact zeros for joint language objectives; V/O/visual gradients nonzero on controlled fixtures. Existing VIM tests unchanged | Accepted L09/N2; Eqs. (4)-(16) |
| JOINT-I10 | Sum of isolated objective gradients agrees with joint gradient on small FP64 control at strict tolerance; full FP32 comparison uses declared elementwise tolerance and also logs relative L2/max discrepancies. Stream/recompute instead of retaining six graphs. A failure requires diagnosis, not N2 exemption | P2D16/17 |
| JOINT-I11 | Visual-only inference with token/FLT/adapter entry points rigged to fail still returns detector and MD logits. Changing supplied language IDs in training-side tests does not change detection/MD values at fixed visual inputs. No targets required at inference | P2D04; p. 4 Section III-A |
| JOINT-I12 | Save/load full future root preserves exact unique counts: visual 41,134,277, FLT 112,758,272, heads 263,682, tau 1, total 154,156,232; inference-only root 41,135,303. Existing identity/tie/source/config invariants retained; no optimizer state invented | P2D01-05/10; derived contract |

For JOINT-I03/I09, nonzero gradients are demanded only on deliberately nonsaturated, asymmetric probes. No expectation that all 154M elements must be nonzero. For CMC B=1 and for identical-distribution KL, zeros are correct. Tests must not pass because every upstream gradient vanished; own-sample/own-branch positive controls are required.

## Numerical, preservation and resource tests

| ID | Proposed execution and required record |
|---|---|
| READY-N01 | Hash-preservation audit of all accepted sources/tests/configs/assets/logs/manifests. Complete old 164-case pytest/JUnit run, exact count, no hidden skip/xfail/override. Future new files only unless explicit exception approved |
| READY-N02 | G2 full FP32 no-update B=1 cases (real/EFS/local) and all B=2 pairs, all six losses and per-loss/joint backward. Log every LE/LD stage and per-module/parameter gradient finiteness, norm and extrema |
| READY-N03 | Re-run accepted N2 F0/F1/F2 using original same weights/inputs and metadata, all required captures/mismatch counts/proportions/max abs/max relative/relative L2. Materially changed evidence blocks a new acceptance claim. No N1 tolerance transfer |
| READY-N04 | Record LR pad/special/content sums, KL teacher/student entropy, CMC scores/tau/duplicate count/row-column entropy, Ipre range and residual sign-zero counts. These are diagnostics, not loss-rescaling operations |
| READY-N05 | Hook/graph lifecycle audit: forward-only captures no autograd graph, no persistent graph-owning object after cleanup, repeated no-update calls do not keep prior outputs alive. Retain only statistics where full values are unnecessary |
| READY-N06 | Fresh-process CPU resource runs separately B=1, B=2, B=8, full shapes. PASS/FAIL/OOM/NOT RUN with explicit reason. RSS baseline/sample interval/peak, runtime, full outputs and gradient status, dependency/backend metadata. No accumulated-microbatch surrogate |
| READY-N07 | GPU run only on supported CUDA environment with exact device/build flags, synchronization and peak allocated/reserved memory. B=1/2/8 independently reported. Current status all NOT RUN; absence of GPU is not a hidden pytest skip |
| READY-N08 | Separately authorized G3 one-step optimizer check, full first-step moments/state allocation and post-update forward/backward. Task versus decay gradient distinction, tau state/positivity, original model snapshot unchanged |
| READY-N09 | Separately authorized G4 20-step B=1/2 trajectories and independent resource-gated B=8. Hard failures and predeclared review triggers stop/report; no clipping/stabilization retries under same result. No accuracy/convergence claim |

READY-N08/N09 are future optimizer tests, **not mandatory executions for a heads/losses-only approval**. A report must explicitly state that optimizer/repeated-step readiness remains NOT RUN when unapproved. Resource tests report unavailable runs outside the mandatory CPU test suite; never add skip marks to make a claimed full suite appear complete. GPU capacity and B=8 cannot be called passed from parameter arithmetic or B=2 results.

## Future source/evidence organization (not files created now)

Keep new head/objective modules and strict scoped configuration separate from accepted modules. Suggested new tests live under a new `tests/objectives` namespace to avoid collisions with accepted files. Each test carries the IDs above plus P2D/equation traceability. Independent expected fixtures belong under that namespace, not generated by production loss functions. Report actual pytest discovery count and parameterization; a test-ID total is not a case-count assertion.

Future implementation deliverables should include resolved new configurations, approval records, original paper equation ledger plus operational interpretation IDs, source/config/fixture SHA-256 manifests, raw stdout/stderr and JUnit (including failed attempts), count/ownership report, per-loss and joint gradients, loss/activation/temperature traces, B-specific resource metadata, preserved regression evidence and a review ZIP. A failure report must identify whether the problem is algebra, target semantics, autograd connectivity, backend numerics, memory or optimizer trajectory. A finite forward/backward run is not described as trained, converged or accurate.

## Decisions needed before tests can be implemented

Approve P2D01-P2D14 and P2D16-P2D18/P2D20 for the proposed heads/losses, scoped configuration and **no-update** verification, or specify changes. P2D15 mask preprocessing remains a future data-contract decision; P2D14 target semantics are enough to verify losses with supplied synthetic masks. P2D19 and READY-N08/N09 need explicit optimizer-smoke authorization separately. Production training, data loaders and benchmark evaluation remain outside all current authorization. No tests, heads or losses have been implemented in this design review.
