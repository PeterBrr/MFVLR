# Phase 2D detection heads and visual adapter contract

Status: **DESIGN REVIEW ONLY, all new decisions PROPOSED**. Phase 2C.2 is accepted by the user's latest instruction as a Windows CPU-verified engineering baseline, including scoped N2. That acceptance does not authorize these heads, objectives, optimizer or data pipeline. No implementation is supplied by this review.

Primary source: [MFVLR.pdf](../papers/MFVLR.pdf), SHA-256 `2afed0516c32121513463d326720153887586ceaa328f26b96263bd39da45307`. References use printed/PDF pages, which agree. Section III-E, Eq. (17) on p. 7 and Eqs. (18)-(27) on p. 8 were checked against rendered pages. Fig. 4 (p. 5), Sections III-A-D (pp. 4-7), Section IV-A (p. 8) and Table X (p. 14) supply context.

**EXPLICIT** means stated by the paper, including incomplete or inconsistent statements. **DERIVED** means a mathematical consequence, conditional on stated assumptions. **ASSUMED** means an engineering choice, even when previously accepted. An accepted choice is not retroactively an explicit author specification. All new executable details below are ASSUMED / PROPOSED unless individually labeled otherwise. Prior accepted components stay unchanged.

Companions: [six-loss contract](phase2d_losses_contract.md), [numerical readiness](phase2d_numerical_readiness.md), [test plan](phase2d_test_plan.md). The decision register below is authoritative for the entire Phase 2D proposal.

## Evidence and accepted interfaces

| Requirement | Classification and provenance | Contract consequence |
|---|---|---|
| Detection MLP consists of one fully connected layer fed by visual forgery features | EXPLICIT, p. 8 Section III-E, Eq. (26); Fig. 4 p. 5 | One affine map; the name MLP is not evidence for a hidden layer |
| Detection outputs are called logits; labels are one-hot | EXPLICIT, Table I and FD paragraph, p. 8 | Preserve raw logits; interpreting Eq. (26)'s log as log-softmax requires a separate assumption |
| Binary categories, d=512 | EXPLICIT, p. 8 Section IV-A | DERIVED two-category affine output, given the one-hot categorical interpretation |
| Adapter is a fully connected layer; predicts a language feature in R^(1xd) from Iv | EXPLICIT, p. 8 Eq. (19) paragraph | Same feature width as Tl; no multilayer projection is specified |
| Iv is the sum of global appearance and residual features | EXPLICIT, p. 6 Section III-B; Fig. 4 | Accepted MVE exposes `MVEOutput.fused_global` `[B,1,512]` |
| Tl is the last LE output token | EXPLICIT, p. 6 Section III-D | Accepted engineering baseline uses physical position 307, shape `[B,1,512]`, including padding |
| Language assists training and is omitted at inference | EXPLICIT, p. 4 Section III-A; Fig. 4 | Image-only route must not require tokens, generator metadata or labels |
| Reconstruction and mask heads operate on appearance local features | EXPLICIT, p. 6 Section III-C | Accepted VD already owns identity RGB head and two-channel mask-logit head; do not implement a second mask head |
| Exact biases, initialization, class order and probability boundary | Not specified by the paper | ASSUMED decisions P2D01-P2D04 |

Accepted implementations reviewed: [MVE](../src/mfvlr/mve.py), [VD](../src/mfvlr/vision_decoder.py), [FLT](../src/mfvlr/flt.py), [external orchestration](../src/mfvlr/visual_language.py). Neither source changes nor modifications to their tests/configuration are proposed. `VisionDecoderOutput.reconstruction` is `[B,3,224,224]` unrestricted finite FP32; `mask_logits` is `[B,2,224,224]`. `FLTOutput.global_language` is `[B,1,512]`; `logits` is `[B,308,49408]`. Normalizing or adapting Iv before accepted VIM would change the baseline and is excluded.

## Proposed detection head: P2D01

Input `fused_global = Iv`: finite FP32 `[B,1,512]`, B positive. Remove only the singleton token axis by indexing `[:,0,:]`, producing `[B,512]`; never unconstrained `squeeze()` that erases B=1. Apply **one** `Linear(512,2,bias=True)`. Mathematical affine output `z_fd = Iv[:,0,:] W_fd^T + b_fd` has shape `[B,2]`. Weights are `[2,512]`, bias `[2]`. No hidden layer, activation, normalization, dropout, temperature or feature normalization belongs inside the head.

Class order is **ASSUMED** `0=Real`, `1=Fake`; it matches the proposed mask background/foreground indices semantically but does not imply detection labels determine all mask pixels. Raw output values are finite real logits, neither bounded nor summing to one. Training CE consumes these logits directly under P2D06. A separate, explicitly named presentation operation may compute `p_fd = softmax(z_fd, class axis)`; `p_fake = p_fd[:,1]`, shape `[B]`. Label presentation uses `argmax(class axis)` with lowest-index tie resolution (Real). This is an interface convention, not approval of benchmark metrics, probability calibration or operating thresholds.

The same affine operation runs in training and inference. `.eval()` adds no activation or loss. Two-logit softmax is recommended over a single sigmoid logit because the paper gives categorical one-hot y and f=2 (p. 8). A one-logit BCE model can express the same decision family but changes output/parameter contracts; it is not the proposed baseline.

## Proposed visual language-alignment adapter: P2D02

Apply independent `Linear(512,512,bias=True)` on the final axis of raw Iv `[B,1,512]`. Result `T_lpre = Iv W_a^T + b_a`, shape `[B,1,512]`, unrestricted finite real feature values. Weights `[512,512]`, bias `[512]`. No activation, LayerNorm, L2 normalization, pooling, extra projection or gating. Eq. (19) explicitly aligns this predicted feature to Tl via feature-axis softmax; the operational temperature and derivatives are proposed in P2D13.

Adapter parameters are independent of detector parameters, Wvoc, visual projection and every VIM matrix. One adapter is shared across all examples, real/fake categories and generator types; there are no appearance/residual-specific or per-language-layer adapters. It consumes the **already fused** feature, not a concatenation or local/patch features. It serves KL only. CMC receives **raw Iv and Tl**, as named in Eqs. (20)-(23); it does not receive T_lpre. FLT/VIM also continue to receive raw Iv.

Gradients from KL reach adapter weights/bias and Iv through this direct path. Under the proposed no-detach policy they also reach Tl, LE, Wvoc and Iv through LE's VIMs. Tl is image-conditioned; calling it a teacher does not make it independent or frozen. Adapter-only training with detached Tl is a different optimization graph requiring separate approval.

At image-only inference the adapter, FLT and all loss modules are unused and need not be instantiated. An existing full training object may still own those weights in memory even when the visual-only route skips them; inactive execution is not parameter removal. No token/metadata fallback is permitted in the inference route.

## Initialization and ownership: P2D03-P2D04

**ASSUMED:** create new heads on CPU in FP32. Biases are zero; both matrices use Xavier-uniform, gain 1.0. One dedicated CPU generator, seed=0 by default, consumes draws in detector-weight then adapter-weight order. Zeroing biases consumes no random draw. Preserve the caller's RNG state during construction, and do not reinitialize or consume the initialization generator of any accepted component. Explicit seed validation: integer, not bool, `0 <= seed < 2^63`. Changing the new head seed is a recorded experimental fixture choice; it never changes accepted weights. Analytical FP64 tests promote copies of these same initialized values.

Proposed ownership is one root registering MVE, VD, FLT, a head bundle (detector plus adapter), and an objective bundle owning only the CMC temperature scalar. Every unique parameter appears once in parameter enumeration. Existing plain `VisualLanguageOrchestrator` can be used as a non-owning reference; do not register accepted modules twice under new aliases, clone them, or change its methods. Shared vocabulary, IE/RE and AD/MD sharing remain the accepted object identities. The root is a future integration proposal, not an implemented training system.

Proposed APIs (interface descriptions, not code):

| Operation | Required inputs | Outputs / behavior |
|---|---|---|
| Detection head | Iv `[B,1,512]` | `detection_logits [B,2]`; strict shape/device/dtype/finite checks |
| Adapter | Iv `[B,1,512]` | `predicted_language [B,1,512]`; same checks |
| Future `visual_only(image)` | Accepted image `[B,3,224,224]` | Existing visual/VD output plus detection logits; no language or target arguments |
| Future `with_language(image, token_ids)` | Image and accepted int64 IDs `[B,308]` | Same visual/detection values, full FLT output and predicted language; language can be diagnostically evaluated in eval mode |
| Future objective call | Structured six-loss inputs/targets | Six scalar losses plus literal unit-weight sum; no optimizer action |

Training-side execution: appearance encoding once -> accepted bottleneck-only VD once -> absolute residual and shared residual encoding -> additive Iv fusion -> detector; raw Iv additionally feeds FLT and the independent adapter. Localization remains on appearance local features. Labels/masks/IDs are loss targets; detection receives neither target nor language feature. The full FLT has access to target prompts through bidirectional LE memory; reconstruction is not an autoregressive generation claim (pp. 6-7 Eqs. (3), (12)-(16)).

## Parameter expectations

**DERIVED from proposed biases/dimensions, not the paper's parameter count.**

| Owner | Formula | Unique parameters |
|---|---|---:|
| Detection | 512x2+2 | 1,026 |
| Adapter | 512x512+512 | 262,656 |
| Head bundle | Sum of the above | 263,682 |
| CMC log-temperature | One scalar, objective owner | 1 |
| New total | Heads plus temperature | 263,683 |
| Accepted visual | MVE 32,078,912 + VD 9,055,365 | 41,134,277 |
| Accepted FLT | Unchanged | 112,758,272 |
| Proposed full training graph | 153,892,549+263,683 | **154,156,232** |
| Image-only deployable graph | Visual plus detector only | **41,135,303** |

Mask/appearance heads are already included in VD; do not count them again. Softmax, target conversion and the other five losses add no parameters. Disabling both new biases would remove 514 parameters, but is not recommended here. A direct-temperature versus log-temperature parameterization still has one scalar. The full proposed count exceeds the reported approximately 141.00M (p. 14 Table IX; p. 15 Table XV) by approximately 13,156,232. This is unresolved reproduction uncertainty, not a reason to shrink accepted modules or infer author fidelity from matching counts.

## Complete Phase 2D decision register

Every new entry is **ASSUMED / PROPOSED**, including mathematical interpretations of an EXPLICIT equation. No entry is approved by acceptance of the previous phase. Implementation is blocked until the relevant entries receive explicit approval. A named alternative is discussed, not authorized. Earlier A-IDs map to [assumptions.md](assumptions.md); old YAML `proposal` values remain historical, unapproved guidance for this scope.

| ID | Recommended choice and evidence boundary | Alternative / consequence | Before which work? | Status |
|---|---|---|---|---|
| P2D01 | One biased 512->2 detector, raw logits, Real=0/Fake=1; FC explicit p. 8 Eq. (26), biases/order assumed (A16/A18) | Bias-free or sigmoid single-logit changes count/interface | Heads | PROPOSED |
| P2D02 | One independent biased 512->512 adapter, identity output, KL only; FC/width explicit Eq. (19), bias/ownership assumed (A16) | Extra layers/normalization or CMC on adapter changes method | Heads | PROPOSED |
| P2D03 | Dedicated seed-0 Xavier/zero-bias initialization; no accepted RNG/state change (A22) | Different initializer affects initial alignment and gradient scales | Heads | PROPOSED |
| P2D04 | Strict APIs and one-owner external composition; train-side language opt-in, image-only inference; pp. 4-8 Fig. 4 (A35) | Language-dependent detector conflicts with image-only intent | Integration | PROPOSED |
| P2D05 | New separately scoped reference/debug schema, strict overlays, exact immutable topology/weights/reductions; no existing config edit (A35) | Hidden defaults or inherited approval obscures scope | Heads/objectives | PROPOSED |
| P2D06 | FD stable two-class CE, sum classes then batch mean; Eq. (26) logit-to-probability interpretation (A18) | Literal log(raw logits) undefined; BCE changes head | FD | PROPOSED |
| P2D07 | LR stable vocabulary CE, all 308 targets including BOS/EOS/pad, token sum/batch mean; Eqs. (24)-(25) (A10/A17/A18) | Ignore pads or token mean changes objective/scaling | LR | PROPOSED |
| P2D08 | CMC raw dot Iv/Tl, no adapter/no normalization; prose explicitly says dot but also cosine, p. 8 (A14) | Cosine-only normalization is plausible but separate, not implemented baseline | CMC | PROPOSED |
| P2D09 | CMC diagonal positives, both directions averaged, same local batch, B=1 zero; Eqs. (20)-(23) (A26) | Multi-positive duplicate handling or all-gather changes negatives | CMC | PROPOSED |
| P2D10 | theta=log(tau), tau=exp(theta), tau initially 0.07, no clamp, reject under/overflow; tau trainable explicit p. 8, parameterization assumed (A15) | Direct tau/inverse-scale updates differ. Proposed no decay on theta requires future optimizer approval | CMC | PROPOSED |
| P2D11 | FL two-class CE on mask indices, mean B,H,W, background=0/foreground=1; Eq. (18) encoding/reduction missing (A18) | Foreground-only term drops background supervision; Dice/focal/BCE not proposed | FL | PROPOSED |
| P2D12 | AR all-element mean square on all images, no range restriction/clipping of reconstruction; Eq. (17) spatial reduction missing (A05/A17) | Pixel/channel sum changes scale by 150,528; real-only AR changes population | AR | PROPOSED |
| P2D13 | KL(text || adapted vision), feature-softmax(z/0.5), feature sum/batch mean, gradients both sides, no T^2; Eq. (19) (A19) | Frozen teacher, reverse KL or feature mean changes derivatives/scaling | KL | PROPOSED |
| P2D14 | Target consumer semantics: real masks zero, EFS one, local manipulation from aligned source difference; consumer requires supplied binary mask, never fabricates one from image label; pp. 7-8 (A01/A23) | Unknown local mask is a validation failure, not all-one fallback | FL/interface fixtures | PROPOSED |
| P2D15 | Proposed future mask preprocessing: float RGB abs-diff, BT.601 coefficients, /255, strict >0.1, native-before-resize; detailed dataset geometry remains deferred; pp. 7-8 do not specify these details (A01/A23) | Different grayscale/threshold/resize order changes targets | Future data approval, not head/loss coding | PROPOSED |
| P2D16 | Literal six coefficients 1, fixed order Eq. (27), no balancing; validated scalar inputs and finite reductions (A17/A18) | Token means or adaptive coefficients change effective objective | Joint objectives | PROPOSED |
| P2D17 | Tests, fixed numerical thresholds, detached controls only in tests, scoped N2 preservation, no production detaches (A07/A19/A31) | Global zero cross-sample loss-gradient rule would incorrectly reject legitimate CMC coupling | Verification | PROPOSED |
| P2D18 | Synthetic no-update gate, seeds/fixtures and diagnostics defined in readiness document; no claim of training readiness from N2 (A31/A34) | Hiding drift/nonfinite results invalidates gate | Head/loss integration verification | PROPOSED |
| P2D19 | Separate future optimizer/repeated-step gate: exact candidate Adam settings and 20-step bounded smoke only after explicit additional authorization (A20/A21) | Optimizer defaults/AdamW/clipping cannot be silently selected | Future optimizer test approval | PROPOSED |
| P2D20 | B=1/2/8 independent CPU/GPU resource records; unsupported cases NOT RUN or FAIL/OOM, never inferred; preservation/manifest checks (A30/A31/A35) | Accumulation, AMP, checkpointing or smaller logits need separate approval | Resource verification | PROPOSED |

Approval of P2D01-P2D18/P2D20 may be scoped to heads/losses and no-update tests. It does not authorize P2D15 dataset implementation or P2D19 optimizer execution by implication. Dataset manifests, crop/resize protocol, training duration/model selection, distributed training, benchmarks and optional cosine/multi-positive ablations remain deferred and require their own explicit authorization. The proposal is ready for review, not implementation.

## Proposed configuration boundary

Future new files could be `heads_losses_reference.yaml` and `heads_losses_debug.yaml` with an independent closed schema. Required fields enumerate architecture, class order, seed/draw order, feature dimensions, scalarization, LR special-token policy, CMC similarity/temperature/positive sets, KL direction/temperature/detach flags and unit coefficients. Reject missing/unknown/duplicate keys; reject bool-as-integer, implicit shape/dtype conversions, changed full dimensions, non-unit weights and unapproved alternatives. Approval records are scoped P2D decisions and cannot be inherited or self-granted by a debug overlay. Reference/debug resolve the same mathematical model; only device, diagnostic batch/seed, threads and output location may differ. Accepted config loaders, snapshots and files are read-only. No config files are created in this design review.

## Design-review preservation check

All 350 pre-existing files captured under `src`, `tests`, `configs`, `docs`, `tools` and `artifacts` retained their original SHA-256 values at this review's verification. The four Phase 2D Markdown files are new. PDF extraction/renders and a before/after hash check are disposable review material under `tmp/pdfs/phase2d`; no production Python, pytest, configuration, optimizer or dataset file was written. The review inspected existing numerical evidence and calculated conditional memory/count estimates; it did not execute the proposed Phase 2D tests or claim new training measurements.
