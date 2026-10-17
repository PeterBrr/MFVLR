# Phase 2D.1 implementation and verification report

Phase 2D.1 is complete and **awaits user review**. Independent heads/adapter and six executable objectives passed **112/112 new cases**, the unchanged accepted regression passed **164/164**, and the combined unfiltered suite passed **276/276**. Exit codes are 0; no failures, errors, skips, xfails, xpasses, deselections or tolerance changes. This is an independent-component result, not full-model six-loss integration, optimizer readiness, training stability, convergence or accuracy.

## Authorization, fidelity and preservation

The latest explicit user instruction approves P2D01-P2D14 and unit-verification portions of P2D16-P2D18/P2D20 **as engineering assumptions for Phase 2D.1 only**. [Scoped approval](../artifacts/phase2d1/approval.json) and [reference configuration](../configs/heads_losses_reference.yaml) record this approval; [debug configuration](../configs/heads_losses_debug.yaml) repeats the same approval and changes diagnostic settings only. P2D15 mask preprocessing and P2D19 optimizer execution remain unapproved. Full synthetic integration/G2 requires the separate Phase 2D.2 handoff. Earlier design documents remain unchanged historical proposals; the scoped record supersedes their approval status only within this authorization.

**EXPLICIT** paper reference: [MFVLR.pdf](../papers/MFVLR.pdf), Section III-E p. 7 Eq. (17), p. 8 Eqs. (18)-(27), single-FC detector/adapter and equal six-objective sum. **ASSUMED, approved engineering baseline**: biases, initialization, hard-index class semantics, probability/logit interpretations, all reductions absent from the paper, raw-dot resolution of conflicting similarity prose, log-temperature parameterization, both-sided KL and special/pad policy. **DERIVED** counts and controlled mathematical expectations are identified below. The original equations are preserved in [loss contract](phase2d_losses_contract.md); implementation never labels an executable interpretation as recovered author code.

SHA-256 preservation: **354/354 pre-existing files unchanged**, including accepted modules, tests, configurations, tokenizer assets/golden fixtures and historical reports/logs/manifests/ZIPs. [Preservation audit](../artifacts/phase2d1/preservation.json) references the pre-edit [baseline](../artifacts/phase2d1/baseline_manifest.json). Unmodified integration regression tests write fixed historical gradient paths: the new runner archives the old bytes, captures fresh evidence separately, and restores those old bytes in `finally`. Accepted N1 remains MVE-T12-only; accepted N2 remains FLT-T17-only. All old 164 case identities and the earlier 119 are present and pass; no old test or fixture was edited.

## Source, interfaces and strict configuration

- [heads.py](../src/mfvlr/heads.py): `DetectionHeads` owns `detector: Linear(512,2,bias=True)` and independent `adapter: Linear(512,512,bias=True)`. Input `[B,1,512]`; outputs raw `[B,2]` Real/Fake logits (0/1) and unrestricted `[B,1,512]` adapted feature. `detect()` is language-independent; `align_language()` is explicit. Presentation-only softmax helpers never feed the approved losses automatically. No extra MD/AD, normalization, activation, dropout or pretrained weights.
- Dedicated CPU generator seed 0 draws detector then adapter weights using Xavier uniform gain 1, zero biases. Caller RNG is restored; accepted owners are never initialized or registered here. Analytical tests promote FP32-initialized weights to FP64.
- [objectives.py](../src/mfvlr/objectives.py): independent functions plus `CrossModalContrastiveLoss` and `SixObjectives`. Only one objective parameter, `cmc.log_temperature`, is registered. `SixObjectives` accepts already-produced tensors and supplied integer targets; it owns no visual/language backbone, detector or adapter. No composed production MFVLR model is introduced.
- [heads_losses_config.py](../src/mfvlr/heads_losses_config.py) and separate immutable snapshot: closed keys/types/values, unique YAML keys, exact shapes/reductions/weights/tolerances/approval; one diagnostics-only overlay level, whole diagnostics replacement, no nested base or changed architecture, no implicit self-approval. Resolved copies are [here](../artifacts/phase2d1/resolved_configs). Accepted schemas/snapshots remain unchanged. The verified execution route is a source checkout; production wheel packaging has not been audited or changed.

| Objective | Approved operation | Reference |
|---|---|---|
| FD | Class log-softmax/gather, B mean; raw logits, int64 Real/Fake IDs | p. 8 Eq. (26), P2D06 |
| LR | Vocabulary log-softmax/gather on `[B,308,49408]`; all token sum then B mean; unshifted IDs including BOS/EOS/pad | p. 8 Eqs. (24)-(25), P2D07 |
| CMC | Raw `V L^T / exp(theta)`, diagonal CE each direction, arithmetic mean; local batch only | p. 8 Eqs. (20)-(23), P2D08-10 |
| FL | Two-class pixel CE with supplied int64 background=0/foreground=1; BHW mean | p. 8 Eq. (18), P2D11/14 |
| AR | `(I-Ipre)^2` mean all RGB elements/images; I in [0,1], unrestricted finite Ipre; both input derivatives retained | p. 7 Eq. (17), P2D12 |
| KL | `KL(text || adapted vision)`, softmax features /0.5, sum 512 then B mean, both distributions differentiated, no T² | p. 8 Eq. (19), P2D13 |
| Total | Literal left-to-right `fd + lr + cmc + fl + ar + kl`, six coefficients 1 | p. 8 Eq. (27), P2D16 unit portion |

Every return is finite rank zero; all required shapes/devices/FP32-or-explicit-FP64 dtypes and integer ranges are validated. Nonfinite intermediates, exp(theta) under/overflow, or finite-input dot/square overflow fail visibly. There is no clipping, balancing, feature normalization, hidden epsilon, teacher detach, target smoothing or ignore mask. Supplied real masks must be zero; explicit EFS masks must be one; local masks may be mixed, empty or full and retain Fake labels. Consumer checks never generate, threshold, resize or infer a source-difference mask (P2D15 remains unimplemented).

## Verification scope and actual executions

All **42 applicable unit IDs** are covered: HEAD-T01-T10; LOSS-FD01-FD04, LR01-LR06, CMC01-CMC07, FL01-FL05, AR01-AR04, KL01-KL06. Parameterization and independent scalar/ownership units produce 112 new pytest cases. [Coverage and exact case IDs](../artifacts/phase2d1/verification.json) distinguish unit scope from future integration portions.

Independent [fixtures](../tests/objectives/reference_fixtures.json) and [Python mathematical oracles](../tests/objectives/oracles.py) use explicit max/logsumexp, sums and hand-derived derivatives, with no production loss imported in the oracle. Full API dimensions are tested. FP64 analytical `atol=1e-10, rtol=1e-8`; FP32 `atol=1e-6, rtol=1e-5`; central finite difference epsilon 1e-6 and derivative `atol=1e-5, rtol=1e-3`. These preapproved thresholds were unchanged.

Tests verify affine sum-of-products and gradients, exact fixed-batch head independence/permutation/zero cross-sample derivatives, initialization/RNG/state ownership, invalid inputs, rank-zero reductions, full all-token LR derivative `(p-onehot)/B`, both-sided KL derivatives and finite extremes. The accepted `shift_low` method is probed directly using a vocabulary-only owner at all segment boundaries; loss targets remain unshifted. A clearly identified independent toy graph checks prompt lookup, direct BOS lookup and tied output gradient additivity. The unchanged full-FLT regressions verify production tying and serialization. New full-FLT LR three-path gradient composition, composed-root head ownership and KL no-LD-dependency probes are deferred to Phase 2D.2, together with all JOINT-I01-I12 and G2; no toy result is presented as a full-model integration pass.

CMC B=1 executes the full formula: exact zero scalar and allocated zero visual/text/theta gradients, with no special detached return. At B=2 controlled nonsymmetric fixtures yield nonzero gradients on all three operands. Deliberate contrastive cross-sample coupling is verified; it is distinct from exact head/model forward isolation. All six independently produced fixture inputs can be scalarized together and differentiated; this unit arithmetic graph contains none of MVE/VD/FLT.

Incremental runs: heads 25, FD/FL/AR 34, alignment 31, LR 17; aggregate new units 112; regression 164; combined 276. Every invocation returned 0 with no test failures. Raw stdout/stderr, command/environment/exit/runtime records and JUnit are under [logs](../artifacts/phase2d1/logs) / [JUnit](../artifacts/phase2d1/junit). Final [stdout](../artifacts/phase2d1/logs/complete_01.stdout.log), [stderr](../artifacts/phase2d1/logs/complete_01.stderr.log), [JUnit](../artifacts/phase2d1/junit/complete_01.xml): **276 passed in 165.269 s**. Failed historical Phase 2A/2C runs and amendment proposals remain preserved; no new failing test attempt was suppressed.

Accepted FLT-T17 FP32 per-stage cross-batch diagnostic statistics and backend match the preserved T17 data exactly in these regression runs: [comparison](../artifacts/phase2d1/n2_regression_diagnostic_comparison.json). Required fixed-batch/permutation/cross-sample gradient and promoted-weight FP64 assertions remain active. This matching drift record is not proof that drift is harmless or that an optimizer trajectory will be stable.

## Parameter ownership

| New owner | Actual unique parameters |
|---|---:|
| Detector | 1,026 |
| Adapter | 262,656 |
| Head subtotal | 263,682 |
| CMC theta | 1 |
| New subtotal | **263,683** |

[Actual ownership report](../artifacts/phase2d1/parameter_ownership.json): five independent Parameter objects/storages; only theta is objective state. Accepted counts pass unchanged regressions: MVE 32,078,912; VD 9,055,365; FLT 112,758,272; visual+FLT 153,892,549. **DERIVED conditional** future composed count 154,156,232; image-only visual+detector 41,135,303. No full new root was instantiated. The conditional count exceeds the paper's approximately 141M by 13,156,232; the discrepancy remains unresolved. Dimensions were not changed to force agreement.

## Independent numerical evidence

[Machine-readable summary](../artifacts/phase2d1/loss_numerical_summary.json), four fresh-process [diagnostics](../artifacts/phase2d1/diagnostics), [gradient CSV](../artifacts/phase2d1/loss_gradient_norms.csv). These are controlled leaf-tensor experiments, **not paper-loss values measured on the accepted full model**.

Uniform baselines: FD/FL ln(2) = 0.69314718056; LR 308 ln(49,408) = 3,328.82323108. At B=2 FP64 the golden real prompt (repeated across the batch) has 552 pad, 16 special and 48 content tokens. Controlled target-score offsets give batch-mean category contributions padding 3120.969268831, special 74.463597390, content 253.388961167, sum **3448.821827387**. The nonuniform value differs from the uniform baseline by design; padding and specials are not ignored.

CMC FP64 promoted theta -2.65926003456 gives tau 0.070000000166; slight decimal deviation from 0.07 reflects FP32 initialization and is preserved in the oracle. Raw max score 0.12, scaled max 1.71428571, loss 0.679973861. Identical controlled feature rows at B=2 yield ln(2) despite diagonal-only positives; no duplicate deduplication/multi-positive labels. This is a duplicate-embedding fixture, not an assumption that identical prompt IDs necessarily yield identical visually conditioned LE features. B=1 is zero, so it does not provide an effective CMC learning signal.

KL probability fixture loss 0.534110809, teacher entropy 6.045579868, student entropy 6.156041747; each probability distribution sums to 1. Both gradients are nonzero on this asymmetric fixture. AR target 0/reconstruction 2 yields loss 4 with finite gradients and unchanged out-of-range reconstruction. Invalid/overflow probes reject intentionally bad inputs instead of modifying the mathematics.

Per-loss B=2 FP64 input/parameter gradient norms (separate graphs; scales depend on the registered fixtures):

| Loss | Variable | L2 | Max abs |
|---|---|---:|---:|
| FD | logits | 0.5 | 0.25 |
| FL | logits | 0.00223214286 | 4.98246173e-06 |
| LR | logits | 12.4096046 | 0.499993862 |
| CMC | visual | 0.21232289 | 0.168770694 |
| CMC | language | 1.01630476 | 0.73732479 |
| CMC | theta | 0.0418718387 | 0.0418718387 |
| KL | text | 0.0446718444 | 0.00139599514 |
| KL | adapted_vision | 0.0625 | 0.001953125 |
| AR | image | 0.00729014804 | 1.32865646e-05 |
| AR | reconstruction | 0.00729014804 | 1.32865646e-05 |

The large difference between LR's token sum and FD/FL means remains a training-readiness risk. These leaf gradients do not measure how the six objectives interact downstream through the unnormalized accepted LD, shared visual graph, reconstruction residual or common vocabulary. N2 and finite independent losses do not establish full-model gradient stability or finite optimizer updates. No additional LayerNorm, AMP, clipping, scaling, altered attention or architectural fix was introduced.

## Resources and limitations

Actual standalone full-shape six-loss diagnostics, sequential independent graphs in fresh processes, Windows CPU:

| B | Dtype | RSS baseline MiB | Sampled peak RSS MiB | Elapsed s | Result |
|---:|---|---:|---:|---:|---|
| 1 | float32 | 278.9 | 774.8 | 1.014 | PASS |
| 1 | float64 | 278.6 | 791.5 | 1.182 | PASS |
| 2 | float32 | 278.4 | 1085.5 | 1.670 | PASS |
| 2 | float64 | 278.7 | 1307.6 | 2.206 | PASS |

RSS is process RSS sampled at nominal 5 ms intervals; actual sample counts are in JSON. Peaks are sampled observations, not exact allocator peaks or GPU capacity. Times include fixture construction/diagnostics/garbage collection, exclude interpreter/Torch startup. Four CPU threads, deterministic algorithms, highest FP32 matmul precision; backend and exact dependency provenance are logged. B=8 standalone head shape test passes, but B=8 six-loss diagnostics and full graphs are **NOT RUN**. CUDA B=1/2/8 are **NOT RUN**: CPU-only build/no device. New full-model six-loss B=1/2/8, G2, optimizer/G3/G4, repeated updates, datasets and benchmarks are **NOT RUN / unauthorized**. Accepted full FLT B=1/2 finiteness regressions pass, which is a distinct verification boundary.

## Reproducible review package and next boundary

[Review ZIP](../artifacts/phase2d1/phase2d1_review_package.zip), [SHA-256 manifest](../artifacts/phase2d1/manifest.json), [external package verification](../artifacts/phase2d1/package_verification.json), [reproduction instructions](../artifacts/phase2d1/REPRODUCE.md). The archive includes full actual source, accepted/new tests, scoped and accepted configurations, tokenizer source/BPE assets/golden fixtures, pinned tokenizer wheels and raw evidence including failed historical artifacts. Prior nested ZIPs are omitted from this ZIP to avoid redundant copies and remain unchanged in the workspace, with baseline hashes verified. CRC and every payload hash are verified. Current manifest/ZIP/package hash records avoid self-referential hashing.

Phase 2D.1 stops here for review. Phase 2D.2 full synthetic integration and G2 no-update tests require a separate handoff after acceptance. Dataset mask preprocessing and any optimizer/training/evaluation remain unapproved. Unresolved reproduction uncertainty includes the author architecture/141M gap, mask preprocessing and joint-loss/LD stability; standalone verification does not resolve them.
