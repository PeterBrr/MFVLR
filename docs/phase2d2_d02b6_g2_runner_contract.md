# D02B6 G2 adapter and execution-readiness contract

2026-10-16, Asia/Saigon. **Static implementation and mock verification only. Phase 2D.2 remains BLOCKED.** B0-D03 execution and B0-D08 finite-failure continuation remain disabled. This document does not promote any variant or accept any of the eight historical failures.

## Authority and implementation boundary

The scientific authority remains [D02B0](phase2d2_d02b0_experiment_contract.md), its [fixture matrix](phase2d2_d02b0_fixture_matrix.md), corrected [A1.1 preregistration](phase2d2_a1_ablation_preregistration_v1_1.md), and approved [loss contract](phase2d_losses_contract.md). The six executable interpretations are approved engineering assumptions, not recovered author implementation. No paper equation is changed. The exact production source and original tests are preserved.

New namespace: `src/mfvlr_research/d02b6`. `dispatch()` and `continue_after_failure()` always raise, independent of arguments. There is no CLI/YAML execution switch, torch import, model factory, worker launch or backward in this namespace. The disabled D02B1/D02B2 and legacy launchers are unchanged. The adapter prepares exact cell/pass/stage specifications, identity-preserving objective argument bindings, strict preflight metadata validation, failure records and bounded mock ingestion into the frozen D02B1 stream. It is **not an authorized runnable full-graph worker**. A future authorized execution layer must implement and validate live observation hooks and materialize actual state/input seals before first use; this package does not pretend mock metadata is those observations.

`primary_matrix()` contains exactly G01–G06 × A0/P3/Q3, using the frozen D02B1 identities `D02B-Gxx-variant`. It retains all 18 cells as NOT RUN. G01 real, G02 efs, G03 local use B1; G04 real_efs, G05 real_local, G06 local_duplicate use B2. All use C4M: CPU FP32, Windows pinned Python 3.14.3 / torch 2.12.1+cpu, intra/inter threads 4/4, MKLDNN on, deterministic algorithms and highest matmul precision. No alternative backend is accepted by the primary schema. The existing golden rows, image seeds, mask rectangle, labels and ordering are unchanged. `historical_g2_anchors.json` copies historical fixture fingerprints and source-summary hashes read-only; no image or model was constructed.

## Objective routing, without replacement formulas

`bind_arguments()` retains the supplied object identities: no detach, cast, masking, feature normalization, rescaling or substitution. The function returns bindings and never calls an objective. Future isolated calls must first retain the original target-pairing and `validate_mask_semantics` checks; joint calls reuse `MFVLRIntegration.compute_objectives` / `SixObjectives` and its original `unit_weight_sum`.

| Pass | Existing callable | Ordered arguments | Preserved semantics |
| --- | --- | --- | --- |
| FD | forgery_detection_loss | detection_logits, labels | Stable 2-class CE, batch mean; Real=0/Fake=1 |
| LR | language_reconstruction_loss | language_logits, token_ids | 49,408 classes, all 308 unshifted targets including BOS/EOS/pad; token sum then batch mean |
| CMC | root.objectives.cmc | visual, global_language | Unadapted Iv; raw dot products; exp(log-temperature), initially 0.07; symmetric diagonal CE |
| FL | forgery_localization_loss | mask_logits, masks | Supplied int64 two-class masks, mean over batch/pixels |
| AR | appearance_reconstruction_loss | image, reconstruction | RGB-element MSE; reconstruction unrestricted, no clipping |
| KL | knowledge_distillation_loss | global_language, predicted_language | KL(text || adapted vision), T=0.5, both distributions differentiable, no T² |
| total | unit_weight_sum inside SixObjectives | FD, LR, CMC, FL, AR, KL | Literal left-associated unit-weight sum, no averaging |

No new loss implementation is introduced. Independent mathematical fixtures test stream arithmetic, not a disguised replacement objective. B1 CMC allocated zeros are retained. Expected None connectivity is distinct from an allocated zero; magnitude alone never establishes a disconnection.

## Full-graph evidence required before execution can be accepted

1. Materialize one canonical seed-0 CPU FP32 integration reference; frozen A1.1 builder makes a private candidate. Observe real tensor bytes, names/shapes, IDs/storage, common parity, new gamma=1/beta=0 norms and tied vocabulary. A0 owns 508 tensors / 154,156,232 elements; P3/Q3 own 550 / 154,177,736. Do not rely on parameter counts alone. Release reference before forwards. New full integration candidate hashes are **NOT MATERIALIZED** in D02B6; N2-only hashes cannot substitute.
2. Seven fresh graphs in FD/LR/CMC/FL/AR/KL/total order. Zero gradients to None between passes without changing parameters. An isolated pass invokes its single objective, not a zero-weighted sum. Collect every parameter's gradient, validate its participation against the full frozen roster, stream contiguous FP32 chunks, then release graph and hooks. All added LD LN parameters participate in LR/total only. Preserve Ipre/Iv/Tl/adapter intermediate derivatives in the future live recorder; they are not inferred from scalar mocks.
3. Frozen `GradientStore` performs ordered FP64 accumulation of isolated FP32 gradients, then the original FP32 comparison with the rounded sum at atol=1e-6/rtol=1e-5. Unrounded FP64-reference metrics are separately labeled and are not full FP64 autograd. Every element, parameter, all 15 isolated-loss pairs and four seeded directions (5101–5104, existing parameter-name salt) are required. Store undefined cosine/relative metrics with reasons, weak nonzero values, sign and zero-reference differences. Preserve ordinary Q/K weights and analytical key-bias residuals; no blanket near-zero pass rule.
4. Singleton VIM Q/K must be connected and allocated exact zero where their loss connects: 38 for LR/total, 24 LE tensors for CMC/KL; unused roles remain None. Ordinary key-bias residuals do not inherit singleton exact-zero criteria. Unexpected None, alias, shape or nonfinite value is mandatory failure.
5. Existing 63-stage schema includes visual outputs, 12 LE outputs, all seven LD self/memory/FF/VIM/output stages, shifted/input embeddings, global feature and full logits. Future live hooks must observe actual post-residual stages respecting A0/P3/Q3 LN placement, using bounded detached statistics only. They must not approximate residual stages by module deltas or retain a graph queue. Record shape, finiteness, RMS/max/L2 for every stage in every pass.
6. Objective records must include each scalar and literal total; LR all-token sum with pad/BOS/EOS/content contributions; CMC raw score range, exp(theta), entropy and diagonal/duplicate-target identity; KL teacher/student feature probability entropy; reconstruction min/max and image target range. No extra category-wise backward or tied-vocabulary path sweep is implied. Those remain separate scopes.
7. Each forward requires **actual** state/input/source/backend/resource seals, including labels and sample-order digests missing from the old G2 summaries. Compare observed current ownership and state before/after every pass. `validate_preflight` and `seal_pass` test supplied-record consistency only; no metadata record, seed or mock PASS counts as live observation. Source inventories and trusted expected seals must come from independent pins, not the record being checked.
8. Matched triples require exact unchanged FD/CMC/FL/AR/KL outputs and common-parameter isolated gradients. LR/total are variant dependent. B1/B2 same-sample output/derivative pairings must respect reductions and cotangents, and exclude intentionally batch-dependent CMC objectives. No N2 FP32 exception transfers to these tests or gradient additivity.

The prior timing proposal remains one no-grad warm-up then three state-reloaded joint forward/backward repetitions after the seven scientific passes. It appears in the plan only, all NOT RUN. Its execution needs explicit approval with the live schedule; mock tests do not execute timing loops.

## Failure, resources and recovery

`failure_record()` writes a finite numerical FAIL unchanged before blocking dependent work. No assertion is replaced by skip/xfail or a new tolerance. B0-D08 remains unconditionally disabled. Provenance, structural, nonfinite, incomplete outputs or resource failures produce HARD STOP. Before launch, insufficient RAM/disk means WAIT/NOT RUN, not a model failure or permission to shrink the graph. Completion/exit 0 never means numerical acceptance.

Reuse D02B1 exclusive coordinator PID + creation-time ownership, permanent output reservations, disjoint byte-preserving interrupted-attempt archive and exact scientific-prefix replay (only resource time/RSS fields excluded). No automatic stale-lock deletion or concurrent workers. Proposed floors remain 5 GiB available RAM for B1, 6 GiB for B2, 8 GiB disk per cell, checked at launch and in worker. These are safeguards, not measured full-graph capacity. GPU/B8 are NOT VERIFIED. No real worker is present in D02B6.

## What was verified now

Read-only N2 reconciliation; static callable boundaries and loss order; tiny NumPy stream math with independent FP64 sums/directions/pair products; all 508/550 roster names with scalar fixtures; None/zero/nonfinite/weak-gradient handling; failure retention; exclusive paths and mock recovery; schema, provenance, backend, inventory and invariant-branch negative controls. The test fixture enforces that torch is not imported, and collection permits only `tests/research_d02b6`.

Actual full integration ownership, hooks, objective statistics, gradients, resource capacity and timing remain **NOT RUN**. The historical eight assertions remain genuine failures. N2 closure alone cannot establish joint six-loss stability or justify production promotion.
