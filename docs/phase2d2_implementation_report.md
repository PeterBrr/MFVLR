# Phase 2D.2 implementation and G2 verification report

**Status: BLOCKED; Phase 2D.2 is not accepted.** The full orchestration is implemented. The final unfiltered suite discovered **321** cases: **313 passed, 8 failed, 0 errors, 0 skips**; exit code 1. The separate unchanged regression run passed **276/276**, exit 0. No xfails, hidden skips, deselections, relaxed thresholds, architecture changes or stabilization methods were used.

All six requested synthetic full-size FP32 fixtures completed their six isolated losses and literal joint backward with finite scalars, features, logits and parameter/intermediate gradients. Every LE/LD block is captured, all connected singleton VIM Q/K gradients remain allocated exact zeros, and weights/identities remain unchanged. **Every fixture fails the full-parameter isolated-sum versus joint-gradient comparison at the unchanged FP32 tolerance.** A separate CMC chain-rule comparison fails at 3/1,024 coordinates. Finite execution is not a passed G2 numerical gate, training stability, optimizer readiness or benchmark performance.

## Scope, approval and preserved baseline

The latest [user request](../artifacts/phase2d2/history/user_phase2d2_request.txt) extends engineering approval only to P2D04/P2D16-P2D18/P2D20 full integration and G2. [Scoped approval](../artifacts/phase2d2/approval.json) records ASSUMED engineering status, not recovered author architecture. P2D15 mask preprocessing and P2D19 optimizer/repeated updates remain UNAPPROVED. No optimizer, scheduler, parameter update, training step, dataset loader or evaluation was implemented. G3/G4 remain future work.

All **452 pre-existing files are byte-identical by SHA-256**, including all accepted MVE/VD/tokenizer/VIM/FLT/head/objective source, 276 tests, configurations, assets, golden fixtures, logs/manifests/review ZIPs and original failures. [Baseline](../artifacts/phase2d2/baseline_manifest.json), [audit](../artifacts/phase2d2/preservation.json). Existing accepted tests emit fixed historical gradient files: the new runner archives old bytes, captures fresh data separately and restores old bytes in `finally`. All 276 old case identities pass in both regression and complete invocations. N1 stays MVE-T12-only; N2 stays FLT-T17-only.

## Full orchestration and physical deployment

[integration.py](../src/mfvlr/integration.py) owns one `MVE`, one bottleneck-only `VisionDecoder`, one accepted `DetectionHeads`, one full `FineGrainedLanguageTransformer`, and one `SixObjectives`. It does not register the plain accepted external orchestrator or duplicate its children. Appearance encoding runs once, VD trunk once, abs reconstruction residual runs through the same MVE, and appearance/residual global features add to raw Iv. Detector, adapter and FLT consume that same raw fusion. The existing two-channel MD and RGB AD are reused, not duplicated. No production detach, feature normalization, rescaling, gating, activation change or temperature change is introduced.

`forward(image)` / `visual_only(image)` return visual/VD output plus raw Real/Fake logits without tokens, targets, FLT, adapter or losses. `with_language(image, token_ids, sample_ids=...)` explicitly opts into full FLT and adapter, including diagnostic eval-mode use. `compute_objectives(predictions, SuppliedTargets)` validates unique ordered sample IDs, original unshifted token IDs and the accepted strict objective input/target semantics. Per-call records are caller-owned tensors, not deep immutable snapshots or root caches; callers must not mutate or reuse records across unrelated inputs/updates.

`ImageOnlyMFVLR` physically instantiates only MVE, VD and a detector. Its dedicated detector initialization reproduces exactly the first accepted head-generator draw; it does not instantiate an adapter/FLT/objective then discard them. [Actual ownership/serialization report](../artifacts/phase2d2/parameter_ownership_serialization.json) verifies full **154,156,232** and image-only **41,135,303** unique parameters, no duplicate registered parameter/storage owners, exact tied vocabulary identity, save/load identity preservation and exact same-runtime logits. Temporary full checkpoints are tested and cleaned; no trained weights are claimed or bundled.

New [reference](../configs/integration_reference.yaml) / [debug](../configs/integration_debug.yaml) configurations and closed [validator](../src/mfvlr/integration_config.py) preserve the immutable approved seed-0 topology and objective policies. Accepted configurations remain unchanged. New roots use the exact accepted constructors/initialization; configuration cannot grant P2D15/P2D19 or alter counts/tolerances. Resolved copies and source hashes are packaged. Source-checkout execution is verified; wheel packaging is not modified or audited.

Paper traceability remains [heads contract](phase2d_heads_contract.md), [loss/equation ledger](phase2d_losses_contract.md), [numerical readiness](phase2d_numerical_readiness.md), [test plan](phase2d_test_plan.md). **EXPLICIT** paper anchors: Fig.4 p.5, III-B/C pp.5-6 shared appearance/residual encoding and decoder roles; III-D pp.6-7 Eqs.(3)-(16) language/VIM; III-E Eq.(17) p.7 and Eqs.(18)-(27) p.8. Executable reductions/probability interpretations, biases/initialization, baseline architecture and orchestration are **ASSUMED, approved engineering choices**. Counts are **DERIVED**, not evidence of author fidelity. The approximately 141M versus 154,156,232 discrepancy remains unresolved; no dimension was reduced to match it. Published equations are unchanged in the preserved reference ledger.

## Synthetic fixtures and full-graph losses

[Fixture definitions](../tools/phase2d2_fixtures.py), [literal provenance/hash records](../artifacts/phase2d2/fixtures). Full RGB224x224 base values use `0.1+0.8*rand`, seeds real17/EFS412/local997. Local changes complement RGB inside `[64:160,80:176]`; the exact rectangle is an oracle mask, not dataset threshold preprocessing. EFS masks are all one, real masks zero. Duplicate local/local uses seed997 and seed412 with identical DiffFace prompt IDs (golden case5) and distinct image perturbations. Using seed412 for the second local is a recorded fixture resolution within the specified seed set, not a recovered paper data protocol. Token IDs come from unchanged golden real/DDPM/DiffFace cases0/1/5 and are independently verified against the accepted builder/tokenizer. Sample IDs remain unique despite duplicated prompts.

The following are **actual losses on the untrained full synthetic model graph**, not benchmark/detection performance or measured training:

| Fixture | B | FD | LR | CMC | FL | AR | KL | Literal total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real | 1 | 6.77836895 | 248989.578 | -0 | 0.72485292 | 0.326568186 | 30.6405811 | 249028.047 |
| efs | 1 | 0.00114780327 | 261166.703 | -0 | 0.664136589 | 0.326553077 | 30.7707138 | 261198.469 |
| local | 1 | 0.00101586222 | 262327.156 | -0 | 0.712015271 | 0.326007158 | 30.6910515 | 262358.875 |
| real_efs | 2 | 3.38975835 | 255077.562 | 206.23291 | 0.694494724 | 0.326560646 | 30.7056503 | 255318.906 |
| real_local | 2 | 3.38969231 | 255658.125 | 309.225586 | 0.718434095 | 0.326287687 | 30.6658173 | 256002.453 |
| local_duplicate | 2 | 0.00109326362 | 262563.531 | 260.358887 | 0.712038994 | 0.325534165 | 30.7314491 | 262855.625 |

All six raw components retain the exact approved interpretations and unit weights, in order FD/LR/CMC/FL/AR/KL. B=1 CMC is the exact connected-zero formula, not a detached constant. B=2 local negatives/diagonal positives are unchanged; duplicated prompts are not deduplicated or relabeled. Per-case outputs record pad/BOS/EOS/content LR sums/counts, CMC raw/scaled scores/tau/row-column entropy/duplicate counts, KL teacher/student entropy and feature masses, Ipre unrestricted range/outside-[0,1] counts and residual positive/negative/equality coverage. These are detached statistics, not additional production normalizations or loss balancing.

## Actual tests and failure history

New discovery: **45 cases** covering JOINT-I01-I12 and READY-N01-N07, including parameterizations. Complete logs/JUnit include every mandatory old/new case, with no filtering. [Raw final stdout](../artifacts/phase2d2/logs/complete_01.stdout.log), [stderr](../artifacts/phase2d2/logs/complete_01.stderr.log), [JUnit](../artifacts/phase2d2/junit/complete_01.xml), [verification/case IDs](../artifacts/phase2d2/verification.json). Original real-case failure, the initial 19-pass/2-fail JOINT run, initial diagnostic source snapshots, all later failed G2 worker exits and final failed pytest logs are preserved.

Passed JOINT controls verify one-owner DAG, exact loss input pairing/literal sum, gradient presence matrix, FD/LR/CMC/KL reconstruction-mediated AD gradients and detached controls, AR/residual derivative additivity and abs zero subgradient, localization/appearance isolation/shared-trunk accumulation, two-sided KL/visual conditioning/no-LD path, all19 VIM instances and 38 exact-zero Q/K task gradients, text-free inference and actual full-root serialization. Independent FP64 CMC partial-derivative and small FP64 sum oracles pass. CMC partial/cross-sample-coupling checks execute before its strict full FP32 chain assertion; metadata is retained even when that assertion fails.

Final failing cases:

- `tests.joint.test_joint::test_joint_i08_cmc_chain_rule_and_batch_coupling`
- `tests.joint.test_joint::test_joint_i10_fp64_oracle_and_full_fp32_gradient_additivity`
- `tests.joint.test_readiness::test_ready_n02_full_g2_no_update[real]`
- `tests.joint.test_readiness::test_ready_n02_full_g2_no_update[efs]`
- `tests.joint.test_readiness::test_ready_n02_full_g2_no_update[local]`
- `tests.joint.test_readiness::test_ready_n02_full_g2_no_update[real_efs]`
- `tests.joint.test_readiness::test_ready_n02_full_g2_no_update[real_local]`
- `tests.joint.test_readiness::test_ready_n02_full_g2_no_update[local_duplicate]`

JOINT-I08: full FP32 `dCMC/dIv` versus direct plus LE-mediated partial derivatives fails 3/1,024 coordinates; initial pytest reports maximum mismatched absolute difference 2.288818359375e-5 and maximum mismatched relative difference 4.72433494e-5. [Detailed comparator](../artifacts/phase2d2/joint/I08.json) separately reports global maxima/relative-L2/counts, which need not equal pytest's maxima restricted to mismatched coordinates. Independent FP64 derivative mathematics and deliberate objective batch coupling pass.

JOINT-I10 and all six READY-N02 cases: a streamed FP64 sum of already rounded isolated FP32 gradients disagrees with the joint FP32 backward under atol=1e-6/rtol=1e-5. Reference is cast once to FP32 for elementwise `isclose`; high-precision error/L2 diagnostics use the uncast accumulator. No N2 exemption or threshold modification applies.

| Fixture | Outside-tolerance parameter elements | Relative L2 joint vs isolated sum | Final LE max | Final LD max | Joint gradient L2 |
|---|---:|---:|---:|---:|---:|
| real | 8,772,234 | 9.16973535e-07 | 135.765411 | 4507.126465 | 37142195.5 |
| efs | 13,636,267 | 1.95149713e-06 | 136.483063 | 4529.701660 | 42382432.9 |
| local | 11,558,833 | 1.53557234e-06 | 136.009735 | 4522.034180 | 44539438.2 |
| real_efs | 14,149,076 | 1.93785674e-06 | 136.483017 | 4529.796875 | 39328342.2 |
| real_local | 12,728,291 | 1.67403775e-06 | 136.009811 | 4522.030762 | 40047723.3 |
| local_duplicate | 14,611,405 | 2.05079921e-06 | 136.691162 | 4522.030762 | 44780500.4 |

The initial real case had 8,772,234 outside-tolerance elements across 293 parameter tensors, global relative L2 about 9.169735e-7; some weak ordinary LE attention Q/K gradients have much larger relative discrepancies. Global norm alone cannot establish harmlessness. Original failure history remains distinct from the later characterization run adding gradient-cosine/contribution statistics.

## Diagnosis and numerical preservation

[Gradient review](../artifacts/phase2d2/gradient_review/review.json) verifies the streamed file's multi-chunk offsets and arithmetic using an independent exact integer-gradient control. On one full real forward graph, direct FD/LR/mask/adapter prediction derivatives match their isolated sums at the unchanged tolerance. Upstream Iv/Ipre comparisons fail. All seven deterministic real forward-statistic records and isolated/joint scalar values match exactly; parameters do not update. This supports an FP32 backward accumulation/rounding-amplification explanation in shared upstream Jacobians. It does not prove every discrepancy harmless, exclude every numerical risk, or establish full promoted-weight FP64 *gradient-additivity* for this new visual graph. No hidden architectural correction or automatic numerical-policy proposal was applied.

The first G2 gate stopped for diagnosis. Subsequent independent fixtures ran as explicitly labeled numerical characterization after the diagnosed finite failure; **the acceptance gate remained BLOCKED throughout**, with each unchanged allclose assertion failing honestly. Any nonfinite/new failure category would halt remaining characterization. There are no parameter updates between cases or calls.

Original N2 F0/F1/F2 were re-executed in both FP32 and promoted-weight FP64, using the unchanged helper, seeds, initial weights, inputs and backend. [Comparison](../artifacts/phase2d2/n2_comparison.json): every scientific field/backend matches preserved accepted evidence exactly, excluding elapsed timing fields only. Original exact fixed-batch/permutation/cross-sample checks and FP64 cross-batch criteria remain active. Old helper PROPOSED labels remain historical; the accepted scoped N2 record is not rewritten. Original N2 agreement does not waive the new failures or prove training stability.

Per-loss/joint [module-gradient CSV](../artifacts/phase2d2/module_gradients.csv), complete parameter gradients and intermediary path norms are preserved under [G2](../artifacts/phase2d2/g2). Selected detached gradient subsets supply pairwise cosine/contribution/cancellation statistics; unused/zero gradients have undefined cosine, not zero. Large LR token-summed values and gradients can dominate other tasks; raw magnitude alone is not a performance claim. The accepted unnormalized LD sensitivity remains unresolved. No LayerNorm, AMP, rescaling, balancing, clipping, changed attention/temperature or teacher detach was introduced.

## Resources and graph lifecycle

All requested CPU B=1/2 cases ran separately in fresh processes. Sampled process RSS (nominal5ms; actual sample count in JSON) and wall time include model construction, seven fresh full forward/backward graphs, reductions, hash checks and streamed diagnostic storage; they are not pure throughput or exact allocator-peak measurements.

| Fixture | B | Sampled peak RSS MiB | Wall s | Status |
|---|---:|---:|---:|---|
| real | 1 | 2280.195 | 120.679 | finite execution; G2 FAIL |
| efs | 1 | 2284.711 | 124.515 | finite execution; G2 FAIL |
| local | 1 | 2276.191 | 104.613 | finite execution; G2 FAIL |
| real_efs | 2 | 2855.996 | 161.428 | finite execution; G2 FAIL |
| real_local | 2 | 2856.809 | 121.531 | finite execution; G2 FAIL |
| local_duplicate | 2 | 2865.500 | 159.156 | finite execution; G2 FAIL |

Batch8 CPU: **NOT RUN**, explicit [resource-gating record](../artifacts/phase2d2/resource_availability.json); available RAM during diagnostics was about 2GiB while even B=1/2 consume multi-GiB working sets, safe B=8 headroom is unestablished and strict G2 is already blocked. No accumulation surrogate or invented OOM/capacity result. GPU B=1/2/8: **NOT RUN**, CPU-only Torch/no CUDA device. READY-N06/07 passing validates reporting of these facts, not execution of unrun capacities.

No six full graphs coexist: each loss gets a fresh deterministic forward, records detached statistics, backpropagates, streams its parameter gradients, releases outputs/hooks and clears gradients. Weakrefs verify requested outputs are gone; all forward hooks are removed. A temporary1,233,249,856-byte FP64 file stores the sum in262,144-element chunks; selected detached gradient vectors are small diagnostic subsets, not graphs. File/selected tensors close/release in `finally`. Full production tensors/gradients remain FP32. Weights and Parameter identities are hash-verified unchanged. This is no-update execution, not repeated optimizer training.

## Reproducibility and review boundary

Recorded versions: {'torch': '2.12.1', 'pytest': '9.1.1', 'PyYAML': '6.0.3', 'psutil': '7.2.2', 'numpy': '2.4.4', 'ftfy': '6.3.1', 'regex': '2026.6.28', 'wcwidth': '0.2.13'}. [Reproduction guide](../artifacts/phase2d2/REPRODUCE.md), [review ZIP](../artifacts/phase2d2/phase2d2_review_package.zip), [payload manifest](../artifacts/phase2d2/manifest.json), [external ZIP verification](../artifacts/phase2d2/package_verification.json). Archive includes full source, all accepted/new tests/configs, tokenizer/BPE assets/golden fixtures/pinned Windows wheels, original paper, complete raw evidence and historical failures/source snapshots. Previous nested ZIPs are omitted only from the new archive to avoid redundant copies and remain unchanged on disk. ZIP CRC and every payload hash are checked; current manifest/ZIP/verification avoid self-reference. Different-platform pinned regex wheels are not verified or silently substituted.

Phase 2D.2 stops here for review. It cannot be accepted under the current numerical criteria. A further explicitly scoped numerical review or approved remedy is needed before acceptance; this report does not approve a policy amendment or architectural stabilization. P2D15/P2D19/G3/G4, datasets, training and benchmarks remain unapproved.
