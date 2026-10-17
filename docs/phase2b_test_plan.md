# Phase 2B Vision Decoder verification plan

Date: 2026-10-11, Asia/Saigon. Revision 2: **APPROVED engineering test requirements under D08, Phase 2B Option B only**, by the explicit user implementation instruction. Companion specification: [phase2b_decoder_contract.md](phase2b_decoder_contract.md). The following remains the acceptance plan; execution outcomes are separate in [phase2b_implementation_report.md](phase2b_implementation_report.md). Phase 2A MVE, its 14 IDs / 15 accepted cases and scoped N1 remain preserved.

Declared implemented case set: VD-T01 has b=1/2, VD-T02–T14 each one case (15 unit cases); VD-I01 and VD-I03 have b=1/2, VD-I02/I04–I08 each one case (10 integration cases); unchanged MVE has 15 cases. Complete acceptance therefore requires **40 cases / 36 IDs**, with zero failures, skips or xfails. Option A-specific positive paths are outside this selected-policy suite, not skipped tests.

## Verification boundary and conventions

Proposed tests cover only VD and its integration with accepted MVE. Use synthetic tensors and deterministic diagnostic probes. No dataset, production loss, classifier, language, optimizer schedule, training loop or benchmark evaluator is needed. A single restored optimizer-step fixture may verify unique shared ownership; that is a unit test, not full-model training.

**ASSUMED / PROPOSED test conventions (D08):** preserve general FP32 `atol=1e-6, rtol=1e-5` and isolated analytical FP64 `atol=1e-10, rtol=1e-8`. Shapes, IDs, counts, record fields, sharing and fixed-batch independence use exact equality. N1's 5e-6 threshold applies only to existing MVE-T12's CPU FP32 cross-batch-size comparison; it is not a new decoder or integration tolerance. Record/investigate any new numerical failure before proposing an amendment. No skips/xfails may conceal a required CPU acceptance check.

Use the full approved topology at b=1 and b=2, no reduced channel/debug model. Fixtures use a private seed stream independent of model initialization; record seeds, versions and device. Decoder seed/order tests use the proposed seed 0. FP64 fixtures may exercise a copied isolated operator on small tensors without changing the production FP32 API or accepted MVE. Do not assert nonzero gradients for every random ReLU parameter; use controlled, nondegenerate fixtures and inspect the intended path.

Policy binding: implement only the explicitly approved baseline initially. For B, A-specific positive-path tests below are not part of that implementation's suite; test B's rejection of explicit skips and independence of unused record fields instead. For A, test all four skip paths. Report the selected policy and applicable checks prospectively, not as runtime skipped/xfail cases. If both variants are later authorized, parameterize over both and verify their distinct counts/configs; do not silently enable the unselected architecture.

## Decoder unit tests

Paper anchors: §III-C p. 6 and Fig. 4 p. 5 for roles/sharing/input; §III-B p. 5 and §IV-A p. 8 for endpoints; assumptions and exact internal choices are D01–D06 in the contract. Test IDs and mechanisms are engineering proposals.

| ID | Purpose and fixture | Acceptance | Traceability |
|---|---|---|---|
| VD-T01 | Full layer hooks on standalone decoder at b=1 and b=2 | Every up/assembly/Conv1/Conv2/head shape matches the selected policy's contract table; output F [b,64,224,224], reconstruction [b,3,224,224], mask [b,2,224,224] | pp. 5/6/8 endpoints; D01/D02/D04 |
| VD-T02 | Independent coordinate/impulse calculation for transpose convolution, including boundaries | Kernel2/stride2/dilation1/padding0/output_padding0; no output crop/interpolation; small FP64 fixture agrees with manual placement and overlap summation | Unspecified upsampling, A02; D02 |
| VD-T03 | Selected-policy dependency | B low-level API rejects any explicit skips; wrapper outputs unchanged when unused skips/tokens/global token change with identical local tensor. A concatenates `[up,S3/S2/S1/S0]` on channels; tagged channels verify order; valid changed skips affect outputs in controlled fixture | p. 6 Iloc input; [44] alternative; D01/D06 |
| VD-T04 | Ownership and one-trunk call hooks; optimizer/state enumeration | Exactly one trunk evaluation per forward; separate head parameter identities; 28 VD parameter tensors; single trunk state subtree, no duplicate optimizer identities and no MVE weight tying | p. 6 sharing; Fig. 4; D03 |
| VD-T05 | Unique parameter report by stage/head | B: trunk 9,055,040; VD 9,055,365; MVE+VD 41,134,277. A: trunk 12,188,480; VD 12,188,805; MVE+VD 44,267,717. Heads 195/130; count shared trunk once | DERIVED count, D02/D03/D04/D08; not author fidelity |
| VD-T06 | Controlled head affine calculation, including negative/above-one reconstruction and arbitrary class scores | Independent 1×1+bias calculation matches; reconstruction can be <0 and >1; two-class logits unconstrained; no sigmoid/tanh/clamp/softmax/argmax or hidden reconstruction-to-input addition | p. 6 heads/output channels; A05/A18 interface, D04 |
| VD-T07 | Architecture enumeration and trace | Exactly four biased transpose convolutions, eight biased 3×3/ReLU layers, two independent biased 1×1 heads; no norm/dropout/resize/extra decoder block; no ReLU immediately after up or heads | ASSUMED detailed topology, D02/D04 |
| VD-T08 | Seed/order replay and initialization | Same-runtime seed replay exact; independent prescribed RNG order recreates all parameters; biases zero; Xavier ranges correct, Kaiming distribution checked on suitable samples; global RNG and existing MVE state unchanged | Scratch statement p. 4; A22/D05 |
| VD-T09 | Invalid input and provenance boundary cases | Reject wrong tensor/type/rank/channel/size, empty batch, wrong dtype/device, nonfinite values and malformed A skips; no broadcast/resize/cast. B wrapper never consumes unused record fields; low-level B rejects explicit skips | A35/D06; accepted MVE boundary |
| VD-T10 | Train/eval and sample independence | Same input train/eval agrees at general thresholds. Holding b=2 fixed, change peer sample's local (and A skips): first output exact. Check batch permutation. Cross-batch-size comparisons, if added, keep general thresholds unless separately approved | No norm/dropout, D02/D08; N1 scope preserved |
| VD-T11 | Actual save/load with a new instance; state only | Reconstruction/mask/F bitwise unchanged on same-runtime/device round trip; sharing/call count preserved; per-call feature records absent from module state; selected policy recorded and topology-mismatched load fails | A35/D03/D06 |
| VD-T12 | Two-head shared-trunk gradient sum | On controlled input, trunk gradients from combined AD/MD probes equal separately computed sums; head-only gradients stay separate; one selected shared parameter update matches expected unique update and is restored | p. 6 sharing; D03/D07 |
| VD-T13 | Dedicated strict decoder profiles/schema and approval gate, after approval | Explicit policy required; no implicit switch/fallback; decoder debug identical topology/init/heads/sharing. Unknown/drifting keys and missing/broadened approval fail; overlay map/list/null semantics preserved; A's historical reference proposal cannot silently instantiate B; accepted MVE profiles/loader unchanged | A35/D01/D06/D08; separate VD validation referencing accepted MVE |
| VD-T14 | Output-record API and lifetime | Field names/shapes/type match; local/output graph remains connected; no implicit clone/detach/cache; no module-owned MVE or recursive residual decoder; outputs valid in caller no-grad mode | §III-A p. 4; Fig. 4; D03/D06/D07 |

For VD-T09, device-mismatch validation can use a meta-device sentinel without requiring CUDA, as in accepted MVE tests; valid GPU execution is a separate resource check. Record provenance cannot be fully established from tensor shapes and remains a caller responsibility. Under A, test swapped/order-mismatched skips and wrong batch as errors, not silent resizes. Under B, record skips remain available to the unchanged MVE even though VD does not read them.

## MVE–VD integration tests

All use the accepted `encode_appearance` → VD → `MVE.forward(..., appearance_encoding=...)` sequence and actual approved VD once available. Hooks must observe role calls without modifying MVE source. Synthetic scalar probes below are solely graph diagnostics and are not implementations of MFVLR losses.

| ID | Test | Acceptance / isolation method | Evidence |
|---|---|---|---|
| VD-I01 | Complete finite forward DAG at b=1/2 | Appearance local/token/Transformer encode once, VD trunk once, each head once, residual local/token/Transformer encode once; no decoder call on residual. Exact `Ir=abs(Ipre−I)` and `Iv=Ig+Ir_g`; all contractual shapes | Fig. 4 p. 5; §§III-B/C pp. 5–6; D06/D07 |
| VD-I02 | Appearance record reused versus recomputed | Same weights/input give equivalent outputs under general thresholds; cached route saves appearance role call. `AppearanceEncoding.skips` remains four intact graph-connected tensors; no MVE interface changes for B | Accepted MVE API/MVE-T07; D01/D06 |
| VD-I03 | Reconstruction-mediated gradient through RE | Probe only `residual_global`, excluding direct appearance-global contribution. Retain gradients on `Ipre` and appearance `local`; require finite nonzero path gradients and representative AD-head/trunk gradients with a nondegenerate fixture. Control using `Ipre.detach()` removes decoder/local path, while RE parameters and direct input path can still have gradients | p. 5 residual, p. 6 sharing; accepted M3/MVE-T08; D07 |
| VD-I04 | Localization dependency isolation | Weighted mask-logit probe reaches mask head, shared trunk and local encoder; appearance head and Transformer-only parameters unused (`None`, not zero assumptions). For A, inspect intended skip branches too | p. 6 MD input/shared trunk; D01/D03/D07 |
| VD-I05 | Reconstruction dependency isolation | AD-output-only probe reaches AD head, trunk and local encoder; mask head and Transformer-only parameters unused. Under B gradients enter local encoder via bottleneck; under A both bottleneck/skips are connected. Structural absence of skip consumption proven separately | p. 6 AD input; D01/D03/D07 |
| VD-I06 | Real VD output outside appearance range | Controlled head fixture produces reconstruction values below 0/above 1; unchanged MVE accepts finite residuals >1. Verify derivative sign away from equality and zero subgradient at equality using existing conventions | §III-B p. 5; A05/D04/D07; accepted MVE-T09 |
| VD-I07 | Shared ownership across integrated state and backward | Accepted MVE count stays 32,078,912; one encoder and one decoder owner. No duplicated encoder/decoder optimizer tensors; different role probes sum correctly. Serial round-trip restores VD policy and MVE+VD outputs without reinitializing MVE | §§III-B/C pp. 5–6; D03/D05/D08 |
| VD-I08 | Accepted Phase 2A regression and preservation | All 14 MVE IDs / 15 cases pass unchanged, including exact fixed-batch independence and narrowly scoped N1. Accepted MVE source/test/config/loader hashes remain preserved; new VD profiles reference the same MVE settings | User-accepted Phase 2A and N1; design-review boundary |

VD-I03 distinguishes decoder-mediated gradients from the direct residual path `−I`: shared visual encoder parameter gradients alone do not prove the decoder path. Inspecting `Ipre`, `Iloc` and decoder parameters, plus the detached control, does. A probe on `fused_global` alone would also include the appearance-global branch, so it is insufficient as the isolated proof. Finite differences must avoid zero abs residuals and ReLU kinks; test abs equality separately.

VD-I04/I05 prove absence of a head dependency, not that all shared weights must receive nonzero numerical gradients. Mask gradients can update the shared trunk and affect future reconstruction after an update even though the current mask probe does not directly use the appearance head. No detector or language module is needed to verify reconstruction→residual-global propagation.

## Execution order after approval

1. Record exact approval of skip policy and D02–D08. Pin the selected architecture/count in separate, scoped decoder reference/debug profiles and a dedicated strict schema; reference the accepted MVE config without changing its loader/tests. Record source/config hashes and environment. Do not reinterpret the existing unapproved Option A reference fields as new approval.
2. Build only VD and its source tests. Run CPU b=1 standalone no-grad forward smoke, then VD-T01/T02/T03/T04/T05/T06/T07/T09/T14. Preserve full widths/resolution.
3. Run CPU b=1 backward/isolation tests VD-T12 and VD-I01–I07, using controlled inputs to separate direct and reconstruction-mediated paths. Retain only current-call traces; clear gradients between independent probes.
4. Run b=2 standalone/integrated shape, train/eval and sample-independence diagnostics. Perform three synthetic forward/backward checks without optimizer updates, reporting scalar purpose explicitly. T12's restored optimizer-step unit fixture is separate from these diagnostics.
5. Run initialization/serialization/configuration tests VD-T08/T11/T13 and integrated serialization checks. Execute the unchanged complete MVE suite and then the complete applicable VD/integration suite, with no required-case skips/xfails or deselection in the final acceptance run.
6. If a supported CUDA device is available, record GPU FP32 b=1/2 forward/backward and memory results; attempt b=8 only after capacity is measured. If absent, report NOT RUN as a resource limitation outside CPU acceptance, never as a passing GPU test. Do not add a required GPU test that gets marked skipped in the CPU suite.

Test selection in intermediate steps accelerates diagnosis; the final accepted run must execute every declared applicable CPU case. Both VD-T01 batch cases are required. The count of cases depends on parameterization and the selected policy and must be enumerated before final verification. Do not promise a numerical reconstruction-quality score from an untrained decoder or treat shape/count tests as reconstruction fidelity.

## Resource checks and artifacts

Use the contract's arithmetic as storage planning, not a measured peak. Measure standalone VD versus MVE+VD, each under no-grad and autograd, so the decoder increment is visible. Baseline B still holds MVE's appearance skips and both encoder role graphs; release per-call records after backward. Option A additionally retains concatenations/wider-convolution inputs. No actual memory ceiling or runtime threshold is specified by the paper.

CPU: report process RSS/peak where supported, allocation measurement method, thread count, b and elapsed intervals; distinguish imported framework overhead from tensor payload. GPU: synchronize timing, reset peak counters around a defined interval, report peak allocated and reserved memory plus device total/capacity, precision and runtime. These APIs measure different quantities; allocator peaks are not the same as physical process memory. Warm-up behavior/workspaces must be described rather than hidden in a single estimate. No GPU benchmark is part of this design review.

Required future verification artifacts:

- Approved decision/skip-policy record, classification/provenance and scoped resolved configs.
- Actual source/tests, version/hash manifest and unchanged accepted MVE regression evidence.
- Raw stdout/pytest logs and JUnit for final complete suites, retaining failures/retries.
- Unique stage/head/integrated parameter counts and parameter identity/state-tree report.
- Forward hooks for every stage/head/encoder role and backward traces for reconstruction/local/skip paths where applicable.
- Isolated gradient-probe and detached-control values, synthetic fixture seeds and scalar definitions.
- Initialization replay and real checkpoint round-trip results.
- CPU/GPU resource measurements and explicit NOT RUN/failed resource checks, without topology reductions.

## Acceptance criteria and current review status

**Proposed implementation acceptance:** selected-policy layer/API/count requirements and all declared applicable CPU unit/integration cases pass; both b=1/2 cases pass; unchanged Phase 2A regression passes; exact sharing, reconstruction-gradient path, output ranges and state round trip are demonstrated; logs/configs/counts/traces are reviewable. New failures are reported explicitly and investigated before any tolerance change. Optional GPU/b=8 limitations remain visible and do not become claims of validated resource capacity.

Original design-review checks: paper/figure evidence reviewed, both variant counts and memory arithmetic checked, source/API/hash boundary inspected, and cross-references reviewed. Actual Phase 2B implementation measurements and test results are recorded separately in the implementation report and raw artifacts. Classifier/FLT/VIM/losses/training/evaluation remain outside authorization. Stop after Phase 2B implementation and verification.
