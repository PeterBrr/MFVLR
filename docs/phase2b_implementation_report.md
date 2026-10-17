# Phase 2B Vision Decoder implementation report

Client date: 2026-10-11, Asia/Saigon. **Acceptance satisfied for the recorded CPU FP32 runtime: 40/40 cases, covering all 36 IDs; zero failures, errors, skips or expected failures.** D01–D08 were explicitly approved for Phase 2B only, selecting **Option B `bottleneck_only`** as an engineering assumption. No claim of recovered author architecture or reconstruction quality is made.

| Suite | IDs | Cases | Actual final result |
|---|---:|---:|---|
| VD-T01–VD-T14 | 14 | 15 (T01 at b=1/2) | PASS |
| VD-I01–VD-I08 | 8 | 10 (I01/I03 at b=1/2) | PASS |
| Accepted MVE-T01–MVE-T14 | 14 | 15 (T01 at b=1/2) | PASS, source/tests unchanged |
| **Complete suite** | **36** | **40** | **40 passed in 35.29 s** |

[Final pytest log](../artifacts/phase2b/logs/05_complete_suite.log) and [JUnit](../artifacts/phase2b/05_complete_suite.xml) record the actual complete run. JUnit includes collection overhead (36.570 s), hence differs from pytest stdout time. The complete unchanged MVE suite also ran independently: [regression log](../artifacts/phase2b/logs/04_mve_regression.log), [JUnit](../artifacts/phase2b/04_mve_regression.xml), 15 passed in 12.89 s.

## Approval and preserved scope

[Decoder contract](phase2b_decoder_contract.md) records D01–D08 as APPROVED for Option B Phase 2B implementation/verification only. Classification remains ASSUMED. [Test plan](phase2b_test_plan.md) retains acceptance conditions and records the applicable 40-case set. Option A is a future PROPOSED ablation, neither implemented nor approved. Whole A01–A35 decisions, losses, FLT/VIM, classifier, datasets, full training and evaluation remain outside authorization.

Accepted MVE module, package exports, configuration loader/schema, diagnostics, test source/fixtures and both authored MVE config files match their Phase 2A N1 manifest hashes. MVE parameters were not reinitialized during decoder construction; initialization replay tests check caller RNG and MVE parameter version counters. Actual MVE tests passed twice in this phase, unchanged. General FP32 thresholds remain atol=1e-6/rtol=1e-5; analytical FP64 thresholds remain 1e-10/1e-8. N1 remains only in existing MVE-T12's CPU FP32 cross-batch-size comparison; it is not used by any decoder/integration test.

Production VD and synthetic integration are implemented; no paper objective, optimizer schedule, training/evaluation runner or learned detector is implemented. One restored SGD unit-test update verifies shared ownership. Resource diagnostics perform zero updates. Original published equations and Phase 1 research records remain unchanged.

## Production source and API

[vision_decoder.py](../src/mfvlr/vision_decoder.py) implements the exact approved topology. Paper anchors are Fig. 4 p. 5, §III-C p. 6 and §IV-A p. 8. Internal layers/initialization are the D01–D08 engineering contract, not explicit author details.

One `trunk` owns four stages. Each stage has one biased 2×2 stride-2 transpose convolution, then two biased padded 3×3 convolutions with ordinary non-inplace ReLU. Widths are 1024→512→256→128→64, with spatial sizes 14→28→56→112→224. There is no activation immediately after the up-convolution, skip concatenation, norm, dropout, crop, interpolation or extra residual block. One trunk evaluation supplies both independent biased 1×1 heads: AD 64→3, MD 64→2. Both output identity values; mask output is raw two-class logits.

`VisionDecoder(skip_policy="bottleneck_only", seed=0)` requires an explicit policy and rejects every other policy before allocation. `VisionDecoder.from_config(config)` requires scoped approval in all referenced decoder profiles. `forward(local, *, skips=None)` rejects any explicit skip argument, including an empty tuple. `decode_appearance(appearance)` accepts the existing `AppearanceEncoding` and reads only its local tensor. Unused skips/tokens/global values are not decoder inputs; the existing MVE continues to validate its full appearance record unchanged.

The immutable per-call `VisionDecoderOutput` has `features`, `reconstruction`, `mask_logits`, with shapes [b,64,224,224], [b,3,224,224], [b,2,224,224]. Tensors retain their graph; no persistent activation cache is stored. Strict validation rejects wrong type/rank/channel/resolution/empty batch/dtype/device and nonfinite inputs/outputs. No implicit conversion, image normalization, clamp or output probability conversion occurs.

Initialization uses a private CPU seed-0 generator in approved stage/head order. Transpose-convolution and head weights use Xavier uniform gain 1; 3×3 ReLU weights use Kaiming normal fan-in/ReLU; every bias is zero. `torch.random.fork_rng` protects the caller's global CPU RNG. The VD is initialized independently of the accepted MVE. Same-runtime exact replay and independent RNG-order reconstruction pass.

## Separate configuration system

[decoder_reference.yaml](../configs/decoder_reference.yaml) and [decoder_debug.yaml](../configs/decoder_debug.yaml) independently record D01–D08, Option B, ASSUMED classification and Phase 2B scopes. Both prohibit full training/evaluation and author-architecture confirmation; Option A remains PROPOSED.

[decoder_config.py](../src/mfvlr/decoder_config.py) uses a separate packaged [decoder_contract_snapshot.json](../src/mfvlr/decoder_contract_snapshot.json) containing required-key/type schemas and locked architecture, initialization, integration, verification and accepted MVE settings. It reuses safe YAML/strict merge helpers from the unchanged accepted loader, without altering them. Duplicate/unknown/missing keys, wrong types, drift, cyclic bases and broadened scope fail. Each profile's approval is checked independently; a proposed base or overlay cannot inherit approval.

Overlay bases resolve relative to their declaring file. Maps merge recursively; sequences/scalars replace; explicit null replaces; inputs are not mutated. Actual debug clears a diagnostic note to null and changes workload to three checks at b=2. Debug architecture/init/integration/tolerances remain identical to reference. The referenced MVE config resolves relative to the base file even when an overlay is elsewhere; its approved settings are checked unchanged. The legacy MVE reference's unapproved Option A decoder blueprint is not used to construct VD.

Resolved decoder configs and independent approval-chain exports are in [resolved_configs](../artifacts/phase2b/resolved_configs). Exports rebase the MVE config path relative to their destination and reload to the same proposal settings. Flattened effective metadata retains the base review; the sidecar preserves every independently validated profile review. Authored profiles remain the source for overlay-chain reproduction.

## Actual parameter report

Measured from constructed production modules in VD-T05 and each resource diagnostic, counted once despite both conceptual decoder roles.

| Owner | Unique parameters |
|---|---:|
| U1 | 6,817,280 |
| U2 | 1,704,704 |
| U3 | 426,368 |
| U4 | 106,688 |
| Shared trunk | 9,055,040 |
| Appearance head | 195 |
| Mask head | 130 |
| **VD total** | **9,055,365** |
| Accepted MVE | 32,078,912 |
| **MVE + VD total** | **41,134,277** |

VD has 28 parameter tensors; MVE+VD has 116 unique parameter tensors. State/optimizer enumeration and call hooks prove one decoder trunk and one shared visual encoder owner, with distinct heads and no encoder/decoder weight tying. [parameter_report.json](../artifacts/phase2b/parameter_report.json) records actual counts and measured source. These are specification consistency checks, not an inference from the paper's rounded full-model totals.

## Unit and integration evidence

Source: [unit tests](../tests/decoder/test_unit.py), [integration tests](../tests/decoder/test_integration.py), [decoder fixtures](../tests/decoder/conftest.py). The accepted parent test fixtures are unchanged, including their Phase 2A startup banner.

| IDs | Verified behavior |
|---|---|
| VD-T01/T07 | Full b=1/2 layer shape trace, exact operators/channels/activation placement; no skips/norm/dropout |
| VD-T02 | Independent small FP64 transpose-convolution placement and gradients, including boundaries |
| VD-T03/T09/T14 | Bottleneck-only input dependency, explicit skip rejection, malformed-input failures, graph-connected output record/no cache |
| VD-T04/T05/T12 | One trunk call, separate heads, unique optimizer/state identities/counts; combined trunk gradient equals AD+MD sums; restored unique update |
| VD-T06/T10 | Independent affine-head calculation, negative/above-one reconstruction and unrestricted logits, train/eval parity, exact fixed-batch independence/permutation |
| VD-T08/T11/T13 | Exact initialization/RNG replay, actual state round trip, mismatch rejection, strict config/schema/overlay/scope checks |
| VD-I01/I02 | Actual finite MVE→VD→residual DAG, cached/recomputed parity, each encoder role once, VD once, exact abs/fusion |
| VD-I03 | Reconstruction-mediated path isolated at both batch sizes, with detached controls |
| VD-I04/I05 | Mask/appearance probe dependency isolation; separate head and Transformer-only parameters unused where appropriate |
| VD-I06 | Actual decoder out-of-range output accepted by unchanged MVE; abs derivative sign/zero subgradient |
| VD-I07/I08 | Integrated ownership/serialization, preserved MVE source/config/test hashes and narrowly scoped N1 |

The initial selected unit-shape run passed 9 cases; selected backward run passed 8; selected serialization/config run passed 5. Intermediate test selection is explicit in logs. The final run executes all 40 cases with no deselection. All new unit/integration tests use the original general tolerances. There were **no failing test runs or unresolved acceptance failures in Phase 2B**. Historical Phase 2A failures remain untouched in their original artifacts.

### Isolated reconstruction-to-residual gradient

VD-I03 probes only `mean(residual_global**2)`, excluding the direct appearance-global contribution. It differentiates with respect to reconstruction, appearance local tensor, AD head, decoder U1 up-convolution and an encoder convolution. All are finite and nonzero for these fixtures; the distinct mask head is unused. Values printed by the actual final pytest run are:

| Target gradient norm | b=1 | b=2 |
|---|---:|---:|
| Reconstruction [b,3,224,224] | 0.056908414 | 0.040240232 |
| Appearance local [b,1024,14,14] | 0.008784664 | 0.006211686 |
| AD head weight | 0.164274797 | 0.164274737 |
| U1 up-convolution weight | 0.383710384 | 0.383710444 |
| Shared encoder first convolution weight | 0.828151941 | 0.828151762 |

Detached reconstruction control gives `None` for the decoder head, U1 up-convolution and appearance local gradients at both batch sizes, while RE/direct-image paths remain nonzero. This separates the decoder-mediated path from the direct `−I` residual path. Shared encoder gradients alone would not prove reconstruction propagation. [gradient_flow.json](../artifacts/phase2b/gradient_flow.json) extracts these actual passed-test values and control observations, with the source log identified.

Mask-only and appearance-only probes also verify the expected unused opposite head and unused Transformer-only parameters. No test requires every ReLU weight to have a nonzero gradient. Synthetic probes are diagnostic scalars, not implementations of paper losses.

## CPU execution and memory measurements

Python 3.14.3; PyTorch 2.12.1+cpu; pytest 9.1.1; PyYAML 6.0.3; psutil 7.2.2; Windows-11-10.0.26200-SP0; CPU FP32, four PyTorch threads, deterministic algorithms. Dependencies were already installed. CUDA is unavailable in this PyTorch runtime: **GPU measurements NOT RUN**, not passing or skipped GPU tests. No batch-8 or mixed-precision capacity claim is made.

[diagnose_phase2b.py](../tools/diagnose_phase2b.py) runs each row below in a separate fresh process. Forward uses no-grad; backward uses actual production VD/MVE and a synthetic scalar. No warm-up is performed. Per-iteration times include input creation, validation, trace hooks and backward where selected, but exclude imports/model construction. b=2 backward rows include all three debug iterations.

| Case | Actual iteration seconds | Sampled interval peak RSS MiB | Windows lifetime peak working set MiB |
|---|---|---:|---:|
| VD b=1 forward | 0.209453 | 389.74 | 389.74 |
| VD b=1 forward+backward | 0.655641 | 471.35 | 472.60 |
| VD b=2 forward | 0.441786 | 443.26 | 443.31 |
| VD b=2 forward+backward | 1.108087 / 1.077360 / 1.110235 | 628.82 | 628.82 |
| MVE+VD b=1 forward | 0.669159 | 566.00 | 566.05 |
| MVE+VD b=1 forward+backward | 1.954817 | 884.55 | 886.68 |
| MVE+VD b=2 forward | 1.174399 | 621.50 | 621.50 |
| MVE+VD b=2 forward+backward | 3.873692 / 3.603399 / 3.620906 | 1,212.12 | 1,214.30 |

RSS is sampled every 10 ms, with boundary samples; short peaks can be missed. Windows `peak_wset` is a process-lifetime measurement including imports, model construction and workspace allocations. Both include framework/process overhead and are not tensor-only storage or GPU VRAM. Small differences between final RSS and sampled peak can arise from polling/teardown allocations. Fresh-process comparisons are more meaningful than subtracting peaks from different cached phases, but are still hardware/runtime-specific observations. Recorded integrated b=2 sampled peak is 1.18 GiB; no inference-quality or throughput benchmark is claimed.

Each backward diagnostic has finite gradients for all 28 VD parameter tensors; integrated diagnostics additionally have finite gradients for all 88 MVE parameter tensors. Three-iteration b=2 runs perform zero optimizer updates. Forward and backward shape events are in each diagnostic's `tensor_trace.json`; retain-grad shapes/norms are in its `summary.json`. [memory_runtime.json](../artifacts/phase2b/memory_runtime.json) aggregates measured rows with original byte counts and intervals. CPU records are actual measurements, distinct from the design document's storage arithmetic.

## Reproduction and review package

From the extracted package root, with recorded dependencies installed:

```powershell
python -m pytest -vv -s --junitxml=artifacts/recheck/complete_suite.xml
python -m pytest -vv -s tests/test_mve.py --junitxml=artifacts/recheck/mve.xml

python tools/diagnose_phase2b.py --component vd --batch 1 --output-dir artifacts/recheck/vd_b1
python tools/diagnose_phase2b.py --component integrated --batch 1 --backward --output-dir artifacts/recheck/integrated_b1
python tools/diagnose_phase2b.py --component integrated --batch 2 --backward --iterations 3 --output-dir artifacts/recheck/integrated_b2
```

Use `artifacts/recheck` to preserve original execution logs. `pyproject.toml` supplies pytest's `src` path and packages the separate decoder schema resource; no editable installation is required for these commands. Production imports are `mfvlr.vision_decoder` and `mfvlr.decoder_config`; the accepted root MVE exports remain untouched. `psutil` is an optional diagnostic dependency, not a model-forward dependency.

[Review ZIP](../artifacts/phase2b/phase2b_review_package.zip) contains actual source/tests, authored/resolved configs, strict schema assets, logs/JUnit, counts, gradient/shape evidence, measured resources, environment and hash manifests, documentation and the supplied paper. [Package instructions](../artifacts/phase2b/REVIEW_PACKAGE.md) and [builder](../tools/build_phase2b_review_package.py) describe verification/rebuilding. The archive uses sorted paths and fixed entry timestamps; rebuilding identical recorded inputs must reproduce its hash. It contains no learned checkpoint or dependency binaries. Temporary serialization checkpoints were actual round-trip test artifacts and are not production trained weights.

## Changed files and stopping point

Created: `src/mfvlr/vision_decoder.py`, `decoder_config.py`, `decoder_contract_snapshot.json`; `configs/decoder_reference.yaml`, `decoder_debug.yaml`; `tests/decoder/{conftest.py,test_unit.py,test_integration.py}`; diagnosis/package tools; Phase 2B artifacts and this report. Updated: Phase 2B contract/test-plan approval records, README, and `pyproject.toml` to package the new schema and declare optional psutil diagnostics. Accepted MVE source/interfaces/tests/configuration behavior and original research documents are preserved.

Remaining limits: no CUDA/b=8 measurement, decoder training, reconstruction-quality assessment or downstream integration beyond synthetic MVE–VD probes. Cached feature provenance remains a caller obligation. Author decoder skip policy/internals remain unknown. Option A and all subsequent components still need separate approval. Phase 2B implementation/CPU verification is complete; stop and wait for review.
