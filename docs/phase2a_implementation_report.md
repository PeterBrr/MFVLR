# Phase 2A MVE implementation and verification

Initial M1–M5 client approval date: 2026-10-10. Scope: MVE only, under approved **engineering assumptions**. **Final acceptance after conditionally approved N1: all 14 test IDs pass, comprising 15 cases including both T01 batch sizes; zero failures, errors, skips or expected failures.** The original strict run had 14 passing cases and one T12 failure. That history is preserved below; the final acceptance section records the scoped amendment and new evidence.

Actual unique parameter count: **32,078,912**, exactly matching the approved specification. This is internal consistency evidence, not evidence of recovered author architecture. No benchmark or author-fidelity claim is made.

## Approval and scope

[implementation_contract.md](implementation_contract.md) and both [paper_reference.yaml](../configs/paper_reference.yaml) and [debug.yaml](../configs/debug.yaml) record M1–M5 as APPROVED for MVE implementation and verification only. Engineering choices retain their ASSUMED classification. Both profiles have independent scoped approval records; approval does not propagate through overlays.

Only relevant portions of A01/A02/A03/A04/A05/A06/A07/A16/A22/A23/A30/A31/A35 are covered. No complete A01–A35 assumption ID is blanket-approved. The full-reference decision register remains PROPOSED outside those portions. Training authorization remains false, author-architecture confirmation remains false, and production skip-decoder approval remains false. The four accepted Phase 1 documents are unchanged; their hashes are recorded in the artifact manifest.

Production Vision Decoder, classifier, FLT, VIM, paper losses, dataset loaders and full training are absent. T08 uses a parameter-free, test-only differentiable reconstruction fixture; it is not a production decoder. T10 checks a single SGD update on selected shared parameters and restores their values; this is a unit test, not authorization of a training pipeline. Synthetic diagnostics perform zero optimizer updates.

## Implemented architecture and interfaces

Implementation: [mve.py](../src/mfvlr/mve.py). The complete approved layer table, proposed image-block formulas and initialization order remain in the contract. Paper anchors are §III-B, PDF pp. 5–6, Eqs. (1)–(2), and §IV-A, p. 8; internal layer choices are approved assumptions rather than explicit paper facts.

One `MVE.shared_encoder` owns all visual parameters. Both appearance and absolute-residual roles call that object. There are five double-convolution stages with widths 64/128/256/512/1024, four max pools, padded biased 3×3 convolutions and non-inplace ReLU. There is no convolutional normalization or dropout. Appearance skips S0–S3 remain graph-connected outputs.

Row-major flattening produces 196 spatial tokens; an affine 1024→512 projection, appended CLS at index 196, and shared learned positions produce 197 tokens. Four independently parameterized pre-LN blocks use eight 64-dimensional heads, separate biased Q/K/V/O projections, key-axis softmax scaled by sqrt(64), and a 512→2048→512 FFN with exact GELU. LN epsilon is 1e-5. There are no masks, dropout or terminal LN. These are visual attention choices; the separate published VIM equation is untouched and VIM is unimplemented.

Initialization uses a private CPU RNG stream seeded with 0, in the approved order: ten convolutions, projection, CLS, positions, then each block's LN1/Q/K/V/O/LN2/FC1/FC2. Conv weights use Kaiming normal fan-in/ReLU; linear weights use Xavier uniform gain 1; biases are zero; LN scales are one; CLS/positions use untruncated Normal(0,0.02). Initialization preserves the caller's global RNG. Same-runtime replay is tested; cross-version bitwise equality is not claimed.

The public API is:

| Method | Result |
|---|---|
| `SharedVisualEncoder.encode_local(image, return_skips=True)` | `LocalEncoding(local, skips)` |
| `SharedVisualEncoder.encode_global(local)` | `GlobalEncoding(tokens, global_token)` |
| `MVE.encode_appearance(image)` | `AppearanceEncoding(local, skips, tokens, global_token)` |
| `MVE.forward(image, reconstruction, appearance_encoding=None)` | `MVEOutput(appearance, residual_image, residual_tokens, residual_global, fused_global)` |
| `MVE.from_config(resolved_config)` | Approved-scope MVE instance; no other component factory |

Forward computes `residual_image = abs(reconstruction - image)`, encodes it with the shared encoder and returns `fused_global = appearance.global_token + residual_global`. Neither operand is detached, clipped, normalized or resized. Supplying an appearance record avoids a second appearance encoding. No persistent activation cache is stored in module state.

Boundary checks reject incorrect rank/channel/resolution, empty batches, non-FP32 tensors, mismatched devices/batches, nonfinite values and malformed appearance records. Appearance RGB/[0,1] scaling remains a caller preprocessing obligation; the model performs no dataset preprocessing or range enforcement. A cached record must come from the identical input and current graph/parameters. Structural checks cannot prove that provenance; it remains the approved caller obligation.

## Parameter count

Measured by construction and checked in MVE-T05 and both diagnostic runs. Each unique parameter is counted once, although the encoder executes twice.

| Owner | Parameters |
|---|---:|
| Local CNN | 18,843,200 |
| Spatial projection | 524,800 |
| CLS | 512 |
| Positions | 100,864 |
| Transformer block 1 | 3,152,384 |
| Transformer block 2 | 3,152,384 |
| Transformer block 3 | 3,152,384 |
| Transformer block 4 | 3,152,384 |
| **Unique total** | **32,078,912** |

There are 88 parameter tensors. [parameter_report.json](../artifacts/phase2a/parameter_report.json) records the measured counts. Checkpoints contain one encoder subtree; optimizer enumeration contains no duplicate identities. No excluded component contributes to this count.

## Configuration loading

[config.py](../src/mfvlr/config.py) implements safe YAML parsing, duplicate/non-string key rejection, required/unknown-key checks, strict scalar types and finite-number checks. [contract_snapshot.json](../src/mfvlr/contract_snapshot.json) is a packaged, static validation schema and approved-value snapshot, not an alternate model implementation. It locks MVE, initialization, verification settings and recorded paper facts/equations against drift.

Debug bases resolve relative to the overlay file, with cycle detection. Only `proposal` receives recursive map merging; lists/scalars replace, explicit null replaces, and neither source is mutated. Unknown overlay keys fail. Metadata and approval records are validated independently rather than merged into authority. The resolved debug architecture, initialization and verification are identical to the reference. A config cannot broaden authorization to training, another component, author fidelity or a skip-connected production decoder.

Full-reference fields for future modules are retained as unimplemented proposals. Loading them is not permission to instantiate or train those modules. The approved tensor count and general thresholds remain unchanged in both configs. N1 now appears as an independently approved review record in each config and a separate, locked `verification.mve_t12_cross_batch` setting. Only that comparison reads the amended threshold; T14 rejects attempts to relax general thresholds or broaden N1's scope.

## Actual execution environment and sequence

Python 3.14.3; PyTorch 2.12.1+cpu; PyYAML 6.0.3; pytest 9.1.1; Windows-11-10.0.26200-SP0. Runs used CPU FP32, four PyTorch threads and deterministic algorithms. CUDA was unavailable. Dependencies were already installed. Logs contain actual process output, including failures; timestamps in JUnit describe the host clock, not the client approval date.

| Order | Execution | Actual outcome | Evidence |
|---|---|---|---|
| 1 | Batch-1 smoke, first attempt | Configuration failure before forward | [01_smoke.log](../artifacts/phase2a/logs/01_smoke.log) |
| 2 | Batch-1 smoke after nullable-field fix | Passed; 0.579 s measured forward interval | [01_smoke_retry.log](../artifacts/phase2a/logs/01_smoke_retry.log), [summary](../artifacts/phase2a/smoke/summary.json) |
| 3 | Shapes/sharing, approved thresholds | 7 passed, T12 failed | [02_shapes.log](../artifacts/phase2a/logs/02_shapes.log), [JUnit](../artifacts/phase2a/02_shapes.xml) |
| 4 | Investigate batch numerical difference | Fixed-batch sample substitution invariant exactly; cross-size max difference 2.861e-6 | [analysis log](../artifacts/phase2a/logs/02_batch_analysis.log), [measurements](../artifacts/phase2a/batch_independence_analysis.json) |
| 5 | Shapes/sharing, exploratory relaxed T12 threshold | 8 passed; not approved acceptance | [02_shapes_retry.log](../artifacts/phase2a/logs/02_shapes_retry.log) |
| 6 | Backward and gradient flow | 4 passed | [03_backward.log](../artifacts/phase2a/logs/03_backward.log), [JUnit](../artifacts/phase2a/03_backward.xml) |
| 7 | Batch-2 full-topology debug | Three finite forward/backward checks; zero updates; 8.103 s measured interval | [04_debug.log](../artifacts/phase2a/logs/04_debug.log), [summary](../artifacts/phase2a/debug/summary.json) |
| 8 | Initialization/serialization/configuration | 3 passed | [05_serialization_config.log](../artifacts/phase2a/logs/05_serialization_config.log), [JUnit](../artifacts/phase2a/05_serialization_config.xml) |
| 9 | Exploratory full suite with relaxed T12 | 15 passed; superseded for acceptance | [06_full_suite.log](../artifacts/phase2a/logs/06_full_suite.log) |
| 10 | Original final full suite, pre-N1 thresholds restored | **14 passed, 1 failed (T12)**; pytest 12.55 s | [07_approved_suite.log](../artifacts/phase2a/logs/07_approved_suite.log), [JUnit](../artifacts/phase2a/07_approved_suite.xml) |
| 11 | N1 conditional-approval numerical investigation | Original difference reproduced; fixed-batch checks exact; cross-sample probe gradients zero; isolated FP64 difference zero | [review log](../artifacts/phase2a/n1_review/numerical_review.log), [evidence](../artifacts/phase2a/n1_review/numerical_evidence.json) |
| 12 | Complete suite with approved, scoped N1 | **15 passed, zero failures/errors/skips**; pytest 12.65 s | [final log](../artifacts/phase2a/n1_review/final_pytest.log), [JUnit](../artifacts/phase2a/n1_review/final_pytest.xml) |

Diagnostic intervals exclude model construction and imports and include tracing overhead. Pytest stdout time differs from JUnit suite time, which also includes collection. Logs have been normalized to UTF-8 without altering their text. The exploratory passing runs remain visible so their changed criterion cannot be mistaken for strict final acceptance.

## All 14 unit-test IDs

Source: [test_mve.py](../tests/test_mve.py); deterministic fixtures: [conftest.py](../tests/conftest.py). Final status refers to the approved-N1 run. All original conditions remain, except the explicitly scoped T12 threshold.

| ID | Verification | Final result |
|---|---|---|
| MVE-T01 | All convolution/pool/skip/bottleneck shapes; invalid inputs; batch 1 and 2 | PASS, two cases |
| MVE-T02 | Independent coordinate-tagged flatten, affine projection, appended CLS and positions | PASS |
| MVE-T03 | Independent float64 attention/normalization/erf-GELU reference; all outputs and gradients; 1e-10/1e-8 tolerances | PASS |
| MVE-T04 | Image-dependent, graph-connected selected CLS | PASS |
| MVE-T05 | Unique sharing/serialization/optimizer identities, counts, absent normalization/dropout/decoder | PASS |
| MVE-T06 | Exact absolute residual and additive fusion; zero-residual behavior | PASS |
| MVE-T07 | Cached/recomputed parity and exactly one shared call per role | PASS |
| MVE-T08 | Isolated differentiable reconstruction path to local features/encoder; detached control removes path | PASS |
| MVE-T09 | Abs derivative sign and zero subgradient; unbounded reconstruction without clipping | PASS |
| MVE-T10 | Combined gradient equals separate-role sum; one shared-weight update verified/restored | PASS |
| MVE-T11 | Exact initialization replay, constants/distributions and caller RNG preservation | PASS |
| MVE-T12 | Train/eval parity at general thresholds; exact fixed-batch independence; CPU FP32 cross-batch-size comparison at N1 thresholds | **PASS after N1; original failure retained** |
| MVE-T13 | Actual state-dict round trip with exact output/sharing; malformed records/reconstructions rejected | PASS |
| MVE-T14 | Strict overlays, null/list semantics, schema errors, cycles, drift and independent approval checks | PASS |

T03 keeps width/head/FFN sizes intact while using a three-token analytical fixture. T08 does not demand every ReLU weight have a nonzero gradient: the isolated local gradient norm was 4.3403477e-5 and disappears in the detached comparison. Serialization uses temporary test checkpoints; no production checkpoint or training state is claimed.

## Forward and backward traces

The smoke [tensor_trace.json](../artifacts/phase2a/smoke/tensor_trace.json) records actual layer forward shapes. The batch-2 [tensor_trace.json](../artifacts/phase2a/debug/tensor_trace.json) contains 612 actual forward/backward hook events across the three iterations. Each iteration reported finite gradients for all 88 parameter tensors.

| Tensor | Batch-2 forward shape | Observed gradient shape where retained |
|---|---|---|
| Input | [2,3,224,224] | [2,3,224,224] |
| Skip0 | [2,64,224,224] | Layer-hook trace |
| Skip1 | [2,128,112,112] | Layer-hook trace |
| Skip2 | [2,256,56,56] | Layer-hook trace |
| Skip3 | [2,512,28,28] | Layer-hook trace |
| Appearance local | [2,1024,14,14] | [2,1024,14,14] |
| Appearance tokens | [2,197,512] | [2,197,512] |
| Appearance global | [2,1,512] | Connected through tokens/fusion |
| External reconstruction | [2,3,224,224] | [2,3,224,224] |
| Absolute residual | [2,3,224,224] | [2,3,224,224] |
| Residual tokens | [2,197,512] | [2,197,512] |
| Residual global | [2,1,512] | Connected through tokens/fusion |
| Fused global | [2,1,512] | [2,1,512] |

Debug uses independent synthetic Uniform[0,1] inputs, external reconstruction `0.75 * image + 0.1` and the non-paper probe `mean(fused_global ** 2)`. Recorded scalar values were 16.6383953, 16.6472225 and 16.6356277; first-convolution gradient norms were 3.1014628, 3.1761124 and 3.1428616. This proves synthetic graph execution, not training quality or decoder correctness.

## Failures, resolutions and unresolved risks

1. **Resolved configuration defect:** the initial smoke rejected `training_source_generator: null` because its schema was string-only. Debug explicitly clears that unused dataset field. The schema now allows string or null for that field, with negative overlay tests retained. No MVE topology or dataset loader changed.
2. **Original acceptance failure, resolved by approved N1:** T12 had one mismatching component out of 512 when comparing batch sizes 1 and 2 under atol=1e-6, rtol=1e-5. At that component absolute difference was 2.0265579223632812e-6 and relative difference 7.265454041771591e-5. Maximum absolute difference over all components was 2.86102294921875e-6. Changing the other sample while keeping batch size 2 produced exactly zero difference. The original failure artifacts are unchanged. The computation still has this numerical difference; the approved acceptance threshold now accounts for it.
3. **N1 history:** initially PROPOSED and unapproved, investigated at atol=5e-6/rtol=1e-5, then removed from the shipped test pending review. The user's subsequent conditional approval required verifying the numerical explanation and retaining exact fixed-batch independence. That evidence is now recorded, N1 is APPROVED within that scope, and the full suite passes. Earlier exploratory passing runs were never retroactively relabeled as approved runs.

CPU FP32 execution is verified; CUDA, mixed precision, batch 8, real image preprocessing, decoder integration and benchmarks are unverified. These do not authorize reducing the topology. Cached-record provenance remains a caller responsibility. The author's exact CNN/Transformer internals remain unknown. Cross-version initialization or CPU/GPU numerical identity is not guaranteed. No mathematical convention or published equation was revised to resolve a test.

## Reproduction commands

Run from the project root in PowerShell with the recorded dependencies installed. The current full suite applies only the approved N1 amendment and passes on the recorded runtime.

```powershell
python -m pytest -vv -s
python -m pytest -vv -s -m shapes
python -m pytest -vv -s -m backward
python -m pytest -vv -s -m serialization
python tools/review_phase2a_n1.py --output-dir artifacts/recheck/n1

$env:PYTHONPATH = Join-Path (Get-Location) 'src'
python -m mfvlr.diagnostics --config configs/debug.yaml --mode smoke --output-dir artifacts/recheck/smoke
python -m mfvlr.diagnostics --config configs/debug.yaml --mode debug --output-dir artifacts/recheck/debug
```

Using `artifacts/recheck` preserves the original run history. `pytest` reads the `src` path from [pyproject.toml](../pyproject.toml). Packaging includes the contract schema resource. No downloads or external MFVLR implementation are needed for these synthetic tests.

## Deliverables and stopping point

Created: `src/mfvlr/{__init__.py,mve.py,config.py,contract_snapshot.json,diagnostics.py}`, `tests/{conftest.py,test_mve.py}`, `pyproject.toml`, `.gitignore`, `README.md`, this report, and `artifacts/phase2a` logs/JUnit/counts/traces/manifest. Modified: `docs/implementation_contract.md`, `configs/paper_reference.yaml`, `configs/debug.yaml`. Phase 1 research documents are preserved.

[Original manifest](../artifacts/phase2a/manifest.json) preserves the pre-N1 hashes and failed acceptance state. The [N1 acceptance manifest](../artifacts/phase2a/n1_review/manifest.json) supersedes that state for current sources/configs and records final outcomes. Phase 2A stops after this acceptance verification. No next-component implementation is authorized or started.

## Final acceptance after N1 review

**Acceptance satisfied under M1–M5 plus N1 for the recorded CPU FP32 runtime.** The user's conditional N1 approval is exercised only after the following evidence from [review_phase2a_n1.py](../tools/review_phase2a_n1.py):

| Check | Actual result |
|---|---|
| Original b=1 versus b=2 fused output difference | Reproduced: max absolute 2.86102294921875e-6; original threshold fails, N1 passes |
| Peer image changed to zeros, ones or independent random values at fixed b=2 | First sample exactly unchanged in all three fixtures |
| Only the other sample's reconstruction changed to 2.0 | First sample exactly unchanged |
| Batch row permutation | Output rows permute exactly |
| Sample-0 scalar gradient to sample-1 image and reconstruction | Exactly zero elements in both gradients |
| Same scalar's own-image / own-reconstruction gradient norms | 1.04717934 / 1.04008603, finite and nonzero |
| First visual block replay, appearance role | Identical inputs; FP32 max difference 1.16229057e-6; isolated FP64 difference 0 |
| First visual block replay, residual role | Identical inputs; FP32 max difference 1.01327896e-6; isolated FP64 difference 0 |

Source inspection confirms that convolutions/pooling act per sample, LayerNorm normalizes the final feature axis, attention operates on `[batch, heads, tokens, head_dim]` with softmax over keys, and fusion/residual operations are elementwise. No operation contracts or normalizes over the batch axis. Together with the controlled fixtures and precision replay, this supports batch-shape-dependent floating-point computation as the cause, with no cross-sample information leakage in the inspected implementation. Precision replay converts only a copied diagnostic block; the actual MVE and acceptance suite stay FP32.

The amendment is exactly `atol=5e-6, rtol=1e-5` for T12's CPU FP32 b=1 versus b=2 comparison. All other general FP32/analytical FP64 tolerances and original test conditions remain. Exact fixed-batch independence remains `atol=rtol=0`. No skip, xfail, test deselection or expected-failure override is used.

The complete final run reports **15 passed in 12.65 s**, covering T01[b=1], T01[b=2], and T02–T14. T03/T08/T09/T10 validate backward mathematics and gradient paths; T05/T07 validate shared ownership and call reuse; T11/T13/T14 validate initialization, serialization and strict configuration consistency. New T14 negative cases reject broadening N1, missing independent approval, or changing general tolerances. Unique parameters remain **32,078,912**. `src/mfvlr/mve.py` is byte-for-byte unchanged from the original implementation manifest (SHA-256 `67adf0c1684427d727bd4c4673df00a165f6259fbcba05a917eaf5cd1ed48a6e`). Parameter identities remain unchanged during the review.

The [review package ZIP](../artifacts/phase2a/phase2a_review_package.zip) contains actual source/tests, authored and resolved configs, approval-chain exports, final pytest/JUnit logs, the measured parameter report, numerical evidence, original failure history, paper/documentation, environment versions and a SHA-256 file manifest. [Build script](../tools/build_phase2a_review_package.py) reproduces the package from the recorded files; [package instructions](../artifacts/phase2a/n1_review/REVIEW_PACKAGE.md) explain rerunning tests and diagnostics after extraction. The source archive uses fixed entry timestamps and sorted paths.

No acceptance failures remain on the tested runtime. CUDA, mixed precision, batch 8 and downstream integration remain unverified, as before. No Vision Decoder, FLT, VIM, training or evaluation implementation was added. Phase 2A acceptance verification is complete and work stops here.
