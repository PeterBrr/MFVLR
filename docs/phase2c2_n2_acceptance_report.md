# Phase 2C.2 N2 acceptance verification

Date: 2026-10-11. N2 verification is complete: **164/164 mandatory cases pass**, with zero failures, errors, skips, xfails, xpasses or deselections. All **119 accepted regression cases** remain present, byte-preserved and passing. Final user acceptance remains pending this review. No subsequent component has started.

## Approval and exact change

The user approved only FLT-T17's universal FP32 cross-batch equality policy. The exact reviewed patch was applied to `tests/language/test_flt.py` after verifying baseline SHA-256 `484efd03bb95725e353c7c2257147f8ee2fe8914d1d5d98abcbf9f2a99c8e6d8`. New SHA-256: `e2d91f5431e64dc7754983bf473294916837ddcd13bd37ade1a8d4d7e9c40184`. The reconstructed unified diff equals the approved patch exactly. Its literal historical “N2 PROPOSED” comment is retained; the [scoped approval record](../artifacts/phase2c2_n2_acceptance/approval.json) supersedes that comment. No architecture or full-model configuration approval was added.

The old failing test is archived verbatim at [test_flt_before_n2.py.txt](../artifacts/phase2c2_n2_acceptance/history/test_flt_before_n2.py.txt). Original failed-run logs, numerical review, proposal, manifests and ZIPs remain unchanged. 281 pre-existing files match their captured hashes; only the authorized test file changed. All 165 Phase 2C.1 manifest payloads match. Integration tests write historical log paths: originals were archived before execution, fresh results copied to the new evidence directory, and original bytes restored. [Preservation audit](../artifacts/phase2c2_n2_acceptance/preservation.json).

## Before and after acceptance policy

| Requirement | Before N2 | After N2 |
|---|---|---|
| Complete suite | 160 passed / 161, T17 failed | 164 passed / 164 |
| FP32 cross-batch allclose | Mandatory; failed | Mandatory complete diagnostic; allclose is recorded, not asserted |
| FP32 diagnostic constants | atol=1e-6, rtol=1e-5 | Unchanged |
| Fixed-batch independence | Exact equality and zero cross-sample visual gradient | Preserved unchanged |
| Batch permutation | Numerical review evidence | Mandatory exact equality for all 27 stages/outputs |
| Cross-sample gradients | Numerical review evidence | Mandatory exact zero in both directions for visual and differentiable lookup tensors |
| Full promoted-weight FP64 | Numerical review evidence | Mandatory deep copy, all stages and both samples; atol=1e-10, rtol=1e-8 |
| Finiteness, full dimensions, ownership and ties | Mandatory | Mandatory, unchanged |
| MVE N1 | MVE-only | MVE-only; never applied to FLT |

The targeted T17 run passed 5/5 cases (JUnit duration 27.70s; pytest stdout 26.53s). The complete suite passed (JUnit duration 126.89s; pytest stdout 125.70s; subprocess wall time 130.11s). These timers cover different execution boundaries. All VIM and FLT tests remain active. Full FP32 forward and backward at batches 1 and 2, serialization, configuration consistency, reconstruction-gradient integration, parameter counts and ownership passed. Exact commands and subprocess durations are in [execution records](../artifacts/phase2c2_n2_acceptance/logs/complete.execution.json); raw [stdout](../artifacts/phase2c2_n2_acceptance/logs/complete.stdout.log), [stderr](../artifacts/phase2c2_n2_acceptance/logs/complete.stderr.log) and [JUnit XML](../artifacts/phase2c2_n2_acceptance/junit/complete.xml) are preserved. No test threshold was raised. The targeted run's 24 deselections are limited to that preliminary run; the complete run has none.

## Diagnostic helper audit

The unchanged `tools/review_phase2c2_n2.py` was inspected and exercised, including negative controls; [audit evidence](../artifacts/phase2c2_n2_acceptance/helper_audit.json) and stdout/stderr are included.

- `forward_capture`: exactly 27 required captures: lookup, LE/LD inputs, 12 LE blocks, seven LD blocks and all five output fields. Shapes and finiteness are checked. A full forward runs under `torch.no_grad()`; stored tensors have no `grad_fn` or `requires_grad`. Hooks are removed. F0 detailed workers additionally capture 12 first-block primitives (39 total).
- `compare`: asserts shape/dtype equality; rejects NaN/Inf; visits all elements in bounded chunks. Original-dtype `torch.isclose` supplies mismatch counts. FP64 accumulation supplies max absolute error, relative errors without epsilon clamping, zero-reference exceptions, RMS and relative L2. Outside proportions are explicitly added in the comparison JSON/CSV. Known analytical values and shape/dtype/nonfinite rejection controls passed.
- `cross_sample_gradients`: `output.low` is the actual differentiable vocabulary lookup (verified `EmbeddingBackward0`), not detached IDs or a fabricated leaf. Both sample directions are tested. Own lookup/visual gradients are finite and nonzero, other-sample gradients exactly zero. Graph retention is only needed between the two backward probes; no parameter gradients are accumulated by this helper.
- `backend`: actual CPU, thread counts, deterministic flags, dtype policy, Torch commit/build, MKL/MKLDNN and environment are recorded and checked against runtime.
- `write`: strict JSON (`allow_nan=False`) round-trips exactly and rejects NaN. No required model output is silently omitted: missing keys/fields, wrong shapes, nonfinite tensors or broken identities fail mandatory checks.
- FP64 pytest deep-copies the live FP32 model and asserts every promoted parameter equals its original FP32 value; no independent FP64 initialization. All captured outputs for both samples pass unchanged FP64 tolerances. The six numerical workers instead promote identically hashed seed-0 FP32 values, matching the earlier review protocol. Their immutable historical `PROPOSED` metadata describes that review routine; current policy is recorded separately in [diagnostic provenance](../artifacts/phase2c2_n2_acceptance/diagnostics/provenance.json).

## Fresh numerical evidence against F0/F1/F2

Six full diagnostic workers ran **after** the complete suite, one FP32/FP64 pair per fixture. Every scientific measurement in each fresh worker matches the corresponding preserved four-thread reference exactly, including all per-stage differences, activation statistics, gradient norms, perturbation results and backend metadata. Only elapsed times differ. Both pytest diagnostic JSONs also match the fresh F0 worker for all 27 stages. No changed numerical evidence requires escalation.

Fixture provenance: F0 uses golden prompt indices 1/5 and FP32 normal visual seed 341, std 0.1; F1 uses indices 0/4 and seed 642, std 0.1; F2 uses indices 8/9 and the accepted untrained MVE+VD fused visual feature from random image seed 412. All use the same seed-0 FLT FP32 state hash `e4233731564ba462fc32423bb4832e996f96331d2bf21509addad8bd0cffaed2`. Literal input IDs, visual values and hashes are preserved per worker. FP64 promotes those identical FP32 fixture values. Backend: Windows 11 AMD64, Python 3.14.3, Torch 2.12.1+cpu, commit `7269437d655783a26cba32aa88195b741ff496aa`, intra/inter-op threads 4/4, deterministic algorithms enabled, highest FP32 matmul precision, MKL 2026.0 / MKLDNN 3.11.2. Exact build strings are in every worker JSON. GPU **NOT RUN** (CPU-only Torch); batch 8 **NOT RUN** in this acceptance run.

Per-stage tables for all LE/LD blocks, embeddings, global feature and vocabulary outputs for **both samples and all fixtures**: [stage_comparison.md](../artifacts/phase2c2_n2_acceptance/stage_comparison.md). Complete FP32/FP64 max absolute/relative differences, outside counts/proportions, relative L2 and zero-reference handling: [numerical_comparison.json](../artifacts/phase2c2_n2_acceptance/numerical_comparison.json) and [CSV](../artifacts/phase2c2_n2_acceptance/stage_differences.csv).

Each logit comparison contains 15,217,664 elements:

| Fixture / sample | FP32 max absolute difference | Outside unchanged tolerance | Relative L2 |
|---|---:|---:|---:|
| F0 / 0 | 0.003082275391 | 3,170,109 (20.8318%) | 3.43608432e-06 |
| F0 / 1 | 0.005348205566 | 4,319,218 (28.3829%) | 6.280215364e-06 |
| F1 / 0 | 0.00150680542 | 2,309,597 (15.1771%) | 2.443390605e-06 |
| F1 / 1 | 0.003684997559 | 3,673,853 (24.1420%) | 4.119867512e-06 |
| F2 / 0 | 0.1786499023 | 8,494,886 (55.8225%) | 2.693932745e-05 |
| F2 / 1 | 0.1029663086 | 5,772,796 (37.9348%) | 1.270704876e-05 |

The embedding lookup and initial position sums remain exact across batch sizes. Earlier evidence of first-block singleton VIM K/V projection rounding is reproduced; divergence accumulates through subsequent LE/LD blocks. Vocabulary projection with an identical hidden tensor remains exact in the isolated batch-shape control, so observed logit drift includes upstream hidden-state drift. Code inspection finds no feature-producing cross-sample reduction or attention across samples. Input-validity and finiteness checks do reduce across the input to accept or reject a call; they do not mix feature values. Exact independence, permutation and zero cross-sample gradients confirm isolation on these valid fixtures. These findings support a batch-shape-dependent FP32 computation explanation, not sample leakage. They do not establish universal numerical safety.

## Stability and parameter preservation

| Fixture | Final LE max abs | Final LD max abs | Logit max abs | All-parameter gradient L2 | Visual gradient L2 |
|---|---:|---:|---:|---:|---:|
| F0 | 27.931324 | 865.81781 | 527.739868 | 3031726.345 | 40303.88947 |
| F1 | 32.985199 | 980.050354 | 520.92627 | 3138625.593 | 43866.78235 |
| F2 | 135.378204 | 4411.21289 | 2645.78101 | 62397308.91 | 114888.56 |

All 387 FLT parameter tensors receive finite gradients for each controlled full-graph probe. All 38 VIM Q/K projection gradients are allocated and exactly zero, as required by singleton attention. Stage/parameter gradient norms, activation norms/maxima and both-direction cross-sample probes are in [parameter_gradient_preservation.json](../artifacts/phase2c2_n2_acceptance/parameter_gradient_preservation.json) and [activation_traces.json](../artifacts/phase2c2_n2_acceptance/activation_traces.json). No activation maximum or gradient norm changed from preserved evidence. No nonfinite activation or gradient was observed in valid full-graph probes; deliberate NaN/Inf helper rejection controls passed.

Counts remain MVE **32,078,912**, VD **9,055,365**, visual **41,134,277**, FLT **112,758,272**, combined **153,892,549**. Shared vocabulary identity, 19 independent VIM instances, unique storage and parameter objects remain intact. The approximately **12.89M** excess over the paper's approximately 141M is still an architectural reproduction uncertainty; dimensions were not adjusted.

The no-LayerNorm LD baseline still amplifies perturbations. Preserved and fresh FP64 fixed-batch probes give final LD RMS gains about 46.79 (F0), 36.53 (F1), and 804.53–840.84 (F2), with memory/visual inputs fixed. This measures the implemented LD's sensitivity; no normalization counterfactual was run, so it does not isolate normalization as the sole cause. FP32 F2 logit drift reaches 0.1786499 and approximately 55.82% outside the old elementwise tolerance for sample 0. Diagnostic success is **not proof that this drift is harmless**. Training, task accuracy, optimization stability and generalization were not tested or authorized. No normalization, clipping, AMP, scaling, attention change or architecture adjustment was introduced.

## Reproduction and review package

See [REPRODUCE.md](../artifacts/phase2c2_n2_acceptance/REPRODUCE.md), [SHA-256 manifest](../artifacts/phase2c2_n2_acceptance/manifest.json), and [review ZIP](../artifacts/phase2c2_n2_acceptance/phase2c2_n2_review_package.zip). The package includes actual production/test source, all scoped configurations, tokenizer artifacts/dependency pins, preserved historical failing evidence, the exact patch and archived failing source, fresh raw logs/JUnit, helper audit and full diagnostic evidence. No pretrained weights are included. Existing source/configuration/tests outside this one approved patch were preserved. Verification is complete; final acceptance awaits the user's review.
