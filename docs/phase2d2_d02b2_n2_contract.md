# D02B2: variant-specific full-language N2 preparation contract

2026-10-16, Asia/Saigon. **Implementation and static/mock verification only. Real numerical N2 execution is NOT RUN. Phase 2D.2 remains BLOCKED.** D02B1 has conditional document-level review; its independent source/ZIP audit remains outstanding. This work neither closes that audit nor enables B0-D03/B0-D08.

## Authority and scientific baseline

The user authorizes isolated N2 preparation for the frozen A0/P3/Q3 research variants. These remain engineering assumptions, not recovered MFVLR author details. Accepted production modules, objectives, original tests, initialization and numerical policies are unchanged. The original disabled research launcher and D02B1 dispatcher are preserved byte-for-byte.

The implementation is `src/mfvlr_research/d02b2/`: `contract.py`, `adapter.py`, `verification.py`, and `frozen_inputs.json`. It uses standard-library utilities and NumPy for records, small arrays and static checks. Imports do not load torch or construct models. There is no CLI/YAML numerical execution option.

`FrozenN2Adapter` binds each variant to the exact pinned `mfvlr_research.decoder_a11.build_variant` source. Its `build`, `promoted_copy`, `execute`, and top-level `dispatch` boundaries unconditionally raise before imports or callbacks. The dormant builder call and deep-copy/promotion call document the future binding and are checked statically; **they were not exercised against a live model**. Actual construction requires a separately approved, reviewed gate change. There is no unlocked private model factory. Metadata preparation is not execution authorization.

## Original definitions and semantics

| Definition | Frozen source and significance |
| --- | --- |
| F0/F1/F2 recipes | `tools/review_phase2c2_n2.py:fixture` |
| Accepted unit-test input | `tests/language/conftest.py:language_inputs`; **F0 only**, not all three fixtures |
| N2 accepted assertions | `tests/language/test_flt.py:test_flt_t17_*`; five T17 cases after the scoped amendment |
| Full FP32 backward positive controls | `test_flt_t16_full_backward`, parameterized B1/B2 on F0 |
| Capture, statistics, comparison | Original helper `Capture`, `forward_capture`, `compare` |
| Genuine lookup derivatives | Original `cross_sample_gradients` differentiates `output.low`, an actual embedding output, plus the visual leaf |
| Historical complete review workers | Original `worker`; F0/F1/F2 × FP32/FP64 at four threads; one-thread F0 is a separate historical diagnostic |
| Policy authority | [N2 acceptance report](phase2c2_n2_acceptance_report.md), original scoped approval, and current user instructions; the literal old “PROPOSED” helper comment is historical, not current policy |
| Variant construction | Frozen A1.1 `build_variant` / `parity_report`; [corrected preregistration](phase2d2_a1_ablation_preregistration_v1_1.md) |

Input IDs remain int64 `[2,308]`, four independent 77-token segments, with original BOS/EOS/padding and token order. No re-tokenization, replacement prompt, truncation, padding mask, or pretrained weights are introduced. Visual inputs are the original FP32 `[2,1,512]` values. B1 cases are **both** exact row slices `[0:1]` and `[1:2]` of the same B2 fixture; the second sample is not replaced by a repeat of the first.

F0/F1 normals are the recorded torch FP32 draws (seeds 341/642, multiplied by 0.1), not NumPy approximations of torch RNG. F2 is the recorded accepted MVE+VD fused visual from image seed 412. **No visual forward is regenerated in D02B2.** The preparation transcribes existing literal JSON values, checks their raw-byte hashes, and returns read-only NumPy arrays for static fixture checks. Future approved FLT-only verification can load those exact values without rerunning MVE/VD; any fresh F2 regeneration would require its own authorization.

All variants start with the same seed-0 CPU FP32 FLT reference. The frozen builder privately deep-copies all common parameters, preserving names/shapes/dtypes/bytes and disjoint ownership; P3/Q3 add only constant-initialized LN parameters. FP64 requires **a deep copy of that candidate's exact FP32 initialization followed by promotion**, never an independent FP64 initializer and never promotion of an already modified candidate. Visual values are promoted from the same recorded FP32 values.

The original A0 N2 FP32 state SHA-256 is `e4233731564ba462fc32423bb4832e996f96331d2bf21509addad8bd0cffaed2`. Its algorithm is ordered `name.encode()`, then `str(tuple(shape)).encode()`, then contiguous CPU parameter bytes. It is **not** the shape-free D02B1/G2 hash or A1.1's serialized metadata/per-tensor-digest hash. The array-only streaming implementation tests this exact convention. P3/Q3 full N2 state digests are **NOT MATERIALIZED**; mock hashes are explicitly test values, not initial-state evidence.

## Architecture and ownership contract

| Variant | FLT tensors / unique elements | LE / LD blocks | LN modules |
| --- | ---: | --- | --- |
| A0 | 387 / 112,758,272 | 12 / 7 | 12 LE; 0 LD |
| P3 | 429 / 112,779,776 | 12 / 7 | 12 LE; 21 LD |
| Q3 | 429 / 112,779,776 | 12 / 7 | 12 LE; 21 LD |

These are static metadata/arithmetic checks against historical A1.1 ownership reports, not a new model recount. The tied vocabulary matrix is `[49408,512]`, one registered owner, and the output projection must use the exact lookup Parameter identity. Every parameter and storage owner is distinct where the contract requires it. There are 19 independent VIM roles and 38 bias-free Q/K parameter tensors subject to allocated exact-zero task gradients.

LE preserves one pre-self-attention LN, attention residual, VIM delta residual, FFN residual. LD ordinary core remains:

- A0: `a=x+S(x); b=a+C(a,M); c=b+F(b); y=c+VIM(c,V)`.
- P3: `a=x+S(N1(x)); b=a+C(N2(a),M); c=b+F(N3(b)); y=c+VIM(c,V)`.
- Q3: `a=N1(x+S(x)); b=N2(a+C(a,M)); c=N3(b+F(b)); y=c+VIM(c,V)`.

All added norms use final width 512, population variance, epsilon `1e-5`, independent gamma=1/beta=0. No normalization wraps VIM or its residual. Static AST checks compare the frozen P3/Q3 expressions with independently written expected expressions; source hashes bind the builder and production blocks. Live hook ordering and actual model ownership remain future execution checks. Matching counts alone is insufficient.

## Mandatory observations and original numerical scope

The basic capture is exactly **27** entries: vocabulary, LE/LD inputs, 12 LE outputs, seven LD outputs, and five final fields (`low`, `encoder_hidden`, `global_language`, `decoder_hidden`, `logits`). F0 review workers additionally capture 12 first-LE primitives, **39** entries total. The adapter preserves both distinctions. In F0 detail, VIM K/V are `[B,1,512]`, not `[B,308,512]`; FFN first projection/GELU are `[B,308,2048]`. Full logits remain `[B,308,49408]`, and global language remains physical slot 307 with shape `[B,1,512]`.

No-backward captures are detached, finite, and must not retain graph ownership. Required stage omission, wrong shape/dtype or retained graph is a hard error. Future actual hooks must be removed in `finally`, following the frozen helper semantics; no live hooks were installed now.

Fixed-batch independence changes the selected sample's token at position 250 to 1234 and adds 0.5 to its visual input while keeping B=2. Both changed-sample directions retain exact unaffected-sample equality. The accepted standalone T17 test specifically changes sample 1 and tests sample 0; the review worker covers both directions. Permutation is `[1,0]` with inverse indexing of outputs and exact equality. No batch-axis attention or mixing is introduced.

The bidirectional derivative probe is `mean(logits[s]^2)+mean(LD_hidden[s]^2)+mean(LE_global[s]^2)`. Differentiate the genuine `output.low` lookup and connected visual leaf; require both other-row gradients exactly zero and finite, nonzero own-row gradients. Detached lookup tensors, integer token IDs, fabricated leaves, missing directions or absent positive controls cannot establish isolation. The legacy fixed-batch hidden-only derivative control remains separate; it is not silently replaced by the richer review probe.

The full backward review probe is `mean(logits^2)+mean(LD_hidden^2)+mean(LE_global^2)`. All 387/429 parameters must have finite allocated gradients; all 38 VIM Q/K gradients must be exactly zero. The seven original T16 positive controls remain positive requirements. Ordinary Q/K weights (52 tensors) may be weak and are preserved; ordinary key-bias tensors (26) retain residuals against analytical zero rather than being forced to structural zero. The new complete-record validator distinguishes absent gradients from allocated zero.

| Category | Original criterion retained | Treatment |
| --- | --- | --- |
| FP32 B2-versus-B1 stages | `atol=1e-6`, `rtol=1e-5` | Mandatory complete diagnostic in the same N2 category; each outside-tolerance comparison remains numerical **FAIL**, not relabeled PASS |
| Promoted-weight FP64 B2-versus-B1 | `atol=1e-10`, `rtol=1e-8` | Mandatory agreement; failure blocks dependent execution |
| Fixed-batch equality / permutation | Exact | Mandatory |
| Both cross-sample derivative directions | Exact zero, connected positive controls | Mandatory |
| Connected singleton VIM Q/K | Allocated exact zero | Mandatory |
| Finiteness, ownership, hashes, complete captures | Exact structural/provenance requirements | Hard stop on failure |

Comparison records retain every stage/sample, outside count/proportion, maximum absolute/nonzero-reference-relative errors, RMS, relative L2 and zero-reference exceptions. No denominator floor is inserted. FP32 diagnostic completeness is not evidence of harmless drift. No exception is transferred to FP64, gradient additivity, ordinary derivative accuracy, losses, MVE or a new test category. N1 is never used here.

## Implementation limits and review status

The verifier accepts closed supplied records and preserves discrepancy classifications. It cannot prove that observations occurred merely because metadata validates; real input/state/capture observations require a separately authorized run. Exact FP64 promotion uses chunked comparison against the original FP32 values cast to FP64, as well as hashes; round-trip FP32 equality alone is deliberately insufficient.

Preparation uses 53 pinned source/config/test/protocol/historical files and a pinned frozen-input bundle. Future requests must name the exact fixture/variant/precision/backend, complete source and input hashes, original reference hash, actual candidate state hash and original hash algorithm. Unknown fields, missing hashes, altered cohorts, source changes and unsealed records fail closed. The live backend is **not measured** by mock metadata checks.

No fixture/model numerical result is generated in this phase. The original 75-case static/mock run and later expanded runs remain preserved. Final counts, command, JUnit, source hashes and exit code are in `artifacts/phase2d2_d02b2/verification.json`. See the [fixture/oracle matrix](phase2d2_d02b2_fixture_and_oracle_matrix.md) and [execution gates](phase2d2_d02b2_execution_gates.md). Stop for review before any real N2 run.
