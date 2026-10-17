# Phase 2D.2-A1.1 implementation and unit verification

**Research correctness: PASS, 54/54 cases. Historical regression coverage: 276/276 cases passed across two runs. Phase 2D.2 remains BLOCKED.** The eight existing A0 numerical gates still fail. No ablation result, production replacement, numerical-policy amendment or training-readiness acceptance is claimed.

**Execution-scope deviation requiring review:** an unchanged legacy VD-T12 test performed one SGD step and restored its parameter before the incomplete optimizer audit was discovered. This violated this phase's no-step restriction. The user subsequently approved **MVE-T10 only**; that approval does not cover VD-T12. MVE-T10 then ran unchanged and passed. Thus this phase executed two restored legacy test updates: one explicitly approved and one unapproved. There were zero optimizer steps in the new research implementation/tests and no training pipeline. See the [incident record](../artifacts/phase2d2_a11/execution_scope_deviation.json) and [sole MVE exception](../artifacts/phase2d2_a11/legacy_mve_t10_approval.json). Do not describe the entire phase as zero-update or fully scope-compliant.

## Authorization and specification correction

A1-D01 alone is approved for isolated A0/P3/Q3 implementation and module verification as **ASSUMED engineering research choices**. A1-D02, I1/A1-D03, R2/A1-D04, production replacement, optimizer/training and numerical amendments remain unapproved. The user-authorized MVE-T10 exception is narrowly recorded above.

The [erratum](phase2d2_a11_erratum.md) was published before variant source. The [A1.1 preregistration](phase2d2_a1_ablation_preregistration_v1_1.md) makes the affine norm unambiguous:

`N(x) = gamma * (x - mean(x)) / sqrt(variance(x) + 1e-5) + beta`

It uses final-axis 512-feature population statistics, independent affine parameters, gamma=1 and beta=0. Beta is outside the division/square root. The original locked A1.0 remains byte-identical, SHA-256 `0989e19fff9053d813a3f65f547ff74a09a412552b170d3ef484d27f90d87230`. Corrected A1.1 SHA-256 is `3577e592a3a4013d136c840093d1d843ac92481b2f4088e4caf66fed7b3f92a3`. The version notice and formula clarification are the only changes; the experimental factors, fixtures and criteria remain intact. Historical proposed-status wording in the retained schedule is not an execution authorization.

Paper Section III-D, p.7, Eqs.(12)–(16) specifies the reference LD relationships but does not establish the added P3/Q3 norms as author architecture. Figure 4's incomplete depiction cannot recover that detail. See the preserved [A1 evidence audit](phase2d2_a1_paper_evidence_audit.md). Parameter totals below are **DERIVED**, not fidelity evidence.

## Isolated implementation and API

New code lives in `src/mfvlr_research/`, with separate `configs/research_a11/{A0,P3,Q3}.yaml`. No accepted production module or loader was edited.

`build_variant(accepted_reference, variant)` accepts a CPU FP32 accepted `MFVLRIntegration` or full `FineGrainedLanguageTransformer`. It deep-copies the reference, verifies every common parameter name, shape, dtype and byte hash, and rejects aliases. A0 retains the exact accepted decoder class and forward. P3/Q3 replace only the copied LD blocks, reusing the accepted ordinary attention, exact GELU FFN and singleton VIM classes:

| Variant | Four stages, in order |
|---|---|
| A0 | `a=x+S(x); b=a+C(a,M); c=b+F(b); y=c+Delta(c,V)` |
| P3 | `a=x+S(LN1(x)); b=a+C(LN2(a),M); c=b+F(LN3(b)); y=c+Delta(c,V)` |
| Q3 | `a=LN1(x+S(x)); b=LN2(a+C(a,M)); c=LN3(b+F(b)); y=c+Delta(c,V)` |

There is no fourth/final/memory/visual norm, dropout, gate, residual scaling, detached pathway or modified initializer. Ordinary attention retains its causal/full-memory roles and sqrt(64) scale; VIM retains its published sqrt(8/512) denominator and singleton softmax graph. Seven independent LD blocks and twelve original LE blocks remain. All existing FLT methods, fields, positions, global shift and tied vocabulary projection are reused unchanged. The copied full root retains accepted visual/head/objective owners and methods; it was instantiated for ownership checks, not used for a variant six-loss sweep.

“Frozen reference” means unchanged values/RNG and ownership. It does not set `requires_grad=False`: A0 and variant derivative checks require connected gradients. New norms consume no random draws. All three complete reference fingerprints equal `f75b84ca38fac8bd668776708769367764df31aac8b16553abe1952357ab3579`.

The strict research config fixes topology, norm semantics and copied initialization and rejects duplicate YAML keys, missing/unknown fields, wrong types, I1/R2 and execution-enabling changes. Checkpoints carry schema and variant identity, use exclusive file creation and strict loading. A0 state dictionaries cannot strictly populate missing P3/Q3 norms. P3/Q3 checkpoints cannot be silently interchanged just because their tensor keys match. The accepted production ownership reporter still has its original A0-only count; use the research parity report for research counts.

## Measured ownership

| Inventory | A0 | P3 | Q3 |
|---|---:|---:|---:|
| Full unique parameters | 154,156,232 | 154,177,736 | 154,177,736 |
| Full Parameter tensors | 508 | 550 | 550 |
| FLT parameters | 112,758,272 | 112,779,776 | 112,779,776 |
| FLT Parameter tensors | 387 | 429 | 429 |
| LD parameters | 36,746,752 | 36,768,256 | 36,768,256 |
| Per-block parameters | 5,249,536 | 5,252,608 | 5,252,608 |
| New norm modules / Parameter tensors | 0 / 0 | 21 / 42 | 21 / 42 |
| VIM roles / tied vocabulary owners | 19 / 1 | 19 / 1 | 19 / 1 |

Each common tensor is independently owned and byte-equal to A0; every new gamma/beta is exactly one/zero. Serialization restores the complete state and preserves the vocabulary parameter identity. The [parameter and initialization report](../artifacts/phase2d2_a11/parameter_initialization_report.json) links detailed per-name hashes. The additional 21,504 elements do not resolve the paper's approximately 141M parameter discrepancy.

## Unit coverage and actual outcomes

All independent FP64 comparisons retain `atol=1e-10, rtol=1e-8`; directional finite differences retain `eps=1e-6, atol=1e-5, rtol=1e-3`. Fixed-batch independence/permutation, cross-sample derivatives and connected singleton Q/K zeros require exact equality. No N1 tolerance was imported and no N2/N3/N3.1 criterion was changed.

| ID | Executed verification | Outcome |
|---|---|---|
| A1-T01 | Full common-state hashes, name/shape checks, disjoint objects/storage, caller RNG and reference preservation | PASS |
| A1-T02 | Independent FP64 population-variance forward and analytic input/gamma/beta VJP; constant, near-constant and ordinary fixtures; nontrivial affine parameters | PASS |
| A1-T03 | Noncommuting sentinels and observed call order; independent arithmetic forward and composed VJPs for ordinary Q/K, all added norms and state/memory/visual | PASS |
| A1-T04 | Causal self-attention, every memory-token derivative, global BOS shift, full seven-block B2 exact independence/permutation and both-direction floating-lookup/memory/visual isolation | PASS |
| A1-T05 | One controlled complete language-module VJP per variant; all 387/429 parameter gradients connected and finite; all 38 VIM Q/K allocated exact zero; all 38 V/O and visual gradients nonzero; new norms connected | PASS |
| A1-T06 | Original controlled full-width ordinary-attention FP64 manual Q/K derivatives and directional finite differences; exact rational key-bias cancellation with floating residual recorded | PASS |
| A1-T07 | Exact inventories, independent depth, tied vocabulary, strict save/load and rejected config mutations | PASS |
| A1-T08 | I1 transformation / R2 residual oracle | NOT APPLICABLE: neither variant is authorized or implemented; a separate rejection guard passed |
| A1-T09 | Known gradient vectors, zero/weak entries, chunk-boundary corruption, missing/nonfinite/shape-invalid negative controls | PASS |
| A1-T10 | Live no-grad capture of 53 required stages, all parameter metadata, actual source/config/input/state changes and tampered/incomplete record rejection; disabled launcher, fixture registry and optimizer-scope guard | PASS |

The LayerNorm oracle derives its VJP explicitly. The composed block oracle uses affine maps, explicit exp-normalization, erf GELU and the mathematical singleton delta; it never invokes the production variant as its reference. Ordinary attention manual derivatives use the unchanged independent N3 helper. Full VIM Q/K graph allocation is tested on the actual forward separately from the simplified value oracle.

Capture hooks observe the actual FLT execution, including all twelve LE outputs, seven LD outputs and four stages per LD block, input embeddings/positions, global language feature, visual input and vocabulary logits. Diagnostic captures retain no autograd graphs. Gradient tests run the normal differentiable path separately. Detailed per-parameter gradient norms and complete activation statistics are in `runs/research_complete_01/evidence/`; these are controlled correctness fixtures, not a numerical-improvement study.

| Recorded run | Cases | Result / exit code | Wall seconds including test-process startup | Sampled peak RSS GiB |
|---|---:|---|---:|---:|
| `primitives_01` | 12 | 12 PASS / 0 | 7.720 | .792 |
| `ownership_01` | 7 | 7 PASS / 0 | 36.481 | 2.139 |
| `graphs_01` | 12 | 12 PASS / 0 | 39.473 | 2.108 |
| `research_complete_01` (final source) | 54 | 54 PASS / 0 | 96.282 | 2.527 |
| `regression_no_optimizer_01` (initial intent in name; VD-T12 deviation disclosed) | 275 | 275 PASS, MVE-T10 explicitly deselected / 0 | 205.012 | 3.332 |
| `mve_t10_approved_01` | 1 | 1 PASS / 0 | 6.321 | .658 |
| `a0_eight_gates_01` | 8 | 8 genuine FAIL / 1 | 11.862 | 2.428 |

Exact historical JUnit case-identity comparison verifies the two regression runs cover all 276 accepted cases with no omissions. The final selected runs cover 338 distinct cases: 330 pass and eight fail. No skips or xfails were used. This is **not** a fresh complete rerun of the historical 328-case suite: 44 other joint/N3 cases were outside this module-only handoff. Neither that historical suite nor the expanded all-tests collection is claimed fully passing.

The fresh original A0 I08 CMC chain-rule comparison still fails. I10 and six READY-N02 cases rerun their original assertions against byte-copied historical N3.1 G2 records and still fail; their G2 gradients were **not regenerated**. [Provenance](../artifacts/phase2d2_a11/a0_gate_reassertion/provenance.json) identifies every copy. These are ordinary failing assertions with original thresholds, not expected-failure overrides.

## Preservation, resources and restricted drivers

The [preservation audit](../artifacts/phase2d2_a11/preservation_audit.json) verifies **1,246/1,246** pre-existing inventoried files unchanged, including accepted source, tests/configs, tokenizer assets/golden fixtures, locked documents and historical evidence. Four fixed-path FLT gradient records written by unchanged regression tests were archived and restored after each test process. The new mandatory N2 FP32 diagnostic is copied into this package with its original path and hash. Production initialization and numerical policies were not edited.

Executed tests used Windows CPU, Python 3.14.3, PyTorch 2.12.1+cpu, four intra-op threads, deterministic algorithms and highest FP32 matmul precision. Actual research backend metadata is in each T10 provenance record. `environment.json` also records the separate metadata collector's defaults, whose deterministic flag is false because it performs no numerical test; it must not be mistaken for the test backend. Timing/RSS samples are approximate process-tree measurements at nominal 50 ms intervals, not guaranteed peaks. System memory use briefly approached 98%; processes ran sequentially.

CPU B1 full research FLT forward/logits and backward and B2 full LD isolation were executed. Accepted A0 full FP32 B1/B2 regression paths also ran. New-variant full visual/six-loss B1/B2 studies, batch 8 and GPU capacity are **NOT RUN**. CUDA was unavailable. No capacity success, reconstruction improvement, convergence or accuracy conclusion follows from these module tests.

`tools/research_phase2d2_a11.py` is an executable **request-preparation launcher** for the locked schedule. It requires explicit experiment identity, variant, fixture and backend; resolves the preregistered fixture registry; seals source/config/spec hashes and numerical criteria; and writes exclusively beneath the new A1.1 prepared tree. Existing destinations and paths into accepted evidence are rejected. Preparation generates neither weights nor fixture tensors. Its expensive execution entry point deliberately raises `PermissionError` without a CLI/YAML bypass. A1-D02's subsequent handoff must authorize and audit numerical worker dispatch before enabling it. There is no claim that the full experimental schedule has been run or validated by these preparation tests.

The updated test runner now audits collected test identities before execution and refuses VD-T12 in this scope. It permits unchanged MVE-T10 only with the explicit approval and original file hash. This guard prevents repeating the disclosed audit error; it does not erase that error or modify either legacy test.

## Review package and next boundary

See [reproduction instructions](../artifacts/phase2d2_a11/REPRODUCE.md), [machine verification](../artifacts/phase2d2_a11/verification.json), [gradient report](../artifacts/phase2d2_a11/gradient_report.json), and the raw stdout/stderr/JUnit under `artifacts/phase2d2_a11/runs/`. The deterministic-order review ZIP includes complete source, tests, configs, tokenizer assets, new verification evidence and historical non-ZIP evidence. Its manifest, member verification and checksum accompany the archive; historical nested ZIPs remain untouched on disk and are represented by their preserved hashes.

Research implementation and unit verification are complete. Review is required for the documented VD-T12 scope deviation and A1.1 acceptance. **Phase 2D.2 remains BLOCKED.** No A1-D02 matrix, I1, R2, production replacement, stabilization, new optimizer, training, dataset or benchmark work was started.
