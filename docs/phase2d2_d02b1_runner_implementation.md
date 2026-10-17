# D02B1 isolated research runner implementation

2026-10-16. **B0-D02 implementation and static/mock verification only. Phase 2D.2 remains BLOCKED.** This work introduces no production architecture or numerical-policy change. The current user handoff approves the isolated infrastructure, not the G2/NG2 experiments described by its records.

## Scope and implementation boundary

New package: `src/mfvlr_research/d02b1/`. It imports standard-library utilities and NumPy, never PyTorch, production MFVLR, the accepted research decoder, or a worker process library. The existing `tools/research_phase2d2_a11.py` launcher and strict research configurations remain unchanged. There is no CLI, YAML execution flag, worker factory, or model-construction callback. `dispatch()` always raises `ExecutionNotAuthorized`; `continue_after_numerical_failure()` also always raises, regardless of arguments.

This is an **ASSUMED engineering evidence implementation** of the accepted [D02B0 contract](phase2d2_d02b0_experiment_contract.md), [matrix](phase2d2_d02b0_fixture_matrix.md), and [gates](phase2d2_d02b0_numerical_gates.md). It is not a recovered author implementation. Source code for an actual full-graph worker, live tensor-to-array observation, removable model hooks, and execution authorization must undergo a subsequent handoff. No real model was constructed to test the infrastructure.

| Module | Implemented responsibility | Boundary |
| --- | --- | --- |
| `schema.py` | Exact 57-cell registry; closed metadata validation; SHA formats/algorithms; complete historical name/shape inventory; counts, ownership, vocabulary, connectivity classes | Supplied metadata is checked against a separate trusted seal; a self-generated digest is not approval or live ownership evidence |
| `runner.py` | Exclusive metadata preparation, disabled dispatch/continuation, independent status axes, complete-output checks, no-update test selection | All scientific execution statuses remain NOT RUN |
| `safety.py` | Resource floors, PID-plus-creation identity, exclusive coordinator and cell claims, byte-preserving archive, complete JSONL prefix comparison | Mock process observations only; no process termination, stale-lock deletion, or automatic replay |
| `gradients.py` | Sequential seven-pass FP32 shards, FP64 ordered accumulation, complete parameter diagnostics and loss pairs; state/identity controls | Receives independent arrays only; no objectives, autodiff, or full graph |
| `pairing.py` | Common-state metadata parity, non-LR branch invariants, explicit sample-pair comparisons | Does not infer variant equality from total losses or aggregate norms |
| `stages.py` | Exact B1/B2 required-stage shape inventory and finite-statistic validation | Validates supplied detached statistics; no activation capture executed |

## Registration and provenance

The registry contains G01–G06 × A0/P3/Q3 (18), C01–C07 × A0/P3/Q3 (21), and N01–N06 × A0/P3/Q3 (18). Cell identities, fixture, batch, backend and partition are indivisible; arbitrary combinations are rejected. `approved_partition` selects which *metadata partition* the validator accepts. It does **not** authorize execution. All 57 entries in `artifacts/phase2d2_d02b1/coverage.json` are NOT RUN.

Required record fields: `identity`, `hashes`, `hash_algorithms`, `backend`, `seeds`, `ownership`, `vocabulary_tied`, `loss_order`. Hash fields cover source/config/protocol/driver, images/tokens/masks/labels/sample ordering, full state and inventory. Missing and unknown fields fail. Backend identity retains CPU/FP32, C4M/C4N/C1M, inter-op 4, deterministic algorithms, highest matmul, explicit attention, Python 3.14.3, torch 2.12.1+cpu and Windows. These are **future scientific cohort requirements**, not a claim that D02B1 imported torch or measured that backend.

G2 seeds remain zero; NG2 FLT seeds are 6101–6103 with visual/VD/heads zero. A seed does not substitute for an input/state hash. NG2 materialization is still NOT RUN. The validator requires an independently pinned complete-record digest. Synthetic tests explicitly use mock seals and mock IDs; those values must never be used as scientific run manifests.

Raw input hashes identify contiguous little-endian bytes; state hashes identify ordered parameter names followed by bytes. Canonical JSON inventory digests and file-byte SHA-256 have separate algorithm identifiers. The historical A1.1 `a11_evidence.state_hash` algorithm is different and is not silently compared with the G2 state digest. `verify_files` checks actual file bytes and rejects paths outside the declared root. Preparation is metadata-only; before any future execution, actual input/state/owner observations and every frozen file must independently match the reviewed seal.

Historical parameter rosters are copied as **metadata** from the three accepted A1.1 `ownership_*.json` records. Source digests accompany the copy. Common shapes are read from those records; new LayerNorm affine shapes are `[512]` under the frozen contract. No new model recount took place. Static arithmetic verifies A0 **508 tensors / 154,156,232 elements**, P3/Q3 **550 / 154,177,736**, 38 singleton VIM Q/K tensors, and one registered vocabulary owner. Object/storage identifiers must be unique; common variant tensors must have matching shape/dtype/hash and disjoint recorded owners. Future live identifiers must be observed in the same process, not fabricated from checkpoint hashes.

## Streamed gradients and independent verification

Storage sequence is exactly `fd, lr, cmc, fl, ar, kl, total`, with every parameter in the sealed inventory for each pass. Missing, repeated or reordered entries fail. Shards are exclusive, little-endian FP32, hash-checked before reading. Partial files remain after an error. None records have no invented tensor; allocated exact zero remains allocated. Expected connectivity is separate from whether a connected gradient happens to be zero. All connected singleton VIM Q/K tensors must be exactly zero; ordinary Q/K weights and ordinary key biases retain their separate categories.

One parameter is analyzed at a time using chunks of **262,144 elements**. FP64 accumulation adds isolated FP32 gradients in the literal loss order. The mandatory elementwise criterion remains FP32 actual against **FP32(FP64 ordered sum)**, `atol=1e-6, rtol=1e-5`. Error norms and directional metrics use the **unrounded** FP64 sum. This is not a promoted full-model FP64 derivative oracle. Finite failures remain `FAIL`, and no continuation is enabled.

Every parameter retains all seven presence/shape/hash/norm/RMS/max/nonzero records; outside counts/proportions; absolute and nonzero-reference relative errors; relative L2; cosine; sign disagreements; reference-zero counts and nonzero actuals on zero references; cancellation numerator/denominator/ratio; four directional projections and errors; all **15** isolated-loss pair dots/cosines with undefined reasons. No small-denominator floor or weak-gradient cutoff is introduced. Ordinary key-bias analytical-zero residuals are explicitly retained rather than forced to zero.

The four directions preserve seeds **5101–5104**, SHA-256 name salt (first four bytes, little-endian), NumPy int8 Rademacher generation with original chunk boundaries, and normalization by square root of parameter size. The D02A local 6701-series directions are not substituted. The scalar controls exercise every 508/550 parameter name and every connectivity slot, while a separate ~1-MiB vector crosses the real chunk boundary. These are synthetic arrays, not model-shaped parameter allocations.

The mathematical controls use six independently defined linear functions and analytical derivatives, unchanged finite-difference tolerances, cancellation/rounding examples, allocated zeros, unused entries, weak signed gradients, corrupted/truncated shards, invalid dtype/shape/order, injected nonfinite values, identity changes and storage aliases. They do not reuse production objectives as their oracle.

## Pairing and stage coverage

For identical fixture/backend/input/common state, compare A0/P3, A0/Q3 and P3/Q3 on FD, CMC, FL, AR and KL outputs and every common isolated gradient. LD-only parameters remain None in these isolated branches. LR and total are variant-dependent. Different NG2 seeds are not paired as invariant models. Separate same-sample B1/B2 output/derivative records must match image/prompt bytes, cotangent and reduction semantics. CMC intentionally couples the batch and is excluded from cross-batch objective equality. Mean-reduced objectives require explicit sample/cotangent accounting; no unregistered multiplication of a loss is hidden in the test.

The stage inventory covers appearance/residual local and global endpoints, fused visual token, reconstruction/residual/mask/detection outputs, adapter, vocabulary lookup, LE input/all 12 outputs/global token, shifted lookup/LD input, all five named substages of each of seven LD blocks, and vocabulary logits. P3/Q3 observations must eventually capture the actual normalized residual values. Duplicate final-output/VIM endpoint names are deliberate coverage aliases, not extra model operations. Missing stages or nonfinite statistics fail validation.

## Verification and deliverables

Fresh execution is restricted to `tests/research_d02b1/` using `--confcutdir=tests/research_d02b1` and disabled external pytest plugin autoload. A local fixture overrides the parent model-owning fixture as defense in depth; collection rejects every other test directory. Tests assert that neither torch nor production MFVLR was imported. MVE-T10 and VD-T12 were not collected or executed. No accepted regression suite is claimed to have run again.

The first run recorded **65/65 passing**, and the expanded second run **78/78 passing**; both original stdout/stderr/JUnits are retained. After the completeness and cross-batch-policy review, the **final run passed 78/78 cases, exit 0, zero failures/errors/skips/xfails**, in 54.465 seconds including process startup. Sampled pytest RSS peaked at **82,927,616 bytes (79.09 MiB)**. This measures array/mock tests only, not model capacity. Python 3.14.3, NumPy 2.4.4 and pytest 9.1.1 ran on Windows 11; neither the harness nor tests imported torch. Tested source hashes stayed unchanged during execution.

The exact final invocation, source hashes, counts and measurements are in [verification.json](../artifacts/phase2d2_d02b1/verification.json), with [raw stdout](../artifacts/phase2d2_d02b1/final_01/stdout.log), [stderr](../artifacts/phase2d2_d02b1/final_01/stderr.log), and [JUnit](../artifacts/phase2d2_d02b1/final_01/junit.xml). Earlier attempts remain separate. `baseline_before.json` and `baseline_after.json` distinguish fresh file hash verification from historical numerical evidence. The compact review package includes source, independent tests/rosters, these documents, accepted design references, raw logs, coverage, approval/status metadata and a non-self-referential SHA-256 payload manifest.

**G01–G06, C01–C07, N01–N06, real full-graph forward/backward, optimizer and training: NOT RUN. GPU/B8: NOT VERIFIED.** Successful mock tests establish infrastructure behavior on controlled arrays, not full-model numerical agreement, resource capacity, or training stability. Stop for review; B0-D03 and B0-D08 remain unauthorized.
