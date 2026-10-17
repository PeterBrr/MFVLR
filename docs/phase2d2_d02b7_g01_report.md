# D02B7 single A0 G01 attempt

2026-10-16, Asia/Saigon. **WAIT/NOT RUN: resource preflight failed. Phase 2D.2 remains BLOCKED.**

The only authorized identity was `D02B-G01-A0`, original real B1 fixture, C4M Windows CPU FP32. The coordinator checked source pins, isolated test evidence and exclusive ownership, then observed **2,092,642,304 bytes (about 1.95 GiB) available RAM**, below the required **5,368,709,120 bytes (5 GiB)**. Free disk was 663,522,033,664 bytes, above the 8 GiB reservation. Following the user's stop instruction, it recorded WAIT/NOT RUN and did not start a worker. There is no retry, waiting loop or other cell execution.

| Status axis | Actual result |
| --- | --- |
| Execution | WAIT/NOT RUN |
| Resource | INSUFFICIENT (RAM); disk gate satisfied |
| Numerical | NOT RUN |
| Structural / live ownership | NOT RUN |
| Source/protocol provenance | File integrity checks passed |
| Live input/state/backend provenance | NOT RUN; no model or fixture materialization |
| Dependent work | BLOCKED |
| B0-D08 continuation | DISABLED |

The execution record's provenance PASS refers to file/registration provenance only. It is **not** a claim that a real 508-tensor model, initialized state, fixture or runtime backend was observed. `verification_scope.json` records this distinction explicitly.

## Implemented source

The new versioned namespace is `src/mfvlr_research/d02b7`, with a separate one-attempt coordinator and worker in `tools/d02b7_run.py` and `tools/d02b7_worker.py`. Previous dispatchers, accepted models, objectives, tests and policies are unchanged. The new coordinator accepts no configurable alternate cell; the worker independently rejects every identity except G01 A0. Both enforce the resource gate before importing/constructing the full model.

The live worker is implemented but **has not been executed**. Its intended measurement path is:

1. Pin Windows CPU C4M metadata, source/protocol/config hashes, original real fixture provenance and historical integration state `1865527facba9adc8a14d443c112d492698d275ce4e8beaa784aee61a514895a` using the existing name-plus-bytes state-hash algorithm.
2. Construct the seed-0 production integration reference, privately copy A0 through the frozen A1.1 builder, verify every common parameter and disjoint ownership, then release the reference. Validate 508 trainable tensors, 154,156,232 elements, exact frozen roster and vocabulary identity.
3. Execute exactly seven fresh full graphs: FD, LR, CMC, FL, AR, KL, total. Use the unchanged production isolated-loss functions and joint objective path. No warm-up, timing repetitions, optimizer, scheduler, detached controls or additional numerical cell is included.
4. Seal graph-entry source/state/input/backend records and a second seal immediately before FLT. Labels, token/mask/image bytes and canonical JSON sample-order digest are included. Observe 63 bounded statistics stages through hooks; A0 post-residual stages are observed at actual subsequent module inputs. A temporary instance-level observer around the unchanged shift method records its returned tensor and restores ordinary method lookup on exit. It neither replaces shift arithmetic nor modifies production source.
5. Independently reconcile actual joint FP32 scalar arithmetic using detached observed outputs. LR records all per-token NLLs/IDs, BOS/EOS/pad/content counts and sums, original token-sum/batch-mean replay, and FP64 category regrouping differences. Regrouped sums are diagnostics, not replacements for the production reduction. FD/FL class semantics, B1 CMC dot/temperature/diagonal behavior, unrestricted AR range and two-sided KL inputs are recorded. Literal FP32 total is replayed in the exact six-loss order. No new loss tolerance is introduced; exact replay must agree. No observer result is used for backward.
6. Observe actual parameter gradients and Ipre/Iv/Tl/adapter gradients; distinguish None from allocated zero and weak nonzero values. Stream all seven passes through frozen D02B1 `GradientStore`: FP64 ordered accumulation of isolated FP32 gradients, original FP32 elementwise comparison at 1e-6/1e-5, all 15 loss pairs, four registered directions, singleton zeros and ordinary Q/K/key-bias records. Verify state/ownership unchanged, remove hooks and release graphs.
7. A finite additivity FAIL is retained with **execution COMPLETE, numerical FAIL, dependent_work BLOCKED**. Structural, provenance, nonfinite or incomplete evidence hard-stops. No continuation occurs regardless of numerical outcome.

## Direct verification performed

The isolated suite ran **18 tests: 18 passed, zero skipped/xfail/failed**, exit code 0. Tests cover exact cell refusal, resource floors, independent status axes, worker AST scope, a small eight-token uniform arithmetic fixture, frozen stream failure handling and actual hook-boundary/cleanup behavior on tiny mock modules. No full MFVLR module was imported by this suite. The small arithmetic fixture uses torch tensors but is not an MFVLR forward or a six-loss model result.

Raw evidence: `preflight.stdout.log`, `preflight.stderr.log`, `preflight_junit.xml`, `preflight_tests.json`, `coordinator.stdout.log`, `coordinator.stderr.log`, `coordinator_exit.json`, `resource_preflight.json` and `execution_record.json`. The single coordinator stopped normally after recording the gate failure; its exit code does not establish numerical acceptance. No worker RSS or full-model runtime measurement exists. The resource readings above are coordinator observations only.

## Deliverable limits and next review

There are **no live six-loss scalars, joint scalar, gradients, additivity comparisons or full-model activation statistics** from D02B7 because the required resource gate prevented execution. These fields remain NOT RUN rather than being populated from mock arrays or historical data. Existing eight G2 failures and all N2 diagnostic failures remain unchanged. The source is ready for independent review, but its full-graph behavior remains unverified.

Before any later attempt, the user must authorize continuation/retry after reviewing this stopped attempt, and sufficient resources must be available. No retry is scheduled. B0-D08 remains disabled; P3/Q3, G02–G06, NG2, optimizer/training, G3/G4, benchmarks, MVE-T10 and VD-T12 were not executed.

The compact package contains complete new source/tests, accepted source dependencies/configs, baseline preservation inventories, original fixture/roster references and actual logs. ZIP CRC and per-file SHA-256 verification are recorded separately. Stop for review.
