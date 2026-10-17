# D02B10 operational wrapper and authorization checklist

Status: static implementation and synthetic tests only. Public live dispatch always raises. No full-model same-forward execution, optimizer or benchmark is authorized. B0-D08 remains DISABLED; Phase 2D.2 remains BLOCKED.

## Binding to the frozen experiment

The new `mfvlr_research.d02b10` namespace wraps unchanged D02B9 `_build_frozen_reference` and `_prepared_body`. The latter reuses frozen D02B1 streamed gradients and D02B7 arithmetic observation. Only original G01-A0, seed 0, B1 real input and Windows CPU C4M are eligible. Required inventory is 508 trainable tensors / 154,156,232 elements; the exact roster is `tests/research_d02b1/fixtures/historical_rosters.json`. Actual tensors, CPU/FP32 dtype, requires_grad, unique object/storage ownership, vocabulary tie, canonical state, fixture bytes and runtime backend are checked, not accepted from caller metadata.

`live.FrozenAdapter` requires an independently trusted pin-file SHA256. Pins cover production/research source, configuration, tokenizer assets, fixture/roster/backend records and R2 original gradient metadata. The package pins and preservation manifest are not execution authorization. The adapter validates the exact canonical state and input hashes from R2. A0 construction and common/private ownership parity use the frozen builder. Every operational seal includes the actual inventory, backend, fixture and state. Source pins are rechecked at seals and completion.

The public `execute`/`execute_real` paths are unconditionally disabled. No YAML/CLI boolean enables them. The internal orchestration engine is implemented for a later explicitly authorized entry point; mock tests inject a synthetic adapter. Underscore functions and mock scope strings are scope controls, not a security boundary against someone writing arbitrary Python. A future handoff must bind the reviewed trusted pin digest and call the live adapter only after explicit user approval.

## Ownership, admission and monitoring

Use an existing, dedicated execution parent and a new simple attempt name. Exclusive creation of `coordinator.json` binds PID, process creation time, cell and attempt name; then create the new attempt directory and immutable claim. Both claims and actual process identity are rechecked. A second coordinator is rejected even if the first process ended. No stale-lock deletion, overwrite, automatic takeover or replay exists. A retry needs a separately reviewed new parent/attempt, preserving the old one.

Before model construction: record actual physical available RAM and temporary disk, require at least 8 GiB each, verify pinned sources, start the resource monitor, recheck ownership. Before the single forward: repeat resources, state/input/ownership/backend seals. The same 8 GiB floor is conservatively enforced during later seals and monitoring too; a resource failure prevents further dependent work, rather than changing precision or model dimensions.

Sample process RSS, available physical RAM, disk, PID/creation time and wall timestamps every 0.5 seconds into an append-only JSONL file. Record elapsed time and sampled peak RSS; sampling does not prove the true instantaneous peak. The monitor latches its first failure. Forward-pre hooks and identity gradient hooks check its status at model/parameter boundaries, so a failure can stop the next boundary without numerical transformations. An in-flight kernel cannot be interrupted safely by this Python monitor. Hooks return the same gradient object and are removed in `finally`. No optimizer or parameter assignment is introduced.

An 8 GiB launch screen is not retained-graph capacity verification. The prior separate-forward RSS of about 2.18 GiB is historical context only. D02B9's memory/storage analysis still applies: each FP32 parameter/gradient vector is 616,624,928 bytes; all saved activations remain live through six isolated traversals. Raw R2 gradients/sums occupy 3,790,126,768 bytes. New graph capacity, runtime, true peak, GPU and B8 remain NOT VERIFIED.

## Traversals, completion and interruption

The frozen recipe performs one full-language forward with the six original losses, six isolated `autograd.grad` passes retaining the graph, then the joint pass freeing it. The wrapper verifies exactly one FLT entry and all three ordered seals: before forward, before traversals, after traversals. Full parameter state/ownership and clean `.grad` are checked before/after; returned gradients are streamed, not assigned to parameters.

Require 3,556 raw pass/parameter records, all 508 FP64 sum shards, exact loss order, retain flags, activation records, arithmetic observation, additivity output and complete R2 pairing. Verify raw file hashes/lengths/shapes, record identities, saved analysis consistency, presence expectations and original elementwise atol=1e-6, rtol=1e-5. Frozen GradientStore retains None/zero/weak gradients, VIM Q/K structural checks, ordinary key-bias residuals, 15 pair metrics and four registered directions. R2 raw comparison verifies historical shard hashes against pinned records. No earlier tolerance exception is broadened.

Finite additivity failure is `execution=COMPLETE`, `numerical=FAIL`, `dependent_work=BLOCKED`. Structural, provenance, nonfinite, missing evidence or interrupted work is HARD STOP. Insufficient pre-forward admission is WAIT/NOT RUN. Historical G01 remains 293 failed tensors and 8,772,234 outside coordinates. No result is upgraded because a worker exited normally.

Catch ordinary failures and KeyboardInterrupt, save stage/error/status and retain all files and the permanent lease. Abrupt process termination or power loss cannot guarantee a final JSON; a claim without complete result is an interrupted attempt, never a successful cell. A retained graph cannot be resumed from disk. Archive and hash partial evidence before any separately authorized new attempt; no replay is implemented or enabled here.

## Checklist before any real attempt

- Explicit approval for exactly one same-forward G01-A0 attempt; no approval currently exists.
- Independent review of D02B10 source, tests, pin SHA256 and compact package.
- Dedicated exclusive execution root; original R2 and all interrupted attempts preserved.
- Fresh measured >=8 GiB RAM and >=8 GiB disk; no capacity inference from current or historical snapshots.
- Trusted complete pin file, exact installed Windows/C4M cohort, original fixture and canonical state.
- Full 508-tensor roster, disjoint common-state construction, tied vocabulary and no changes to A0/objectives.
- Journal/resource sampler active; exactly one forward and seven retained-graph traversals.
- Complete gradients, hashes, comparisons and statuses; stop regardless of outcome, B0-D08 disabled.

The operational code is prepared, but this checklist is not a declaration that runtime gates have passed. Only static/synthetic controls have executed.
