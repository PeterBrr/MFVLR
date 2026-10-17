# D02B7-R2: actual G01-A0 seven-pass result

2026-10-17, Asia/Saigon. **Execution COMPLETE; numerical FAIL; dependent work BLOCKED.** Phase 2D.2 remains BLOCKED. B0-D08 remains DISABLED.

Exactly one authorized cell ran: `D02B-G01-A0`, original real B1 fixture, pinned Windows CPU C4M FP32. The unchanged D02B7 coordinator and worker were reused through a small wrapper changing only the exclusive output directory. The original D02B7 and R1 artifacts were preserved. No second cell, timing repetition, optimizer, training, backend control or normalization variant ran.

## Live results

| Objective | Actual FP32 scalar |
| --- | ---: |
| FD | 6.778368949890137 |
| LR | 248989.578125 |
| CMC | -0.0 |
| FL | 0.7248529195785522 |
| AR | 0.3265681862831116 |
| KL | 30.640581130981445 |
| Literal total | 249028.046875 |

All seven values exactly reproduce the recorded historical G01 scalars. They are actual full-graph measurements, not mock fixture values. Every isolated pass used a fresh graph and the unchanged production objective; total used the existing joint objective. There were exactly seven forwards and seven backwards, with no optimizer steps.

The live independent arithmetic observer replayed FD/FL class reductions, LR all-token sum then batch mean, raw-dot symmetric B1 CMC, unrestricted reconstruction MSE, KL(text||adapted vision) at 0.5, and the literal FP32 total order. All replayed scalar values agreed exactly. The FP32 total steps were 6.778368949890137 → 248996.359375 → 248996.359375 → 248997.078125 → 248997.40625 → 249028.046875.

LR includes 4 BOS, 4 EOS, 276 padding and 24 content tokens. Summing recorded token NLL values in FP64 by category gives a value 0.00128173828125 below the production FP32 LR reduction. This regrouping difference is preserved as a diagnostic, not used to replace the approved reduction or alter the loss. Complete token values, category sums, CMC scores/temperature, KL entropies and reconstruction ranges are in `attempt_01/live_arithmetic.json`.

## Actual gradient coverage

508 trainable tensors / 154,156,232 elements were verified with exact frozen roster, common-state parity, private storage, seed-0 state and tied vocabulary. Before/after identities and bytes matched. Each of the seven passes includes all 508 allocation records; unused None gradients are not silently converted to allocated zeros.

| Pass | Allocated | None | Allocated exact zero | Nonzero |
| --- | ---: | ---: | ---: | ---: |
| FD | 116 | 392 | 0 | 116 |
| LR | 501 | 7 | 38 | 463 |
| CMC | 333 | 175 | 333 | 0 |
| FL | 46 | 462 | 0 | 46 |
| AR | 46 | 462 | 0 | 46 |
| KL | 334 | 174 | 24 | 310 |
| total | 508 | 0 | 39 | 469 |

Connected singleton VIM Q/K gradients were exact zero: 38 for LR/total, 24 LE tensors for CMC/KL. B1 CMC remains connected with all 333 allocated gradients exactly zero. Ordinary Q/K weights and key-bias residuals are retained without magnitude filtering. All recorded allocated gradients were finite. Actual Ipre/Iv/Tl/adapter derivatives, including unused None paths, are preserved separately per pass. Each pass recorded all 63 activation stages.

Frozen D02B1 streaming accumulated isolated FP32 gradients in FP64 in FD/LR/CMC/FL/AR/KL order. The original FP32 comparison against the rounded sum, atol=1e-6 and rtol=1e-5, found **8,772,234 outside-tolerance coordinates across 293 of 508 parameters**. No threshold or scientific policy changed. Complete per-parameter data includes all 7,620 loss-pair records and 2,032 directional pairs. This is not FP64 full-model autodiff.

`live_additivity.junit.xml` is generated from actual comparison records: **508 cases, 293 numerical failures, zero skips**. It does not represent a newly run pytest suite. The earlier isolated D02B7 preflight suite remains 18/18 passing and was reused by exact source hashes, not rerun or counted as gradient evidence. Exit code 0 records successful computation, not numerical acceptance. The original eight historical assertions remain unresolved genuine failures.

## Resources and preservation

The initial inspection observed approximately 5.94 GiB available RAM; both coordinator and worker then recorded their own passing 5 GiB RAM / 8 GiB disk launch gates. Exact readings are in `resource_preflight.json` and `attempt_01/resources_before_import.json`. Sampled peak worker RSS was **2,341,023,744 bytes (2.18025 GiB)**; worker elapsed time was **119.255064 s**, coordinator wall time **120.568979 s**. RSS was sampled every 0.25 s and is not a guaranteed instantaneous maximum. Per-pass RSS, available RAM, free disk and timing are retained. The 5 GiB gate applies before launch; declining available RAM as the model allocates memory is not a changed launch threshold.

Raw FP32 gradient and FP64 sum shards total **3,790,126,768 bytes**. Every shard's SHA-256 was verified in a read-only audit. They remain on disk and are excluded only from the compact ZIP; `excluded_binary_inventory.json` records each path, size, hash, parameter, pass and format. Independent recomputation from every gradient element requires those original binary shards. The compact package permits complete inspection of coverage, scalar/stage records, per-parameter/pair/directional diagnostics and source provenance; summaries alone cannot reconstruct omitted raw gradients.

Frozen source/protocol pins and both earlier attempts were rehashed unchanged. State, input and backend seals precede each graph and FLT call. `live_evidence_audit.json` records the separate post-run verification. No new model computation was used to build the report.

## Wrapper digest clarification

The original R2 `retry_scope.json` preserved an LF source-text SHA-256 under the generic `wrapper_sha256` key. Windows wrote CRLF bytes, so this was not the on-disk file digest. The original record was not overwritten. `wrapper_hash_clarification.json` records the actual file-byte hash and verifies that newline normalization exactly recovers the original text digest. No code content changed. Frozen coordinator/worker/scientific source pins passed unchanged; this bookkeeping clarification does not modify numerical results or create a new acceptance policy.

## Stop

The single attempt is finished. Finite additivity FAIL is retained and dependent work is BLOCKED. No G02–G06, P3/Q3, NG2, optimizer, training, G3/G4, benchmark, MVE-T10 or VD-T12 execution occurred. Neither reproduction of historical values nor a finite graph establishes training stability or accuracy. Stop for independent review; no further attempt is scheduled.
