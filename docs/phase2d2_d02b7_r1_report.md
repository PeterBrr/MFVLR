# D02B7-R1 resource recovery attempt

2026-10-16, Asia/Saigon. **WAIT/NOT RUN. Phase 2D.2 remains BLOCKED.**

The user's retry instruction was treated as authorization for one conditional G01-A0 reattempt. Its initial resource check observed **5,139,156,992 bytes = 4.7862 GiB available physical RAM**, below the frozen **5 GiB = 5,368,709,120 bytes** requirement. The shortfall was 229,552,128 bytes. Free disk was **663,519,948,800 bytes = 617.95 GiB**, above the 8 GiB requirement. No other Python numerical worker was observed. No unrelated process was terminated and no scientific or system setting was changed.

Because the initial gate failed, no second gate was attempted and no numerical worker, model or fixture was constructed. R1 stops here, in accordance with the user's explicit failure policy.

| Axis | R1 result |
| --- | --- |
| Identity | D02B-G01-A0 only |
| Execution | WAIT/NOT RUN |
| Numerical | NOT RUN |
| Resource | INSUFFICIENT RAM; sufficient disk |
| Source/protocol file integrity | PASS, 197 pinned files |
| Live state/input/backend provenance | NOT RUN |
| Structural checks / 508-parameter gradient participation | NOT RUN |
| FD, LR, CMC, FL, AR, KL and total | NOT RUN; no measured values |
| Gradient additivity | NOT RUN |
| Actual model memory peak | Not measured; no model process |
| Dependent work | BLOCKED |
| B0-D08 | DISABLED |

The initial process inventory contained only the short-lived resource-inspection Python process. `coordinator.lock.json` in the new R1 directory identifies the separate evidence recorder; it is not evidence of a numerical worker. The prior D02B7 lock and execution record were untouched.

## Preservation and evidence

New exclusive directory: `artifacts/phase2d2_d02b7_r1`. The original D02B7 directory's 18 files were fingerprinted and rechecked unchanged, including its review ZIP, lock, execution record, resource logs and manifests. All 224 payload hashes of the original manifest were checked against current files; all matched. The frozen worker and scientific code were reused as source references only, with no edits or execution. No original numerical thresholds, losses, fixtures, initializers or historical failures changed.

The R1 resource and process observations are saved in `resource_preflight.json`. `resource_check.stdout.log` is a transcription of the initial tool observation and stop decision, **not captured stdout from a model worker**; `log_provenance.json` makes that distinction machine-readable. There are no worker stdout/stderr logs, numerical JUnit cases or new pytest results because no worker or new test suite was run. The compact package includes the earlier D02B7 test logs with their original paths, clearly historical rather than fresh R1 passes.

Requested live metrics are explicitly null/NOT RUN in `requested_measurements.json`, rather than filled from old or mock results. `execution_record.json` separates execution, numerical, structural, provenance and resource outcomes. File preservation is independently inspectable through `previous_attempt_preservation.json`, `source_protocol_verification.json` and `preservation_after.json`.

## Stop boundary

No G02–G06, P3/Q3, NG2, F0/F1/F2 rerun, optimizer, training, G3/G4, benchmark, MVE-T10 or VD-T12 execution occurred. This resource outcome establishes neither numerical acceptance nor a new numerical failure. No retry is scheduled. Stop for review; another attempt requires further authorization and the unchanged resource gates.
