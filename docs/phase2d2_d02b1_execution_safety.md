# D02B1 execution safety and recovery record

2026-10-16. **Implementation + mock verification only. All numerical dispatch disabled.**

## Approval ledger

| Decision | Current status |
| --- | --- |
| B0-D02 isolated runner/schema/storage and mock tests | **APPROVED**, current user instruction only |
| Preserve accepted scientific baseline, configurations, initialization, objectives and thresholds | **REQUIRED**; before/after file audit |
| B0-D03 G01–G06 real primary execution | **NOT AUTHORIZED / NOT RUN** |
| C01–C07 backend controls | **NOT AUTHORIZED / NOT RUN** |
| NG2 materialization and N01–N06 | **NOT AUTHORIZED / NOT RUN** |
| B0-D08 finite numerical-failure continuation | **DISABLED**, requires separate explicit approval |
| New N2/N3/local oracle execution | **NOT AUTHORIZED / NOT RUN** |
| MVE-T10 and VD-T12 | **EXCLUDED**, neither executed nor collected |
| I1/R2, production promotion, G3/G4, optimizer/training/datasets/evaluation | **NOT AUTHORIZED / NOT RUN** |

The user approves infrastructure for the proposed partition, not an executable approval token. `dispatch` and finite-failure continuation unconditionally refuse. There is no CLI/YAML switch that activates them. The legacy launcher/configs remain untouched. Source changes will be needed for any future real worker and require a new review.

## Resource and coordinator safeguards

`Reservations` enforces minimum available RAM of **5 GiB for B1**, **6 GiB for B2**, plus **8 GiB free temporary storage per cell**. Configuration may increase floors but cannot lower them. Missing, noninteger, negative or untrusted readings fail closed; insufficient resources yield NOT RUN. D02B1 tests inject synthetic resource readings and allocate no full-cell reservation. They do not claim present RAM suffices or validate GPU/B8 capacity.

A coordinator claims one schedule root with exclusive-create `coordinator.lock.json`, recording PID **and creation time**. A liveness provider must confirm the exact owner before a cell can reserve its exclusive path. Duplicate coordinators/cells fail; PID reuse fails. No worker-launch API exists, so actual concurrent full-model workers cannot start through this package. A future live resource/process observer is not silently approximated by the mock provider.

There is no automatic stale-lock deletion. Completed or abandoned reservations cannot be overwritten. Recovery operates on an explicit new disjoint archive/attempt path and preserves the original. A live owner, reused PID, or inaccessible process observation blocks recovery; an absent observed PID does not establish an OOM or power-off cause. Root ownership must be maintained across the full approved schedule by the future coordinator; choosing another root is not authorization for duplicate scientific work.

`archive_attempt` copies every original file byte-for-byte using exclusive destinations, records SHA-256, checks source/destination digests and the file set, rejects linked evidence paths and overlapping roots, and retains partial files on errors. It never removes the original attempt. Recovery provenance records cause as UNKNOWN unless separately evidenced.

`read_complete_prefix` accepts only complete newline-terminated scientific JSONL records and separately hashes an unterminated tail. A malformed complete line is an error. Replay compares every original complete record in order, including comparison identity, failed statuses, weak-gradient values and all scientific fields. Only explicitly named `resources.seconds` and `resources.rss_bytes` are excluded. Missing/reordered/changed records fail. The comparison record is written before raising on divergence or incomplete replay. No failed original comparison is dropped or promoted because replay matches.

## Independent status axes

Execution/provenance, structure, numerical agreement and resources remain separate. Missing or mismatched provenance, state/ownership changes, unexpected gradient presence, nonfinite values, missing stages/parameters/passes and incorrect singleton structural zeros are hard errors. Finite failed additivity retains numerical FAIL. A successful file-write or process exit is never numerical acceptance. This phase cannot report a scientific cell COMPLETE: every real cell remains NOT RUN.

The original full-gradient threshold is FP32 `1e-6 + 1e-5 * abs(reference)`. No N1 tolerance is imported. N2 remains the FLT-T17 cross-batch FP32 policy only; fixed-batch independence/permutation, zero cross-sample derivatives, promoted-weight FP64 tests and the eight original A0 failed assertions retain their scope. Ordinary key-bias residuals are diagnostics against analytical zero, not a license to force or ignore weak gradients. B0-D08 continuation is not implemented behind a hidden boolean.

## Verification isolation and audit

Reproduce only the authorized new tests:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest --confcutdir=tests/research_d02b1 tests/research_d02b1 -q --junitxml=<new-output-path>/junit.xml
```

Use a new output directory for a reproduction. The root accepted pytest fixtures construct MVE and are excluded by `--confcutdir`. The local fixture additionally overrides the model-owning autouse fixture. The collection guard accepts only the new research test directory, and tests assert no torch/production-module import. No broad regression invocation occurred. Array controls with historical 508/550 names allocate scalar fixtures, not full parameter tensors.

`artifacts/phase2d2_d02b1/baseline_before.json` includes the freshly matched 1,594-file historical preservation inventory, existing source/config/tests/tools/docs, D02B0 records and final D02A/lite archive/manifests. The matching after-audit verifies exact bytes, not scientific re-execution. Old source, tests, failed logs and original manifests are neither replaced nor regenerated. New verification evidence is in a separate directory.

The completed after-audit matched **1,637/1,637 frozen files**, zero mismatches, covering 4,711,823,963 bytes. Baseline manifest SHA-256 is `c0b0d25561e1bc563bb6882669beef77e8df793950a283a1c560ff5f9afb7945`. See [baseline_after.json](../artifacts/phase2d2_d02b1/baseline_after.json) and [guard_verification.json](../artifacts/phase2d2_d02b1/guard_verification.json). The fresh final test run is 78/78 passing, exit 0; prior 65-case and 78-case runs remain historical artifacts, not extra scientific cells.

The new compact package manifest hashes every included payload; the verification record checks ZIP CRC, duplicate/path safety and all included SHA-256 values. Archive and manifest hashes are recorded outside the archive to avoid self-reference. Large old archives/states are referenced by hashes, not copied into this compact infrastructure package. Historical result claims remain explicitly historical.

**Phase 2D.2 stays BLOCKED.** No full-model feasibility or numerical acceptance is inferred from passing static/mock tests. Stop for user review before B0-D03 or B0-D08.
