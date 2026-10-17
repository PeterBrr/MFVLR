# D02B2 execution gates and verification record

2026-10-16. **Phase 2D.2 BLOCKED. No numerical N2 execution is authorized.**

## Approval and prerequisite ledger

| Gate | Status |
| --- | --- |
| New isolated N2 preparation, metadata/array algorithms, static/mock tests | APPROVED by current D02B2 instruction |
| D02B1 document-level review | CONDITIONALLY REVIEWED; independent source/ZIP audit still outstanding |
| A0/P3/Q3 model construction, full-language forward/backward, real N2 comparisons | NOT AUTHORIZED / NOT RUN |
| Real P3/Q3 initialization/promotion/ownership verification | NOT RUN; metadata tests are not substitutes |
| B0-D03 primary G01–G06 | DISABLED / NOT RUN |
| B0-D08 finite-failure continuation | DISABLED |
| NG2, G2 controls, optimizer, training, evaluation | NOT AUTHORIZED / NOT RUN |
| MVE-T10, VD-T12 | NOT COLLECTED / NOT RUN; historical exception does not transfer |
| N1/N2/N3/N3.1 policy changes | NOT AUTHORIZED; original scope retained |

The new adapter's construction, promotion and execution methods raise unconditionally. The frozen builder call is dormant behind the construction barrier. No callback, record field, YAML switch, CLI option or supplied hash unlocks it. Enabling execution would require a subsequent reviewed code change plus explicit authorization. No existing dispatcher was edited.

## Preconditions for a future authorization

1. Review this package and the outstanding D02B1 independent source/ZIP audit. Approve an explicit subset of fixture/variant/precision cells and any additional original supplemental controls; do not infer approval of the 18-row prepared matrix.
2. Recheck frozen source/config/test/protocol hashes and actual backend/thread/build/environment values. A declared backend in a mock record is not an observed backend. Any changed cohort needs review.
3. Verify complete original ID/visual hashes and B1/B2 ordering. F2 is a hash-verified historical visual input; do not run MVE/VD merely to construct the N2 fixture without separate scope.
4. Construct one shared CPU FP32 reference, then use the frozen variant builder, verify all common bytes and disjoint ownership, additional LN initialization, 12 LE/7 LD/19 VIM roles and tied vocabulary. Materialize actual P3/Q3 N2 state hashes before forward; none were generated here.
5. For FP64, deep-copy that exact FP32 candidate and promote exact values. Preserve source state/identities. Round-trip FP32 equality is necessary but insufficient: direct FP64 equality to the cast original is also required. The promotion receipt requires the expected FP64 digest from original FP32 arrays.
6. Establish a sequential, exclusive execution owner, resource preflight and interruption retention appropriate to the authorized N2 scope. D02B1 full-six-loss 5/6-GiB reservations are not evidence of N2/FP64 capacity. CPU capacity for this new variant run, GPU and B8 remain NOT VERIFIED; no automatic worker or new resource claim exists in D02B2.
7. Pin an independent request seal and preserve complete raw outputs before validation. No model result may be replaced by the mock fixtures or their fabricated object/storage IDs/hashes.

## Hard stops and numerical failures

Missing source/input/state hashes, incomplete parameter or stage coverage, wrong shapes/dtypes, invalid tied ownership, incorrect LN placement, unregistered backend, nonfinite values, state mutation, unexpected missing gradients, nonzero VIM structural zeros, fixed-batch leakage and permutation/cross-sample derivative failures are hard errors. Their evidence must be retained; do not substitute another fixture, omit a weak tensor, or continue silently.

FP64 cross-batch disagreement remains a mandatory numerical failure at `1e-10/1e-8`. FP32 cross-batch differences retain their numerical FAIL count and original `1e-6/1e-5` statistics under the same N2 diagnostic category. The complete-record validator may report that the supplied record is complete while its numerical comparisons FAIL; it never grants acceptance or proves execution occurred. FP32 diagnostics are not a new exemption for losses, gradient additivity or ordinary Q/K accuracy. No MVE N1 tolerance is applied.

Ordinary Q/K weakness and ordinary key-bias cancellation residuals remain visible, including nonzero values arbitrarily close to zero. There is no small-gradient cutoff, epsilon floor, normalization, clipping, rescaling, AMP, changed initializer or hidden mathematical change. VIM Q/K requires an allocated exact-zero gradient, not None. No optimizer state or step is implemented.

## Direct verification versus historical evidence

Fresh D02B2 checks are source hashing, historical-input byte reconstruction, static AST placement checks and independent synthetic array/mock record tests. Full model-shaped **metadata** is tested without allocating full model-shaped weights, activations or logits. The largest array test is a small vector crossing the original 262,144-element chunk boundary. Neither torch nor production MFVLR is imported in the test process.

The first isolated run passed 75 cases; the expanded run passed 98 cases. All original logs and JUnits are retained separately. The final reproducible run **passed 98/98 cases, exit 0, zero failures/errors/skips**, in 4.998 seconds including process startup. No expected-failure overrides were used. Python 3.14.3, NumPy 2.4.4 and pytest 9.1.1 ran on Windows 11; torch remained unimported. Tested source hashes were unchanged during the run. Exact invocation, environment and hashes are in [verification.json](../artifacts/phase2d2_d02b2/verification.json), with [stdout](../artifacts/phase2d2_d02b2/final_01/stdout.log), [stderr](../artifacts/phase2d2_d02b2/final_01/stderr.log), and [JUnit](../artifacts/phase2d2_d02b2/final_01/junit.xml). The original tests are unchanged and are not counted as freshly executed. Existing 78-case D02B1 results remain submitted historical evidence, not an independent source audit or a new regression run.

Use only the authorized isolated test selection:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest --confcutdir=tests/research_d02b2 tests/research_d02b2 -v --junitxml=<new-output-directory>/junit.xml
```

The root pytest fixtures can construct MVE; `--confcutdir` excludes them, a local autouse fixture overrides the model dependency as defense in depth, and collection rejects other directories. Tests require torch/production modules to remain unimported. Do not invoke the original N2 review worker, the original FLT tests, or the legacy research launcher during this preparation.

`before.json` / `after.json` provide fresh SHA-256 preservation evidence, including accepted source, original tests, D02B1 code/package and prior numerical archives. This is byte integrity verification, **not closure of the outstanding independent D02B1 audit**. The compact review package contains complete source, the original helper/builder/configs/tests used as references, frozen input records, new tests, raw logs/JUnit and a manifest. Its CRC and every included payload hash are checked; large historical archives/checkpoints remain external and are referenced by preserved hashes.

The completed preservation audit matched **1,674/1,674 files**, with zero mismatches. The before-inventory SHA-256 is `ac60ce057c52ca7ce1b0942f4451ec115b7377be700dc3b4c83ce41a0d6f8bd5`; see [after.json](../artifacts/phase2d2_d02b2/after.json). The separate frozen-source verifier also matched all 53 pinned dependencies. These checks include the disabled D02B1 dispatcher and do not alter any historical scientific result.

Stop for user review. No new numerical N2, G01–G06, NG2, optimizer or training run has occurred. Full numerical acceptance, repeated-step stability, convergence and accuracy remain unestablished.
