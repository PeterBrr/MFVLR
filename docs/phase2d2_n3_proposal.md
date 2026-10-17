# N3 verification proposal — unapplied and blocked for acceptance

**Recommendation: do not accept Phase 2D.2 or replace its failing assertions yet.** The evidence supports a numerical mechanism for gradient additivity discrepancies, but the newly measured percent-scale fixed-cotangent sensitivity and remaining FP64 LD coordinate failures require more than reclassifying the old FP32 test as a diagnostic. No higher tolerance is proposed.

The universal FP32 elementwise gradient-additivity assertion is not an architecture-independent mathematical correctness criterion: it compares differently rounded evaluations of an exact-real identity, including analytically canceling coordinates. It remains a useful recorded regression experiment. Replacing it can be justified only as part of an independently justified composite verification policy. The current evidence is insufficient to assign quantitative full-graph error budgets without fitting the observed failures.

This proposal concerns verification only. All model dimensions, parameters, initialization, six objective definitions, unit coefficients, temperature, raw feature scales, tokenizer, singleton VIM, N1 and N2 remain unchanged. No architecture stabilization or training is authorized by this document.

## Before and proposed staged outcome

| Item | Current mandatory policy | Proposed diagnostic addition | Future replacement policy |
|---|---|---|---|
| JOINT-I08 FP32 CMC chain | Existing allclose remains failing | Add preserved FP64/local derivative evidence | BLOCKED pending scoped approval and an independently justified criterion |
| JOINT-I10 FP32 full additivity | Existing allclose remains failing | Add same-forward identities, per-parameter/cotangent controls | BLOCKED; cannot waive with global relative L2 |
| Six READY-N02 fixtures | Existing summary must be PASS; currently fails | Retain raw failures and attach complete diagnostic audit | BLOCKED; finite execution alone is insufficient |
| 276 accepted regression cases | Unchanged | Unchanged | Must remain unchanged |
| N1/N2 | Existing accepted narrow scope | No extension | No extension |
| Numerical thresholds | FP32 1e-6/1e-5; FP64 1e-10/1e-8; existing FD 1e-5/1e-3 | Identical | No new number specified or approved |
| Phase 2D.2 status | BLOCKED | BLOCKED | Requires explicit policy approval and a complete subsequent suite |

No accepted test is removed, skipped, xfailed or reweighted. The seven-case supplemental patch would not resolve the eight original failures. Its isolated dry run passing must not be described as a passed 328-case mandatory suite; that suite has not been installed or executed.

## Complete candidate policy design

The following is **PROPOSED design**, not implemented acceptance logic.

1. **Mandatory provenance and architecture gates.** Hash accepted source/configs/assets and weights; verify dimensions/counts, unique ownership, tied vocabulary, live input identities, unchanged parameters and zero updates. Every numeric record carries fixture seed, exact values or hashes, precision, backend flags, thread count and tool source hash. Changes in environment are explicit new evidence, not presumed equivalent.
2. **Mandatory finiteness and connectivity.** All six scalars, requested activations, logits, intermediates and all present parameter gradients must be finite. Preserve loss-specific presence/absence controls, reconstruction-mediated gradients, detached diagnostic controls and allocated exact-zero singleton VIM Q/K. Unexpected None versus zero is a failure. Intentional CMC cross-sample coupling is not sample leakage.
3. **Independent derivative correctness.** Keep all accepted loss FP64 mathematical oracles, finite differences and FLT/VIM tests at unchanged tolerances. Add exact rational ordinary-key-bias derivation, independent ordinary Q/K local formulas and directional finite differences. Preserve the full 12-LE promoted-weight FP64 CMC probe. Clearly separate local FP64 scope from full visual-model correctness. The new LD FP64 failures remain visible and require adjudication, not automatic exemption.
4. **Full FP32 discrepancy evidence.** Continue measuring the original elementwise assertion outcome even if an eventual replacement changes its gate status. Require per-parameter count/proportion outside the original tolerance, maximum absolute/relative error, error/reference norms, presence, zero-reference behavior, signs and directional projections. Stream all 508 parameter vectors for full-module error aggregation; do not label the current selected-vector aggregates full coverage. Intermediate outputs and scalar loss nodes get separate records.
5. **Normwise and directional checks.** Compare each nonzero parameter/group and seeded directions under separately justified budgets. A whole-model norm dominated by LR is never sufficient. Retain absolute norms and errors when denominator is small. Do not assign an epsilon denominator floor or a minimum-gradient exemption without separate approval. Four existing random directions are useful diagnostics, not complete proof; include loss-specific and fixed-cotangent directions before selecting a mandatory protocol.
6. **Canceling derivatives.** VIM structural zeros remain exact. Ordinary key biases remain present and unchanged; their exact-real zero derivative is independently verified, while finite-precision residuals are judged by a separately derived scale-aware absolute error model, not relative error to zero. Ordinary Q/K weights receive nonzero-derivative evidence and cannot inherit a bias exemption. Fixture-dependent near-zero parameters are explicitly listed and reviewed rather than dropped.
7. **Multiple fixtures and backends.** Retain all six full synthetic fixtures and add the selected same-forward matrix. Preserve MKLDNN on/off distinctions; these are not two platforms. A future supported-platform run must record actual results. GPU/other-platform checks remain NOT RUN until executed; lack of a backend must not be counted as a pass.
8. **No-update numerical gate before later work.** A composite gate must distinguish derivative connectivity from finite losses, stable VJPs and later optimizer behavior. Any new nonfinite value, missing gradient, identity mismatch or unsupported error-budget excursion stops acceptance. G3/G4, optimizer finiteness, repeated-step stability and convergence remain separate unapproved work.

### How quantitative budgets could be justified

Do not choose a threshold from the observed maximum, its rounded decimal or a multiplier that makes these fixtures pass. A possible next study would derive local backward-error bounds for the executed affine, matmul, reduction and softmax Jacobian operations, using actual accumulation dimensions, operand norms and error propagation. The classical `gamma_n = n*u/(1-n*u)` relation is a starting point for a specified floating-point reduction model, **not** a proven bound on this entire graph or an approved threshold. Optimized kernels, cancellation and the stored rounded softmax probabilities need explicit treatment.

Use independent FP64/manual local references to test such bounds and predeclare new held-out cotangent/input fixtures before evaluating a candidate budget. Report both guaranteed bounds and empirical estimates as distinct quantities. If a bound is too loose to detect meaningful defects, it is not a useful acceptance criterion. If independent local references still fail the unchanged gate, diagnose them before using them as a policy foundation. A reviewer may ultimately choose a documented diagnostic policy with manual adjudication, but that would be a new explicit approval, not a numerical proof or automatic acceptance of this baseline.

No quantitative replacement threshold is supplied because the present study does not justify one. In particular, **16.6%**, a global relative-L2 threshold above 2.1e-6, and wholesale removal of ordinary attention biases are not recommendations.

## Unapplied test changes and repair

[proposed_tests.patch](../artifacts/phase2d2_n3/proposal/proposed_tests.patch) is a valid new-file unified diff for `tests/joint/test_n3_controls.py`. It adds exact rational cancellation, independent FP64 ordinary attention derivatives, two Q/K directional finite differences, streamed-comparison self-tests with an injected-error control, complete recorded diagnostic checks and the promoted CMC record check. It removes or modifies **no** existing assertion. [Metadata](../artifacts/phase2d2_n3/proposal/proposal_metadata.json) pins the current two mandatory joint-test source hashes and requires the proposed new target not to exist. The patch was reconstructed/compiled only under `artifacts/phase2d2_n3/proposal/dryrun/`; all seven cases there were executed and passed. Diagnostic completeness tests validate the content they read, but a future mandatory workflow must freshly regenerate evidence and verify its provenance before invoking them; static JSON alone is not proof of a fresh run.

The separate [proposed_metadata_repair.patch](../artifacts/phase2d2_n3/proposal/proposed_metadata_repair.patch) fixes the new diagnostic's transient loss-`grad_fn` wrapper IDs by holding named wrappers before sampling IDs. It is syntax-checked and **UNAPPLIED**. The original tool and result files remain unchanged; the metadata limitation is documented in the review. This defect affects optional recorded addresses, not gradients. No production implementation defect was identified by the bounded controls, so no production repair is proposed.

If either patch is approved, verify its expected source/target state before applying it, preserve original records, regenerate affected metadata/evidence and rerun the appropriate complete suite. Approval of these additions alone would leave the eight original failures mandatory. Do not imply approval of a numerical waiver from approval of a metadata repair.

## Approval decisions

| ID | Decision | Classification | Status | Approval boundary |
|---|---|---|---|---|
| N3D01 | Add the seven supplemental independent/diagnostic tests from the unapplied patch | ASSUMED verification design | PROPOSED | Explicit approval before adding to the mandatory test tree |
| N3D02 | Repair transient loss-grad-function identity metadata, preserve the originals and rerun affected metadata capture | ASSUMED diagnostic engineering | PROPOSED | Separate unapplied repair; no model arithmetic change |
| N3D03 | Adopt the composite policy structure above | ASSUMED verification design | PROPOSED | Design approval only; does not supply passing budgets |
| N3D04 | Replace the universal FP32 CMC/additivity assertion as an acceptance gate | ASSUMED policy amendment | BLOCKED | Independent bounds/adjudication, full-module evidence and explicit policy approval still required |
| N3D05 | Select new per-parameter/module numerical thresholds | ASSUMED | DEFERRED | No threshold justified or proposed solely from these failures |
| N3D06 | Extend fixed-cotangent/high-precision diagnosis to new predeclared fixtures and full streamed directional/module coverage | ASSUMED diagnostic protocol | PROPOSED | Remains no-update diagnosis; no optimizer or architecture change |
| N3D07 | Apply N1/N2 to the new six-loss gradient gates | Neither authorized nor justified | BLOCKED | Existing amendments retain their accepted original scope |
| N3D08 | Alter LD normalization, attention, precision, loss scaling, clipping or parameters | ASSUMED architecture/stabilization change | DEFERRED / UNAPPROVED | Not part of this proposal or implementation |
| N3D09 | Declare Phase 2D.2 accepted | Acceptance decision | BLOCKED | Requires explicit remedy/policy approval and a complete new mandatory suite |
| N3D10 | Begin G3/G4, optimizer, scheduler, datasets, training or evaluation | Future implementation | DEFERRED / UNAPPROVED | Separate authorization required |

The numerical review is complete and stops here. All existing acceptance failures and historical evidence remain intact. See the [numerical review](phase2d2_n3_numerical_review.md) and [reproducible review package](../artifacts/phase2d2_n3/phase2d2_n3_review.zip).
