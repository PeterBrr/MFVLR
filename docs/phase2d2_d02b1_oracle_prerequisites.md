# D02B1 oracle prerequisites and ordering reconciliation

2026-10-16. **No oracle or numerical experiment is executed in this phase.** Historical records below are reused only for their named claims. Their existence is not a fresh PASS for P3/Q3 full integration. Phase 2D.2 remains BLOCKED.

## Original ordering versus G2-first proposal

The [corrected A1.1 preregistration](phase2d2_a1_ablation_preregistration_v1_1.md), “Frozen controls” and final recommended authorization sequence, calls for source/ownership and primitive checks, local held-out comparisons, full-language N2, then G2 and NG2. The [D02B0 matrix](phase2d2_d02b0_fixture_matrix.md) proposes G01–G06 as the first **full-six-loss partition**, not as proof that all earlier variant-specific prerequisites have passed.

D02A completed the registered local A0/P3/Q3 comparison scope; it did not execute P3/Q3 full-language N2 or six-loss G2. The accepted A0 N2 policy cannot be inherited as an observed P3/Q3 numerical result. Therefore **G2-first inside the 57-cell matrix does not waive the pre-G2 N2 obligation**. Before B0-D03 execution, the user must separately authorize and review outstanding variant-specific prerequisites, or explicitly approve a documented sequencing amendment. D02B1 implements neither an amendment nor an implicit waiver. No YAML field or mock seal can satisfy that gate.

## Evidence classification

| Prerequisite/control | Classification and named sufficiency | Outstanding work / gate |
| --- | --- | --- |
| A0 N2 F0/F1/F2 FP32 and promoted-weight FP64, same FP32 initialization | **Previously verified**, sufficient to describe accepted A0 numerical behavior and original N2 scope; preserved in `artifacts/phase2d2_n31/n2/` and the N2 acceptance records | No fresh execution; does not establish normalized variant behavior or full visual FP64 gradients |
| P3/Q3 full-language N2 F0/F1/F2, exact fixed-batch/permutation and both cross-sample directions | **Requires new variant-specific verification** under the unchanged controls | **NOT RUN / BLOCKED** before claiming original sequence complete. Not included in 57 cells, not authorized here |
| A0 N3 same-forward versus separate-graph controls | **Previously verified**, sufficient to show historical finite additivity failures persist beyond separate-forward rounding; `artifacts/phase2d2_n31/same_forward/` | Numerical FAIL remains FAIL; cannot mark the joint-gradient requirement PASS |
| P3/Q3 same-forward controls involving changed LD/LR/total | **Requires new variant-specific verification** if used to establish those claims | **NOT RUN**; outside proposed primary partition and separately authorized; no inference from lower local activation scales |
| N3 CMC chain FP32/promoted-weight FP64 | **Previously verified** for the unchanged A0 CMC functional path and historical discrepancy, not a fresh variant/full-gradient result | CMC uses LE/fused vision, not LD; exact variant-invariant CMC outputs/common gradients are a required future integration check. A new primitive CMC sweep is an **optional diagnostic outside primary** unless dependency validation fails |
| N3.1 ordinary-attention analytical Q/K derivatives, exact key-bias mathematics, finite differences | **Previously verified**, sufficient for the unchanged primitive mathematics at recorded fixtures/backend; original tolerances and failures retained | Does not guarantee all weak full-model Q/K gradients accurate; no new local sweep needed merely to repackage metadata |
| A1.1/D02A ordinary/constant/near-constant LN forward and VJP controls | **Previously verified**, sufficient for the frozen P3/Q3 LN primitive and approved placement/ownership module tests | Actual full-graph capture/ownership/connectivity remains **new variant-specific verification**, NOT RUN |
| D02A independent ordinary Q/K primitive oracle checks | **Previously verified**, sufficient for the frozen ordinary-attention primitive prerequisite; `artifacts/phase2d2_d02a/oracles/junit.xml` records 12 passing controlled cases | 12 is the primitive test-case count, not the held-out derivative-comparison count |
| D02A 414-job local A0/P3/Q3 held-out matrix | **Previously verified**, sufficient for completion/provenance of that local schedule only | 399,522 raw numerical FAILs, 4,907 separate cross-batch FAILs and both Q3 FP64 NLD failures remain genuine; not a full-model gate PASS |
| D02A 2,736 native FP64 ordinary-attention oracle comparisons | **Previously verified**, sufficient for recorded local oracle role coverage | 864 Q/K-weight +432 key-bias +864 input/query-bias +576 LN-role comparisons; not 2,736 Q/K-weight checks, not universal accuracy |
| D02A 58,752 allocated singleton Q/K zero checks | **Previously verified** for their local fixtures | Future six-loss connectivity must independently verify 38 total/LR zeros, 24 LE zeros for CMC/KL, and correct unused None values; **NOT RUN** |
| N3.1 visual probes / unchanged visual modules | **Previously verified**, sufficient for historical visual behavior and N1's narrowly accepted scope | New visual-only sweeps are **optional diagnostics outside primary**; each future full cell still needs exact preservation and no disconnection |
| G2 complete variant pairing and full 508/550-parameter gradients | **Requires new variant-specific verification** | All 18 primary cells NOT RUN; no legacy eight-failure waiver |
| NG2 fixtures/full initial states and variant-specific integration | **Requires new materialization and verification** | NOT MATERIALIZED / NOT RUN; local LD checkpoints cannot substitute for whole-model state |
| LR token-category parameter-gradient decomposition, isolated tied-vocabulary path probes | **Outstanding separate study scope**, not supplied by scalar category statistics | Outside the 57-cell primary/control schedule. Future approval must enumerate these; do not claim complete A1.1 study coverage |

## Evidence boundaries

The historical A1.1 review preserves 54 research cases and its independent ownership/parity reports. D02B1 reads and hashes those records; it does not rerun them. The accepted [A1.1 snapshot](phase2d2_a11_acceptance_snapshot.md) and [incident closure](phase2d2_a11_incident_closure.md) preserve the unauthorized legacy VD-T12 step. No historical approval is inferred from restoration, and the earlier narrowly scoped MVE-T10 exception does not transfer.

The [N3.1 review](phase2d2_n31_heldout_validation.md) retains eight actual failed assertions, including CMC chain, full gradient additivity and all six A0 G2 fixtures. The [D02A report](phase2d2_d02a_numerical_pilot_report.md) retains both normalized-variant improvements and regressions, weak gradients and key-bias cancellation residuals. None is reclassified by this implementation.

The new independent array tests establish only streaming arithmetic, bookkeeping, recovery, strict schemas and refusal behavior. Their PASS cannot fill a model-specific prerequisite. GPU, B8, finite optimizer updates, repeated-step stability, convergence and accuracy remain unverified and unauthorized.
