# Phase 2D.2-A1.1 LayerNorm erratum

The user conditionally approved A1-D01, requiring an unambiguous correction before research implementation. This erratum supersedes only the LayerNorm formulation in the locked A1.0 preregistration. A1.0 remains byte-for-byte preserved. No experimental factor, fixture, threshold or schedule has changed.

The canonical definition is:

`N(x) = gamma * (x - mean(x)) / sqrt(variance(x) + 1e-5) + beta`

Mean and **population variance** reduce the final 512 features independently for each sample/token. Beta is added **after** the division and affine multiplication; it is not inside the denominator. Gamma and beta are separate trainable parameters in each norm, initialized to exactly one and zero. None is shared. Initialization uses constants, without random draws.

The corrected [A1.1 preregistration](phase2d2_a1_ablation_preregistration_v1_1.md) replaces the ambiguous displayed expression with separate mean/variance definitions and explicitly grouped affine output. The original [A1.0](phase2d2_a1_ablation_preregistration.md) remains historical evidence. Exact SHA-256 values and the supersession relationship are recorded in [preregistration_versions.json](../artifacts/phase2d2_a11/preregistration_versions.json). The A1.0 hash is `0989e19fff9053d813a3f65f547ff74a09a412552b170d3ef484d27f90d87230`.

These are **ASSUMED engineering research choices**, not recovered author details. Paper Section III-D, p.7, Eqs.(12)–(16) does not specify these added LD norms. The approved singleton VIM and ordinary attention definitions remain unchanged.

| Decision | A1.1 authorization |
|---|---|
| A1-D01 | APPROVED: isolated A0/P3/Q3 implementation and controlled module correctness tests, after this correction |
| A1-D02 | BLOCKED: full preregistered experiments require separate approval |
| A1-D03 / I1 | UNAPPROVED: no initialization transformation implemented |
| A1-D04 / R2 | UNAPPROVED: no double-residual variant implemented |
| Production replacement, policy amendments, optimizer/training | UNAPPROVED |

Phase 2D.2 remains **BLOCKED**. Unit correctness cannot establish numerical improvement or training readiness. The version notice in A1.1 explains why unchanged historical proposed-status statements remain in its schedule.
