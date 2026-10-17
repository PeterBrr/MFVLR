# D02B6 N2 evidence closure

2026-10-16, Asia/Saigon. **Read-only closure; Phase 2D.2 remains BLOCKED.**

All 18 registered cells completed. The following statuses audit existing observations; none was rerun in D02B6. Execution and structural checks do not convert FP32 diagnostic FAILs into PASS. The original tolerances remain FP32 1e-6/1e-5 and promoted FP64 1e-10/1e-8. N1 is not transferred.

| Cell | Execution | Exact/structural | Comparisons | Numerical FAILs | Outside coordinates | Promoted FP64 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| D02B3-F0-A0-float32 | COMPLETE | PASS, record audit | 78 | 44 | 7894336 | NOT APPLICABLE |
| D02B3-F0-P3-float32 | COMPLETE | PASS, record audit | 78 | 44 | 991971 | NOT APPLICABLE |
| D02B3-F0-Q3-float32 | COMPLETE | PASS, record audit | 78 | 44 | 89532 | NOT APPLICABLE |
| D02B3-F0-A0-float64 | COMPLETE | PASS, record audit | 78 | 0 | 0 | PASS |
| D02B3-F0-P3-float64 | COMPLETE | PASS, record audit | 78 | 0 | 0 | PASS |
| D02B3-F0-Q3-float64 | COMPLETE | PASS, record audit | 78 | 0 | 0 | PASS |
| D02B4-F1-A0-float32 | COMPLETE | PASS, record audit | 54 | 44 | 6394218 | NOT APPLICABLE |
| D02B4-F1-P3-float32 | COMPLETE | PASS, record audit | 54 | 44 | 983289 | NOT APPLICABLE |
| D02B4-F1-Q3-float32 | COMPLETE | PASS, record audit | 54 | 44 | 88423 | NOT APPLICABLE |
| D02B4-F1-A0-float64 | COMPLETE | PASS, record audit | 54 | 0 | 0 | PASS |
| D02B4-F1-P3-float64 | COMPLETE | PASS, record audit | 54 | 0 | 0 | PASS |
| D02B4-F1-Q3-float64 | COMPLETE | PASS, record audit | 54 | 0 | 0 | PASS |
| D02B5-F2-A0-float32 | COMPLETE | PASS, record audit | 54 | 46 | 14948266 | NOT APPLICABLE |
| D02B5-F2-P3-float32 | COMPLETE | PASS, record audit | 54 | 46 | 1344978 | NOT APPLICABLE |
| D02B5-F2-Q3-float32 | COMPLETE | PASS, record audit | 54 | 46 | 163449 | NOT APPLICABLE |
| D02B5-F2-A0-float64 | COMPLETE | PASS, record audit | 54 | 0 | 0 | PASS |
| D02B5-F2-P3-float64 | COMPLETE | PASS, record audit | 54 | 0 | 0 | PASS |
| D02B5-F2-Q3-float64 | COMPLETE | PASS, record audit | 54 | 0 | 0 | PASS |

F0 records 39 captures (27 core + 12 detailed), while F1 and F2 each record 27. All nine promoted-FP64 cells have zero outside-tolerance comparisons; all nine FP32 cells retain failed comparisons. Complete per-cell hashes, actual before/after inventories, derivative records and allocation/zero evidence are indexed in the JSON closure matrix. The compact package includes those source records.

## Prerequisites demonstrated and outstanding

The former D02B1 outstanding P3/Q3 full-language N2 prerequisite is now demonstrated for F0/F1/F2: seed-0 common FP32 initialization, privately owned variants, exact deep-copy promotion, both B1 slices, exact fixed-batch independence/permutation, both genuine cross-sample derivative directions and connected own-sample controls. All 18 cells preserved state and tied vocabulary. FP32 B1/B2 full gradients were finite and allocated, including exact singleton VIM Q/K zeros. FP64 full parameter-backward sweeps were not part of N2 and remain NOT RUN. Weak ordinary Q/K and key-bias residuals remain observations, not universal accuracy proofs.

A1.1/D02A LN forward/VJP and ordinary-attention primitive controls remain previously verified for their named local scope. The 414-job local schedule remains complete with every original failure retained. D02B6 does not rerun or strengthen those controls. Original A0 N3 same-forward and CMC-chain evidence still includes genuine failures. P3/Q3 same-forward six-loss controls are NOT RUN, separately scoped; optional new unchanged-path CMC/visual controls are NOT RUN. Backend controls, NG2, LR-category parameter-gradient decomposition and isolated tied-vocabulary path probes remain separately authorized/outstanding. Scalar LR category statistics cannot satisfy those decomposition requirements.

## Boundary of closure

The pre-G2 full-language N2 observation gap is closed under the already scoped diagnostic policy, not by a new waiver. The historical eight failing assertions (including the six A0 G2 fixtures) remain unresolved and unchanged. There is no full-six-loss numerical acceptance, optimizer/update stability, convergence, capacity or accuracy claim. All G01–G06 × A0/P3/Q3 cells remain NOT RUN. B0-D03 and B0-D08 remain disabled pending separate approval.
