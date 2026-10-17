# D02B2 fixture and oracle matrix

2026-10-16. **Preparation only. All new A0/P3/Q3 numerical cells are NOT RUN.** Historical A0 evidence remains historical and does not fill P3/Q3 results.

## Exact input lineage

| Fixture | Golden prompt rows, in order | Original visual recipe | Current preparation |
| --- | --- | --- | --- |
| F0 | 1, 5 | `torch.randn(2,1,512, generator=seed341) * 0.1`, FP32 | Literal historical values, hash checked |
| F1 | 0, 4 | Same operation, seed 642 | Literal historical values, hash checked |
| F2 | 8, 9 | Accepted FP32 MVE+VD fused visual; random RGB batch `[2,3,224,224]`, seed 412 | Literal historical fused values; **no MVE/VD execution** |

Source: `tools/review_phase2c2_n2.py:fixture`; original rows are in `tests/foundation/fixtures/golden_tokens.json`. Existing paired records are `artifacts/phase2d2_n31/n2/fixture{0,1,2}_{float32,float64}_threads4.json`. Both precision records have identical FP32 fixture descriptions and original initialization hashes. Those records were read, not rerun or modified.

| Fixture | int64 token-byte SHA-256 | FP32 visual-byte SHA-256 |
| --- | --- | --- |
| F0 | `2cf0f655563da2cffecd51b59d127031772322f0fb5ba08c1d33d074d69810d5` | `f8cf91ba3c84f6d0d2c9be4964a2c0489980c412229c8dc6f3e5bd510861f69f` |
| F1 | `a860a1bd09980157ecfb7f0afd4c629238cf82348c801811adfb4621cea304d9` | `68cd54f1fdae7602eb57ae0e6eb821c018fadf2e5fef55c927c040293033e10d` |
| F2 | `be9475b3515453a299929a95fd160bbf08d7a997ff9ba5b8fe9a4f9bfa86a1cf` | `27a62aad20c5c1a3cc0abd2c537137a356a53c13460597e886afcc71f18a838e` |

`src/mfvlr_research/d02b2/frozen_inputs.json` SHA-256: `3e4bc2ab73652b3c62d246e4c61edd5c12f0a56e84c5968bde2042ed199bf0d2`. This is an input/metadata transcription, not a model checkpoint or new scientific result. Its historical source hashes are included; old “PROPOSED” record labels are not overwritten.

## Prepared future cell identities

Each row below expands to A0, P3 and Q3, in that order. Identity format: `D02B2-N2-F<index>-<variant>-<precision>`. The complete machine-readable matrix has **18 rows**, all NOT RUN. Registration is preparation, not authorization to execute this schedule.

| Fixture / precision | Variants | Batch pairings | Capture coverage | New status |
| --- | --- | --- | --- | --- |
| F0 FP32 | A0/P3/Q3 | B2, B1 row 0, B1 row 1 | 27 core +12 detailed | NOT RUN |
| F0 promoted FP64 | A0/P3/Q3 | Same | Same | NOT RUN |
| F1 FP32 | A0/P3/Q3 | Same | 27 core | NOT RUN |
| F1 promoted FP64 | A0/P3/Q3 | Same | 27 core | NOT RUN |
| F2 FP32 | A0/P3/Q3 | Same | 27 core | NOT RUN |
| F2 promoted FP64 | A0/P3/Q3 | Same | 27 core | NOT RUN |

All cells pin the recorded Windows CPU cohort: Python 3.14.3, torch 2.12.1+cpu, commit `7269437d655783a26cba32aa88195b741ff496aa`, intra/inter-op 4/4, deterministic algorithms, highest matmul precision, MKLDNN enabled, and complete recorded build/environment metadata. A different machine/version/backend is not a silent replacement. This is a future required cohort, not a fresh torch backend measurement. One-thread F0, GPU and B8 are outside this prepared matrix.

## Controls and their provenance

| Control | Original A0 evidence | P3/Q3 requirement | D02B2 verification basis |
| --- | --- | --- | --- |
| Same initialized weights across fixtures/B1/B2 | Original seed-0 N2 hashes; accepted test model reused | Same common FP32 reference, private frozen-builder candidate; actual candidate hash still needed | Historical metadata and mock parity checks |
| Promoted FP64 | Original deep-copy test and recorded workers | Deep-copy each variant's own FP32 state, exact promotion | Small independent arrays + mocked complete promotion receipts; no model promotion |
| B1/B2 token/visual rows | Original helper and T17 | Same exact row order/slices for both samples | Original literal bytes and golden fixtures checked |
| Exact fixed-batch independence | Original T17 changed-sample-1 test and richer both-direction review | Preserve mutations and unchanged-row checks | Independent polynomial controls + complete metadata rejection tests |
| Exact permutation | Original T17, 27 core captures | Same `[1,0]` permutation/inverse indexing | Small independent arrays, no model forward |
| Both derivative directions | Original genuine lookup/visual probes | Both directions, no fabricated lookup leaf, own positive controls | Analytical block-diagonal Jacobian and negative metadata controls |
| Full FP32 gradient participation | Original review B2 probe; separate T16 B1/B2 on F0 | Same probe and full 387/429 coverage, connected VIM zeros | Mocked parameter records, weak gradients and cancellation preserved |
| Ordinary Q/K / key-bias behavior | Existing N2 parameter norms; N3/D02A local oracles | Retain weak values/residuals; no local derivative threshold waiver | Record-category tests; **no new ordinary-attention numerical oracle run** |
| LN placement and initialization | Frozen A1.1 builder and original independent LN oracle evidence | Exact 21 added norms, correct pre/post placement, common state unchanged | Static AST comparison, pinned source, roster arithmetic and negative controls |
| All stages / projection | Original helper 27/39 captures; projection-only review diagnostic | No omitted LE/LD/global/logit outputs | Schema checked against original shape records |

The current preparation's FP32 complete-record schema includes the original **B2** full-backward and bidirectional derivative review probes. It does not claim fresh B1 full-backward evidence. The separate accepted T16 F0 B1/B2 tests remain unchanged; any variant-specific live repetition must be explicitly named in the execution handoff. Full FP64 parameter-gradient sweeps are not silently added: original N2 full-backward/cross-derivative workers ran in FP32. Promoted FP64's mandatory scope is cross-batch output agreement.

Projection-only and FP64 fixed-input LD perturbation studies remain historical supplemental diagnostics from the original review worker. They are not automatically dispatched by this preparation, not newly measured, and not substituted for mandatory checks. A future request must explicitly enumerate them if reruns are wanted. No new precision, seed, perturbation scale or cotangent sweep is introduced.

## Oracle and prerequisite reconciliation

Historical A0 F0/F1/F2 records are sufficient for their original recorded claims; they are not P3/Q3 PASS results. A1.1/D02A primitive LN and ordinary-attention controls support the unchanged primitive definitions at their recorded fixtures. They do not prove normalized full-language sample independence or FP64 agreement. New P3/Q3 N2 observations remain **required and NOT RUN** before claiming the original pre-G2 sequence is satisfied.

The D02B0 G2-first proposal refers to the first full-six-loss partition. It does not override this outstanding prerequisite. D02B2 only prepares it; B0-D03, NG2, B0-D08 and the broader A1.1 schedule remain disabled. See [D02B1 prerequisite reconciliation](phase2d2_d02b1_oracle_prerequisites.md).

Historical numerical failures, including all eight A0 integration assertions, D02A weak-gradient failures and Q3 FP64 local failures, remain unchanged. A matched fixture or valid metadata record is not evidence of a successful new numerical run.
