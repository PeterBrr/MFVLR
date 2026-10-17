# D02B5 controlled F2 full-language numerical pilot

2026-10-16, Asia/Saigon. **Phase 2D.2 remains BLOCKED.** Only the six explicitly authorized F2 FLT-only cells were eligible. Frozen D02B1/D02B2 dispatchers, production models, objectives, fixtures and original tests remain unchanged. No F0/F1, six-loss G2, NG2, optimizer, training, G3/G4 or benchmark was executed.

## Actual execution

| Cell | Execution | Numerical comparison | Outside coordinates | Wall seconds | Sampled peak RSS MiB |
| --- | --- | --- | ---: | ---: | ---: |
| D02B5-F2-A0-float32 | COMPLETE | FAIL | 14948266 | 42.11 | 2268.4 |
| D02B5-F2-P3-float32 | COMPLETE | FAIL | 1344978 | 43.11 | 2301.1 |
| D02B5-F2-Q3-float32 | COMPLETE | FAIL | 163449 | 41.36 | 2299.3 |
| D02B5-F2-A0-float64 | COMPLETE | PASS | 0 | 54.38 | 3564.7 |
| D02B5-F2-P3-float64 | COMPLETE | PASS | 0 | 53.91 | 3633.4 |
| D02B5-F2-Q3-float64 | COMPLETE | PASS | 0 | 51.40 | 3625.9 |

Execution completion is distinct from numerical PASS. FP32 cross-batch failed comparisons remain genuine failures under the original N2 diagnostic scope. Any promoted FP64 disagreement or structural failure stops dependent cells. NOT RUN rows are not successes. See `execution_matrix.json` and each `result.json` for precise failure traces and status.

## Provenance and live checks

A new versioned D02B5 layer allows only the exact six cell IDs. A single exclusive coordinator owns the sequence using PID plus process creation time. A fresh worker per cell independently initializes the canonical seed-0 CPU FP32 reference, verifies its original N2 state hash, then calls the frozen A1.1 builder. All common values and disjoint ownership are verified by the frozen parity checker. The source reference is released before captures. Private candidates retain 387/429 parameter tensors, 12 LE blocks, seven LD blocks, 19 VIM roles and tied vocabulary; counts and per-tensor hashes are in each live inventory.

FP64 cells deep-copy the actual candidate and compare every promoted tensor, chunk by chunk, with its actual FP32 source converted to FP64. They do not trust a supplied expected digest or merely round-trip to FP32. Original F2 token/visual bytes are loaded from the hash-pinned historical input bundle. Before every model forward, a live pre-hook checks parameter bytes/identities and runtime backend and writes an exclusive state/input/source/backend seal. Forward values are observed under no-grad for numerical captures; hooks are removed and captures checked for graph retention.

Each completed cell records all 27 original F2 core captures (no additional F0 detailed captures), both B1 row slices versus B2, exact fixed-batch independence in both directions, exact permutation and genuine visual/embedding-output cross-sample autograd derivatives with connected own-sample positive controls. The user-authorized derivative isolation checks are performed for both precisions; no FP64 full parameter-gradient sweep is added. FP32 cells additionally execute full probe backward at B1 and B2. Every parameter gradient is recorded with finite norms, maxima, nonzero counts and hashes; all 38 singleton VIM Q/K gradients must be allocated exact zeros. Ordinary Q/K weights and ordinary key-bias residuals remain unfiltered in those records. No ordinary-attention derivative accuracy oracle or training loss is substituted for the original N2 probe.

The probe is `mean(logits^2)+mean(decoder_hidden^2)+mean(global_language^2)`. Cross-sample probes use the corresponding sample only and differentiate the actual `output.low` embedding boundary and connected visual leaf. The seven original positive-control parameter roles and visual gradient remain mandatory. A state check after all probes verifies unchanged parameters/ownership.

## Policy and resources

Original comparison constants remain FP32 atol=1e-6/rtol=1e-5 and promoted FP64 atol=1e-10/rtol=1e-8. Fixed-batch equality, permutation and cross-sample zeros are exact. N1 is not used. No policy extends to additivity, six losses or another category. `live_results.junit.xml` represents raw observed outcomes, including failed FP32 diagnostic comparisons; it is not a pytest suite claiming those discrepancies passed. `preflight_junit.xml` is the separately executed isolated 25-case static/mock pytest suite.

The operational launch reservation is 5 GiB available RAM and 8 GiB free disk for either precision. This conservative engineering safeguard is not a measured guarantee of capacity. Insufficient resources are recorded WAIT/NOT RUN. Workers run sequentially; a second in-worker resource/ownership check occurs before importing torch. Available RAM, disk, PID/creation, runtime build/thread metadata, process time and sampled RSS are preserved. Peak RSS is sampled, not a guaranteed instantaneous maximum. No GPU/B8 capacity claim is made.

All outputs use exclusive paths. No interrupted attempt is overwritten or auto-replayed. Unexpected interruption has UNKNOWN cause unless evidence identifies it. Source preservation uses before/after SHA-256 inventories. The compact package includes complete source, isolated tests, original fixture/source references, raw cell records/logs, actual failure traces, JUnit, manifests and preservation audits; old large archives are referenced by hashes rather than duplicated.

## Interpretation and limits

These are bounded no-update F2 observations. `activation_scales.json` retains every capture's RMS/max/norm; `paired_sensitivity.json` preserves lower, higher, equal and undefined metrics for every matched stage/sample/precision, rather than selecting favorable layers. Compare each variant with its own B1 reference: the variants represent different functions. Do not infer architecture superiority from activation magnitudes, total failed comparisons or global relative L2. FP32 diagnostic success cannot establish harmless numerical drift. Full numerical agreement, repeated-step stability, convergence, accuracy and author architecture fidelity are not established. A mandatory stop leaves later authorized cells NOT RUN pending diagnosis and review; authorization does not permit weakening thresholds or silently retrying modified conditions.

Stop for user review. Phase 2D.2 remains BLOCKED.

## Final verification findings

Six of six cells completed. All 162 promoted-FP64 comparisons passed. FP32 retained 138 failed stage/sample comparisons and 16,456,693 outside-tolerance coordinates. Raw live JUnit has 330 entries, 192 passing and 138 failing, with no skips. These are genuine FP32 diagnostic failures, distinct from 25/25 actual isolated pytest passes.

All 108 A0 historical F2 stage/sample metric records matched exactly, as did both precisions' activation statistics. All 2,125 baseline files were unchanged. Actual pre-forward input hashes were reconciled with the exact frozen F2 bytes, row slices, permutation and prescribed independence perturbations; every promotion record was reconciled with both live inventories.

F2 loaded golden prompt rows 8/9 and the stored FP32 fused visual tensor. No MVE/VD forward or input regeneration occurred. Token SHA-256: `be9475b3515453a299929a95fd160bbf08d7a997ff9ba5b8fe9a4f9bfa86a1cf`; visual SHA-256: `27a62aad20c5c1a3cc0abd2c537137a356a53c13460597e886afcc71f18a838e`.

| Variant | B2 probe | Full parameter gradient L2 | Final LD RMS | Final LD max | Logits max |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 | 1859507.8 | 62397309 | 1242.3115 | 4411.2129 | 2645.781 |
| P3 | 17904.137 | 274440.98 | 114.80462 | 353.49515 | 241.2644 |
| Q3 | 2027.0868 | 26601.073 | 4.233046 | 14.191059 | 8.0339947 |

All 387 A0 / 429 P3/Q3 parameter gradients were allocated and finite at B1 and B2. The 38 singleton VIM Q/K tensors were exactly zero in each sweep: 228 tensor checks in total. Both genuine cross-sample derivative directions passed in all six cells with connected positive controls. Ordinary Q/K and key-bias residuals remain unfiltered. These are quadratic-probe gradients, not six-loss training results or new ordinary-attention accuracy oracle results.

All workers exited 0 and passed the operational resource gates; exit code does not imply numerical acceptance. No numerical attempt was interrupted or replayed. An initial UTF-8 decoding issue while creating the report helper was fixed before aggregation, without changing frozen source or numerical data. GPU/B8 capacity is NOT RUN/NOT VERIFIED.

The separate F0/F1/F2 summary uses the common 27-stage subset alongside original totals. The package includes the source records for that read-only comparison. No F0/F1 cell was rerun. Overall N2 acceptance and training stability are not established. Phase 2D.2 remains BLOCKED.
