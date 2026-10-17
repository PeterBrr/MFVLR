# D02B3 controlled F0 full-language numerical pilot

2026-10-16, Asia/Saigon. **Phase 2D.2 remains BLOCKED.** Only the six explicitly authorized F0 FLT-only cells were eligible. Frozen D02B1/D02B2 dispatchers, production models, objectives, fixtures and original tests remain unchanged. No F1/F2, six-loss G2, NG2, optimizer, training, G3/G4 or benchmark was executed.

## Actual execution

| Cell | Execution | Numerical comparison | Outside coordinates | Wall seconds | Sampled peak RSS MiB |
| --- | --- | --- | ---: | ---: | ---: |
| D02B3-F0-A0-float32 | COMPLETE | FAIL | 7894336 | 43.11 | 2274.3 |
| D02B3-F0-P3-float32 | COMPLETE | FAIL | 991971 | 46.15 | 2299.6 |
| D02B3-F0-Q3-float32 | COMPLETE | FAIL | 89532 | 60.48 | 2289.8 |
| D02B3-F0-A0-float64 | COMPLETE | PASS | 0 | 73.42 | 3557.0 |
| D02B3-F0-P3-float64 | COMPLETE | PASS | 0 | 68.09 | 3619.2 |
| D02B3-F0-Q3-float64 | COMPLETE | PASS | 0 | 57.47 | 3625.6 |

Execution completion is distinct from numerical PASS. FP32 cross-batch failed comparisons remain genuine failures under the original N2 diagnostic scope. Any promoted FP64 disagreement or structural failure stops dependent cells. NOT RUN rows are not successes. See `execution_matrix.json` and each `result.json` for precise failure traces and status.

All six cells completed with no mandatory hard stop, interruption, OOM or resource WAIT. The 468 stage/sample cross-batch comparisons contain **132 numerical failures**, all FP32: **44/78 per variant**. The 234 promoted-FP64 comparisons pass the unchanged tolerances. Counting the six mandatory cell outcomes as well, the generated raw-results JUnit contains **474 entries: 342 passing, 132 failing, zero skipped**. These failures are deliberately preserved; this is not an all-pass numerical suite. The separate implementation preflight pytest passed **22/22 cases**. None of the original regression tests was rerun.

| FP32 variant / sample | Logit max absolute difference | Outside proportion | Logit relative L2 |
| --- | ---: | ---: | ---: |
| A0 / 0 | 0.003082275391 | 20.8318% | 3.43608432e-6 |
| A0 / 1 | 0.005348205566 | 28.3829% | 6.28021536e-6 |
| P3 / 0 | 0.000039100647 | 2.73855% | 5.80572348e-7 |
| P3 / 1 | 0.000039100647 | 2.80540% | 5.80062540e-7 |
| Q3 / 0 | 0.000001937151 | 0.000545419% | 6.40217698e-7 |
| Q3 / 1 | 0.000001817942 | 0.00151797% | 6.64960901e-7 |

Both P3 and Q3 show lower FP32 relative-L2 drift than A0 in 16 matched capture/sample comparisons, equal values in 60, and **higher drift in two**: the first LD output for both samples. P3 first-block relative L2 is 4.93333e-7 / 5.01353e-7, Q3 is 5.42626e-7 / 5.54740e-7, versus A0 4.09756e-7 / 4.12852e-7. This retains a regression rather than selecting only later favorable stages. Many captures are shared or duplicate endpoints, so these are bookkeeping counts, not independent trials.

FP32 final LD max magnitudes are A0 865.81781, P3 86.93282 and Q3 3.49030. B2 probe parameter-gradient L2 norms are approximately 3,031,726.345, 16,445.382 and 1,474.573 respectively. These belong to different variant functions and do not establish training stability or a winner. All 38 VIM Q/K gradients are allocated zero in every B1/B2 full-backward probe. Complete weak ordinary Q/K and key-bias values are in `gradient_categories.json`, including tiny nonzero residuals.

The new A0 FP32 cross-batch metrics match every preserved original F0 metric exactly; activation statistics and aggregate parameter-gradient L2 also match exactly. This is a read-only comparison with existing evidence, recorded in `a0_historical_comparison.json`, not an additional execution. All six live evidence inventories passed the subsequent read-only completeness audit.

## Provenance and live checks

A new versioned D02B3 layer allows only the exact six cell IDs. A single exclusive coordinator owns the sequence using PID plus process creation time. A fresh worker per cell independently initializes the canonical seed-0 CPU FP32 reference, verifies its original N2 state hash, then calls the frozen A1.1 builder. All common values and disjoint ownership are verified by the frozen parity checker. The source reference is released before captures. Private candidates retain 387/429 parameter tensors, 12 LE blocks, seven LD blocks, 19 VIM roles and tied vocabulary; counts and per-tensor hashes are in each live inventory.

FP64 cells deep-copy the actual candidate and compare every promoted tensor, chunk by chunk, with its actual FP32 source converted to FP64. They do not trust a supplied expected digest or merely round-trip to FP32. Original F0 token/visual bytes are loaded from the hash-pinned historical input bundle. Before every model forward, a live pre-hook checks parameter bytes/identities and runtime backend and writes an exclusive state/input/source/backend seal. Forward values are observed under no-grad for numerical captures; hooks are removed and captures checked for graph retention.

Each completed cell records all 39 F0 captures (27 core plus 12 detailed), both B1 row slices versus B2, exact fixed-batch independence in both directions, exact permutation and genuine visual/embedding-output cross-sample autograd derivatives with connected own-sample positive controls. The user-authorized derivative isolation checks are performed for both precisions; no FP64 full parameter-gradient sweep is added. FP32 cells additionally execute full probe backward at B1 and B2. Every parameter gradient is recorded with finite norms, maxima, nonzero counts and hashes; all 38 singleton VIM Q/K gradients must be allocated exact zeros. Ordinary Q/K weights and ordinary key-bias residuals remain unfiltered in those records. No ordinary-attention derivative accuracy oracle or training loss is substituted for the original N2 probe.

The probe is `mean(logits^2)+mean(decoder_hidden^2)+mean(global_language^2)`. Cross-sample probes use the corresponding sample only and differentiate the actual `output.low` embedding boundary and connected visual leaf. The seven original positive-control parameter roles and visual gradient remain mandatory. A state check after all probes verifies unchanged parameters/ownership.

## Policy and resources

Original comparison constants remain FP32 atol=1e-6/rtol=1e-5 and promoted FP64 atol=1e-10/rtol=1e-8. Fixed-batch equality, permutation and cross-sample zeros are exact. N1 is not used. No policy extends to additivity, six losses or another category. `live_results.junit.xml` represents raw observed outcomes, including failed FP32 diagnostic comparisons; it is not a pytest suite claiming those discrepancies passed. `preflight_junit.xml` is the separately executed isolated 22-case static/mock pytest suite.

The operational launch reservation is 5 GiB available RAM and 8 GiB free disk for either precision. This conservative engineering safeguard is not a measured guarantee of capacity. Insufficient resources are recorded WAIT/NOT RUN. Workers run sequentially; a second in-worker resource/ownership check occurs before importing torch. Available RAM, disk, PID/creation, runtime build/thread metadata, process time and sampled RSS are preserved. Peak RSS is sampled, not a guaranteed instantaneous maximum. No GPU/B8 capacity claim is made.

All outputs use exclusive paths. No interrupted attempt is overwritten or auto-replayed. Unexpected interruption has UNKNOWN cause unless evidence identifies it. Source preservation uses before/after SHA-256 inventories. The compact package includes complete source, isolated tests, original fixture/source references, raw cell records/logs, actual failure traces, JUnit, manifests and preservation audits; old large archives are referenced by hashes rather than duplicated.

The after-audit matched **1,707/1,707 preserved files**, zero mismatches, including accepted production code, D02B1/D02B2 code and disabled dispatchers, original tests and historical archives. This run does not substitute for unrelated independent audits. The new driver manifest pins the actual worker/coordinator version used for execution; post-run aggregation is separately included in the package manifest.

## Interpretation and limits

These are bounded no-update F0 observations. `activation_scales.json` retains every capture's RMS/max/norm; `paired_sensitivity.json` preserves lower, higher, equal and undefined metrics for every matched stage/sample/precision, rather than selecting favorable layers. Compare each variant with its own B1 reference: the variants represent different functions. Do not infer architecture superiority from activation magnitudes, total failed comparisons or global relative L2. FP32 diagnostic success cannot establish harmless numerical drift. Full numerical agreement, repeated-step stability, convergence, accuracy and author architecture fidelity are not established. A mandatory stop leaves later authorized cells NOT RUN pending diagnosis and review; authorization does not permit weakening thresholds or silently retrying modified conditions.

Stop for user review. Phase 2D.2 remains BLOCKED.
