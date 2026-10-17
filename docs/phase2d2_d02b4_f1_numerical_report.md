# D02B4 controlled F1 full-language numerical pilot

2026-10-16, Asia/Saigon. **Phase 2D.2 remains BLOCKED.** Only the six explicitly authorized F1 FLT-only cells were eligible. Frozen D02B1/D02B2 dispatchers, production models, objectives, fixtures and original tests remain unchanged. No F0/F2, six-loss G2, NG2, optimizer, training, G3/G4 or benchmark was executed.

## Actual execution

| Cell | Execution | Numerical comparison | Outside coordinates | Wall seconds | Sampled peak RSS MiB |
| --- | --- | --- | ---: | ---: | ---: |
| D02B4-F1-A0-float32 | COMPLETE | FAIL | 6394218 | 49.65 | 2267.1 |
| D02B4-F1-P3-float32 | COMPLETE | FAIL | 983289 | 45.63 | 2301.6 |
| D02B4-F1-Q3-float32 | COMPLETE | FAIL | 88423 | 44.12 | 2297.8 |
| D02B4-F1-A0-float64 | COMPLETE | PASS | 0 | 49.87 | 3561.9 |
| D02B4-F1-P3-float64 | COMPLETE | PASS | 0 | 54.90 | 3632.2 |
| D02B4-F1-Q3-float64 | COMPLETE | PASS | 0 | 53.65 | 3626.8 |

Execution completion is distinct from numerical PASS. FP32 cross-batch failed comparisons remain genuine failures under the original N2 diagnostic scope. Any promoted FP64 disagreement or structural failure stops dependent cells. NOT RUN rows are not successes. See `execution_matrix.json` and each `result.json` for precise failure traces and status.

## Provenance and live checks

A new versioned D02B4 layer allows only the exact six cell IDs. A single exclusive coordinator owns the sequence using PID plus process creation time. A fresh worker per cell independently initializes the canonical seed-0 CPU FP32 reference, verifies its original N2 state hash, then calls the frozen A1.1 builder. All common values and disjoint ownership are verified by the frozen parity checker. The source reference is released before captures. Private candidates retain 387/429 parameter tensors, 12 LE blocks, seven LD blocks, 19 VIM roles and tied vocabulary; counts and per-tensor hashes are in each live inventory.

FP64 cells deep-copy the actual candidate and compare every promoted tensor, chunk by chunk, with its actual FP32 source converted to FP64. They do not trust a supplied expected digest or merely round-trip to FP32. Original F1 token/visual bytes are loaded from the hash-pinned historical input bundle. Before every model forward, a live pre-hook checks parameter bytes/identities and runtime backend and writes an exclusive state/input/source/backend seal. Forward values are observed under no-grad for numerical captures; hooks are removed and captures checked for graph retention.

Each completed cell records all 27 original F1 core captures (no additional F0 detailed captures), both B1 row slices versus B2, exact fixed-batch independence in both directions, exact permutation and genuine visual/embedding-output cross-sample autograd derivatives with connected own-sample positive controls. The user-authorized derivative isolation checks are performed for both precisions; no FP64 full parameter-gradient sweep is added. FP32 cells additionally execute full probe backward at B1 and B2. Every parameter gradient is recorded with finite norms, maxima, nonzero counts and hashes; all 38 singleton VIM Q/K gradients must be allocated exact zeros. Ordinary Q/K weights and ordinary key-bias residuals remain unfiltered in those records. No ordinary-attention derivative accuracy oracle or training loss is substituted for the original N2 probe.

The probe is `mean(logits^2)+mean(decoder_hidden^2)+mean(global_language^2)`. Cross-sample probes use the corresponding sample only and differentiate the actual `output.low` embedding boundary and connected visual leaf. The seven original positive-control parameter roles and visual gradient remain mandatory. A state check after all probes verifies unchanged parameters/ownership.

## Policy and resources

Original comparison constants remain FP32 atol=1e-6/rtol=1e-5 and promoted FP64 atol=1e-10/rtol=1e-8. Fixed-batch equality, permutation and cross-sample zeros are exact. N1 is not used. No policy extends to additivity, six losses or another category. `live_results.junit.xml` represents raw observed outcomes, including failed FP32 diagnostic comparisons; it is not a pytest suite claiming those discrepancies passed. `preflight_junit.xml` is the separately executed isolated 24-case static/mock pytest suite.

The operational launch reservation is 5 GiB available RAM and 8 GiB free disk for either precision. This conservative engineering safeguard is not a measured guarantee of capacity. Insufficient resources are recorded WAIT/NOT RUN. Workers run sequentially; a second in-worker resource/ownership check occurs before importing torch. Available RAM, disk, PID/creation, runtime build/thread metadata, process time and sampled RSS are preserved. Peak RSS is sampled, not a guaranteed instantaneous maximum. No GPU/B8 capacity claim is made.

All outputs use exclusive paths. No interrupted attempt is overwritten or auto-replayed. Unexpected interruption has UNKNOWN cause unless evidence identifies it. Source preservation uses before/after SHA-256 inventories. The compact package includes complete source, isolated tests, original fixture/source references, raw cell records/logs, actual failure traces, JUnit, manifests and preservation audits; old large archives are referenced by hashes rather than duplicated.

## Interpretation and limits

These are bounded no-update F1 observations. `activation_scales.json` retains every capture's RMS/max/norm; `paired_sensitivity.json` preserves lower, higher, equal and undefined metrics for every matched stage/sample/precision, rather than selecting favorable layers. Compare each variant with its own B1 reference: the variants represent different functions. Do not infer architecture superiority from activation magnitudes, total failed comparisons or global relative L2. FP32 diagnostic success cannot establish harmless numerical drift. Full numerical agreement, repeated-step stability, convergence, accuracy and author architecture fidelity are not established. A mandatory stop leaves later authorized cells NOT RUN pending diagnosis and review; authorization does not permit weakening thresholds or silently retrying modified conditions.

Stop for user review. Phase 2D.2 remains BLOCKED.

## Final observed results

All six cells completed with mandatory checks passing. All 162 promoted-FP64 stage/sample comparisons passed. Each FP32 variant retained 44/54 failed comparisons: A0 6,394,218, P3 983,289 and Q3 88,423 outside-tolerance coordinates. The 7,465,930 total is a count of compared coordinates across recorded stages/samples, not unique model elements.

Raw live JUnit contains 330 entries: 198 passing and 132 genuine FP32 diagnostic failures, zero skips. This is separate from 24/24 actual isolated pytest cases. No numerical failure has been relabeled PASS.

The read-only historical comparison found exact equality for every original A0 F1 cross-batch metric and activation statistic in both precisions. The preservation audit verified 1,910 pre-existing files with zero mismatches, including frozen D02B3 evidence.

| Variant | B2 probe | Full parameter gradient L2 | LD block 6 RMS | LD block 6 max | Logits max |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 | 80639.422 | 3138625.6 | 258.56355 | 980.05035 | 520.92627 |
| P3 | 798.02643 | 17576.672 | 24.424105 | 86.095276 | 46.123116 |
| Q3 | 79.643028 | 1549.281 | 1.0066631 | 3.4153011 | 1.8765646 |

These quadratic-probe gradients are not six-loss training measurements. All 387/429 parameter gradients were allocated and finite at B1 and B2. Every one of 38 singleton VIM Q/K tensors was exactly zero in each sweep (228 tensor checks across six sweeps). Both cross-sample derivative directions passed for all six cells with positive connected own-sample controls. Ordinary Q/K weights and key-bias residuals are preserved in full, including weak values; these are observations, not independent derivative-accuracy oracle passes.

Paired FP32 sensitivity direction counts over all 54 matched sample/stage records per variant:

```json
{
  "P3": {
    "max_absolute_difference": {
      "equal": 36,
      "higher": 2,
      "lower": 16
    },
    "relative_l2_difference": {
      "equal": 36,
      "higher": 2,
      "lower": 16
    },
    "outside_proportion": {
      "equal": 36,
      "lower": 17,
      "higher": 1
    }
  },
  "Q3": {
    "max_absolute_difference": {
      "equal": 36,
      "lower": 18
    },
    "relative_l2_difference": {
      "equal": 36,
      "higher": 2,
      "lower": 16
    },
    "outside_proportion": {
      "equal": 36,
      "lower": 18
    }
  }
}
```

Lower sensitivity is not architecture superiority. All seven decoder blocks and all remaining captures are retained, including regressions. GPU and B8 were NOT RUN. No optimizer, full integration objective, F0 rerun or F2 cell was executed. The initial RAM wait is preserved in resource records; all six workers subsequently passed both launch gates.

Phase 2D.2 remains BLOCKED. F1 is complete for review; overall N2 acceptance is not claimed.
