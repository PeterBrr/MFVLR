# Read-only F0/F1/F2 scientific comparison

All values come from existing records; no F0/F1 computation was rerun. F0 has 39 capture stages, including 12 detailed stages; F1/F2 have 27. Common-core columns use exactly the same 27 stage names and both samples. Full columns preserve every original failure. Different fixtures have different prompt and visual bytes, so cross-fixture totals are not controlled architecture comparisons.

| Fixture | Variant | Precision | Full failed / comparisons | Core failed / 54 | Full outside | Core outside |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| F0 | A0 | float32 | 44/78 | 44 | 7894336 | 7894336 |
| F0 | P3 | float32 | 44/78 | 44 | 991971 | 991971 |
| F0 | Q3 | float32 | 44/78 | 44 | 89532 | 89532 |
| F0 | A0 | float64 | 0/78 | 0 | 0 | 0 |
| F0 | P3 | float64 | 0/78 | 0 | 0 | 0 |
| F0 | Q3 | float64 | 0/78 | 0 | 0 | 0 |
| F1 | A0 | float32 | 44/54 | 44 | 6394218 | 6394218 |
| F1 | P3 | float32 | 44/54 | 44 | 983289 | 983289 |
| F1 | Q3 | float32 | 44/54 | 44 | 88423 | 88423 |
| F1 | A0 | float64 | 0/54 | 0 | 0 | 0 |
| F1 | P3 | float64 | 0/54 | 0 | 0 | 0 |
| F1 | Q3 | float64 | 0/54 | 0 | 0 | 0 |
| F2 | A0 | float32 | 46/54 | 46 | 14948266 | 14948266 |
| F2 | P3 | float32 | 46/54 | 46 | 1344978 | 1344978 |
| F2 | Q3 | float32 | 46/54 | 46 | 163449 | 163449 |
| F2 | A0 | float64 | 0/54 | 0 | 0 | 0 |
| F2 | P3 | float64 | 0/54 | 0 | 0 | 0 |
| F2 | Q3 | float64 | 0/54 | 0 | 0 | 0 |

Paired core sensitivity counts against A0 within each identical fixture and precision (lower / higher / equal; undefined shown separately in JSON):

| Fixture | Variant | Precision | Max absolute | Relative L2 | Outside proportion |
| --- | --- | --- | --- | --- | --- |
| F0 | P3 | float32 | 16 / 2 / 36 | 16 / 2 / 36 | 16 / 2 / 36 |
| F0 | Q3 | float32 | 18 / 0 / 36 | 16 / 2 / 36 | 18 / 0 / 36 |
| F0 | P3 | float64 | 16 / 0 / 38 | 16 / 2 / 36 | 0 / 0 / 54 |
| F0 | Q3 | float64 | 18 / 0 / 36 | 16 / 2 / 36 | 0 / 0 / 54 |
| F1 | P3 | float32 | 16 / 2 / 36 | 16 / 2 / 36 | 17 / 1 / 36 |
| F1 | Q3 | float32 | 18 / 0 / 36 | 16 / 2 / 36 | 18 / 0 / 36 |
| F1 | P3 | float64 | 16 / 2 / 36 | 16 / 2 / 36 | 0 / 0 / 54 |
| F1 | Q3 | float64 | 18 / 0 / 36 | 16 / 2 / 36 | 0 / 0 / 54 |
| F2 | P3 | float32 | 16 / 1 / 37 | 16 / 2 / 36 | 18 / 0 / 36 |
| F2 | Q3 | float32 | 18 / 0 / 36 | 18 / 0 / 36 | 18 / 0 / 36 |
| F2 | P3 | float64 | 16 / 1 / 37 | 16 / 2 / 36 | 0 / 0 / 54 |
| F2 | Q3 | float64 | 18 / 0 / 36 | 18 / 0 / 36 | 0 / 0 / 54 |

Lower and higher error counts preserve improvements and regressions without choosing favorable stages. Full per-stage data, including weak values, remains available in the included core records and each original raw table. Neither lower activation magnitudes nor lower total failure counts establish superiority, harmless drift, training stability or overall N2 acceptance. Phase 2D.2 remains BLOCKED.

## Activation scale context

F2 uses the recorded visual-path output; F0/F1 use their own historical visual fixtures. These input scales differ and are not interchangeable. The following values are read from original B2 FP32 records; they do not measure training stability.

| Fixture | Variant | Final LD RMS | Final LD max | Logits max |
| --- | --- | ---: | ---: | ---: |
| F0 | A0 | 254.91487 | 865.81781 | 527.73987 |
| F0 | P3 | 24.063414 | 86.932816 | 45.865864 |
| F0 | Q3 | 1.0051123 | 3.4903042 | 1.9408846 |
| F1 | A0 | 258.56355 | 980.05035 | 520.92627 |
| F1 | P3 | 24.424105 | 86.095276 | 46.123116 |
| F1 | Q3 | 1.0066631 | 3.4153011 | 1.8765646 |
| F2 | A0 | 1242.3115 | 4411.2129 | 2645.781 |
| F2 | P3 | 114.80462 | 353.49515 | 241.2644 |
| F2 | Q3 | 4.233046 | 14.191059 | 8.0339947 |
