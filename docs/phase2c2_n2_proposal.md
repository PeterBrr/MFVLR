# N2 proposal: full-FLT cross-batch numerical verification

**Status: PROPOSED. Implementation: BLOCKED pending explicit user approval. Applied: NO. Phase 2C.2 acceptance: BLOCKED.** This proposal is an engineering verification policy for the approved baseline, not a published MFVLR requirement or confirmation of the authors' implementation.

## Decision requested

Approve replacing **only** `test_flt_t17_cross_batch_original_tolerance`'s universal FP32 cross-batch `assert_close` requirement with a **mandatory, complete numerical diagnostic**. Keep its existing FP32 `atol=1e-6, rtol=1e-5` values as diagnostic comparison thresholds. Record violations; do not make them disappear with a larger tolerance. Keep every other existing test and threshold unchanged, including exact fixed-batch independence and the narrow MVE N1 amendment.

Strengthen the batch-correctness evidence with three explicit T17 companion cases: fixed-shape batch permutation, both-direction exact-zero cross-sample gradients, and full-FLT promoted-weight FP64 cross-batch comparisons at unchanged FP64 tolerances. The existing analytical FP64 ordinary-attention and VIM tests remain independently mandatory.

Evidence and limitations are in the [numerical review](phase2c2_numerical_review.md). A first value-projection difference of approximately `2.98e-8` amplifies through the approved depth. Across three fixtures, exact fixed-batch and permutation checks pass; both directions of cross-sample visual/lookup derivatives are exactly zero; every full-FLT FP64 cross-batch stage passes its existing tolerance. The integration-scale FP32 fixture still reaches a logit difference of approximately 0.179 and high directional sensitivity. N2 does not classify those risks as solved.

## Proposed mandatory policy

| Requirement | Proposed acceptance behavior |
|---|---|
| Existing fixed-B=2 independence | Keep the current test byte-for-byte. Perturb the other sample's IDs/visual feature; unaffected outputs remain exactly equal; its visual gradient remains exactly zero. |
| Batch permutation | New explicit case jointly swaps IDs/Iv at fixed B=2, undoes the swap, and requires exact equality at lookup/position boundaries, all 19 blocks and all output fields on the reference CPU fixture. No tolerance introduced. |
| Cross-sample derivative isolation | New explicit case probes each sample in turn through full logits, LD hidden and LE global. Both the other sample's Iv and raw lookup gradients must be allocated exact zeros. Own-sample derivatives must be finite and nonzero. Integer IDs are not differentiated. |
| FP64 analytical checks | All existing independent attention/VIM tests and tolerances stay unchanged. |
| Full-FLT FP64 batch comparison | New explicit case deep-copies the same FP32-initialized parameters and promotes the same inputs. Compare B=2 against each B=1 sample at every recorded stage using `atol=1e-10, rtol=1e-8`. Any failure remains a real failure. |
| FP32 cross-batch diagnostic | Mandatory execution of full-width/depth B=2 and each B=1 sample; complete stage records and metadata must be written successfully. Finite outputs, matching shapes and unchanged parameter identities remain hard checks. `allclose=False` alone is not a correctness failure under N2. |
| Finiteness and full backward | Existing T01/T16, integration checks and all other numerical gates remain mandatory. Any nonfinite valid output/gradient is a failure requiring diagnosis; no clipping/normalization is allowed as a silent response. |
| Dimensions, ownership, initialization, serialization, configuration | All existing gates remain mandatory: 112,758,272 FLT parameters, 19 independent VIMs, exact vocabulary tie, 153,892,549 combined count, unchanged approved topology and initialization. |
| Existing regression and N1 | All 119 cases remain unchanged and must pass. N1 remains exclusive to the old MVE-T12 comparison; it is never applied to FLT. |
| Complete execution | After explicit approval and application, a new complete suite must pass without skips/xfails or threshold overrides. The proposed collection is 164 cases, subject to actual collected-count verification. |

The fixed-shape exact requirements above are reference verification gates, not claims of bitwise equality between arbitrary hardware/platforms or different batch shapes. A new backend requires its own complete evidence. Failure of an exact gate cannot be reclassified by N2 without a separate review.

## Required diagnostic content

For both B=2 sample slices against independent B=1 execution, retain:

- Lookup, position-added encoder/decoder input, each of twelve LE and seven LD outputs, LE global at 307, decoder hidden and full vocabulary logits; their shapes/dtypes and finite status.
- Original FP32 comparison thresholds, exact-equality flag, outside-tolerance count and element count, maximum absolute difference, maximum relative difference, zero-reference exception counts, RMS absolute difference and relative L2 difference.
- Relative error convention: absolute difference divided by absolute B=1 reference, without an epsilon floor. Store an explicit infinite-relative-error flag/count when a zero reference differs; do not emit nonstandard JSON infinity or conceal such values.
- Python/PyTorch/build/OS/CPU metadata, device, thread counts, deterministic flag, matmul precision and available/enabled math backends; relevant selected threading environment settings.
- Fixture identities, matching parameter/input provenance and the full parameter ownership checks in the surrounding review package.

The proposed runtime test stores its current-run JSON in a new uniquely named directory under `artifacts/phase2c2_n2_acceptance`, and verifies that the complete record can be read back. A passed diagnostic means that the prescribed experiment and artifact checks succeeded, **not** that the FP32 outputs became equal or that drift is universally harmless. The three-fixture review matrix and activation/gradient records remain part of acceptance review, alongside the permanent reference-fixture tests.

No replacement universal FP32 error ceiling is proposed. Selecting an arbitrary larger threshold would be unsupported by the paper and by these few fixtures. Numerical evidence must still receive review: a changed topology/backend/fixture or materially different error pattern requires diagnosis, not an automatic claim that any finite difference is acceptable. This is a disclosed human-review obligation, not an undocumented numerical threshold or an authorization for unstable training.

## Exact unapplied changes

[proposed_t17.patch](../artifacts/phase2c2_n2/proposed_t17.patch) is a concrete unified diff against the preserved current `tests/language/test_flt.py`. It is **not applied**. The baseline test-file SHA-256 and syntax-only validation are recorded in [proposal_status.json](../artifacts/phase2c2_n2/proposal_status.json).

The diff replaces exactly one existing function and adds three functions immediately before T18:

| Current/proposed function | Exact change |
|---|---|
| `test_flt_t17_fixed_batch_independence` | No edit |
| `test_flt_t17_cross_batch_original_tolerance` | Replace with `test_flt_t17_cross_batch_numerical_diagnostic`, retaining FP32 constants and recording all comparisons instead of asserting universal equality |
| `test_flt_t17_fixed_batch_permutation` | Add exact fixed-shape paired permutation check |
| `test_flt_t17_bidirectional_cross_sample_gradients` | Add both-direction visual/raw-lookup derivative isolation |
| `test_flt_t17_promoted_weight_fp64_cross_batch` | Add deep-copy/promoted-value full-FLT comparisons at unchanged FP64 tolerances |

The patch reuses the new review-only helpers in [review_phase2c2_n2.py](../tools/review_phase2c2_n2.py); those helpers call the unchanged production model and perform diagnostic capture/statistics only. They are not a replacement model, and their presence does not activate the proposed tests. The proposed code was compiled as an in-memory syntax check, not installed into the test tree or executed as an amended suite. No production file, existing configuration, numerical constant, accepted test or old manifest is edited by the proposal.

After approval, record N2 in a new scoped verification approval record. There is no need to change runtime architecture/tolerance values or broaden the existing configuration schema: N2 changes the interpretation of one verification gate. Preserve all pre-N2 results and packages, then create a new acceptance report/manifest after the complete suite. This document supplies no advance approval for such a report to declare success.

## Before/after acceptance comparison

| Item | Before N2, actual | After N2 if approved/applied |
|---|---|---|
| Complete suite | 161 collected; 160 pass, 1 fail | Expected 164 collected; **NOT RUN** |
| Original cross-batch FP32 result | Real failing assertion, preserved | Same discrepancy recorded completely; no raised tolerance |
| Exact fixed-batch gate | Pass | Unchanged mandatory gate |
| Permutation | New review evidence exact across all tested stages/fixtures | Explicit mandatory permanent case |
| Cross-sample gradients | Existing one-direction visual gate passes; new review also verifies both directions and lookup boundary | Original gate preserved plus new explicit both-direction case |
| Analytical FP64 | Existing tests pass | Unchanged |
| Full promoted-weight FP64 cross-batch | New review passes all stages, both samples, three fixtures | Additional mandatory reference-fixture case |
| Source/config/parameters | Approved baseline | Identical production baseline |
| Earlier regression | 119/119 pass; files preserved | All 119 must pass unchanged again |
| Tolerances | FP32 `1e-6/1e-5`, FP64 `1e-10/1e-8`; scoped MVE N1 only | Identical numeric values and N1 scope |
| GPU/batch-8 capacity, training stability, parameter-count gap | Unresolved | Unresolved; N2 supplies no waiver or success claim |
| Phase 2C.2 status | BLOCKED | Remains BLOCKED until approval, application, a complete passing suite and review |

The expected case count is `161 - 1 replaced case + 1 diagnostic case + 3 new companion cases = 164`. It is not a fabricated test result. The only new pytest execution in this review reruns unchanged tests and reports **14 passed, 1 failed**; all existing historical records keep their original outcomes.

## Approval boundary

Requested decision: approve **N2 only**, as the disclosed replacement of the universal FP32 cross-batch assertion with mandatory diagnostics plus the three specified correctness cases. N2 is currently **PROPOSED / BLOCKED**, and the old failed test remains active. No full-model acceptance, architecture adjustment, loss/classifier/dataset/training work, GPU or batch-8 success is authorized or claimed. Work stops at this review package until explicit user approval.
