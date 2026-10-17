# Phase 2D.2 N3.1 held-out validation

**Phase 2D.2 remains BLOCKED.** The final mandatory suite ran **328 cases: 320 passed, 8 failed, 0 errors, 0 skips**. The seven supplemental cases separately ran **7/7 passing**. All **276/276 accepted regression cases pass**; all original 321 identities remain present. Original eight failure identities preserved: **True**. New failure identities: `[]`. See [verification](../artifacts/phase2d2_n31/verification.json), [full stdout](../artifacts/phase2d2_n31/logs/complete_01.stdout.log), [stderr](../artifacts/phase2d2_n31/logs/complete_01.stderr.log), [JUnit](../artifacts/phase2d2_n31/junit/complete_01.xml).

Actual nonpassing mandatory cases (pytest exit code 1):

- `tests.joint.test_joint.test_joint_i08_cmc_chain_rule_and_batch_coupling`
- `tests.joint.test_joint.test_joint_i10_fp64_oracle_and_full_fp32_gradient_additivity`
- `tests.joint.test_readiness.test_ready_n02_full_g2_no_update[real]`
- `tests.joint.test_readiness.test_ready_n02_full_g2_no_update[efs]`
- `tests.joint.test_readiness.test_ready_n02_full_g2_no_update[local]`
- `tests.joint.test_readiness.test_ready_n02_full_g2_no_update[real_efs]`
- `tests.joint.test_readiness.test_ready_n02_full_g2_no_update[real_local]`
- `tests.joint.test_readiness.test_ready_n02_full_g2_no_update[local_duplicate]`

## Locked protocol and approved patches

[Registration](../artifacts/phase2d2_n31/heldout_preregistration.json) and its [SHA lock](../artifacts/phase2d2_n31/preregistration_lock.json) precede experiments. No fixture definition was altered after observing results. Attention initialization/input/cotangent/direction seeds are 2101/2102/2103/2104–2105; `[1,308,512]`, magnitudes 0.02/0.2/2, cotangent scales 0/1e-4/1/1e4. LD uses copied actual real/EFS accepted decoder inputs at multipliers 0.25/1/2, cotangent seeds 3101/3102, cuts 1/4/7 and scales 1e-4/1/1024. Visual controls use new image seed 4101 and cotangent seeds 4102/4103 at the same three nonzero scales, with MKLDNN on/off. All remain finite no-update diagnostics. Zero, analytically canceling and generally nonzero derivatives are explicitly distinct.

The exact approved N3D01 new-file test patch and N3D02 metadata patch were applied only after verifying expected source hashes. [Applied-patch record](../artifacts/phase2d2_n31/applied_patches.json) contains before/after hashes and the original patch hashes. Original helper bytes and both diffs are archived. Existing 321 tests/assertions, 276 accepted regressions, production sources and configs are unchanged. Fresh primary records retain seven distinct loss-grad-function wrappers and live tensor/storage identities; the old defective metadata remains historical.

## Fresh evidence integrity

The final workflow seals the required record inventory, source/test/config hashes, locked protocol and raw execution logs. A pytest session plugin verifies the fresh run nonce, exact sealed record hashes, all required filenames, regenerated synthetic image/token/mask provenance, initialized parameter hashes, ownership/counts, repaired graph metadata and no-update/finiteness flags **before tests run**. The external-evidence checks within the seven supplemental cases consume the newly generated N3.1 directory via environment variables; the other supplemental cases execute live mathematical controls. No N3 historical JSON is substituted.

[Negative provenance tests](../artifacts/phase2d2_n31/provenance_negative_tests.json) reject wrong-run nonce, missing records, incomplete inventory, altered source hashes and altered evidence. Integrity is not a numerical proof: the seal explicitly grants no acceptance, and numerical failures remain recorded and failing. A local SHA manifest is an integrity/reproducibility mechanism, not a signed adversarial attestation.

The workflow refreshed four same-forward N3 cases, CMC FP32/FP64, all six G2 fixtures plus one alternate backend, six N2 F0/F1/F2 precision cases and the newly held-out local experiments. No production FP64 MVE/VD interface was introduced. Local FP64 promotion copies FP32-initialized parameters and stored inputs; it is not full-model FP64 validation.

The declared repeated scientific fields match the preserved evidence exactly: **True**. This compares selected A/B gradient statistics and comparisons for all four N3 runs, both CMC records, all six original-backend G2 records and six N2 records. Timing/process identities are excluded; the unordered legacy cosine-entry list alone is canonicalized. The corrected seven distinct loss-wrapper identities are separately checked. See [exact comparison audit](../artifacts/phase2d2_n31/historical_scientific_comparison.json). Held-out seeds/magnitudes and the newly complete alternate-backend G2 are new experiments, not expected to match old fixtures.

## Independent attention results

There are **24** precision/magnitude/scale attention records. Independent FP64 formula comparisons, key-bias zero-reference comparisons and every failed FP32/local comparison remain in [heldout_attention.json](../artifacts/phase2d2_n31/heldout_attention.json) and [flat comparison CSV](../artifacts/phase2d2_n31/heldout_comparisons.csv). No failing weak or nonzero Q/K derivative is omitted.

Conditional local affine/backward-contraction/bias-reduction envelopes: **33** checked, **0** coordinates outside. These use dimensions/roundoff models fixed before observations, not maxima-fitted thresholds. The envelopes cover frozen operands only, are computed in FP64 without interval rounding, and do not certify optimized kernels or a full network. Stored probability defects and their Jacobian row-sum consequences are separately recorded.

Central finite differences use epsilon 1e-6 and unchanged atol 1e-5 / rtol 1e-3. They compare independent functional perturbations, with no in-place Parameter edit:

| Input magnitude | Parameter | Absolute directional error | Existing check |
|---|---|---:|---|
| 0.02 | q.weight | 3.53651e-11 | PASS |
| 0.02 | k.weight | 5.06107e-10 | PASS |
| 0.2 | q.weight | 1.39822e-09 | PASS |
| 0.2 | k.weight | 8.81953e-10 | PASS |
| 2.0 | q.weight | 7.00628e-07 | PASS |
| 2.0 | k.weight | 9.75458e-07 | PASS |

Finite differences are approximate references, subject to truncation and subtractive rounding. Exact rational mathematics verifies ordinary key-bias invariance separately. Allocated singleton VIM Q/K zeros retain their exact requirement when connected; ordinary attention biases do not receive a structural-zero reinterpretation.

## Seven-block LD held-out controls

There are **54** predeclared LD precision/magnitude/cut/scale records. The table reports scale-one VJP additivity relative L2, without changing the original elementwise criterion. Full absolute/count/zero-reference/norm/cosine evidence at all scales remains in [heldout_LD.json](../artifacts/phase2d2_n31/heldout_LD.json).

| Input multiplier | Precision | Cut | State relative L2 | Memory relative L2 | LD1 Q-weight relative L2 |
|---|---|---|---:|---:|---:|
| 0.25 | torch.float32 | LD01 | 1.155088e-05 | 7.908334e-07 | 1.026599e-05 |
| 0.25 | torch.float32 | LD04 | 0.0001795829 | 0.0001156669 | 0.0001356558 |
| 0.25 | torch.float32 | LD07 | 0.004081029 | 0.004373742 | 0.003964704 |
| 0.25 | torch.float64 | LD01 | 1.884041e-14 | 1.447851e-15 | 1.678406e-14 |
| 0.25 | torch.float64 | LD04 | 2.819071e-13 | 1.447983e-13 | 1.888877e-13 |
| 0.25 | torch.float64 | LD07 | 8.623266e-12 | 8.284141e-12 | 8.310812e-12 |
| 1.0 | torch.float32 | LD01 | 0.0001916819 | 8.625526e-07 | 0.0001915624 |
| 1.0 | torch.float32 | LD04 | 0.003004616 | 0.00293439 | 0.002509503 |
| 1.0 | torch.float32 | LD07 | 0.02380211 | 0.0541279 | 0.01677164 |
| 1.0 | torch.float64 | LD01 | 2.999477e-13 | 1.549716e-15 | 2.772717e-13 |
| 1.0 | torch.float64 | LD04 | 7.223151e-12 | 9.479581e-12 | 6.971172e-12 |
| 1.0 | torch.float64 | LD07 | 2.429275e-11 | 1.124797e-10 | 1.74658e-11 |
| 2.0 | torch.float32 | LD01 | 0.0007164736 | 1.440096e-06 | 0.0006306328 |
| 2.0 | torch.float32 | LD04 | 0.003736652 | 0.004596129 | 0.003157273 |
| 2.0 | torch.float32 | LD07 | 0.0786628 | 0.05447743 | 0.08365885 |
| 2.0 | torch.float64 | LD01 | 1.24831e-12 | 2.738109e-15 | 1.100167e-12 |
| 2.0 | torch.float64 | LD04 | 9.314259e-12 | 2.256905e-11 | 6.337892e-12 |
| 2.0 | torch.float64 | LD07 | 3.107471e-11 | 9.772437e-11 | 6.238282e-11 |

These are fixed local Jacobians with independent state/memory/visual leaves, not full LE/visual gradients. Each cut's u, v and sum VJPs use one forward graph. The FP32 u+v includes input-sum rounding; promoted FP64 sums promoted operands. Different precisions recompute forward intermediates, so precision improvement does not isolate every error source. No threshold is inferred from these measurements. A count of completed diagnostics must not be presented as passing numerical comparisons.

Across all local manual/VJP/zero-reference comparisons, **125 of 504** are outside their unchanged coordinate tolerances. They are listed in [nonpassing comparisons](../artifacts/phase2d2_n31/heldout_nonpassing_comparisons.json). They are exploratory comparison outcomes, not extra pytest case counts, and include near-zero cancellation as well as nonzero derivative disagreement.

## Complete parameter and module coverage

Each G2 worker streams six isolated FP32 gradients to the existing FP64 accumulator and runs the literal joint backward. An added diagnostic pass reads the same stream without modifying the legacy comparison or its result. All **508 parameter tensors / 154,156,232 elements** are covered per run. Exact original coordinate mismatch counts are cross-checked against the unchanged comparator. Counts, zero-reference coordinates, absolute/relative/norm errors, signs, cosine and four seeded directions are reported per parameter and module. Module directions concatenate parameter-specific Rademacher vectors and normalize by module size. This is full module coverage, not N3's selected subset.

The original elementwise gate casts the accumulated reference back to FP32 before comparing with the joint FP32 gradient. The new norm, absolute-error, zero-reference and directional diagnostics instead compare the joint values promoted to FP64 against the stored FP64 sum. These are explicitly different arithmetic views of the same stream; no change to the legacy reference or tolerance is implied. The FP64 sum contains already-rounded isolated FP32 gradients and is not a full-model FP64 derivative oracle.

| Fixture/backend | Outside elements | Global relative L2 (diagnostic only) | LE8 Q-weight relative L2 | Peak RSS MiB | Worker seconds |
|---|---:|---:|---:|---:|---:|
| real | 8,772,234 | 9.169735e-07 | 0.005703393 | 2271.2 | 118.40 |
| efs | 13,636,267 | 1.951497e-06 | 0.006087369 | 2280.8 | 152.89 |
| local | 11,558,833 | 1.535572e-06 | 0.005908343 | 2267.3 | 121.59 |
| real_efs | 14,149,076 | 1.937857e-06 | 0.004808017 | 2885.2 | 148.74 |
| real_local | 12,728,291 | 1.674038e-06 | 0.004482767 | 2848.9 | 176.58 |
| local_duplicate | 14,611,405 | 2.050799e-06 | 0.003516976 | 2878.8 | 153.58 |
| real_efs_native | 14,247,101 | 2.0276e-06 | 0.01130299 | 3045.8 | 164.10 |

[Complete parameter CSV](../artifacts/phase2d2_n31/complete_parameter_discrepancies.csv), [complete group CSV](../artifacts/phase2d2_n31/complete_group_discrepancies.csv), [G2 summaries](../artifacts/phase2d2_n31/complete_G2_summary.json). Every full JSON retains vector sufficient statistics and directional values. Whole-model norms never replace weak-parameter evidence. Undefined zero-reference relative L2/cosine remains null. Classification distinguishes analytical ordinary key-bias zeros, structural VIM zeros and other parameters. The independent stream control detects a deliberately injected +1 gradient error across a chunk boundary and checks zero-reference handling.

The MKLDNN-disabled run is the same CPU/platform with a backend flag changed, not an independent platform. Full G2 uses actual unit-weight losses and original scale; cotangent-scale sensitivity is independently measured in the held-out LD and visual controls. No synthetic scaling was applied to production objectives or gradients.

## Preservation and resource limits

All 942 pre-existing files were audited. The only changed pre-existing file is the explicitly approved diagnostic metadata helper; its original bytes are archived. Every other pre-existing source/config/test/historical artifact remains byte-identical. The new test file exactly matches its approved patch. [Preservation](../artifacts/phase2d2_n31/preservation.json).

CPU batch 1 and 2 were actually exercised. GPU and CPU batch 8 are NOT RUN; no capacity is inferred. Full workers have sampled RSS/runtime records including disk streaming/statistics, not pure model throughput or exact allocator peaks. No simultaneous seven full graphs are retained. All parameters and their identities remain unchanged within runs. No optimizer, update, training, benchmark, dataset, clipping, normalization, AMP or parameter freezing was implemented.

The mandated unchanged regression suite includes the previously accepted **MVE-T10** test: it performs one controlled SGD update on three MVE parameters, checks the exact result and restores their original bytes in `finally`. That pre-existing unit test was retained unchanged. The no-update claim applies to every N3.1 experiment and full G2 worker; it does not erase this historical unit-test behavior. No new optimizer, full-graph optimization, G3/G4 execution or training step was introduced.

The original orchestration stopped when the first fresh N2 worker could not write its result into a missing output directory. Its original stderr/exit record is preserved. The recovery created that directory, verified the original pinned source snapshot and reran all six N2 workers under distinct retry log names; no numerical source or fixture changed. See [recovery record](../artifacts/phase2d2_n31/resume_record.json). During recovery the system had only 565,764,096 bytes of available physical memory (96.7% use); the first FP32 worker took about 1,705 seconds. The user then freed RAM. This is a resource-confounded wall time, not representative model throughput. See [live observation](../artifacts/phase2d2_n31/resume_resource_observation.json) and actual per-worker execution logs. The reproduction guide explicitly creates the output directory before starting.

The [numerical budget](phase2d2_n31_numerical_budget.md) explains conditional bound limitations; the [proposal](phase2d2_n31_acceptance_proposal.md) gives the stopping decision. Review ZIP and manifests are under [artifacts/phase2d2_n31](../artifacts/phase2d2_n31).
