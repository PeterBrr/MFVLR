# Phase 2C.2 numerical acceptance review

2026-10-11. **Recommendation: propose N2; do not accept Phase 2C.2 yet.** The fixed FP32 `atol=1e-6, rtol=1e-5` cross-batch allclose assertion is not a suitable universal correctness gate for this approved baseline across batch shapes/backends. The observed failure arises from shape-dependent floating arithmetic and its amplification through depth. Exact fixed-batch independence, exact zero cross-sample derivatives, exact tested batch permutation, independent analytical tests, and promoted-weight FP64 comparisons support sample correctness. They do **not** establish FP32 training stability.

N2 is **PROPOSED**, with implementation **BLOCKED pending explicit user approval**. [The proposal](phase2c2_n2_proposal.md) contains the narrowly scoped gate change and a concrete, unapplied patch. Production code, existing tests, configurations, reports, logs, ZIPs and manifests remain unchanged. The existing complete-suite result remains **160 passed, 1 failed / 161 cases**; this review does not relabel that failure.

## Evidence and scope

Inspected sources: [LE/ordinary attention](../src/mfvlr/language_encoder.py), [LD](../src/mfvlr/language_decoder.py), [full FLT](../src/mfvlr/flt.py), [accepted VIM](../src/mfvlr/vim.py), [T17 and full-FLT tests](../tests/language/test_flt.py), [analytical/LE tests](../tests/language/test_encoder_stage.py), and the accepted [language contract](phase2c_language_contract.md). Historical [FP32](../artifacts/phase2c2/batch_numerics.json), [FP64](../artifacts/phase2c2/batch_numerics_fp64.json), resource, gradient and JUnit evidence was read without alteration.

The new [numerical summary](../artifacts/phase2c2_n2/numerical_summary.json) links seven fresh-process review runs: three matched FP32/FP64 fixtures at four CPU threads, plus a one-thread FP32 control for the original fixture. [Raw JSON and command/exit logs](../artifacts/phase2c2_n2/run_01/index.json) retain both samples, every LE/LD block, lookup/position boundaries, global language and vocabulary output, first-block primitive details for the original fixture, and full FP32 backward measurements. These are diagnostic executions, not an applied amended test suite.

The user reports independent Linux corroboration of the FP32 difference and a successful promoted-weight FP64 comparison. No raw Linux logs were supplied to this workspace review, so those results are attributed to the user and are not merged into local numerical measurements. GPU and batch-8 checks were not run in N2; their prior NOT RUN status is unchanged.

## Matched fixtures and reproducibility

Every worker initializes the exact approved full model in FP32 with seed 0, then records the same parameter-state SHA-256: `e4233731564ba462fc32423bb4832e996f96331d2bf21509addad8bd0cffaed2`. FP64 workers promote those FP32 values; they do not initialize directly in FP64. IDs and visual inputs are generated in FP32, hashed and recorded literally in JSON, then promoted as needed. Matching IDs/visual hashes are checked across precision pairs. Parameters are rehashed after the probes; no optimizer/update occurs. The full 112,758,272-parameter topology, parameter-object identities and tied vocabulary remain intact.

| Fixture | Token IDs from unchanged golden cases | Visual values before promotion | Visual RMS / maximum |
|---|---|---|---:|
| F0, original T17 | 1/5: DDPM / DiffFace | `randn`, seed 341, multiplied by 0.1 | 0.097405 / 0.334322 |
| F1, prompt/input variation | 0/4: genuine / StyleGAN3 | `randn`, seed 642, multiplied by 0.1 | 0.099717 / 0.366209 |
| F2, visual integration scale | 8/9: Diffae / LatentTransformer | Actual accepted FP32 MVE→VD→residual-fusion outputs on two seed-412 random images | 4.245593 / 15.184306 |

F2 is an untrained synthetic-image integration fixture, not a dataset or reconstruction-quality experiment. Its visual values are computed once per worker in the accepted FP32 visual modules and match exactly between language precision workers. No visual normalization/scaling is inserted to make FLT behave better. All cases retain `[B,308]`, width 512, 12 LE/7 LD blocks and full `[B,308,49408]` logits.

Local backend: Windows 11 / AMD64, Python 3.14.3, PyTorch 2.12.1+cpu, deterministic algorithms enabled, float32 matmul precision `highest`, MKL and MKLDNN available/enabled, four intra-op/inter-op threads for the main cases. Build metadata identifies oneAPI MKL 2026.0 and MKLDNN 3.11.2; complete compiler/build/capability information and selected threading environment variables are stored per worker. No specific internal BLAS microkernel was identified. The one-thread control retains the same forward cross-batch difference metrics as the four-thread F0 case; lowering thread count did not remove the effect. No backend setting was changed in production/configuration.

## Batch-dimension audit

| Operation | Mathematical batch behavior | Evidence / caveat |
|---|---|---|
| Vocabulary lookup and positions | Each sample indexes Wvoc independently; Pe/Pd broadcast constants across samples | Lookup and position-added inputs are exact across batch sizes in these probes |
| LayerNorm | `normalized_shape=(512,)`; normalization over features of one token | No batch statistics or BatchNorm; first LE LN output is exact in F0 |
| Ordinary linear/FF maps | Matrix contracts only the final feature axis; GELU is elementwise | GEMM may flatten leading rows for execution, which can change rounding without mathematical row mixing |
| Head reshape and transpose | `[B,N,512]→[B,8,N,64]`; B remains a leading independent index | No flattening of B into the attention key/query length |
| Ordinary attention | `Q @ K.transpose(-2,-1)` contracts head features, then softmax over last/key axis | `[B,8,N,N]`; self/memory operands have matching B; no softmax/reduction over B |
| Singleton VIM | `[B,8,N,64] @ [B,8,64,1]`, key-axis softmax, V from the same sample | One finite key gives probability one; K/Q rounding cannot affect the delta, but V/O rounding can |
| LD shift/causal mask | Concatenation and slicing along token axis only; mask `[N,N]` broadcast across B/head | Shift crosses segment boundaries, never sample boundaries |
| Residuals/global selection | Elementwise additions; `[:,307:308,:]` is a view/slice | No pooling across samples; global selection introduces no new arithmetic |
| Vocabulary projection | Final-axis contraction with the same Wvoc parameter | No output softmax or batch reduction; tying parameters does not tie sample activations |
| Validation/finite checks | Global boolean reductions over a submitted tensor/batch | These may reject an invalid batch as a whole; for valid finite inputs they neither modify values nor create a differentiable cross-sample path |
| Backward to shared parameters | Contributions from samples sum into shared parameter gradients | Expected parameter sharing; leakage checks must use sample-specific input/intermediate derivatives, not merely shared-weight gradient presence |

There is no learned feature operation that mixes the batch dimension. Both directions of sample-local probes have allocated exact-zero derivatives to the other sample's visual input **and differentiable raw lookup tensor**. Integer IDs themselves have no gradient. Own-sample derivatives are nonzero, preventing vacuous zero-gradient evidence.

## Root cause and stage differences

PyTorch documents that floating arithmetic is order-dependent and that batched and sliced computations can differ even for mathematically identical inputs. Deterministic execution does not provide a cross-shape or cross-platform bitwise guarantee. This is supporting library evidence, not a blanket justification for arbitrary error or removal of numerical verification. [PyTorch 2.12 numerical accuracy](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html#batched-computations-or-slice-computations), [reproducibility](https://docs.pytorch.org/docs/2.12/notes/randomness.html).

For F0, sample 0, vocabulary lookup, Pe addition, first LN, all first ordinary self-attention projections and first VIM query projection are exact between B=2 and B=1. The **first measured discrepancies are the first VIM K and V linear outputs**, each with maximum difference `2.98023224e-8`. The visual input and projection weights are identical. These maps operate on one versus two rows of 512 features; this directly localizes a shape-dependent floating result before any later activation growth. Identifying this effect does not establish whether a particular GEMM/GEMV or microkernel dispatch was used internally.

K has no forward influence through a finite singleton key softmax; its task gradient is exactly zero. V's difference propagates through VIM O (`7.45058060e-8`), FF and residual addition; the first complete LE block differs by `4.76837158e-7`. Subsequent LE blocks accumulate and amplify rounding differences. The LD starts with identical shifted raw lookup/Pd states but receives already-different LE memory plus its own shape-dependent VIM value computations. All seven LD blocks are unnormalized under the approved engineering baseline.

The global feature selects a subset of the final LE difference. The vocabulary projection is linear in the LD state. In a separate control, projecting **identical LD values** at B=2 versus B=1 yields exactly equal outputs on this backend in all seven workers. Thus the measured final logit discrepancy here is inherited from LD values; this does not promise that projection kernels on every backend are cross-shape identical.

| Stage, F0 sample 0 | FP32 max absolute difference | Outside original FP32 tolerance / 157,696 | FP64 max absolute difference |
|---|---:|---:|---:|
| LE1 | 4.7683716e-07 | 0 | 7.2164497e-16 |
| LE2 | 2.2649765e-06 | 29 | 4.6629367e-15 |
| LE3 | 3.3676624e-06 | 329 | 6.6613381e-15 |
| LE4 | 5.2452087e-06 | 1,074 | 9.3258734e-15 |
| LE5 | 7.1525574e-06 | 1,686 | 1.1990409e-14 |
| LE6 | 8.8214874e-06 | 2,472 | 1.5987212e-14 |
| LE7 | 1.2874603e-05 | 3,911 | 2.1316282e-14 |
| LE8 | 1.4305115e-05 | 3,449 | 2.6645353e-14 |
| LE9 | 1.8119812e-05 | 5,281 | 3.1974423e-14 |
| LE10 | 2.4199486e-05 | 3,965 | 4.0856207e-14 |
| LE11 | 3.0517578e-05 | 5,349 | 5.1514348e-14 |
| LE12 | 3.3020973e-05 | 8,318 | 6.0396133e-14 |
| LD1 | 2.18153e-05 | 1,830 | 4.2632564e-14 |
| LD2 | 0.0001115799 | 12,517 | 1.563194e-13 |
| LD3 | 0.00020599365 | 12,629 | 3.0908609e-13 |
| LD4 | 0.00029945374 | 15,861 | 7.6738615e-13 |
| LD5 | 0.00065803528 | 18,112 | 1.1084467e-12 |
| LD6 | 0.001159668 | 17,952 | 3.2116532e-12 |
| LD7 | 0.0058555603 | 33,924 | 8.6011198e-12 |

Each table entry above uses original F0 sample 0, a reference B=1 result, and the unchanged FP32 `1e-6 + 1e-5*abs(reference)` comparison. Detailed first-block maps, both samples and all three fixtures are in the JSON. “Outside” counts retain the original dtype's `torch.isclose` arithmetic. Maximum absolute/relative and norm statistics use FP64 diagnostic reductions to avoid losing information in the measurement itself; no production arithmetic is promoted by that measurement.

| Output, F0 sample 0 | FP32 max absolute difference | Outside original tolerance | Max relative difference to nonzero reference | FP64 max absolute difference |
|---|---:|---:|---:|---:|
| Global language | 2.52723694e-5 | 31 / 512 | 6.14653676e-4 | 4.61852778e-14 |
| Vocabulary logits | 0.00308227539 | 3,170,109 / 15,217,664 | 20.4 | 5.00222086e-12 |

The large maximum elementwise relative errors occur near zero reference values and must be read with absolute error and normwise error. No denominator epsilon is silently introduced: JSON records maximum relative error on nonzero references, zero-reference/nonzero-difference count, and an explicit infinite-relative-error flag. All requested outputs/stages are retained even after an earlier comparison is outside tolerance.

Across three four-thread fixtures, fixed-B=2 changes to either sample leave **all recorded stages of the other sample exactly unchanged** in both precisions. Swapping both IDs and visual features, then undoing the permutation, also gives exact equality at every recorded stage. Full promoted-weight FP64 cross-batch checks satisfy unchanged `atol=1e-10, rtol=1e-8` for **both samples at every stage**, including the integration-scale fixture. The largest F2 FP64 logit difference is `3.85853127e-10`; this is compatible with the relative-plus-absolute bound, not evidence of an altered absolute tolerance.

These results distinguish observed sample correctness from accumulated numerical error. The replicated Linux behavior reported by the user strengthens the cross-platform relevance, but no universal error bound or worst-case conditioning guarantee follows from these finite fixtures.

## Stability and fixture variation

The following activation statistics cover all 19 block outputs at B=2. RMS is `L2_norm/sqrt(numel)`; raw L2 norms, maxima and finite flags are retained for every stage in JSON.

| Stage, B=2 | F0 RMS / max absolute | F1 RMS / max absolute | F2 RMS / max absolute |
|---|---:|---:|---:|
| LE1 | 0.63440 / 2.20337 | 0.64918 / 2.20374 | 4.58661 / 18.90462 |
| LE2 | 1.21318 / 4.45168 | 1.21031 / 4.47936 | 7.35823 / 22.84324 |
| LE3 | 1.83324 / 6.03098 | 1.81389 / 6.48534 | 10.05310 / 32.36319 |
| LE4 | 2.31953 / 7.68473 | 2.31246 / 7.78706 | 12.54798 / 41.23435 |
| LE5 | 3.04091 / 9.50408 | 3.05617 / 10.11278 | 15.84891 / 53.75494 |
| LE6 | 3.57592 / 12.00794 | 3.64428 / 14.11096 | 18.74078 / 67.37709 |
| LE7 | 4.24394 / 13.44102 | 4.29531 / 12.93666 | 21.87531 / 69.42026 |
| LE8 | 4.99907 / 14.69992 | 5.10262 / 16.62720 | 25.85570 / 79.94816 |
| LE9 | 5.66041 / 20.29292 | 5.78021 / 20.31506 | 29.27421 / 80.61993 |
| LE10 | 6.82561 / 21.85952 | 6.98169 / 23.09562 | 34.63193 / 108.03067 |
| LE11 | 7.68846 / 26.36524 | 7.84722 / 29.19417 | 39.20419 / 130.70332 |
| LE12 | 8.75369 / 27.93132 | 8.91509 / 32.98520 | 44.74036 / 135.37820 |
| LD1 | 9.86059 / 28.73666 | 9.93950 / 34.32529 | 51.25371 / 153.52513 |
| LD2 | 19.02246 / 59.20166 | 19.34344 / 73.71176 | 98.90698 / 280.66541 |
| LD3 | 32.47733 / 107.50298 | 32.53911 / 106.16492 | 172.05552 / 527.15558 |
| LD4 | 54.46169 / 193.46141 | 55.55766 / 196.86263 | 307.91040 / 937.62793 |
| LD5 | 92.23660 / 319.42596 | 92.78169 / 324.84167 | 492.47906 / 1794.89343 |
| LD6 | 150.19496 / 526.79749 | 150.51682 / 474.10342 | 787.56181 / 2404.27979 |
| LD7 | 254.91487 / 865.81781 | 258.56355 / 980.05035 | 1242.31151 / 4411.21289 |

| Fixture | Maximum FP32 logit cross-batch difference, samples 0 / 1 | Logit relative L2 difference, samples 0 / 1 | Full parameter-gradient L2 |
|---|---:|---:|---:|
| F0 | 0.00308228 / 0.00534821 | 3.43608e-6 / 6.28022e-6 | 3.03173e6 |
| F1 | 0.00150681 / 0.00368500 | 2.44339e-6 / 4.11987e-6 | 3.13863e6 |
| F2 | 0.17864990 / 0.10296631 | 2.69393e-5 / 1.27070e-5 | 6.23973e7 |

Gradients use the same diagnostic scalar as the prior full-FLT checks: mean squared logits + mean squared LD hidden + mean squared LE global. This is instrumentation, not a paper loss or optimizer/training step. All 387 FLT parameters receive finite gradients in each FP32 fixture; all 38 VIM Q/K matrices receive allocated exact-zero gradients. Norms for every named parameter, every retained LE/LD block output, raw lookup, position-added input, full logits and visual feature are stored. These norms depend on the chosen probe and its reduction; they are not scale-free measures of training quality.

For example, F0's LE1/LE12 output-gradient L2 norms are approximately 12,991.86 / 2,886.27; LD1/LD7 are 805.91 / 1.095. For F2 they are approximately 84,555.46 / 18,424.46 and 22,596.22 / 5.326. F0's largest parameter norm is vocabulary weight at about 1.2065e6; F2's largest is LE6 FF1 weight at about 1.0893e7. These are finite, but substantial depth/input-scale sensitivity remains.

### Controlled decoder perturbation

To isolate LD sensitivity from changing backend shape and LE memory, each FP64 worker holds B=2, LE memory and visual features fixed. A seed-123 RMS-one direction is added to the initial shifted-plus-position LD state at magnitudes `1e-6` and `1e-7`. All seven actual approved LD blocks execute unchanged. This is an input-perturbation experiment only; the reference production forward is not rescaled.

| Fixture | Final LD RMS difference / actual input RMS perturbation, epsilon=1e-6 | Same ratio, epsilon=1e-7 |
|---|---:|---:|
| F0 | 46.7913 | 46.7873 |
| F1 | 36.5345 | 36.5348 |
| F2 | 840.839 | 804.526 |

The approved no-LN decoder configuration **does magnify small perturbations on these fixtures**, substantially more for integration-scale visual inputs. F2's gain varies with perturbation magnitude; the direction crosses a more nonlinear local response. These are finite-direction sensitivities, not exact Jacobian spectral norms, worst-case condition numbers or evidence that removing LN alone caused the amplification. No normalized counterfactual was implemented or approved; weights, residual composition, attention and FF also contribute.

No valid new probe produced nonfinite outputs or gradients. The unchanged VIM negative overflow-rejection test deliberately exercises overflowing inputs and passes by raising its expected error; it is separate from valid-model stability measurements. **Finiteness is not a convergence result.** N2 should classify the cross-batch gate appropriately while retaining the large activations, gradients and sensitivity as unresolved risks for any future loss/training design. No LayerNorm, clipping, AMP, scaling, attention replacement or reduced dimensions were inserted.

## Verification, preservation and conclusion

[Unchanged targeted pytest execution](../artifacts/phase2c2_n2/verification_01/pytest.log), [JUnit](../artifacts/phase2c2_n2/verification_01/pytest.xml): **14 passed, 1 failed in 10.76 seconds**. It reruns all 12 existing VIM cases, the independent FP64 ordinary-attention oracle, exact fixed-batch T17, and the original failing FP32 cross-batch T17. The failure is retained as an ordinary pytest failure with exit code 1, no skip or xfail. Existing analytical tolerances remain untouched. A new complete suite under N2 has **not** run because the proposal is not applied.

[Preservation evidence](../artifacts/phase2c2_n2/preservation_after.json) matches all **249 pre-review files**, including every source/test/config/document/tool/artifact captured before this review and all old archives/manifests. It also verified the 242 payload hashes of the Phase 2C.2 review manifest. Generated bytecode/cache files are excluded from this preservation inventory. The review adds only new tools, two new documents and a new N2 artifact directory.

Reproduction, from the existing documented source/dependency environment:

```powershell
python tools/review_phase2c2_n2.py
python tools/verify_phase2c2_n2.py
python tools/summarize_phase2c2_n2.py
```

The review driver creates fresh numbered run directories. The targeted test command currently returns 1 for the real T17 failure. The summarizer deliberately targets the initial preserved `run_01` and `verification_01` evidence for this proposal; it does not silently replace that evidence with later measurements. It generates an unapplied patch and checks its Python syntax only. Reproducing on another platform requires the same accepted tokenizer dependencies; bundled Windows wheels are not substitutes for unavailable compatible wheels elsewhere.

**N2 is justified as a narrow verification-policy amendment, not a tolerance increase or architectural fix.** It should preserve exact sample checks and finite/analytical/identity requirements, add explicit permutation and bidirectional derivative gates, retain promoted-weight full-FLT FP64 cross-batch verification, and require complete FP32 stage diagnostics. Approval, application and a successful new complete suite remain separate future steps. Phase 2C.2 is still blocked, and training stability, GPU/batch-8 capacity and the 141M paper parameter discrepancy remain unresolved.
