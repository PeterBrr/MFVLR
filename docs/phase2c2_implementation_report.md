# Phase 2C.2 full FLT implementation and verification

2026-10-11. **Full FLT is implemented; strict acceptance is BLOCKED by FLT-T17's cross-batch FP32 comparison.** The final mandatory suite runs every case, with no skips, xfails or tolerance overrides. All 119 accepted regression cases pass unchanged. The new language and integration scope has 42 cases; its sole failure is the cross-batch comparison. Full FP32 forward/backward at batch sizes 1 and 2 remains finite for both standalone FLT and the complete visual-language route.

The implementation has **112,758,272 unique FLT parameters**, including 19 independent VIMs, and **153,892,549 combined visual/language parameters**. Parameter counts verify this approved engineering baseline, not fidelity to unpublished author code. Detection heads, adapter, paper objectives, datasets, optimizers, training loops and benchmark evaluation remain unimplemented.

## Approval and preservation

Authority is the user's explicit Phase 2C.2 instruction. Every completion choice below is an **ASSUMED, APPROVED engineering baseline**. Explicit paper operations and dimensions retain their original classifications in the [language contract](phase2c_language_contract.md); approval does not make an assumption an author-confirmed fact.

| Decisions | Current status and scope |
|---|---|
| L01–L03 | APPROVED in accepted Phase 2C.1: hierarchical prompts, immutable CLIP-compatible tokenizer assets, four independently framed 77-slot segments |
| L04 | APPROVED for this baseline: bidirectional unmasked LE; physical slot 307 global feature |
| L05 | APPROVED: twelve independent LE blocks, one specified pre-self-attention LN, caller-owned residuals |
| L06 | APPROVED: seven independent LD blocks; causal self-attention, full LE memory, FF then VIM; no LD norms |
| L07 | APPROVED: global one-position shift of raw lookup embeddings, BOS prepended |
| L08 | APPROVED: eight-head biased ordinary MHA; 512→2048→512 exact-GELU FF; zero dropout |
| L09 remaining scope | APPROVED: nineteen independent instances of the unchanged singleton-key primitive |
| L10 | APPROVED: exact same vocabulary parameter for lookup/output; independent Pe/Pd; trainable row zero |
| L11–L13 remaining scope | APPROVED: specified scratch initialization, full-FLT APIs, separate closed configurations and verification/orchestration |
| Any new numerical amendment | PROPOSED only if later submitted; none applied or authorized here |
| Losses, heads, adapter, datasets, training and evaluation | DEFERRED; no authorization from this phase |
| Phase 2C.2 acceptance | BLOCKED by T17; awaiting review |

Current approval records live in [flt_reference.yaml](../configs/flt_reference.yaml), [flt_debug.yaml](../configs/flt_debug.yaml) and the [closed snapshot](../src/mfvlr/flt_contract_snapshot.json). Historical design and Phase 2C.1 reports deliberately keep their original pending/full-FLT-unimplemented statements; this new scoped record supersedes those historical statuses. No older approval record or accepted payload was rewritten.

The [preservation audit](../artifacts/phase2c2/preservation.json) checks **all 165 payload files** in the accepted Phase 2C.1 manifest, including the earlier 110-file Phase 2B baseline. Zero mismatches were found before implementation and after verification. This includes accepted MVE/VD/VIM/tokenizer source, every old test, golden fixtures, configurations, snapshots and historical artifacts. N1 remains confined to the original MVE-T12 cross-batch comparison. New language tests use FP32 `atol=1e-6, rtol=1e-5` and FP64 `atol=1e-10, rtol=1e-8`; structural zeros and unaffected fixed-batch outputs use exact equality.

## Source, operations and interfaces

| New file | Implemented responsibility | Paper/contract traceability |
|---|---|---|
| [language_encoder.py](../src/mfvlr/language_encoder.py) | Vocabulary/positions, ordinary attention/FF, twelve LE blocks, stage-one owner | §III-D pp. 6–7, Eq. (3); dimensions §IV-A p. 8; L04/L05/L08/L10/L11 |
| [language_decoder.py](../src/mfvlr/language_decoder.py) | Seven LD blocks and their four caller-owned residual additions | Eqs. (12)–(16), p. 7; L06/L08/L09 |
| [flt.py](../src/mfvlr/flt.py) | Full owner, embedding shift, encode/decode, tied raw logits, parameter inventory | Shift prose p. 7; Eq. (24), p. 8; L07/L10/L12 |
| [flt_config.py](../src/mfvlr/flt_config.py) | Closed reference/debug schema, exact approval and tokenizer verification | L13; no change to accepted visual/foundation loaders |
| [visual_language.py](../src/mfvlr/visual_language.py) | External plain-object orchestration of accepted MVE/VD and optional language call | §III-A–D pp. 4–7; Fig. 4 p. 5; L12 |

The reference equations and their mathematical ambiguities remain unchanged in the accepted [paper analysis](paper_analysis.md) and [VIM analysis](phase2c_vim_analysis.md). Executable details implement the approved interpretation; they are not replacements for printed equations.

`FineGrainedLanguageTransformer.from_config(config)` creates the full model; `forward(ids, visual, return_logits=True)` accepts int64 `[B,308]` IDs and matching finite FP32 `[B,1,512]` visual features. It returns `FLTOutput(low, encoder_hidden, global_language, decoder_hidden, logits)`. `encode`, `shift_low`, `decode`, and `project_vocabulary` expose the corresponding boundaries without detaching them. `return_logits=False` is explicitly a feature-only interface; all mandatory resource probes use the default full vocabulary route. FP64 copies exist only for analytical diagnostics. Invalid dimensions, dtype, batch, device and vocabulary ranges raise; no silent casts or resizing occur.

The LE computes lookup plus Pe, then for each block: LN → self-attention delta plus input → VIM delta plus current tokens → FF delta plus current tokens. There is exactly one affine LN per LE block (`eps=1e-5`), with neither pre-VIM/pre-FF nor final LN. Ordinary attention uses separate biased Q/K/V/O, eight heads of width 64, denominator `sqrt(64)`, key-axis softmax and explicit matrix operations. No padding masks or segment boundaries constrain attention. The global feature is the physical view `encoder_hidden[:,307:308,:]`, including when the input at 307 is padding.

The LD globally shifts raw lookup embeddings by prepending existing vocabulary row 49406 and removing the final raw lookup. Independent Pd is then added. The shift does not reset at indices 77, 154 or 231. Each LD block performs causal self-attention → full-memory attention → FF → VIM, with one residual at each operation and no LN. Causal self-attention masks only future keys; the bidirectional LE memory still exposes future source tokens, as the contract explicitly acknowledges. FF is two biased affine layers with exact GELU and no output activation/dropout. Vocabulary logits are raw `F.linear(hidden, vocabulary.weight, bias=None)`; the output property returns the exact embedding parameter, with no duplicate registration.

All nineteen VIMs use the accepted [vim.py](../src/mfvlr/vim.py) byte-for-byte. Each sees the same fused visual tensor but owns independent Q/K/V/O matrices. The published denominator `sqrt(r/d)=0.125`, one visual key/value and key-axis softmax remain intact. Delta-only output has no internal residual, gate, norm or bias. Full Q/K computation graphs are preserved; their task gradients are allocated exact zeros, not disconnected `None` values.

Initialization uses a private seed-0 CPU generator and preserves caller RNG state. Wvoc/Pe/Pd receive normal `(0,0.02)` in that order; linears receive Xavier uniform gain 1 and zero biases; LE LN weights/biases receive one/zero. Draw order then follows LE blocks (self Q/K/V/O, VIM Q/K/V/O, FF1/2), then LD blocks (self Q/K/V/O, memory Q/K/V/O, FF1/2, VIM Q/K/V/O). No pretrained tensor is imported. The stage-one LE owner reserves and initializes Pd before LE to preserve this exact draw prefix, but never consumes Pd and has no LD; its reserved-inclusive parameter count is 76,011,520. That reservation is not an additional parameter in the final FLT.

External orchestration calls appearance encoding → accepted bottleneck-only appearance decoder → accepted absolute reconstruction residual/shared encoding/fusion → FLT. The orchestrator registers no learned parameters and stores no last graph. `VisualLanguageOrchestrator(mve, vd)` can omit FLT construction entirely; image-only `visual_only(image)` never accesses IDs, prompts, tokenizer or FLT. Calling `with_language` without supplying FLT raises clearly. This is a visual feature/reconstruction/mask route, not a detection or classification implementation. Standalone `.eval()` still computes FLT outputs for testing; train-only language usage is a caller-route policy.

## Shapes and actual parameter ownership

| Boundary | Shape |
|---|---|
| IDs / packed tokenizer segments | `[B,308]` / `[B,4,77]` |
| Wvoc / Pe / Pd | `[49408,512]` / two separate `[1,308,512]` parameters |
| LE/LD states, ordinary Q/K/V before splitting | `[B,308,512]` |
| Ordinary Q/K/V heads / attention maps | `[B,8,308,64]` / `[B,8,308,308]` |
| VIM query heads / singleton K/V / maps | `[B,8,308,64]` / `[B,8,1,64]` / `[B,8,308,1]` |
| FF hidden / block output | `[B,308,2048]` / `[B,308,512]` |
| Language global / fused visual input | `[B,1,512]` / `[B,1,512]` |
| Full vocabulary logits | `[B,308,49408]` |
| Accepted visual local / reconstruction / mask logits | `[B,1024,14,14]` / `[B,3,224,224]` / `[B,2,224,224]` |

Actual [tensor and backward traces](../artifacts/phase2c2/forward_backward_traces.json) record every block, FF expansion/GELU and VIM output in full-depth batch-1/2 resource runs, with finite flags, maxima/norms, and per-parameter gradient shapes. T01 additionally hooks every affine layer/LN/GELU/VIM; T08 and the unchanged VIM tests independently verify internal attention dimensions/algebra.

| Component | Unique parameters |
|---|---:|
| Shared vocabulary | 25,296,896 |
| Encoder / decoder positions | 157,696 each |
| Twelve LE blocks | 12 × 4,199,936 = 50,399,232 |
| Seven LD blocks | 7 × 5,249,536 = 36,746,752 |
| **FLT** | **112,758,272** |
| Nineteen VIMs, already included | 19,922,944 |
| VIM Q/K subset, already included | 9,961,472 |
| Accepted MVE / VD | 32,078,912 / 9,055,365 |
| **Visual + FLT** | **153,892,549** |

The [actual inventory](../artifacts/phase2c2/parameter_report.json) checks parameter identity and distinct storage, 387 FLT parameter tensors/state entries, independent depth instances and disjoint visual/language ownership. Serialization restores the tie without adding an output matrix. Combined count exceeds the paper's approximately 141.00M (Tables IX/XV, pp. 14–15) by **12,892,549** before future heads. The accepted [parameter sensitivity audit](phase2c_parameter_audit.md) remains relevant and unchanged. No dimensions, FFs or VIM parameters were removed to force a match.

## Executed tests and failure history

Environment: Windows 11, Python 3.14.3, PyTorch 2.12.1+cpu, pytest 9.1.1, four CPU threads, deterministic algorithms. The accepted test hook still prints its historical “Phase 2A” banner; case names and JUnit identify this phase. Test scalars such as mean squared logits/hidden features are bounded diagnostic probes, not production paper objectives or a training pipeline.

| Stage / attempt | Actual result | Raw evidence |
|---|---|---|
| Stage-one LE, before implementing LD | 7 passed; 6.99 s | [log](../artifacts/phase2c2/logs/encoder_01.log), [JUnit](../artifacts/phase2c2/junit/encoder_01.xml) |
| Full-FLT unit attempt 1 | 24 passed, 1 failed; 35.11 s | [log](../artifacts/phase2c2/logs/unit_01.log), [JUnit](../artifacts/phase2c2/junit/unit_01.xml) |
| Full-FLT unit attempt 2 | 25 passed, 1 failed; 37.77 s | [log](../artifacts/phase2c2/logs/unit_02.log), [JUnit](../artifacts/phase2c2/junit/unit_02.xml) |
| Integration | 9 passed; 27.91 s | [log](../artifacts/phase2c2/logs/integration_01.log), [JUnit](../artifacts/phase2c2/junit/integration_01.xml) |
| Unchanged regression | 119 passed; 41.48 s | [log](../artifacts/phase2c2/logs/regression_01.log), [JUnit](../artifacts/phase2c2/junit/regression_01.xml) |
| Complete attempt 1 | Collection error; no completed suite | [log](../artifacts/phase2c2/logs/complete_01.log), [JUnit](../artifacts/phase2c2/junit/complete_01.xml) |
| Complete attempt 2 | 160 passed, 1 failed; 117.73 s | [log](../artifacts/phase2c2/logs/complete_02.log), [JUnit](../artifacts/phase2c2/junit/complete_02.xml) |
| Final complete suite, attempt 3 | **160 passed, 1 failed; 113.27 s** | [log](../artifacts/phase2c2/logs/complete_03.log), [JUnit](../artifacts/phase2c2/junit/complete_03.xml), [test summary](../artifacts/phase2c2/test_summary.json) |

Every executed attempt and exit status is retained. Initial unit failure occurred in T17's batch-1-vs-batch-2 `encoder_hidden` comparison. T17 was split into independent fixed-batch and cross-batch cases so that a failed comparison cannot prevent exact independence/backward checks from running. Inputs, topology and tolerance stayed unchanged. This increased the unit count from 25 to 26; it did not suppress a failure.

Complete attempt 1 found the new language `test_integration.py` collided with the accepted decoder file's same import basename. Only the new file was renamed to [test_flt_integration.py](../tests/language/test_flt_integration.py), and its stage runner path updated. The unchanged accepted pytest configuration and decoder files remain intact. The complete rerun collects all 161 cases.

After complete attempt 2, the external orchestrator API made its FLT argument explicitly optional, and I06 added an omitted-FLT construction check. This changes no learned operation or parameter. The final complete rerun verifies this final API; resource measurements predate that optional-argument guard and use the same explicitly supplied-FLT numerical graph.

Coverage: FLT-T02–T05 reuse the unchanged tokenizer/golden tests; T07–T09 run in the encoder-stage file; remaining T01/T06/T10–T20 run in the full-FLT unit file. All twenty IDs are exercised, with T17 overall failed. All eight FLT-I IDs pass (I01 has two batch cases). All ten unchanged VIM-T IDs pass in twelve cases. T01 and T16 both execute full FP32 batch sizes 1 and 2. Ordering oracles use noncommuting test doubles plus actual modules; tests also cover exact initialization, causal masking, full-memory gradients, trainable padding row, tied-gradient sums, full state reload, reference/debug consistency and closed-schema rejection.

## T17 diagnosis: acceptance remains blocked

The final FP32 comparison retains `atol=1e-6, rtol=1e-5`. It fails at the same LE output with **8,318 / 157,696** elements outside tolerance (5.3%), maximum absolute difference **3.3020973e-5**. Separate diagnostics inspect every requested output after the first assertion, rather than stopping at LE:

| Same sample, B=2 versus B=1 | FP32 maximum absolute difference | Outside original FP32 tolerance | FP64 promoted-weight diagnostic maximum |
|---|---:|---:|---:|
| LE hidden | 3.3020973e-5 | 8,318 | 6.0396133e-14 |
| Physical last-token global | 2.5272369e-5 | 31 | 4.6185278e-14 |
| LD hidden | 0.0058555603 | 33,924 | 8.6011198e-12 |
| Vocabulary logits | 0.0030822754 | 3,170,109 | 5.0022209e-12 |

Evidence: [FP32 diagnosis](../artifacts/phase2c2/batch_numerics.json), [FP64 precision diagnosis](../artifacts/phase2c2/batch_numerics_fp64.json), and their raw [FP32](../artifacts/phase2c2/logs/batch_numerics.log)/[FP64](../artifacts/phase2c2/logs/batch_numerics_fp64.log) logs. FP64 uses identical initialized FP32 weights/inputs promoted before arithmetic; it does not replace the production test. All four comparisons satisfy the already specified FP64 tolerance in that diagnostic.

Lookup is exactly equal across batch shapes. First LE block differs by only 4.7683716e-7, initially within tolerance; subsequent blocks accumulate differences, and the unnormalized LD amplifies them. Exact fixed-B=2 output equality holds for encoder/global/decoder/logits when only the other sample changes. A sample-zero-only probe gives allocated exact-zero gradient to sample one's visual input and nonzero gradient to its own input. Attention operations preserve the batch axis. Together with the FP64 reduction, this is strong evidence of batch-shape-dependent floating arithmetic rather than cross-sample information leakage; the specific backend kernel/reduction implementation was not instrumented.

This diagnosis is **not an acceptance amendment**. N1 is not applied. No clipping, rescaling, normalization, dropout, masked padding, singleton shortcuts or arithmetic backend substitutions were inserted. No activations or task gradients became nonfinite on valid full-topology probes. The unchanged VIM overflow-rejection test deliberately exercises invalid overflowing arithmetic as a negative test, separately from these successful valid probes. Unnormalized decoder magnitude growth is still a risk: the first full smoke fixture's final LE/LD maxima were about 25.58 / 819.20 and logits about 492.95. Finiteness on these inputs does not establish training stability or convergence.

## Gradient pathway evidence

Actual integration measurements are stored in [FLT-I02](../artifacts/phase2c2/gradients/FLT-I02.json), [FLT-I03](../artifacts/phase2c2/gradients/FLT-I03.json), [FLT-I04](../artifacts/phase2c2/gradients/FLT-I04.json) and [FLT-I05](../artifacts/phase2c2/gradients/FLT-I05.json). These record measurements only after asserted checks succeed; the final complete execution may overwrite them with the same deterministic measurement.

- Full tied-logit probe reaches fused visual, residual image and reconstruction: gradient norms approximately 27,072.34 / 7,874.98 / 7,874.98. Appearance-head and shared decoder-trunk weight norms are approximately 19,624.05 / 52,766.33; mask-head gradients are absent, as expected for this route.
- Isolating `appearance_global.detach()+residual_global` yields nonzero reconstruction gradient (approximately 38,645.18), exactly equal elementwise to residual gradient times `sign(reconstruction-image)`. Reusing detached reconstructed values removes every VD gradient while residual shared-encoder gradients remain nonzero.
- Detaching fused visual removes every MVE/VD gradient while FLT gradients remain. A Q/K-only singleton path with V/O detached has allocated exact-zero Q/K and visual gradients.
- Detached raw lookup and direct LD visual, with connected LE memory, isolate memory propagation: memory gradient norm approximately 3,606.40, visual gradient approximately 45,044.38. Detaching memory too removes LE/visual paths; retaining only direct LD visual restores a nonzero visual gradient (approximately 1,552.43) with LE gradients absent.

Fresh resource runs summarize every named parameter's gradient shape, finite flag and norm. All 38 VIM Q/K matrices have allocated exact-zero task gradients in full batch-1/2 backward; V/O, vocabulary, positions, ordinary attention and FF gradients remain finite. Shared parameter identity is preserved across serialization and orchestration.

## Actual CPU resources and unexecuted capacity checks

The [resource summary](../artifacts/phase2c2/resource_summary.json) and [complete raw records](../artifacts/phase2c2/resources.json) contain four separate subprocess runs, each with one full warmup plus three measured iterations for no-grad forward and forward+backward. Measurements use FP32, full logits, four threads, no AMP/fused attention/checkpointing/chunking or optimizer. Every measured backward iteration releases all hooked output references after graph cleanup; results are `[0,0,0]` in all four cases.

| Route | Batch | Status | Peak process RSS MiB | Mean forward s | Mean forward+backward s | Full logits MiB |
|---|---:|---|---:|---:|---:|---:|
| FLT | 1 | PASS | 1,780.06 | 0.784 | 1.933 | 58.051 |
| FLT | 2 | PASS | 2,082.94 | 1.744 | 4.646 | 116.102 |
| MVE→VD→residual fusion→FLT | 1 | PASS | 2,332.80 | 1.603 | 4.469 | 58.051 |
| MVE→VD→residual fusion→FLT | 2 | PASS | 2,925.84 | 3.381 | 8.157 | 116.102 |
| Full batch 8 CPU capacity | 8 | **NOT RUN** | No measured peak | — | — | 464.406 analytical only |
| GPU capacity / numerical tests | Any | **NOT RUN** | CPU-only PyTorch; CUDA unavailable | — | — | — |

RSS is sampled every 2 ms from before model construction through diagnostic cleanup, and Windows lifetime peak working set is also recorded. Both peak measures agreed on this host. These process measurements include tensor-statistic/finite-check overhead and allocator/interpreter memory; they are not deployment-only activation peaks. Short sampler peaks may be missed. Timings exclude post-execution statistic/gradient summaries, include production finite checks, and have no performance acceptance threshold.

Host RAM is 16,901,771,264 bytes. Batch-8 preflight observed 5,372,989,440 available bytes (about 5.00 GiB), below a disclosed conservative requirement of 8 GiB available for integrated full-FP32/autograd execution. **This preflight is an engineering resource policy, not an observed OOM or a proven 8-GiB requirement.** Batch-8 capacity remains unverified; no smaller run is represented as success. GPU absence is a resource result, not a skipped pytest case.

Actual FLT parameter and gradient tensors each occupy 430.139 MiB; combined parameter tensors occupy 587.053 MiB. The integrated diagnostic has no gradients for the unused two-class mask head, reducing actual gradient allocation by 130 FP32 elements (520 bytes). Logits have exactly 15,217,664 / 30,435,328 elements for batch 1/2. The accepted contract's illustrative activation inventories and optimizer estimates remain estimates, not these measured peaks; no optimizer was instantiated.

## Configuration, dependencies and reproduction

[flt_reference.yaml](../configs/flt_reference.yaml) and [flt_debug.yaml](../configs/flt_debug.yaml) resolve identical topology, initialization, assets, tolerances and approval. Debug overlays permit only the exact diagnostic replacement and resolve the fixed base in their own directory. Duplicate, unknown/missing, wrong-typed, changed-architecture, unsafe base, nested/cyclic and approval/tolerance escalation inputs are rejected by the closed schema. [Resolved snapshots](../artifacts/phase2c2/resolved_configs) include both language profiles and accepted visual/foundation profiles.

The [hash-pinned FLT dependency lock](../configs/flt_requirements.lock) exactly preserves the accepted foundation pins: ftfy 6.3.1, regex 2026.6.28, wcwidth 0.2.13. [Dependency evidence](../artifacts/phase2c2/dependency_report.json) records installed versions and the accepted wheel manifest. Regex's bundled compiled wheel is **CPython 3.14 Windows amd64 only**; compatible wheels for other platforms/Python versions are unavailable in this review package, with external availability unverified. No alternate dependency version is substituted. Torch/pytest/PyYAML/psutil runtime versions are reported rather than silently installed. Source imports use `src` and workspace-local `.phase2c1_deps`; accepted project metadata remains unchanged, so an installed-wheel packaging workflow is not claimed verified.

Tokenizer manifest/golden SHA-256 values remain `179b071a483620d95f4c16e591bd2e64fd334621c13f76508492569a73d06949` / `1e8950d386d765c5cd13e628e751356afe4ac11e35482701eadc7a696761d0e6`. Official tokenizer revision remains `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`. No pretrained CLIP or language-model weights are downloaded or loaded. No model forward requires a network request.

From the extracted workspace on the documented compatible platform, with the reported PyTorch/pytest/PyYAML/psutil environment available:

```powershell
python -m pip install --no-index --find-links artifacts/phase2c1/wheels --require-hashes --no-deps --target .phase2c1_deps -r configs/flt_requirements.lock
python tools/run_phase2c2_review.py --stage encoder
python tools/run_phase2c2_review.py --stage unit
python tools/diagnose_phase2c2.py
python tools/analyze_flt_batch_numerics.py
python tools/analyze_flt_fp64_numerics.py
python tools/run_phase2c2_review.py --stage integration
python tools/run_phase2c2_review.py --stage regression
python tools/run_phase2c2_review.py --stage complete
python tools/collect_phase2c2_evidence.py
python tools/package_phase2c2_review.py
```

The unit/complete commands currently return **exit code 1 for the disclosed T17 failure**; do not add exclusions, xfails or tolerance changes to make them return zero. The resource command must precede integration because I08 checks its actual fresh-process evidence. Stage runner logs commands, PYTHONPATH, stdout/stderr and exit statuses; subsequent attempts create new numbered files. The LE-first order was observed during development; on an extracted completed workspace the stage remains independently runnable.

## Review package and final status

The [review ZIP](../artifacts/phase2c2/phase2c2_review_package.zip) includes actual accepted/new source, all tests, immutable tokenizer artifacts and independent golden fixtures, hash-pinned wheels, scoped and resolved configurations, historical failures, final raw pytest/JUnit logs, parameter/gradient/numerical/resource reports, approved contract documents and this report. It excludes generated bytecode, installed dependency trees, model checkpoints and nested old ZIPs. Payload hashes and fixed ZIP timestamps are verified by [package_phase2c2_review.py](../tools/package_phase2c2_review.py); [manifest](../artifacts/phase2c2/manifest.json), [ZIP checksum](../artifacts/phase2c2/phase2c2_review_package.zip.sha256) and [package verification](../artifacts/phase2c2/package_verification.json) distinguish package integrity from test acceptance. Historical failures are retained, and the package explicitly records BLOCKED acceptance.

**Ready for review; not ready for strict acceptance.** Remaining issues are T17's failed original FP32 cross-batch requirement, unverified batch-8/GPU capacity, the paper parameter-count discrepancy and unnormalized-depth numerical/training risks. No new tolerance or architectural amendment is requested or preapproved by creating this report. The implementation remains at the approved baseline and stops here for review; no next component is implemented.
