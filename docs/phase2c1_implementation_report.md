# Phase 2C.1 implementation and verification report

2026-10-11. **Tokenizer and standalone VIM foundation is implemented and verified; awaiting user acceptance.** The complete suite passed **119/119 cases**: 79 new foundation cases and all 40 unchanged accepted visual cases. No failed tests, skips, xfails or tolerance amendments occurred. Full FLT remains unapproved and unimplemented.

## Scoped approval record

Authority: the user's explicit Phase 2C.1 instruction. These are approved engineering assumptions, not confirmation of the authors' undocumented implementation. The [Phase 2C design](phase2c_language_contract.md) remains the historical full-architecture proposal; this narrower record supersedes its pending status only for the portions listed here.

| Decision | Status for Phase 2C.1 | Exact scope |
|---|---|---|
| L01 | APPROVED | Hierarchical L1–L4 prompts, Fig. 6's nine-generator registry, strict fake metadata checks |
| L02 | APPROVED | CLIP-compatible tokenizer, pinned tokenizer-only source/data/dependencies, independent golden fixtures; no model weights |
| L03 | APPROVED | Four independent 77-slot segments, BOS/EOS/right-padding, 308-token concatenation, overflow rejection |
| L09 portion | APPROVED | One standalone 512-dimensional, eight-head singleton-key VIM primitive; bias-free Q/K/V/O; published scale; key softmax; delta-only output |
| L11 portion | APPROVED | Private seed-0 CPU Xavier initialization, FP32 runtime and FP64 analytical copies, unchanged numerical tolerances |
| L12 portion | APPROVED | Prompt/tokenizer/VIM APIs and direct differentiable visual-feature boundary tests |
| L13 portion | APPROVED | Foundation-only configuration, provenance, tests and review artifacts |
| Remaining L09/L11–L13 portions | PROPOSED | No approval for 19-block allocation, full FLT initialization/configuration/orchestration |
| L04–L08, L10, full FLT | PROPOSED | No LE/LD, pooling/mask/shift/FFN/vocabulary-embedding implementation |
| Losses, adapter, classifier, datasets, full training/evaluation | DEFERRED | Not implemented or authorized |

Machine-readable approval is present independently in [reference](../configs/foundation_reference.yaml) and [debug](../configs/foundation_debug.yaml) configurations and the [foundation snapshot](../src/mfvlr/foundation_contract_snapshot.json). The overlay cannot broaden approval or change architecture, assets or tolerances. Accepted M1–M5, D01–D08 and N1 retain their original scopes.

## Implemented components

| Source | Behavior / paper traceability |
|---|---|
| [prompts.py](../src/mfvlr/prompts.py) | `PromptMetadata` and `PromptBuilder`; templates/depicted labels from MFVLR §III-D, Fig. 6 p. 6; exact registry and strict validation are L01 assumptions |
| [tokenizer.py](../src/mfvlr/tokenizer.py) | Independent byte-BPE implementation plus offline adapter; tokenizer reference §III-D p. 6 and s/n dimensions §IV-A p. 8; framing/packing are L02/L03 assumptions |
| [vim.py](../src/mfvlr/vim.py) | `VisionInjectionDelta` and per-call `VIMTrace`; Eqs. (4)–(11), §III-D p. 7; no internal residual, normalization, bias, gate or learned scale |
| [foundation_config.py](../src/mfvlr/foundation_config.py) | Closed schema with explicit scoped approval; one-level debug overlay permits only diagnostic overrides |

The prompt builder preserves batch order, repeats the real-face sentence four times for genuine metadata, and rejects unknown/aliased/inconsistent fake generator/type/family combinations. Unused fake fields do not affect genuine prompts. Tokenization returns CPU int64 `ids [b,308]`, `segment_ids [b,4,77]`, payload lengths and absolute EOS positions `[b,4]`, plus a boolean padding-position mask `[b,308]`. A frozen record prevents rebinding its fields; its tensors remain caller-owned and mutable. Padding is identified by position, not by treating vocabulary ID0 as a dedicated PAD symbol. A 75-token payload fits; 76 tokens raise. There is no truncation option.

VIM accepts matching FP32 `[b,n,512]` language and `[b,1,512]` visual tensors with positive b/n. FP64 module copies support analytical tests. It validates shape/dtype/device/finiteness, computes all four projections and attention explicitly, returns a delta, and stores no last-call graph. Finite-input numerical overflow raises `FloatingPointError`; it is not hidden with a constant singleton shortcut. Initialization consumes a private CPU generator in Q/K/V/O order and preserves caller RNG state.

For r=8,d=512 the published `sqrt(r/d)` denominator is 0.125. Softmax over the single key gives exactly one for finite scores. The implementation nevertheless retains Q/K operations and their allocated zero gradients. V/O and visual-input gradients remain nonzero on controlled fixtures. Residual ownership is tested through an external identity-add fixture, not through unapproved LE/LD blocks.

## Tokenizer provenance and independent oracle

Official upstream revision: [OpenAI CLIP commit d05afc436d78f1c48dc0dbf8e5980a9d471f35f6](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6), commit date 2026-03-25. The immutable source URLs, sizes, source/BPE/license hashes, decompressed BPE hash, ordered vocabulary hash and ordered byte-map hash are recorded in [tokenizer_assets/manifest.json](../src/mfvlr/tokenizer_assets/manifest.json). The MIT license is preserved alongside the artifacts. The inspected official [tokenizer](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/simple_tokenizer.py) and [tokenize wrapper](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/clip.py) are tokenization evidence, not sources of learned weights.

| Artifact | SHA-256 |
|---|---|
| Official `simple_tokenizer.py` | `d1b09f10ed3d1e2c343619cb98cbfacb92363dd8607526a43aa7fe2d60d15bad` |
| Official `clip.py`, reference only | `9540f200fbf8145479fa655382a56dab048d238cc698b9cbd8df3b6d86d3f1b6` |
| BPE gzip | `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` |
| Decompressed BPE | `67603cfda2e032ad77b5f8808af37789d590db664b26df8705d2bf8b3c553fc8` |
| Final tokenizer manifest | `179b071a483620d95f4c16e591bd2e64fd334621c13f76508492569a73d06949` |
| Golden fixtures | `1e8950d386d765c5cd13e628e751356afe4ac11e35482701eadc7a696761d0e6` |

The golden generator [prepare_phase2c1_fixtures.py](../tools/prepare_phase2c1_fixtures.py) imports **no production mfvlr module**. It executes the unmodified pinned official `SimpleTokenizer` and AST-extracts only the official `tokenize` function, avoiding the surrounding model-loading module. It generates literal metadata/prompt cases independently of `PromptBuilder`, then produces 10 complete four-level prompt fixtures and 26 text fixtures including Unicode, repair/escaping/whitespace, punctuation, empty input and exact overflow boundaries. Existing golden bytes cannot be overwritten by a changed oracle without an explicit intervention. The production adapter's merge implementation is separately authored and is compared against those stored arrays.

The runtime checks the manifest digest, every pinned file, ordered vocabulary/byte map and dependency versions. Dependencies are pinned to ftfy 6.3.1, regex 2026.6.28 and wcwidth 0.2.13; [the hash lock](../configs/foundation_requirements.lock) and [wheel hashes](../artifacts/phase2c1/dependency_wheels.json) accompany the wheels. No pretrained CLIP/model tensor was downloaded, loaded or initialized. No runtime network access occurs; an explicit offline test replaces socket connection with a raising sentinel.

Acquisition initially encountered the sandbox's network restriction and succeeded after tool-mediated network permission. The missing ftfy dependency was installed into workspace-local `.phase2c1_deps`, without altering the accepted project dependency file. These were environment preparation steps, not failed numerical tests.

## Actual verification results

Environment: Windows, Python 3.14.3, PyTorch 2.12.1+cpu, pytest 9.1.1, four CPU threads and deterministic algorithms. The unchanged root test hook still prints its historical “Phase 2A” banner; collection names and JUnit identify the actual stage. No GPU tests ran because this host's PyTorch build is CPU-only.

| Executed stage | Cases | pytest-reported duration | Log / JUnit |
|---|---:|---:|---|
| Tokenizer, prompts, assets/goldens | 45 passed | 1.03 s | [log](../artifacts/phase2c1/logs/tokenizer_01.log), [JUnit](../artifacts/phase2c1/junit/tokenizer_01.xml) |
| VIM-T01–T10 | 12 passed, 10 IDs | 0.98 s | [log](../artifacts/phase2c1/logs/vim_01.log), [JUnit](../artifacts/phase2c1/junit/vim_01.xml) |
| All new foundation tests | 79 passed | 3.24 s | [log](../artifacts/phase2c1/logs/foundation_01.log), [JUnit](../artifacts/phase2c1/junit/foundation_01.xml) |
| Unchanged accepted visual regression | 40 passed | 34.82 s | [log](../artifacts/phase2c1/logs/regression_01.log), [JUnit](../artifacts/phase2c1/junit/regression_01.xml) |
| **Complete suite** | **119 passed** | **36.26 s** | [log](../artifacts/phase2c1/logs/complete_01.log), [JUnit](../artifacts/phase2c1/junit/complete_01.xml) |

JUnit includes different session bookkeeping intervals; its complete-suite `time` is 37.408 s. The structured [test summary](../artifacts/phase2c1/test_summary.json) retains the actual values, names and outcomes. Every stage has zero failures/errors/skips/xfails and exit code 0. VIM-T01 runs n=1,b=1 and n=308,b=1/2. The 12 VIM cases cover all ten applicable IDs; hypothetical 19-block totals are arithmetic checks, not an allocated language architecture. Tokenizer tests implement applicable FLT-T02–T05 portions; six FDN-C IDs cover scoped configuration/API/persistence/preservation. Full-FLT-only tests remain unimplemented rather than being disguised as skipped cases.

VIM tests exercise raw published logits (plus a test-only two-key mathematical scale example), finite singleton probabilities, FP64 output/derivative formulas, Q/K invariance, residual identity, head layout, parameter independence, deterministic initialization, invalid boundaries, overflow rejection, finite differences and exact fixed-batch independence. Production rejects multi-key visual input. FDN-C06 passes the accepted MVE fused feature directly to the primitive and proves reconstruction gradients disappear with a detached visual control; it does not construct a training pipeline.

FP32 tolerances remain `atol=1e-6, rtol=1e-5`; FP64 remain `atol=1e-10, rtol=1e-8`. Structural zeros/ones and fixed-batch unaffected outputs use exact equality. N1 was not applied to foundation tests. Its only use remains the unchanged accepted MVE-T12 cross-batch-size test.

## Parameter and gradient evidence

The [parameter report](../artifacts/phase2c1/parameter_report.json) confirms:

| Implemented component | Unique parameters |
|---|---:|
| Standalone VIM, four matrices | **1,048,576** |
| Q/K subset, included above | 524,288 |
| Accepted MVE | 32,078,912 |
| Accepted VD | 9,055,365 |
| Accepted visual subtotal | 41,134,277 |
| Visual + one standalone VIM, arithmetic subtotal | 42,182,853 |
| Prompt builder / tokenizer | 0 learned parameters |

All forward/backward tensor shapes and parameter/input gradient norms are recorded in [vim_diagnostics.json](../artifacts/phase2c1/vim_diagnostics.json). Additional actual full-length probes independently compare the explicit attention path to its singleton algebraic identity:

| dtype / batch | Delta max absolute reference error | Visual-gradient max absolute error | Q/K gradient norms | Visual gradient norm |
|---|---:|---:|---|---:|
| FP32 / 1 | 9.537e-7 | 4.657e-8 | exactly 0 / 0 | 0.151857 |
| FP32 / 2 | 1.311e-6 | 1.770e-8 | exactly 0 / 0 | 0.107225 |
| FP64 / 1 | 2.609e-15 | 6.744e-17 | exactly 0 / 0 | 0.141311 |
| FP64 / 2 | 3.109e-15 | 4.163e-17 | exactly 0 / 0 | 0.111141 |

All comparisons pass the unchanged combined absolute/relative criterion. A maximum absolute difference exceeding `atol` alone does not mean the `atol + rtol×abs(reference)` criterion failed. V/O gradient norms are positive and finite in every case. Language-to-delta gradients are exactly zero; the external residual identity carries language gradients in VIM-T06/T09.

The same diagnostic recorded single-call forward+backward times of about 0.0103/0.0141 s in FP32 b=1/2 and 0.0156/0.0255 s in FP64 b=1/2. These are observations, not warmed benchmark estimates. Process RSS before/after is logged, explicitly **not** claimed as peak memory. No batch-8 full-language/logit capacity or GPU performance is established by these standalone tests.

## Preservation and sensitivity findings

[preservation.json](../artifacts/phase2c1/preservation.json) verifies all **110** files from the accepted Phase 2B manifest remain byte-identical, including MVE/VD source, existing tests/configurations, the original paper and accepted reports. No existing 40-case test was edited. The three Phase 2C design documents also remain unchanged; the approval scope is recorded separately here.

[The parameter audit](phase2c_parameter_audit.md) provides formulas and counterfactual totals for FFN structures/widths, normalization, biases, VIM sharing and vocabulary tying. Its 153,892,549 baseline is hypothetical, not an instantiated FLT. Large uncertainty comes from FFN structure/width and depth sharing; bias/norm choices have much smaller count effects. No architecture dimensions were adjusted to match the reported 141.00M. All alternatives remain unapproved, and no parameter-count resemblance is treated as evidence of author fidelity.

## Reproduce and inspect

Use the source checkout/review ZIP root with the recorded Python/PyTorch/pytest/PyYAML environment. The accepted `pyproject.toml` and package manifest are unchanged; this foundation's supported review workflow is source-based via `PYTHONPATH`, not a newly verified distributable wheel.

On the recorded Windows CPython 3.14 platform, install the bundled tokenizer dependencies offline:

```powershell
python -m pip install --no-index --find-links artifacts/phase2c1/wheels --require-hashes --target .phase2c1_deps -r configs/foundation_requirements.lock
python tools/run_phase2c1_review.py --stage complete
python tools/diagnose_phase2c1.py
python tools/audit_phase2c_parameters.py
```

The runner explicitly uses workspace-local dependencies and `src`, retains sequential attempt logs/JUnit, and returns pytest's actual exit code. The supplied regex wheel/hash is Windows CPython 3.14-specific; another platform needs a separately recorded wheel/hash for the same dependency version, not a silent version substitution. Core PyTorch/pytest/PyYAML wheels are not bundled; their versions are recorded in the accepted environment and test logs.

To replay the independent token oracle using already bundled official assets:

```powershell
$env:PYTHONPATH = "$PWD\.phase2c1_deps;$PWD\src"
python tools/prepare_phase2c1_fixtures.py
```

Optional acquisition replay uses `python tools/acquire_phase2c1_tokenizer.py` followed by the oracle preparation command; only those preparation steps need network access. Normal test/model use is offline. The oracle generator refuses changed golden fixtures and records its environment. Do not replace golden arrays with production-tokenizer output.

The [review ZIP](../artifacts/phase2c1/phase2c1_review_package.zip) contains actual source/tests, scoped and resolved configurations, pinned tokenizer source/BPE/license, golden arrays, dependency wheels/hash lock, accepted visual evidence required by preservation tests, new logs/JUnit, diagnostic/audit JSON and this report. The [package manifest](../artifacts/phase2c1/manifest.json) hashes every payload file; its own hash and the ZIP are excluded from its self-list. [The ZIP checksum](../artifacts/phase2c1/phase2c1_review_package.zip.sha256) is separate to avoid self-reference. [Package tooling](../tools/package_phase2c1_review.py) uses sorted paths/fixed ZIP timestamps and verifies archived file hashes after writing.

## Limitations and stopping point

There are **no unresolved failures in the approved Phase 2C.1 CPU test scope**. Singleton VIM's zero Q/K task gradients are expected mathematical behavior, not a repaired failure. The generator registry intentionally rejects unreviewed dataset aliases/names. Cross-platform dependency packaging and GPU execution are not verified. Full FLT parameter fidelity, normalization/FFN choices, padding/pooling/shifting policies and training behavior remain unresolved design questions.

No LE, LD, trainable vocabulary embedding/output, paper loss, classifier, adapter, dataset loader or full training/evaluation pipeline was implemented. Phase 2C.1 verification is complete and submitted for review; the next component requires approval.
