# MFVLR reproduction specification and implementation plan

Status: proposed, pending research review. This is documentation only. No implementation, model training, dataset download, experiment run, or test execution is authorized by this document itself. The user's current request ends after these four research documents.

Source: [MFVLR.pdf](../papers/MFVLR.pdf), PDF pp. 1–18. [paper_analysis.md](paper_analysis.md) contains architectural requirements `R01`–`R29`, all published equations, numerical targets and discrepancies. [architecture.md](architecture.md) specifies tensors and data dependencies. [assumptions.md](assumptions.md) lists open decisions `A01`–`A35`.

**EXPLICIT** denotes paper requirements; **DERIVED** denotes mathematically necessary consequences; **ASSUMED** denotes proposed implementation, testing or experiment conventions. All software module names, verification tolerances, phase gates and artifact formats below are **ASSUMED engineering plans**, even when they verify an explicit requirement. They have not been executed.

## Reproduction scope and evidence standard

The proposed primary target is an independent PyTorch reconstruction of the paper's specified graph: shared appearance/residual encoding, shared appearance/mask decoding, summed visual embeddings, hierarchical prompt encoding/reconstruction with VIM, and the six-term objective. It must retain image-only inference and from-scratch training (**EXPLICIT**, p. 4 §§II-B/III-A; pp. 5–8 §§III-B–E).

Two distinct claims will be assessed:

1. **Structural/mathematical reproduction:** implementation satisfies the paper's explicit shapes, graph, sharing, block ordering and selected, fully disclosed interpretations of incomplete equations. This is possible with documented assumptions.
2. **Empirical replication:** reported ACC/AUC/mIoU are comparable under matched data, training and evaluation protocols. Exact replication is currently unresolved because splits, training duration, architecture internals and reductions are absent. Passing unit tests does not establish empirical replication.

The primary reference formulation must preserve the published singleton VIM and sqrt(r/d) scale. A corrected-scale or richer-attention experiment can be studied later, but must have a separate configuration and claim. Scalar CE/AR/FL interpretations must be recorded as interpretations, because the literal equations do not fully specify an executable objective. No existing MFVLR code is a source or dependency.

## Frozen paper settings

| Setting | Value | Classification / source |
|---|---|---|
| RGB input | 3×224×224 | EXPLICIT p. 5 §III-B |
| Image labels | Real=0, fake=1; two classes | EXPLICIT Fig. 4 p. 5 and p. 8 §IV-A |
| Local image representation | 1024×14×14 | EXPLICIT p. 8 §IV-A |
| Feature/token width d | 512 | EXPLICIT p. 8 |
| Image transformer depth B | 4 | EXPLICIT p. 8; best tested depth Table XI p. 14 |
| LE depth E | 12 | EXPLICIT p. 8; Table XVI p. 15 |
| LD depth D | 7 | EXPLICIT p. 8; Table XVI p. 15 |
| Vocabulary s | 49408 | EXPLICIT p. 8 |
| Token count n | 308 | EXPLICIT p. 8; FF++ and ablation length changes not stated |
| Batch b | 8 | EXPLICIT p. 8 |
| Visual token count | 14×14+1=197 | DERIVED from Eq. (1), local shape and class token |
| VIM visual K/V length | 1 | EXPLICIT p. 7 §§III-D, Eqs. (8)–(9) |
| VIM scale denominator | sqrt(r/d) | EXPLICIT printed Eq. (10), p. 7 |
| Initial CMC temperature | 0.07, trainable | EXPLICIT p. 8 Eqs. (20)–(23) |
| KL temperature | 0.5 | EXPLICIT p. 8 Eq. (19); application convention A19 |
| Mask construction threshold | 0.1 | EXPLICIT pp. 7–8; sensitivity 0.05,0.10,0.15 in Fig. 10 |
| Loss terms / coefficients | FD, LR, CMC, FL, AR, KL, all coefficient 1 | EXPLICIT Eq. (27), p. 8; internal reductions remain unresolved |
| Optimizer | Adam | EXPLICIT p. 8; not AdamW |
| Initial LR | 1e-4 | EXPLICIT p. 8 |
| Weight decay | 1e-3 | EXPLICIT p. 8 |
| LR schedule | Reduce by factor 10 every 15 epochs | EXPLICIT p. 8; exact scheduler boundary A20 |
| Initialization regime | Train from scratch | EXPLICIT p. 4 §II-B; distributions A22 |
| Framework / reported hardware | PyTorch / Tesla V100 GPU | EXPLICIT p. 8; versions/device count not stated |
| Primary dataset | GenFace | EXPLICIT pp. 8–9, citation [42] |
| Cross-dataset training | FF++, L1 and L2 prompts only | EXPLICIT p. 10 §IV-B |
| Cross-dataset evaluation | FF++, Celeb-DF, DFDC, DF-1.0 | EXPLICIT pp. 9–10, Table VIII p. 13 |
| Metrics | ACC, AUC, mean class-wise IoU | EXPLICIT p. 8; aggregation/thresholds A24 |
| Reported full parameter count | 141.00M | EXPLICIT Tables IX/XV pp. 14–15; accounting scope unknown |
| Reported class-token model FLOPs | 173.04G | EXPLICIT Table XV p. 15; profiling convention unknown |

No paper value is available for attention head counts, FF widths, convolutional stage definitions, dropout, full norm layout, head biases/kernels, image normalization/augmentation, optimizer beta/epsilon, total epochs, warmup, clipping, gradient accumulation, AMP, seeds, validation selection, train/test sample counts, or exact corruption severities. These cannot be hidden framework defaults; the resolved configuration must include them as ASSUMED or explicitly unknown (A01–A35).

## Operational objective contract for review

The following is an **ASSUMED candidate interpretation**, not a replacement transcription of Eqs. (17)–(27). It gives reviewers a concrete objective to accept or revise. The original equations remain in paper_analysis.md.

Let `z_fd`, `z_mask`, and `z_tok` denote raw FC/convolution/vocabulary logits, and let `CE(z,k)` mean the negative log-softmax at class k. Integer targets are algebraically equivalent to one-hot categorical targets (**DERIVED**). Define:

| Loss | Candidate executable meaning | What is paper-explicit / what is assumed |
|---|---|---|
| FD | Average `CE(z_fd[u],y_index[u])` over b | Binary FC output and batch average explicit, Eq. (26); log-softmax interpretation A18 |
| AR | Average `(Ipre-I)^2` over b×3×224×224 | Squared error and batch average explicit, Eq. (17); channel/spatial mean A17 |
| FL | Average two-class `CE(z_mask[u,:,h,w],M[u,h,w])` over b×224×224 | Two classes and mask supervision explicit; CE normalization, class expansion and spatial mean A18 |
| KL | `P=softmax(Tl/0.5)`, `Q=softmax(Tlpre/0.5)`; average `sum_d P*(log P-log Q)` over b | Direction and feature softmax temperature explicit Eq. (19); division convention and no detachment A19 |
| CMC | Candidate raw score `C=Iv Tl^T/tau`; `(CE_rows(C,diagonal)+CE_rows(C^T,diagonal))/2` | Batch pairing/symmetric average explicit Eqs. (20)–(23); raw dot versus cosine A14 remains review choice |
| LR | Sum `CE(z_tok[u,x],token_id[u,x])` over all n positions, then mean over b | Tied output Eq. (24) and token sum Eq. (25) explicit; log-softmax and all-position target handling A10/A18 |
| Total | Add all six scalars | EXPLICIT Eq. (27); no balancing weights or temperature-square factor |

CMC reverse probabilities must be a row softmax of the transposed raw scores; simply transposing already normalized forward probabilities is generally wrong (**DERIVED**, Eqs. (20)–(21)). Use the raw `Iv`/`Tl` for their other consumers even if cosine normalization is selected solely for CMC.

Use numerically stable log-softmax/log-sum-exp identities as an **ASSUMED engineering implementation** of the selected categorical interpretation, not explicit materialization of `log(softmax)` when avoidable. No implementation should evaluate logs of negative raw logits and silently clamp them to make the total finite.

For review, the candidate reference retains mean AR/FL and token-summed LR exactly as disclosed above. A token-mean alternative, image/pixel-sum alternatives, normalized CMC, and alternate padding policies must be separate run variants. They must never be relabeled as the same objective after observing test results.

## Proposed package boundaries

All paths below are **ASSUMED future organization**; they are not files created in this task.

| Future responsibility | Suggested module boundary | Requirements and interfaces |
|---|---|---|
| Immutable configuration/provenance | `config`, `manifest` | Resolve EXPLICIT/DERIVED/ASSUMED fields and source IDs; A35 |
| Dataset metadata and split loading | `data/datasets`, `data/splits` | Image authenticity, forgery type/family/generator, source-pair IDs and grouping; R01/R02/R21/R29 |
| Mask construction/preprocessing | `data/masks`, `data/transforms` | Explicit pair-difference rule and declared grayscale/crop/resize semantics; A01/A23 |
| Prompt construction/tokenization | `text/prompts`, `text/tokenizer` | L1–L4 / FF++ L1–L2, pinned 49408 vocabulary, exact 308-slot policy; R02/R13 |
| Local visual encoder | `models/visual_encoder` | `encode_local(image)` → local map and declared skip state; R03 |
| Global visual transformer | same module or `models/visual_transformer` | `encode_global(local)` → 197-token sequence / global token; R04/R05 |
| Shared visual decoding | `models/vision_decoder` | local state → reconstruction and two-channel mask logits; R06/R07 |
| Visual orchestration | `models/mfvlr` | appearance encode/decode → abs residual → SAME encoder → sum; R08–R10 |
| VIM primitive | `models/vision_injection` | text states + one global visual token → injection delta; declared scale/axis/residual A12/A13 |
| LE/LD | `models/language` | Exact block order, shift policy, LE memory, final pooling; R14–R18 |
| Tied vocabulary/heads | `models/heads` or owners above | Vocabulary sharing; single-FC adapter/detection; R11/R12/R19 |
| Objectives | `losses` | Six separately observable scalar terms; R20/R22–R26 |
| Training/checkpoints | `train`, `checkpoint` | From-scratch Adam, scheduler, selection, RNG persistence; R27/A20–A23/A35 |
| Inference/export | `infer` | image-only API returning mask/detection logits; R28 |
| Metrics/protocol runners | `evaluate`, `metrics` | Explicit split manifests, thresholds, aggregations, reported reference cells; R29/A24 |

Exported visual artifacts should contain exactly the modules needed for the visual graph, including AD and RE. The training checkpoint may also contain language, adapter and objective state. Trainable temperature registration must survive checkpoint save/load.

## Implementation stages and gates

1. **Freeze the research contract.** Resolve critical choices A01–A03, A07–A19, A21 and A23; record all other defaults. Acceptance: no unresolved value required to instantiate the selected baseline, scalar objective or training protocol; each departure is labeled. No need to resolve optional qualitative/alternate-token ablations before the main baseline.
2. **Prepare deterministic data and targets.** Build manifests, pair verification, mask and prompt/token pipelines. Acceptance: exact synthetic mask/token fixtures; no split/source/video leakage; provenance hashes and sample counts by class/type/generator. Obtain data through authorized sources later; do not guess missing pairs.
3. **Build the visual graph.** Implement shared U-Net/transformer, AD/MD, residual and detection. Acceptance: shape, sharing, forward dependency and finite-gradient tests below. Start with small diagnostic batches, then verify b=8 before benchmark training.
4. **Build FLT and VIM.** Implement LE/LD, shift, selected padding/attention behavior, tied vocabulary, adapter and temperatures. Acceptance: analytical VIM behavior, ordering, shift and weight-tying tests. Do not treat zero singleton-Q/K gradients as a bug to repair.
5. **Integrate objective and trainer.** Implement the selected scalar-loss contract, Adam schedule, checkpoint/resume and reporting. Acceptance: independent loss fixtures and total-loss/gradient checks, scheduler boundary test, deterministic resume at the selected precision.
6. **Run a small overfit diagnostic.** This is an ASSUMED engineering sanity test, not a published protocol. Use a fixed, disjoint diagnostic fixture containing real, EFS, AM and FS examples with valid masks/metadata. Acceptance: finite training, meaningful reduction of declared supervised losses and correct sample associations; no hard promise that all six losses reach zero because CMC negatives and shared reconstruction goals can conflict. Record curves and inspect masks/residuals. Diagnostic settings must not be reported as the benchmark recipe.
7. **Run the primary benchmark and selected ablations.** Start with DiffFace training and FSLSD/LatDiff/FaceSwapper evaluation to overlap Tables VI/X/XI/XIII, then complete Tables II–VIII if data are available. Acceptance: source-only checkpoint selection, frozen configuration, all targets reported with deltas, and no target-test tuning.
8. **Profile and report.** Profile training/inference separately and disclose assumptions. Acceptance: unique parameter accounting, reproducible metric computation, complete deviation ledger. Stop claims at “documented reimplementation” if matching-data evidence for empirical replication remains unavailable.

These phases are a future plan. The present research task is complete when the four documents are internally consistent and available for review; it does not execute phases 1–8.

## Verification tests and acceptance criteria

Every test below is planned, not run. Tolerances are **ASSUMED** numerical criteria: use exact equality for IDs, binary masks, shapes and parameter identity; FP64 small analytical fixtures with approximately `atol=1e-10, rtol=1e-8`; FP32 forward equivalence approximately `atol=1e-6, rtol=1e-5`, adjusted only with a documented numerical reason. Benchmarks have separate statistical criteria below.

| Test ID | Test | Acceptance criterion | Evidence / decision |
|---|---|---|---|
| V01 | Source-pair mask construction | Real is all zero; EFS all one; identical source/fake gives zero; synthetic changed region agrees with declared grayscale and threshold; threshold boundary covered | pp. 7–8 §III-E, A01 |
| V02 | Geometry and split integrity | Same geometric crop/resize for paired data; binary-mask resizing preserves class IDs; no forbidden source/video/identity intersections; missing pair fails visibly | p. 4 data dependencies; A01/A23 |
| V03 | Prompt/metadata fidelity | Each Fig. 6 template matches its class/type/family/generator; real repeats L1–L4; FF++ excludes unsupported L3/L4 | Fig. 6 p. 6; §IV-B p. 10 |
| V04 | Tokenizer and sequence packing | Pinned artifacts; IDs in [0,49407]; declared 308 positions, truncation and special tokens exactly match fixtures | pp. 6/8; A08–A10 |
| V05 | Visual shape trace | I→[b,1024,14,14]→[b,197,512]; Ipre/Ir [b,3,224,224]; Mpre [b,2,224,224]; global tokens [b,1,512]; logits [b,2] | pp. 5–8; R01/R03–R11 |
| V06 | Exact weight sharing | IE/RE parameter objects identical; AD/MD trunk parameter objects identical; separate 3-channel and 2-channel heads; optimizer contains each trainable tensor once | §§III-B/C pp. 5–6 |
| V07 | Residual operator and fusion | Controlled inputs give absolute difference elementwise; identical reconstruction gives zero; `Iv=Ig+Ir_g` exactly; no detach under chosen A07 | pp. 5–6 §III-B |
| V08 | No recursive/circular computation | Appearance encoder and residual encoder each execute one role call per image batch; residual is not reconstructed again; decoder inputs are appearance features | §III-A p. 4, Fig. 4 |
| V09 | LE block order | Trace LN/MHA+residual → VIM+residual → FF+residual, E=12; last-token index matches selected policy | pp. 6–7 §III-D, Eq. (3), A09 |
| V10 | LD shift and causal self-attention | Declared BOS prepend/drop-last matches original targets; at a fixed self-attention input, perturbing future positions cannot change earlier masked-self-attention outputs | p. 7 Eqs. (12)–(13), A10/A11 |
| V11 | LD order and memory dependency | D=7; masked MHA→memory MHA→FF→VIM with declared residual ownership; changing memory can change reconstructed outputs | Eqs. (13)–(16), A11/A13 |
| V12 | VIM analytic singleton behavior | Softmax weights one along single key; with dropout disabled, injection equals broadcast `Iv Wval Wfc`; changing Q/K only leaves output unchanged | p. 7 Eqs. (4)–(11), DERIVED under A12 |
| V13 | VIM gradients and scaling | Q/K task-loss gradients numerically zero in singleton fixture; V/output gradients can be nonzero; synthetic multi-key formula test matches sqrt(r/d), not the library default | Eq. (10); A12. Multi-key fixture checks math only, not main-model architecture |
| V14 | Exactly one residual addition | Zero injection produces unchanged text at the VIM boundary, not doubled text; independent LE/LD traces confirm the choice | Fig. 5, Eq. (16), A13 |
| V15 | Tied vocabulary | Input embedding and output projection use one matrix object; controlled logits equal Trec×Wvoc^T; gradients from both usages reach it | Eq. (24), R19 |
| V16 | Detection/localization CE interpretation | Tiny two-class logits agree with manual log-softmax CE; both background and foreground penalize errors; large logits remain finite | Eqs. (18)/(26), A18 |
| V17 | Loss reduction scaling | Duplicating batch leaves averages unchanged; repeating token positions doubles LR sum; repeating spatial domain leaves candidate mean AR/FL unchanged; any alternate reduction passes its own named expectation | Eqs. (17)/(18)/(25), A17/A18 |
| V18 | KL direction, axis and gradients | Equal distributions give zero; nonnegative within tolerance; asymmetric fixture matches KL(Ptext||Padapter); sum over 512 then mean batch; both operands receive gradients under A19 | Eq. (19) |
| V19 | CMC matching and reverse direction | Uniform scores give ln(b); b=1 gives zero CMC; strong correct diagonal lowers loss; joint pair permutation preserves loss; mispairing changes it; reverse uses raw-score transpose | Eqs. (20)–(23) |
| V20 | Similarity and temperatures | Initial effective tau=0.07; KL effective temperature=0.5; selected raw-dot/cosine behavior independently checked; tau trainable and restored on resume | p. 8, A14/A15/A19 |
| V21 | LR objective | Known token logits match token-summed then batch-mean CE; selected pad policy works; uniform full-length logits yield 308×ln(49408) if all positions count | Eqs. (24)–(25), A10/A17 |
| V22 | Total objective | Total equals sum of exactly six independently reported scalars; no hidden weights, auxiliary loss or T² multiplier | Eq. (27) |
| V23 | Gradient-path audit | FD can reach AD via residual and shared encoders; FL reaches mask/decoder/local encoder but not distinct TE-only parameters; LR reaches vocabulary/LE/LD/visual VIM paths; do not require Q/K singleton gradients | §§III-B–E; architecture.md gradient map |
| V24 | Image-only inference parity | Full training model in eval mode and stripped visual model agree on image/mask logits; changing/removing metadata/text has no effect; source images not requested | §III-A p. 4, Fig. 4 |
| V25 | From-scratch provenance | No pretrained weight loader in baseline; initialization seed/scheme recorded; tokenizer assets distinguished from learned weights | §II-B p. 4; A22 |
| V26 | Optimizer and scheduler | Adam hyperparameters match resolved manifest; LR drops ×0.1 at agreed 15-epoch boundaries; no unintended AdamW substitution | §IV-A p. 8, A20 |
| V27 | Resume equivalence | Fixed fixture resumed checkpoint reproduces next loss/LR/update within declared precision tolerance, including optimizer, scheduler, RNG and tau | A35 engineering control |
| V28 | Metric fixtures | Known confusion matrices and rank scores give expected ACC/AUC/class IoUs; absent-class/all-real/all-fake policies explicit; no test-label threshold selection | §IV-A p. 8, A24 |
| V29 | Baseline resource check | Full b=8,n=308 shapes run within recorded hardware capacity; unique parameter/activation counts reported; mismatch to 141M explained, not hidden | Table IX p. 14, Table XV p. 15, A30/A31 |
| V30 | Ablation dependency validation | A configuration with RE but no available AD fails or explicitly supplies a named variant; no-LD removes LR; no-LE removes dependent text terms; module removal distinct from zeroing a loss | Tables VI/X, A28 |

Tests V10 and V11 intentionally distinguish LD self-attention causality from the full text reconstructor: a full-encoder memory can expose future target tokens. A test demanding strict causality of the entire FLT would reject the published data flow incorrectly.

V13 must isolate task-loss gradients before weight decay. An optimizer can alter a parameter with zero task gradient, so comparing weights after a decay step is not a correct singleton-attention gradient test.

## Experimental execution matrix

This table specifies the reported studies to reproduce later (**EXPLICIT paper protocols**). Their run orchestration and order are **ASSUMED**.

| Experiment | Separate training sources | Evaluation targets | Source |
|---|---|---|---|
| Cross-forgery | EFS; AM; FS | Each of the two other types | Table II p. 9 |
| Cross-diffusion | DDPM; LatDiff; CollDiff; DiffFace; Diffae | Each other diffusion generator, ACC/AUC/mIoU | Tables III–IV pp. 10–11 |
| AM cross-generator | Diffae; LatTrans | Other listed AM generators including IAFaces | Table V p. 12 |
| FS cross-generator | DiffFace; FSLSD | Other listed FS generators including FaceSwapper | Table VII p. 13 |
| Cross-dataset | FF++ with L1/L2 | FF++, Celeb-DF, DFDC, DF-1.0 | Table VIII p. 13; §IV-B p. 10 |
| Corruption robustness | GenFace, source mixture unspecified | DF-1.0 with seven corruptions × five severities plus clean | Fig. 7 p. 9, §IV-B p. 10; A25 |
| Module additions | DiffFace | FSLSD, LatDiff | Table VI p. 12 |
| Loss subsets | DiffFace | FSLSD, FaceSwapper | Table X p. 14 |
| B sweep 0–5 | DiffFace | FSLSD, FaceSwapper | Table XI p. 14; A28 for B=0 |
| VIM absent/LE/LD/both | CollDiff | DDPM,LatDiff,DiffFace,Diffae | Table IX p. 14 |
| VIM placement | DiffFace is a candidate based on surrounding ablation context and matched full-model results | FSLSD, LatDiff | Table XII p. 14; training source is ASSUMED, not stated in its caption |
| Prompt levels/omissions | DiffFace | FSLSD,LatDiff | §V-G p. 13, Table XIII p. 15 |
| VIM fusion type | DiffFace | FSLSD,LatDiff | Table XIV p. 15 |
| VIM K/V token type | DiffFace is a candidate based on surrounding context/matched results | FSLSD,LatDiff | Table XV p. 15; training source ASSUMED |
| LE/LD depth | DiffFace | FSLSD,LatDiff | Table XVI p. 15 |
| Language decoder removed/added | CollDiff | DDPM,LatDiff,Diffae,DiffFace | Fig. 8 p. 12 |
| GT threshold 0.05/0.1/0.15 | DiffFace | DDPM,CollDiff,LatDiff,Diffae | §V-J / Fig. 10 p. 15 |

Baseline-comparison code for CLIP, MFCLIP or other methods is outside the primary independent MFVLR implementation. Their published table results can be retained as reported references without claiming to have rerun them. Introducing VIM into those external baseline models is a separate secondary study.

## Benchmark acceptance and reporting

**ASSUMED acceptance policy A34:** a benchmark run is technically valid only when data manifests, preprocessing, selected assumptions, training duration, source-domain validation, checkpoint choice and metric definitions are frozen before viewing target test outcomes. Report each target metric against its own table cell, not only a selected average. Report exact sample counts and the unit of evaluation (image/frame/video).

There is no paper-specified tolerance, standard deviation or seed count. Do not invent a “within 1%” success rule or promise table-level equivalence. Proposed final reporting uses at least three seeds where resources permit, with individual runs, mean and dispersion; identity/video-aware uncertainty estimation must respect correlated samples if used. These are engineering research controls, not a recovered author protocol.

Acceptance levels:

- **Documentation accepted:** all stated architecture choices carry EXPLICIT/DERIVED/ASSUMED status and references; contradictions remain visible. This is the scope of the current deliverable.
- **Implementation accepted:** critical decisions resolved and logged; V01–V30 applicable tests pass; no hidden pretrained weights, topology changes, inference text, loss rescaling or test leakage.
- **Reproduction evaluated:** valid benchmark outputs and all deviations from the paper are reported, including negative results and missing data. Numerical proximity alone is insufficient to claim fidelity.
- **Empirical replication supported:** matched-protocol evidence and a prospectively agreed comparison criterion support the claim. Without undisclosed author settings/splits, describe results as a documented independent reimplementation rather than exact replication.

Particular conflicting targets (CollDiff→LatDiff AUC 99.99/99.98; duplicate Table XII/XIII/XVI settings) must retain both references. Do not average them into an invented ground truth. Any candidate choice selected using test results becomes a tuning experiment and requires a fresh held-out evaluation for a generalization claim.

## Planned run artifacts

**ASSUMED A35:** each eventual run should save its resolved configuration with classification/source IDs; paper hash; code revision; dataset and tokenizer hashes; class/type/generator counts; source-pair and split manifests; initialization seed; runtime versions/hardware; precision policy; per-loss scalars and selected gradient diagnostics; effective temperatures/LR; optimizer/scheduler/RNG checkpoint; checkpoint-selection evidence; image scores and masks aligned to sample IDs; metric aggregation settings; and numerical deltas to the relevant paper cells.

Separate the faithful-graph baseline with disclosed interpretations from engineering approximations and methodological extensions. An implementation cannot be made paper-faithful by renaming an approximation after it performs well.
