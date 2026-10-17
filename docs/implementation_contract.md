# Phase 1.5: proposed implementation contract

Revision: 3, Phase 2A N1 acceptance review. **M1–M5 are APPROVED for Phase 2A MVE implementation and verification only**, by the user's explicit Phase 2A instruction. **N1 is APPROVED only for the CPU FP32 cross-batch-size comparison in MVE-T12**, following conditional user approval and verified numerical evidence. They remain engineering assumptions, not confirmed MFVLR-author architectural details. Unrelated decisions, full-model training, and a skip-connected production decoder remain unapproved. Phase 1 remains the accepted research record.

Primary source: [MFVLR.pdf](../papers/MFVLR.pdf), arXiv:2605.10071v1, PDF pages 1–18, SHA-256 `2afed0516c32121513463d326720153887586ceaa328f26b96263bd39da45307`. Source equations remain unchanged in [paper_analysis.md](paper_analysis.md). The accepted [architecture](architecture.md), [reproduction specification](reproduction_spec.md), and [assumptions register](assumptions.md) remain unchanged as the Phase 1 record.

Machine-readable proposals: [paper_reference.yaml](../configs/paper_reference.yaml) and [debug.yaml](../configs/debug.yaml). The filename `paper_reference` means a proposed implementation of the reference formulation with disclosed assumptions, not an approved or uniquely recovered author configuration.

## Classification, status, and scope

Evidence classification and review status are independent:

- **EXPLICIT:** a statement in MFVLR, cited by page/section/equation. It can still be mathematically incomplete.
- **DERIVED:** a mathematical consequence of stated facts or of an explicitly identified proposed architecture.
- **ASSUMED:** our engineering selection. A cited precedent can support it without making it EXPLICIT in MFVLR.

Decision statuses have the following meanings:

| Status | Meaning | Current use |
|---|---|---|
| PROPOSED | Concrete recommendation awaiting explicit user approval | A01–A35 portions outside the approved MVE scope |
| APPROVED | Explicitly approved implementation choice, with approval recorded | M1–M5, MVE portions only; explicit user Phase 2A approval |
| BLOCKED | Work cannot start because a named prerequisite is missing | Benchmark work lacks approval and data manifests |
| DEFERRED | Work intentionally outside the next implementation scope | Secondary ablations, corruption recipes and qualitative reproduction |

Every new choice stays PROPOSED even when work using it is BLOCKED or DEFERRED. Thus the tables distinguish **decision status** from **work readiness**; postponing language work does not approve its proposed settings.

**Approved Phase 2A scope:** implement and unit-test MVE only: shared local encoder, visual tokenization/Transformer, appearance-encoding record, absolute residual, residual encoding and additive fusion. The production appearance/mask decoder, classifier, language modules, loss functions, data loaders, training loop and benchmark runner remain later work. MVE tests may use synthetic reconstructions and an explicitly test-only differentiable reconstruction fixture.

## Recorded MVE approval

All five bundles below are APPROVED as engineering assumptions for MVE only. Approval source: explicit user message commencing Phase 2A, client date 2026-10-10. A whole assumption ID is not approved when only its MVE portion is covered. Both configuration profiles record the same scoped approval independently.

| Bundle | Decisions requiring approval now | Concrete choice |
|---|---|---|
| M1: local encoder | A02, visual part of A03 | Five double-convolution stages, widths 64/128/256/512/1024, four max pools, padding 1, ReLU, convolution bias, no convolutional normalization/dropout |
| M2: global encoder | Visual parts of A03/A04/A16 | Four pre-LN blocks, width 512, eight heads, head width 64, FFN 2048, GELU, bias-enabled projections, no dropout/final norm; row-major spatial tokens, appended class token |
| M3: sharing and graph | A06/A07; A05 boundary | One registered encoder shared between appearance/residual calls; gradients through absolute residual; decoder remains external; no detach/clamp/residual renormalization |
| M4: input and API | Tensor boundary of A01/A23; A35 | RGB float32 [b,3,224,224], appearance scaled [0,1]; explicit per-call appearance record passed across external reconstruction; complete API below |
| M5: initialization and verification | MVE parts of A22/A30/A31/A35 | Kaiming convolution initialization; Xavier per linear matrix; normal tokens/positions; deterministic seed 0; FP32; tests and full-topology debug profile below |

Approving MVE does **not** approve native-resolution mask construction, a dataset split, the proposed decoder internals, A08–A19 language/loss settings, or a training budget. Those cannot block isolated MVE coding after M1–M5 approval. A02 is split: its encoder and skip-output interface are needed now; its decoder implementation can wait. A05 is reviewed now because it controls the future residual distribution, but only the no-clipping external-reconstruction boundary is needed for MVE.

## Additional primary-source evidence

The following primary papers were consulted for specific details, without examining any MFVLR implementation. Source precedents are not MFVLR requirements.

| Evidence | Finding and boundary |
|---|---|
| S1: MFVLR reference [44], [U-Net §2, PDF p. 4](https://arxiv.org/pdf/1505.04597) | Supports double 3×3 convolutions, ReLU, pooling, expanding paths and skip concatenation. Its original convolutions are unpadded; our padded 224→14 topology is an adaptation, not inherited fidelity. No normalization layers are specified in that description. |
| S2: reference [23], [ViT §3.1, PDF pp. 3–4](https://arxiv.org/pdf/2010.11929) | Supplies a precedent for pre-LN attention and a two-layer GELU MLP. ViT also has a final normalization; we explicitly omit it in this proposal because MFVLR specifies extraction from the block output without defining one. Eight heads are our choice, not established by this citation. |
| S3: reference [21], [MFCLIP §III-D](https://arxiv.org/html/2409.09724#S3.SS4) | Explicitly embeds four separately tokenized 77-position sentences and concatenates them into 308 positions. This is stronger evidence for A08 than the arithmetic 4×77 alone. Its transfer to MFVLR is still ASSUMED; it does not settle MFVLR padding, decoder shifting or pooling. |
| S4: reference [34], [CLIP §2.2–2.3, PDF pp. 3–4](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf) | Supplies precedents for normalized contrastive features, a log-parameterized scale, BPE and EOS pooling. These do not override MFVLR's conflicting similarity wording or its last-token instruction. No pretrained weights are proposed. |

Further useful evidence would be GenFace [42] for source-pair identity, file layout and official splits; FF++ [53] and DF-1.0 [56] for dataset/corruption conventions; and the exact CLIP tokenizer artifact for special IDs and merge rules. These may narrow choices, but cannot establish unpublished MFVLR training settings. Such artifacts have not been downloaded or pinned in this phase.

## Review of all 35 assumptions

Recommendation cells retain their Phase 1.5 **ASSUMED / PROPOSED** record outside the approved MVE portions. The approval table and configuration `mve_approval.portions` supersede their status only for M1–M5. “Now” identifies MVE portions; “later” identifies unapproved work. Each row compares alternatives, selects one baseline, states its consequence and identifies helpful evidence.

| ID; scope; status | MFVLR explicit / missing detail | Alternatives and recommended baseline | Consequence and additional evidence |
|---|---|---|---|
| A01; input now, targets later; PROPOSED | pp. 7–8 §III-E: RGB absolute source difference, grayscale /255, threshold 0.1; real zero/EFS one. Missing alignment, grayscale coefficients, resize order and boundary. | Recommend already aligned native-resolution face crops; compute signed float RGB differences before resizing; weighted grayscale (0.299,0.587,0.114), strictly >0.1; resize binary mask nearest to 224. Resize RGB bilinear with antialiasing and scale /255. Alternative: difference after resizing, which yields different targets. | Native targets avoid blurring differences before thresholding; registration/compression artifacts remain. Reject unaligned/mismatched source pairs. GenFace [42] could identify pairing/cropping, not the absent operator order. MVE only needs the image tensor boundary. |
| A02; encoder now, decoder later; PROPOSED | pp. 5–6 §§III-B/C and p. 8: U-Net family, 1024×14×14 bottleneck, 224 outputs; stages/skips/norms absent. | Recommend exact padded 64→128→256→512→1024 encoder below, no BN/GN. Recommend matching skip decoder later. Alternatives: original valid-convolution U-Net, BN/GN, no skips, bilinear decoder. | Preserves paper endpoints, avoids shared-domain BN statistics, retains high-resolution decoder inputs. Skip use is an extra assumption relative to the written Iloc-only interface and may facilitate reconstruction shortcuts. S1 supports the family, not the adaptation. |
| A03; visual now, language later; PROPOSED | Eq. (2) gives four image blocks; LE/LD order in pp. 6–7, but heads/FF width/image norms absent. | Recommend visual pre-LN MHA+residual, pre-LN 512→2048→512 GELU FFN+residual; 8 heads, dropout 0, eps 1e-5, no terminal LN. Later language candidate uses the same head/FF sizes, but only the explicitly placed LE pre-MHA LN and no additional LD norms. Alternatives: post-LN, single-affine FF, extra language norms. | Visual choice provides a concrete stable block; text choice stays close to written norm locations but can be unstable. No stock block may introduce extra norms implicitly. S2 helps visual structure; S4 supports a language precedent but not MFVLR fidelity. |
| A04; visual now, text later; PROPOSED | p. 5 Eq. (1): learned class token and image positions; pp. 6–7 text positions. Missing layout/init/pooling norm. | Recommend row-major h,w flatten, affine 1024→512, class token at index 196, learned [1,197,512] positions; no embedding/final LN or positional interpolation. Later learned independent Pe/Pd. Alternative class-first/EOS-derived pooling. | Requires strict 224 input; no hidden resize. Fixed class position is easy to audit. S2 shows another placement but does not resolve MFVLR's “appended” wording. |
| A05; boundary now, decoder later; PROPOSED | p. 6: AD/MD convolution outputs, channels 3/2. Missing kernels and output activation. | Recommend decoder trunk at 64×224×224 with separate biased 1×1 heads; identity reconstruction output, mask logits. Alternatives sigmoid/tanh reconstruction or larger kernels. | No saturation/clipping alters residual; reconstruction can leave [0,1] and Ir can exceed 1. MVE must accept this. S1 supports a 1×1 segmentation head only; it cannot resolve reconstruction activation. |
| A06; encoder now, decoder later; PROPOSED | pp. 5–6: IE/RE share architecture/weights, AD/MD share decoder trunk. | Recommend one registered shared encoder, called twice sequentially; no cloned role-specific modules. Later decoder trunk once then two heads. Alternatives alias-registration or independent copies/concatenated role batches. | Unique optimizer/checkpoint ownership; no BN state or stochastic differences under proposed encoder. Independent copies violate sharing. No external paper is needed to establish MFVLR sharing. |
| A07; now; PROPOSED | p. 5: Ir=abs(Ipre−I); no gradient-stop/freeze rule. | Recommend joint autograd through reconstruction and residual, abs subgradient 0 at equality, no detach. Alternatives detached residual/frozen AD/real-only pretraining are separate method changes. | Detection/text losses can update AD through RE; residual and appearance paths accumulate into one encoder. Test with an external differentiable reconstruction fixture. Cited papers cannot establish an unstated detach policy. |
| A08; language later; PROPOSED | p. 6 Fig. 6 and p. 8: four prompt levels, CLIP tokenizer, 308 positions, vocabulary 49408. Packing absent. | Recommend four independent 77-slot tokenizations concatenated L1,L2,L3,L4; right-padding per level, BOS/EOS per segment; reject overflow instead of silently truncating. Alternative joined 308-slot prompt. | S3 directly supports the precursor packing. Internal padding and multiple BOS/EOS remain deliberate assumptions; tokenizer artifacts/IDs still unpinned. For FF++, recommend two 77-slot segments (154 positions) in its separate model; that length is not MFVLR-explicit. |
| A09; language later; PROPOSED | p. 6: last LE token supplies Tl; last physical slot versus EOS unspecified. | Recommend physical slot 307 for GenFace, including if padded; no terminal norm/projection. Alternative last EOS or last nonpad. | Bidir attention lets the last pad position aggregate context if visible; it is not ordinary EOS pooling. S4 supports the alternative, not a silent substitution. |
| A10; language later; PROPOSED | p. 7 shift with begin token, preserve n; Eq. (25) token sum. Padding/BOS handling absent. | Recommend one global right shift across all 308 slots, prepend BOS embedding, remove final embedding; original IDs remain targets, including padding and existing BOS/EOS. Alternatives segment-wise shifts, excluding BOS/pads, token mean. | Explicitly permits BOS,BOS at decoder start because original sequence already has BOS. Predicts segment boundaries and many pads; can dominate LR. Tokenizer evidence establishes IDs, not shift choice. |
| A11; language later; PROPOSED | pp. 6–7: LE MHA; LD masked self-MHA and attention to Thig_e. Other masks absent. | Recommend bidirectional LE; causal LD self-mask; no key-padding masks anywhere; unmasked full LE memory in decoder. Alternatives masked padding, causal LE, causal memory. | Aligns last physical-slot pooling and all-slot LR; pads influence representations. LD is a text autoencoder, not a fully causal generator. S4 differs on LE masking and therefore cannot be imported wholesale. |
| A12; language later; PROPOSED | p. 7 Eqs. (4)–(11): one visual K/V token and sqrt(r/d) denominator; softmax axis absent. | Recommend softmax over the singleton key axis, printed scale, 8 VIM heads, no attention dropout, four bias-free projections. Alternative conventional scale or patch/query-axis attention must be labeled modified. | DERIVED: visual delta is word-independent; Q/K task gradients zero. Keep explicit Q/K computation for auditability rather than deleting “unused” parameters. No cited paper can repair the equation under the label faithful. |
| A13; language later; PROPOSED | Fig. 5 includes VIM residual; Eq. (16) adds external residual. Ownership ambiguous. | Recommend VIM API returns delta; LE/LD caller adds input exactly once. Alternative residualized module plus external addition doubles input. | Prevents unintentional double skip; interpretation remains declared. Only MFVLR clarification can resolve author intent. |
| A14; losses later; PROPOSED | p. 8 Eqs. (20)–(21): called cosine, defined dot. | Recommend raw dot for the equation-oriented baseline; cosine-normalized CMC only as named sensitivity variant. Never normalize Iv globally. | Magnitudes can drive logits and saturate contrastive loss. S4 supports cosine as a reasonable alternative, not proof for MFVLR. |
| A15; losses later; PROPOSED | p. 8: trainable tau, initial 0.07; parameterization absent. | Recommend theta=log(tau), tau=exp(theta), theta initially log(0.07), no clamping, no decay on theta. Alternative direct positive-constrained tau or inverse-scale parameter. | Guarantees positive real-valued temperature mathematically; finite precision overflow/underflow must fail visibly. Parameterization changes updates. S4 offers a log-scale precedent; our log-denominator convention is explicit. |
| A16; visual projection now, heads later; PROPOSED | pp. 7–8: one FC adapter and detection, VIM matrix equations. Bias unspecified. | Recommend biases for visual projection, visual MHA/FFN and later adapter/classifier; none in VIM; tied vocabulary projection has no bias. Alternative all bias-free. | Fixes count and initialization without using rounded paper totals to infer bias. No citation uniquely identifies MFVLR biases. |
| A17; losses later; PROPOSED | Eqs. (17)/(25): batch average AR, token-summed LR; spatial AR reduction absent. | Recommend AR mean over all image elements, LR sum over n then mean b; keep total coefficients one. Alternative AR sum or LR mean separately named. | LR scale may exceed other losses by orders of magnitude. This is accepted as a risk of the reference candidate, not silently balanced. No cited paper settles missing MFVLR reduction. |
| A18; losses later; PROPOSED | Eq. (18) lacks class encoding/spatial reduction; (25)/(26) log unrestricted logits. | Recommend stable categorical CE for FD/LR; two-class FL CE averaged over b,h,w. Alternative literal log logits is invalid generally; single-channel BCE changes head. | Makes a scalar objective with background supervision; explicitly an interpretation of incomplete equations. S1 offers CE precedent, not MFVLR's reduction. |
| A19; losses later; PROPOSED | Eq. (19): KL(text||adapted vision), softmax temperature .5. No detach/reduction convention. | Recommend softmax(z/.5), feature sum then batch mean, gradients into both distributions, no T² multiplier. Alternatives teacher-detach, inverse KL or feature mean are different. | Image-conditioned text receives gradients and can move with student; tau .5 sharpens under this convention. Cited work cannot settle undocumented teacher policy. |
| A20; training later; PROPOSED | p. 8: Adam, LR 1e-4, decay 1e-3, LR ×.1 every 15 epochs. | Recommend beta (.9,.999), eps 1e-8, coupled decay, no AMSGrad, no warmup/clipping; decay all learned tensors except log-tau; schedule after each completed epoch. Alternative AdamW/excluding norms or biases. | Decay on log-tau would push tau toward 1, so explicitly exempt it. This refines Phase 1's all-parameter-decay candidate. Framework defaults must be pinned, not called paper facts. |
| A21; training later; PROPOSED | p. 8 has no epoch limit or selection rule. | Recommend 45 epochs, no early stop, source-validation AUC selection, tie by lower validation FD CE then earlier epoch. Alternatives 30/60 epochs or last checkpoint. | Three 15-epoch LR stages provide a finite starting budget; 45 is not inferred from the paper. Target metrics cannot select epochs. No cited paper can establish MFVLR duration. |
| A22; MVE init now, repeats later; PROPOSED | p. 4 §II-B: from scratch; schemes/seeds absent. | Recommend initialization below, baseline seed 0; final repeats seeds 0,1,2. No pretrained weights. Alternatives PyTorch defaults or CLIP initialization. | Explicit scheme avoids accidental backend defaults; seeds do not guarantee cross-platform bitwise identity. Sources can justify choices but not recover author seed. |
| A23; tensor boundary now, data later; PROPOSED | pp. 8–10: GenFace and cross-source protocols, no manifests/cropping/augmentation details. | Recommend official splits when obtainable; official validation if present, otherwise reserve 10% source-training groups with seed 0; keep identity/source/video groups disjoint. Use supplied aligned crops, no extra detector, resize directly to 224, no augmentation. | Actual manifest construction BLOCKED by absent data/metadata. Reject missing grouping instead of claiming leakage-free splits; no invented test sets. GenFace [42] and dataset primary papers can materially help. |
| A24; evaluation later; PROPOSED | p. 8: ACC, AUC, class-wise mIoU; aggregation absent. | Recommend image/frame-level fake softmax score, argmax detection/mask; dataset confusion matrix, mean over classes with nonzero union, report support; undefined AUC on single-class sets. Alternatives video pooling/per-image IoU. | Includes real and fake masks in mIoU; logs absent classes and both class IoUs. Do not equate it to unreported author aggregation. Dataset protocols could clarify reporting units. |
| A25; robustness deferred; PROPOSED | Fig. 7 corruption names/severity counts; Fig. 10 GT thresholds; numeric recipes absent. | Recommend defer exact Fig. 7 replication until recipes known; baseline uses clean images. For threshold studies use fixed .1 evaluation GT, with changing-GT study separate. Alternative adopt declared external corruption suite. | No arbitrary severity ladder is called a reproduction. DF-1.0 [56] may help but cannot prove MFVLR's exact corruption choices. |
| A26; training later; PROPOSED | p. 8: b=8, diagonal CMC positives. Batch sampling unspecified. | Recommend single-device batches of eight, uniform shuffled training records, no class balancing, drop last incomplete train batch; pair-diagonal labels. Alternatives balanced batches/global negatives/multi-positive CMC. | Preserves batch-pair definition; imbalance and semantic false negatives remain. Accumulation alone is not eight-pair CMC. GenFace metadata informs sampling counts only. |
| A27; language later; PROPOSED | pp. 6–7 per-block VIM and Table IX increments versus “two VIMs” wording. | Recommend independent VIM at each of 12 LE and seven LD blocks. Alternatives two modules total or sharing across depth. | More parameters than shared-depth VIM; matches per-block notation, supported but not proven by count deltas. No matching-count claim of architecture fidelity. |
| A28; ablations deferred; PROPOSED | Tables VI/X/XI omit exact module-loss coupling and B=0 pooling. | Recommend main B=4 model first; reject B=0 or incompatible module removal until a separate approved ablation contract. Alternative invent mean pooling or keep constant CLS. | Prevents mislabeled ablations and residual without reconstruction. Cited architectures cannot identify missing author ablation behavior. |
| A29; alternate VIM deferred; PROPOSED | Tables XIV/XV show fusion/KV variants without full construction. | Recommend singleton reference only; block patch/all-token runs until source/fusion/rescaling contract exists. Alternative explicit research extension later. | Does not substitute conventional attention into baseline. Published performance/FLOPs do not determine patch construction. |
| A30; MVE audit now, full profiling later; PROPOSED | Tables IX/XV: 141M, 173.04G, scope unknown. | Recommend exact count from our declared MVE topology; full graph and inference profiles separate; count unique parameters once. Alternative force widths to match paper count is rejected. | Proposed MVE count 32,078,912 is a self-consistency test only. Profiler and FLOP conventions remain later choices. |
| A31; MVE debug now, optimizations later; PROPOSED | p. 8 PyTorch/V100; AMP/DDP etc. absent. | Recommend single-device FP32, deterministic debug CPU, no AMP/checkpointing/compilation/distribution; full MVE shape with batch 2 in debug. Alternatives reduced architecture or AMP. | CPU debug can be slow but preserves topology. Full b=8 memory must be measured before training; no claim it fits an unspecified GPU. |
| A32; visuals deferred; PROPOSED | Figs. 2–3/9/11–12 omit complete plotting/probe protocols. | Recommend defer quantitative FID/t-SNE/heatmap replication; later sample-ID-fixed qualitative diagnostics with named scaling/method. Alternative invent settings and label exploratory. | Pictures cannot validate fidelity. Cited visualization methods could specify algorithms but not author sample choices. |
| A33; reporting later; PROPOSED | Tables XII/XIII/XVI duplicate inconsistent settings; prose/table errors in Phase 1 ledger. | Recommend retain both entries and source locations, no reconciliation by averaging. Alternative select favorable target is rejected. | Uncertainty about author runs remains; author clarification could resolve it. |
| A34; validation later; PROPOSED | No reported seed variance or replication tolerance. | Recommend structural tests first, then disclose all benchmark deltas and three-seed results; no numeric “replicated” threshold yet. Alternative arbitrary ±1% is rejected. | Honest evaluation remains possible without claiming exact recovery. More author run metadata would help statistical criteria. |
| A35; interface/config now, artifacts later; PROPOSED | No package/API/config/checkpoint schema in paper. | Recommend typed per-call records, one parameter owner, strict input checks, full provenance and approval manifest; configs are declarative until a reviewed loader exists. | Changes software organization, not math; prevents accidental authorization or stale caches. No cited paper required. |

## Exact proposed MVE specification

All MVE internal topology, initialization and API choices in this section are **ASSUMED / APPROVED under M1–M5**, unless explicitly labeled. Dimensions copied from MFVLR retain their EXPLICIT provenance. The deferred VD specification remains PROPOSED and unapproved; no language decision is required to instantiate MVE.

### Input and local encoder

**EXPLICIT anchors:** I and Ir have three 224×224 channels, Iloc has 1024×14×14, shared IE/RE architecture and weights (pp. 5–6 §III-B; p. 8 §IV-A). **Proposed boundary:** NCHW float32 tensors, b≥1, finite values, one device and dtype. Appearance input comes from RGB/255 without mean/std normalization; this scaling belongs to preprocessing, not an operation inside the encoder. Residual inputs are finite, nonnegative and may exceed 1. The shared encoder performs no range clamp or renormalization and no implicit conversion/resize.

Each convolution: kernel 3×3, stride 1, padding 1 with zeros, dilation 1, groups 1, bias enabled. Each is followed by ordinary ReLU, not in-place. No BatchNorm, GroupNorm, InstanceNorm, residual CNN block, dropout or pooling after stage S4. Each max pool: kernel 2×2, stride 2, padding 0, dilation 1, ceil_mode=false. Keep S0–S3 outputs as ordered appearance skip tensors; keep S4 as Iloc. Skips are ordinary graph-connected references, not clones/detached caches.

| Step | Exact operation | Output [b,C,H,W] | Encoder parameters |
|---|---|---|---:|
| Input | RGB / residual tensor | [b,3,224,224] | 0 |
| S0.1 | Conv 3→64; ReLU | [b,64,224,224] | 1,792 |
| S0.2 | Conv 64→64; ReLU; save skip0 | [b,64,224,224] | 36,928 |
| P0 | MaxPool2d | [b,64,112,112] | 0 |
| S1.1 | Conv 64→128; ReLU | [b,128,112,112] | 73,856 |
| S1.2 | Conv 128→128; ReLU; save skip1 | [b,128,112,112] | 147,584 |
| P1 | MaxPool2d | [b,128,56,56] | 0 |
| S2.1 | Conv 128→256; ReLU | [b,256,56,56] | 295,168 |
| S2.2 | Conv 256→256; ReLU; save skip2 | [b,256,56,56] | 590,080 |
| P2 | MaxPool2d | [b,256,28,28] | 0 |
| S3.1 | Conv 256→512; ReLU | [b,512,28,28] | 1,180,160 |
| S3.2 | Conv 512→512; ReLU; save skip3 | [b,512,28,28] | 2,359,808 |
| P3 | MaxPool2d | [b,512,14,14] | 0 |
| S4.1 | Conv 512→1024; ReLU | [b,1024,14,14] | 4,719,616 |
| S4.2 | Conv 1024→1024; ReLU; Iloc | [b,1024,14,14] | 9,438,208 |

All spatial shapes and counts in this table are **DERIVED from the proposed topology**, not evidence that the authors used these layers. No residual-call skip tensors need be returned, though normal autograd activations remain for backward.

### Tokenization and image Transformer

**EXPLICIT anchors:** flatten/project Iloc, append learned class token, add learned positions, apply B=4 blocks and select class token (p. 5 Eqs. (1)–(2)); d=512 (p. 8). Proposed detailed sequence:

| Step | Specification | Shape |
|---|---|---|
| Flatten | Spatial index t=14×row+column; channels last | [b,196,1024] |
| Projection | One affine Linear(1024,512), bias enabled | [b,196,512] |
| Append CLS | Parameter [1,1,512] expanded across b at end | [b,197,512] |
| Add Pi | Parameter [1,197,512] shared by IE/RE | [b,197,512] |
| Blocks 1–4 | Independent block parameters across depth; same blocks for both image roles | [b,197,512] after each |
| Extract | Last token, retain singleton axis; no extra norm/projection | Ig or Ir_g: [b,1,512] |

The following are **new proposal formulas**, not renumberings or modifications of MFVLR Eq. (2). For each block input X:

\[
U=LN_1(X),\quad Q=UW_Q+b_Q,\ K=UW_K+b_K,\ V=UW_V+b_V,
\]
\[
A=\operatorname{softmax}_{key}(Q_hK_h^T/\sqrt{64}),\quad
Y=X+(\operatorname{Concat}_{heads}(AV_h)W_O+b_O),
\]
\[
Z=Y+\bigl(\operatorname{GELU}(LN_2(Y)W_1+b_1)W_2+b_2\bigr).
\]

There are eight heads; Qh,Kh,Vh are [b,8,197,64]; logits/weights are [b,8,197,197]. Q/K/V/output matrices are 512×512, independent, all biases enabled. No causal or padding mask; every visual token attends to all 197 tokens. LN normalizes the last 512 elements, affine scale/bias enabled, epsilon 1e-5. W1 maps 512→2048; W2 maps 2048→512; exact erf GELU (`approximate=none`). All attention, embedding, FFN and residual dropout probabilities and stochastic-depth rates are zero. No terminal LayerNorm, layer scaling or extra learned projection.

The visual attention denominator sqrt(64) is a **proposed conventional choice for the otherwise unspecified image block**. It does not replace the published VIM denominator sqrt(r/d). VIM remains a separate later module with a different, preserved equation. Packed QKV is only an allowable later representation optimization if each submatrix initialization and all outputs/gradients match this separate-projection reference; it is not the initial recommended representation.

### Initialization and sharing

| Parameter | Exact proposed initialization |
|---|---|
| Encoder Conv2d weights | Kaiming normal, mode=fan_in, nonlinearity=relu; mean 0, variance 2/fan_in |
| All visual linear matrices, including separate Q/K/V/O | Xavier uniform, gain 1, limits ±sqrt(6/(fan_in+fan_out)) |
| All convolution/linear biases | Zero |
| LN scales / biases | One / zero |
| CLS and Pi | Independent untruncated Normal(mean=0,std=0.02) |
| RNG | Seed 0 for initial reference/debug run; initialize unique modules once in declared stage/block order; no second role-specific initialization |

Proposed initialization order: CNN layers S0.1 through S4.2, local-to-token projection, CLS, Pi, then blocks 1–4 with LN1,Q,K,V,O,LN2,FC1,FC2. Constants do not consume random draws. Use one documented CPU RNG stream for initialization before device transfer. Cross-version bitwise initialization is not guaranteed; record the runtime version later.

`MVE` owns exactly one `SharedVisualEncoder`, which owns the CNN, projection, CLS, Pi and four blocks. Both roles call this object. There are no separately registered `image_encoder` and `residual_encoder` copies, independent positional tables, domain-specific normalizations or depth-shared blocks. Checkpoints contain one encoder subtree; optimizer lists contain each tensor once. Role helper methods do not own parameters.

**DERIVED from this proposal:** CNN 18,843,200 parameters; local projection 524,800; CLS 512; Pi 100,864; each Transformer block 3,152,384, four total 12,609,536. Unique MVE total **32,078,912**, counted once despite two calls. Decoder, classifier, adapter, language and loss temperature are excluded. This count tests our own specification; matching or failing to match the paper's 141M total proves nothing about author fidelity.

### Forward API and external decoder boundary

These are interface descriptions, not implementation code. Tensors preserve their graph unless a caller explicitly leaves training mode; no object-level activation cache is stored.

| Interface | Inputs | Outputs / behavior |
|---|---|---|
| `SharedVisualEncoder.encode_local(image, return_skips)` | [b,3,224,224] | `LocalEncoding(local=[b,1024,14,14], skips=(S0,S1,S2,S3) or empty)` |
| `SharedVisualEncoder.encode_global(local)` | [b,1024,14,14] | `GlobalEncoding(tokens=[b,197,512], global_token=[b,1,512])` |
| `MVE.encode_appearance(image)` | image | `AppearanceEncoding(local, skips, tokens, global_token)`; one local and one global encoding |
| `MVE.forward(image, reconstruction, appearance_encoding=None)` | image and graph-connected Ipre, both [b,3,224,224]; optional appearance record from same input/model/mode | Reuse record or compute it once if absent. Ir=abs(Ipre−I); shared encoder processes Ir; return `MVEOutput(appearance, residual_image, residual_tokens, residual_global, fused_global)` with fused_global=Ig+Ir_g |

`AppearanceEncoding` is an immutable container by convention, but its tensors retain gradients. A supplied record must come from the identical image tensor and current forward graph/parameters; reuse across samples or optimizer steps is invalid. Shape/device/dtype checks cannot prove provenance, so this is a caller contract covered by integration tests. Avoid data-dependent output copying and stale persistent caches.

Future full-model call sequence: encode appearance once → external VD reconstructs/localizes from appearance local/skip state → call MVE.forward with that same record and Ipre → external classifier/FLT consume Iv. The record avoids a second appearance encoding. The residual encoder does not invoke VD; no recursive graph exists. The MVE API accepts reconstruction because VD is outside its ownership; only the eventual full MFVLR inference API accepts a lone image (§III-A p. 4).

MVE forward validates matching shapes/devices/dtypes, floating inputs, nonempty batch, finite inputs and fixed resolution. It does not clamp appearance/reconstruction/residual, normalize output vectors, detach tensors or apply a classifier. `abs` has the proposed zero subgradient at exactly zero; tests away from zero verify sign gradients. The same `forward` semantics apply in train/eval; no BN/dropout means mode alone introduces no numerical difference.

**Deferred proposed VD compatible with this interface (A02/A05/A06):** four stages, ConvTranspose2d(kernel=2,stride=2,padding=0,output_padding=0,bias=true) with channels 1024→512→256→128→64. At each stage concatenate the corresponding appearance skip, then two padded biased 3×3 ReLU convolutions: 1024→512→512 at 28²; 512→256→256 at 56²; 256→128→128 at 112²; 128→64→64 at 224². No norms/dropout. One shared trunk, two biased 1×1 heads 64→3 and 64→2, both identity outputs. This is a recommendation to review later, not permission to implement VD with MVE.

## Later language and objective contract

A08–A19/A27 have concrete recommendations in the decision table and YAML, but approval can wait. The recommended reference deliberately preserves difficult consequences: singleton VIM, raw-dot CMC, all-position token-summed LR, and physical final-position pooling. These are not claims that the design is optimal.

For clarity the unchanged published VIM expression is:

\[
T^{glo}_{j,i}=\delta\left(\frac{Q_{j,i}K_{j,i}^{T}}{\sqrt{r/d}}\right)V_{j,i}
\qquad\text{(MFVLR Eq. (10), p. 7).}
\]

With key length 1 and proposed key-axis softmax, the attention weight is one (**DERIVED**). Keep single K/V, the printed scale, and projection parameters; multi-token attention is not an implementation fallback. VIM returns a delta; one residual belongs to each caller.

The six published losses, including their defects, remain in the Phase 1 equation ledger. The separate **proposed executable interpretation** is FD categorical CE batch mean; AR all-element mean squared error; FL two-class CE pixel/batch mean; LR categorical CE sum over all 308 positions then batch mean; symmetric diagonal CMC on raw dot scores with learned positive tau; KL(text||adapter) using feature-axis softmax(z/0.5), feature sum then batch mean and gradients through both operands. Coefficients remain one. No silent T², Dice/BCE substitution or balancing factor is introduced.

The candidate training budget is 45 epochs: epochs 1–15 use LR 1e-4, 16–30 1e-5, 31–45 1e-6; stepping after epoch 45 sets 1e-7 only for a hypothetical next epoch. Validate after each epoch on the source domain. This finite budget and checkpoint criterion are proposed, not reported MFVLR settings. Actual training remains blocked by data/tokenizer artifacts and separate approval.

## MVE unit-test acceptance criteria

Tests below are APPROVED Phase 2A engineering checks under M5. Actual execution outcomes are recorded separately in `phase2a_implementation_report.md` and artifacts; the table remains the acceptance specification. FP32 comparisons: atol=1e-6, rtol=1e-5 unless explicitly tighter; float64 analytical fixtures: atol=1e-10, rtol=1e-8. Shapes/IDs/ownership/counts use exact equality. Run at b=1 and b=2 initially; b=8 is a later resource gate, not a reason to shrink topology.

| ID | Acceptance criterion | Source/decision |
|---|---|---|
| MVE-T01 | Every convolution, pool, skip and bottleneck matches the layer table at b=1/2; wrong rank/channel/resolution/empty batch is rejected | MFVLR pp. 5/8; A02/A35 |
| MVE-T02 | Coordinate-tagged local features flatten in row-major order; biased projection maps 1024→512; CLS is token 196; exactly 197 positions | Eq. (1); A04 |
| MVE-T03 | Blocks match an independent small attention/FFN calculation with pre-LN, eight heads, key-axis softmax, sqrt(64), exact GELU and no final LN; output/gradient agreement | Eq. (2) unresolved internals; A03 |
| MVE-T04 | Selected CLS responds to changed image tokens; output is [b,1,512] and graph-connected; it is not an untouched learned constant | Eq. (2); A04 |
| MVE-T05 | Appearance and residual roles use the same parameter identities; checkpoint has one encoder subtree; unique parameter total is 32,078,912; no duplicate optimizer references | §III-B pp. 5–6; A06/A30 |
| MVE-T06 | Given controlled I/Ipre, Ir is exact abs difference and Iv=Ig+Ir_g; Ipre=I gives zero Ir but need not give zero Ir_g because CLS/positions are learned | §III-B; A07 |
| MVE-T07 | Cached path executes appearance CNN/TE once and residual CNN/TE once; no internal decoder; recomputed and reused appearance paths agree with identical parameters/input | §III-A/III-B; A35 |
| MVE-T08 | With a differentiable test-only reconstruction from local features, a fused-feature scalar propagates through Ipre into local encoder parameters; detached comparison removes that path | §III-B; A07; production VD not required |
| MVE-T09 | Isolated abs derivative matches sign away from zero and zero at equality; no reconstruction clipping or residual min-max normalization | Published residual; A05/A07 |
| MVE-T10 | Gradient of two encoder-role objectives equals the sum of their separately computed parameter gradients on controlled independent inputs; shared weights update once | §III-B; A06 |
| MVE-T11 | Initial biases zero, LN scale one/offset zero, parameter distributions/ranges consistent with specified schemes; resetting seed in same runtime reproduces initialization exactly | p. 4 scratch training; A22 |
| MVE-T12 | With no dropout/BN, train/eval outputs agree; a sample's output agrees alone or within a batch within tolerance; no cross-sample attention | A02/A03/A06 |
| MVE-T13 | Serialization round-trip preserves outputs and sharing; records do not persist as module state; invalid reconstruction dtype/device/shape/nonfinite values fail visibly | A35 |
| MVE-T14 | Debug resolves to identical encoder topology, initialization, scale and sharing; only declared workload/runtime/diagnostic overrides differ | A31/A35 |

MVE-T08/T10 use meaningful numerical fixtures and inspect isolated paths; they must not demand a nonzero gradient for every CNN weight in every random ReLU input. Parameter count is a self-consistency assertion, never a fidelity test against 141M. Expected zero VIM Q/K gradients are a later FLT test, not an MVE failure.

**N1 / APPROVED under the user's conditional Phase 2A review instruction:** use atol=5e-6, rtol=1e-5 only for MVE-T12's batch-size-1 versus batch-size-2 CPU FP32 comparison. General FP32 thresholds remain 1e-6/1e-5, analytical FP64 thresholds remain 1e-10/1e-8, and fixed-batch sample independence remains exact (atol=rtol=0). The amendment is independently recorded in both configuration review sections and locked in `proposal.verification.mve_t12_cross_batch`; it does not extend to other tests, devices or dtypes.

Approval condition verified: [numerical evidence](../artifacts/phase2a/n1_review/numerical_evidence.json) reproduces the original maximum absolute error 2.86102294921875e-6, shows identical inputs to the first Transformer block, and shows its cross-batch-size difference disappearing in isolated FP64 replay. Multiple fixed-batch peer-image/reconstruction changes and batch permutation preserve sample outputs exactly. A scalar-gradient probe has exactly zero gradients to the other sample's image and reconstruction; source review confirms no batch-axis normalization, contraction or attention. This establishes the rounding explanation for the tested CPU runtime and fixtures; it is not a universal numerical guarantee.

The original 1e-6-threshold failure and earlier exploratory runs are retained unchanged. The final approved-N1 suite passes all 14 IDs / 15 cases, with no skips, expected failures or overrides. See the [final acceptance report](phase2a_implementation_report.md#final-acceptance-after-n1-review). MVE source, architecture, initialization, mathematical conventions and parameter identities are unchanged by N1.

## Configuration semantics and current readiness

`paper_reference.yaml` separates immutable evidence facts from a `proposal` tree. Its default classification remains ASSUMED, with specific field paths annotated EXPLICIT/DERIVED. Scoped MVE adoption is APPROVED; unrelated portions remain PROPOSED. Phase 2A implements a strict loader and MVE builder only. Approval flags record the user's decision; they do not establish author fidelity or authorize other components.

`debug.yaml` is a declared overlay: resolve its base relative to the debug file; recursively merge maps only under `proposal`; replace sequences/scalars; explicit null replaces rather than inherits; reject unknown keys. Metadata/review status from either file never grants approval. The effective MVE settings must remain identical. Debug uses synthetic batch 2 with independent uniform [0,1] input RNG seed 0, three finite forward/backward checks, an external affine reconstruction fixture and a diagnostic scalar, not MFVLR training. No reduced widths, fewer blocks, lower resolution or alternative attention is hidden in debug.

| Work | Status | Reason / exit condition |
|---|---|---|
| MVE implementation | APPROVED | Explicit user approval of M1–M5; Phase 2A implementation and verification only |
| VD/classifier integration | DEFERRED | Outside next MVE scope; proposed decoder/head details need separate approval |
| FLT and six-loss implementation | DEFERRED | A08–A19/A27 approval and tokenizer artifact pinning remain |
| Benchmark training | BLOCKED | Dataset/split/source-pair manifests, tokenizer artifacts/runtime pins and training/protocol approval absent |
| Secondary ablations/robustness/visuals | DEFERRED | Missing procedures and outside initial baseline |

Remaining risks: padded/skipped U-Net and pre-LN image blocks may differ from the authors; no normalization may change optimization; the identity reconstruction head permits unbounded residuals; sparse or misregistered source-pair masks can change localization targets; physical final-pad pooling/all-pad losses and raw-dot CMC can create training shortcuts or imbalance; singleton VIM does not provide query-dependent selection; equations and conflicting tables still prevent exact-replication claims. Additional cited-paper evidence narrows choices, but approval chooses a transparent baseline rather than certifying recovered author code.

**MVE coding is authorized for Phase 2A only.** Stop after MVE implementation and verification, then wait for approval of any next component. Accepted Phase 1 files are preserved; Phase 2A execution evidence is recorded in its separate report.
