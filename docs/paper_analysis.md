# MFVLR: paper analysis and evidence ledger

Status: research analysis only; no model, training, or evaluation implementation has been written. This document describes the supplied paper, including its inconsistencies, rather than an existing MFVLR implementation.

## Source and classification

Primary source: [papers/MFVLR.pdf](../papers/MFVLR.pdf), 18 pages, arXiv:2605.10071v1, dated 11 May 2026 on p. 1. Authors: Yaning Zhang, Tianyi Wang, Zan Gao, Yibo Zhao, Chunjie Ma, and Meng Wang. SHA-256: `2afed0516c32121513463d326720153887586ceaa328f26b96263bd39da45307`. The repeated September 2020 journal header is template text, not the manuscript date. Page references below use PDF pages, which agree with printed page numbers.

All 18 pages were read, including the reference list [1]–[62] and author biographies. Figures 1–12 and the displayed equations were also inspected in rendered pages. There is no supplementary appendix in this PDF. Missing details mean absent from this supplied version; they are not claims about material that might exist elsewhere.

- **EXPLICIT**: directly stated or displayed in the supplied paper. This does not imply internally consistent or executable as written.
- **DERIVED**: follows mathematically from explicit statements, with any necessary interpretation stated.
- **ASSUMED**: an unresolved choice or proposed engineering convention, not an author-specified setting.

These labels apply to individual claims, not to whole modules. A component can be explicitly required while its internal layout remains assumed. [architecture.md](architecture.md) gives tensor contracts and data flow; [reproduction_spec.md](reproduction_spec.md) gives the proposed implementation and verification plan; [assumptions.md](assumptions.md) records decision IDs `A01`–`A35`.

## Research claim and scope

**EXPLICIT — pp. 1–4, §§I–III-A, Figs. 1–3.** MFVLR performs binary face forgery detection and pixel-level localization. Its core proposal is to combine appearance and reconstruction-residual representations, train a language encoder/decoder on hierarchical forgery descriptions, and align vision and language representations. Language assists training; inference uses images alone. The network is trained from scratch (§II-B, p. 4), rather than initialized as a pretrained CLIP model.

**EXPLICIT — pp. 5–8, §§III-B–E.** The residual is the absolute difference between the input and its learned U-Net appearance reconstruction. There is no diffusion denoising process, timestep schedule, diffusion inversion, FFT branch, Sobel branch, or SAM backbone in the specified MFVLR architecture. Diffusion describes the image generators being detected. FFT and Sobel are comparison priors in Fig. 2.

**DERIVED — pp. 4–7.** The architecture is a finite directed graph, despite the compact inference diagram looking cyclic: appearance encoding → reconstruction → residual encoding → summed global feature. Localization uses the appearance encoder's local features, not the fused global feature directly. Language losses affect localization indirectly through shared trainable visual parameters.

## Reading map

| Pages | Material read | Consequence for reproduction |
|---|---|---|
| 1–3 | Abstract, introduction, contributions; Figs. 1–3 | Defines DFFDL, residual motivation, language-guided training |
| 3–4 | Related work §§II-A–C | Distinguishes MFVLR from MFCLIP, CLIP, DIRE, SAM-based systems |
| 4–5 | Overview and MVE; Fig. 4; Eqs. (1)–(2) | Train/inference distinction, weight sharing, image feature extraction |
| 6–7 | VD and FLT; Figs. 5–6; Eqs. (3)–(16) | Shared decoder trunk, prompts, LE/LD ordering, VIM math |
| 7–8 | Losses, notation, setup; Eqs. (17)–(27), Table I | Six losses, mask construction, numerical hyperparameters |
| 9–11 | Main evaluations; Tables II–IV; Fig. 7 | Cross-forgery, cross-diffusion, cross-dataset and corruption protocols |
| 12–15 | Tables V–XVI; Figs. 8–10; §V | Module/loss/depth/prompt/fusion/mask-threshold ablations |
| 15–16 | Figs. 11–12, §VI, conclusion/limitations | Qualitative localization and residual evidence; computation/generalization limits |
| 16–18 | References [1]–[62], biographies | Bibliographic dependencies; no extra training specifications |

## Architectural traceability table

Rows state **EXPLICIT** requirements unless the status column says otherwise. Unspecified internals must not be filled in as paper facts.

| ID | Component / requirement | Paper location | Status and open boundary |
|---|---|---|---|
| R01 | RGB image input; binary image labels and dense masks | p. 4 §III-A; p. 5 §III-B; p. 8 §IV-A; Fig. 4 | EXPLICIT; preprocessing A01, A23 |
| R02 | Fine-grained text generator using hierarchical labels | p. 6 §III-D, Fig. 6; citation [21] | EXPLICIT; sequence packing A08 |
| R03 | U-Net encoder produces local appearance features | p. 5 §III-B; Fig. 4; citation [44] | EXPLICIT; exact U-Net A02 |
| R04 | Flatten spatial features, project channels, add class token and learned positions | p. 5 §III-B, Eq. (1) | EXPLICIT; flatten order and class position A04 |
| R05 | Four image transformer blocks; select output class token | p. 5 Eq. (2); p. 8 §IV-A; Table XI p. 14 | EXPLICIT; block internals A03 |
| R06 | AD: U-Net decoder plus appearance convolution | p. 6 §III-C; Fig. 4 p. 5 | EXPLICIT; skips and output range A02, A05 |
| R07 | MD: same decoder architecture/weights as AD, separate localization convolution | p. 6 §III-C; Fig. 4 | EXPLICIT; two output channels p. 8; trunk execution A06 |
| R08 | Absolute residual `abs(Ipre - I)` | p. 5 §III-B | EXPLICIT; gradient treatment A07 |
| R09 | RE shares IE architecture and weights | pp. 5–6 §III-B; Fig. 4 | EXPLICIT; same learned encoder, not copied independent weights |
| R10 | Sum global appearance and residual embeddings | p. 6 §III-B | EXPLICIT; no concatenation or learned gate specified |
| R11 | Detection head consists of one fully connected layer | pp. 4, 8 §§III-A/E; Eq. (26) | EXPLICIT despite the name “MLP”; output probability interpretation A18 |
| R12 | Adapter consists of one fully connected layer, predicts language features | p. 8 §III-E, Eq. (19); Fig. 4 | EXPLICIT; bias A16 |
| R13 | CLIP-referenced tokenizer and learned vocabulary embedding | p. 6 §III-D; p. 8 §IV-A; citation [34] | EXPLICIT; artifact and special-token handling A08–A10 |
| R14 | LE: 12 blocks, LN → MHA + residual → VIM + residual → FF + residual | pp. 6–7 §III-D, Eq. (3), Fig. 4 | EXPLICIT; other norms, heads, FF A03, A11 |
| R15 | LE last output token supplies global language embedding | p. 6 §III-D | EXPLICIT; last position versus EOS A09 |
| R16 | VIM language queries, one fused visual class token as K/V, head split, attention, concatenate and output projection | p. 6 Fig. 5; p. 7 Eqs. (4)–(11) | EXPLICIT; softmax axis A12; singleton consequence DERIVED |
| R17 | Shift low-level language embeddings using a begin token, preserve length, add decoder positions | pp. 7–8 §III-D | EXPLICIT; special-token convention A10 |
| R18 | LD: 7 blocks, masked self-attention → encoder-memory attention → FF → VIM, residual after each | p. 7 Eqs. (12)–(16); Fig. 4; Table XII p. 14 | EXPLICIT; VIM residual ownership A13 |
| R19 | Reconstruct vocabulary logits with transpose of input vocabulary matrix | p. 8 Eq. (24) | EXPLICIT tied matrix; no independent language head specified |
| R20 | Appearance reconstruction loss | p. 7 Eq. (17) | EXPLICIT expression; scalar reduction absent A17 |
| R21 | Source-pair mask generation, threshold 0.1; real zero/EFS one | pp. 7–8 §III-E; Fig. 10 p. 15 | EXPLICIT; grayscale/alignment/order A01 |
| R22 | Localization loss | p. 8 Eq. (18) | EXPLICIT expression; class encoding and reduction incomplete A18 |
| R23 | KL language-to-adapted-vision direction, temperature 0.5 | p. 8 Eq. (19) | EXPLICIT; detachment A19 |
| R24 | Bidirectional batch-pair contrastive loss, learned temperature initially 0.07 | p. 8 Eqs. (20)–(23) | EXPLICIT; dot/cosine ambiguity A14 and temperature parameter A15 |
| R25 | Batch-mean, token-summed language reconstruction loss | p. 8 Eq. (25) | EXPLICIT written reductions; probability and padding A10, A18 |
| R26 | Detection loss; six equally weighted terms in total objective | p. 8 Eqs. (26)–(27) | EXPLICIT written form; logits/probabilities A18 |
| R27 | Train from scratch; Adam and stepwise learning-rate decay | p. 4 §II-B; p. 8 §IV-A | EXPLICIT; duration/initialization/optimizer details A20–A22 |
| R28 | Image-only inference retains IE, RE, AD, MD and detection head | p. 4 §III-A; p. 5 Fig. 4 | EXPLICIT; adapter and FLT unnecessary at inference |
| R29 | Protocols and ACC/AUC/mIoU | pp. 8–13 §IV, Tables II–VIII | EXPLICIT names and tasks; splits and metric aggregation A23–A25 |

## Equation ledger: preserve published equations

This transcription preserves the published operations, including defects. Composition-chain intermediate lines in (2), (3), and (12) are omitted here; their initial and final expressions are retained. Superscripts/subscripts are typeset consistently without changing meaning. No softmax, missing class axis, scalar reduction, or corrected attention scale has been inserted into these equations.

**EXPLICIT — p. 5, Eqs. (1)–(2):**

\[
I^{tra}_1=I_{tok}+P_i.\tag{1}
\]
\[
TE(I^{tra}_1)=TB^i_B\circ TB^i_{B-1}\circ\cdots\circ TB^i_1(I^{tra}_1)=I_{TE}.\tag{2}
\]

The prose defines `Itok = App(Proj(Flat(Iloc)))`, extracts `Ig` from the class token, defines `Ir = |Ipre - I|`, and defines `Iv = Ir_g + Ig` (pp. 5–6). These are published but unnumbered operations.

**EXPLICIT — p. 6, Eq. (3):**

\[
LE(T^{tra}_1)=TB^e_E\circ TB^e_{E-1}\circ\cdots\circ TB^e_1(T^{tra}_1)=T^e_{hig}.\tag{3}
\]

Here `Ttra_1 = Tlow_e + Pe`, and `Tl` is the last token of `Thig_e`. The visual argument to each LE block is implicit in (3), made explicit in the VIM description.

**EXPLICIT — p. 7, Eqs. (4)–(11):**

\[
q_j=T^{tok}_jW^{que}_j,\tag{4}
\]
\[
k_j=I_vW^{key}_j,\tag{5}
\]
\[
v_j=I_vW^{val}_j,\tag{6}
\]
\[
\{Q_{j,i}\in\mathbb R^{n\times d/r}\}_{i=1}^r=Pa(q_j),\tag{7}
\]
\[
\{K_{j,i}\in\mathbb R^{1\times d/r}\}_{i=1}^r=Pa(k_j),\tag{8}
\]
\[
\{V_{j,i}\in\mathbb R^{1\times d/r}\}_{i=1}^r=Pa(v_j),\tag{9}
\]
\[
T^{glo}_{j,i}=\delta\left(\frac{Q_{j,i}K_{j,i}^{T}}{\sqrt{r/d}}\right)V_{j,i},\tag{10}
\]
\[
T^{glo}_j=Cat(\{T^{glo}_{j,i}\}_{i=1}^r)\in\mathbb R^{n\times d}.\tag{11}
\]

The denominator in the rendered Eq. (10) is **sqrt(r/d)**, not sqrt(d/r). The subsequent prose defines `Tadd_j = Tglo_j Wfc_j + Ttok_j` and `Ttra_(j+1) = FF^e_j(Tadd_j) + Tadd_j`. Before VIM, `Ttok_j = MHA^e_j(LN^e_j(Ttra_j)) + Ttra_j` (p. 6).

**EXPLICIT — p. 7, Eqs. (12)–(16):**

\[
LD(T^{tra}_{t1},T^e_{hig},I_v)=TB^d_D\circ\cdots\circ TB^d_1(T^{tra}_{t1},T^e_{hig},I_v)=T^d_{rec}.\tag{12}
\]
\[
T^{mmha}_{tj}=MMHA^d_j(T^{tra}_{tj})+T^{tra}_{tj},\tag{13}
\]
\[
T^{mha}_{tj}=MHA^d_j(T^{mmha}_{tj},T^e_{hig})+T^{mmha}_{tj},\tag{14}
\]
\[
T^{ff}_{tj}=FF^d_j(T^{mha}_{tj})+T^{mha}_{tj},\tag{15}
\]
\[
T^{tra}_{t(j+1)}=VIM^d_j(T^{ff}_{tj},I_v)+T^{ff}_{tj}.\tag{16}
\]

**EXPLICIT — pp. 7–8, Eqs. (17)–(19):**

\[
\mathcal L_{ar}=\frac1b\sum_{u=1}^b(I^u-I^u_{pre})^2,\tag{17}
\]
\[
\mathcal L_{fl}=\frac1b\sum_{u=1}^b-(M^u)^T\log(M^u_{pre}),\tag{18}
\]
\[
\mathcal L_{kl}=\frac1b\sum_{u=1}^b\delta(T_l^u)^T\log\frac{\delta(T_l^u)}{\delta(T_{lpre}^u)}.\tag{19}
\]

Temperature 0.5 is specified for the softmax in (19). The division and logarithm are elementwise, followed by the displayed vector inner product (**DERIVED** dimensional interpretation).

**EXPLICIT — p. 8, Eqs. (20)–(23):**

\[
S^{um}_{v2l}(I_v,T_l)=\frac{\exp(sim(I_v^u,T_l^m)/\tau)}{\sum_{m=1}^b\exp(sim(I_v^u,T_l^m)/\tau)},\tag{20}
\]
\[
S^{um}_{l2v}(T_l,I_v)=\frac{\exp(sim(T_l^u,I_v^m)/\tau)}{\sum_{m=1}^b\exp(sim(T_l^u,I_v^m)/\tau)},\tag{21}
\]
\[
\mathcal L_{v2l}=\frac1b\sum_{u=1}^b-(y^u_{pa})^T\log S^u_{v2l}(I_v,T_l),\tag{22}
\]
\[
\mathcal L_{l2v}=\frac1b\sum_{u=1}^b-(y^u_{pa})^T\log S^u_{l2v}(T_l,I_v).\tag{23}
\]

The paper defines `Lcmc = (Lv2l + Ll2v)/2`, diagonal positive pair labels, and a trainable `tau` initialized at 0.07. It calls these cosine similarities but defines `sim` as a dot product. Normalization is not stated.

**EXPLICIT — p. 8, Eqs. (24)–(27):**

\[
T_{pre}=T^d_{rec}W_{voc}^{T}\in\mathbb R^{n\times s},\tag{24}
\]
\[
\mathcal L_{lr}=\frac1b\sum_{u=1}^b\sum_{x=1}^n-(T_{gt}^{u,x})^T\log(T_{pre}^{u,x}),\tag{25}
\]
\[
\mathcal L_{fd}=\frac1b\sum_{u=1}^b-(y^u)^T\log(y^u_{pre}),\tag{26}
\]
\[
\mathcal L=\mathcal L_{fd}+\mathcal L_{lr}+\mathcal L_{cmc}+\mathcal L_{fl}+\mathcal L_{ar}+\mathcal L_{kl}.\tag{27}
\]

The six coefficients are one in (27). Different reductions inside each loss would still change effective weighting; equal displayed coefficients do not identify those reductions.

## Mathematical audit

1. **DERIVED, conditional on conventional key-axis softmax — Eq. (10), p. 7.** Each head has one key, so `QK^T` has shape `[n,1]`. Softmax over that singleton key axis is identically one. Every token receives the same projected value, independently of its query or the key. Consequently VIM's attention branch reduces to broadcasting `Iv Wval Wfc` across words. With finite inputs and no extra unspecified operations, the task-loss gradients of `Wque` and `Wkey` through this branch are zero; weight decay can still update them. The total output remains token dependent through its text residual and later FF/self-attention. This does not imply that visual conditioning is useless; it contradicts the claimed query-dependent word-level selection. Changing the softmax axis or using patch tokens is a different architecture. See A12.
2. **EXPLICIT discrepancy / DERIVED implication — Eq. (10).** Dividing by sqrt(r/d) multiplies by sqrt(d/r); conventional scaled dot attention divides by sqrt(d/r). For head width `dh=d/r`, the published logits are `dh` times conventionally scaled logits. Singleton attention hides the forward difference, but all-token/patch ablations expose it. Preserve the printed formula in the reference formulation; any conventional-scale variant must be labeled a correction.
3. **EXPLICIT ambiguity — Eqs. (24)–(26), p. 8.** Linear projections produce unrestricted real logits. Their logarithm is not a valid categorical cross entropy and can be undefined. Stable log-softmax CE is a proposed interpretation (A18), not present in these equations. The same normalization question affects the localization output in (18).
4. **DERIVED dimensional issue — Eqs. (17)–(18).** (17) only averages over the batch, leaving image dimensions unreduced under elementwise squaring. (18) uses a 2D binary mask against a two-channel prediction without specifying one-hot expansion, spatial summation, or averaging. Direct literal tensor expressions do not define a scalar total objective. A binary foreground-only interpretation would give real masks zero localization penalty and is not justified by the two-class setup.
5. **EXPLICIT conflict — Eqs. (20)–(21).** Dot similarity and cosine similarity agree only after feature normalization. No norm or epsilon appears. Record both candidate interpretations, not an automatic CLIP default (A14).
6. **DERIVED — Eqs. (19), (25), (27).** Loss scaling can dominate optimization. At uniform predictions, FD is ln(2) ≈ 0.693 and CMC at b=8 is ln(8) ≈ 2.079. The written token sum with n=308, s=49408 gives LR ≈ 3328.82 if all positions count; token averaging gives ≈ 10.81. Sum versus mean reconstruction differs by 150528; sum versus mean pixel CE by 50176. KL over features must sum the 512 dimensions before batch averaging, not average all elements.
7. **DERIVED — §III-D, p. 7, Eqs. (12)–(14).** A causal LD self-attention mask does not make the complete system a causal language model: encoder memory is computed from the complete target prompt. Reconstruction can copy or exploit future-token information through memory. This is not test-image label leakage because the entire language branch is removed at inference. Adding denoising, corrupted prompts, or a memory-causal mask would be an extension (A11).
8. **DERIVED — Eqs. (19)–(23), VIM.** `Tl` is image-conditioned. Alignment is not against an independent text-only teacher. Duplicate prompts do not guarantee identical `Tl`, but the diagonal-only CMC objective still treats semantically equivalent examples as negatives. Class-aware multi-positive contrastive loss would change the paper (A26).
9. **EXPLICIT ambiguity — Fig. 5 and Eq. (16).** Fig. 5 includes a residual within the illustrated VIM; (16) adds the LD residual outside `VIM`. A literal reuse of an already residualized LE module can add text twice. Separate the injection delta from residual ownership in a declared interpretation (A13).
10. **DERIVED — §III-D and Table IX.** Four d-by-d VIM matrices give 1048576 weights/block at d=512, excluding biases. Twelve instances give 12.583M; seven give 7.340M, near Table IX's increments 12.59M and 7.34M. This supports one VIM per block, despite §V-E's wording about “two VIMs” (LE and LD placements). It does not prove exact bias or sharing choices (A27).
11. **DERIVED — §V-F, p. 13.** Cross-attention map cost is proportional to `n_text * n_visual * d`; one visual token is linear in text length. Calling all-token cross-attention categorically quadratic omits the two distinct sequence lengths. Projection and FF costs remain, and 173.04G FLOPs is not an inference latency figure.
12. **EXPLICIT / DERIVED — §III-E.** Reconstruction targets are the input images for both real and fake samples; no real-only pretraining, detach, or reconstruction bottleneck regularizer is specified. A successful autoencoder can reduce forgery residuals as training progresses. Claims of a useful residual prior are empirical, not guaranteed by (17).

## Experiments and numerical reference targets

All values in this section are **EXPLICIT reported results**, in percent; they are targets for later comparison, not results reproduced here. Missing splits, preprocessing, seeds, and selection procedures prevent a statistical replication claim from these numbers alone. Main protocol details are pp. 8–10 §IV; measurement ambiguities are A23–A25.

### Cross-forgery: Table II, p. 9

Train on one GenFace forgery type and test on the other types. Genuine examples are needed for binary metrics (**DERIVED**), but their selection/counts are unspecified.

| Train → test | ACC | AUC |
|---|---:|---:|
| EFS → AM | 60.13 | 80.72 |
| EFS → FS | 79.26 | 84.03 |
| AM → EFS | 56.73 | 89.03 |
| AM → FS | 54.62 | 83.40 |
| FS → EFS | 61.13 | 85.98 |
| FS → AM | 64.92 | 83.73 |

### Cross-diffusion: Tables III–IV, pp. 10–11

Each training row corresponds to a separate training run. DDPM, LatDiff and CollDiff are EFS; DiffFace is FS; Diffae is AM (Fig. 6).

| Train | Test | ACC | AUC | mIoU |
|---|---|---:|---:|---:|
| DDPM | LatDiff | 60.32 | 98.41 | 59.34 |
| DDPM | CollDiff | 55.09 | 78.04 | 58.69 |
| DDPM | DiffFace | 82.65 | 99.98 | 58.49 |
| DDPM | Diffae | 69.88 | 99.90 | 73.87 |
| LatDiff | DDPM | 100 | 100 | 74.89 |
| LatDiff | CollDiff | 56.34 | 99.50 | 55.71 |
| LatDiff | DiffFace | 100 | 100 | 73.20 |
| LatDiff | Diffae | 99.91 | 100 | 75.63 |
| CollDiff | DDPM | 100 | 100 | 98.56 |
| CollDiff | LatDiff | 96.70 | 99.99 | 91.37 |
| CollDiff | DiffFace | 92.64 | 99.98 | 72.94 |
| CollDiff | Diffae | 99.96 | 99.99 | 71.32 |
| DiffFace | DDPM | 99.99 | 99.99 | 65.60 |
| DiffFace | LatDiff | 86.64 | 93.60 | 71.68 |
| DiffFace | CollDiff | 61.97 | 98.43 | 57.28 |
| DiffFace | Diffae | 54.63 | 98.97 | 75.47 |
| Diffae | DDPM | 94.65 | 99.76 | 73.84 |
| Diffae | LatDiff | 68.97 | 99.99 | 58.09 |
| Diffae | CollDiff | 61.79 | 79.73 | 58.09 |
| Diffae | DiffFace | 84.60 | 99.83 | 79.15 |

High AUC with low ACC appears repeatedly. This can reflect threshold/calibration shift; it is not evidence that a reproduction should choose thresholds on test labels. MFVLR is not the best result in every table cell: e.g. DDPM → CollDiff AUC 78.04 is below MSCCNet's 82.56 (Table III); DiffFace → LatDiff AUC 93.60 is below MFCLIP's 99.94 (Table IV).

### AM / FS cross-generator and cross-dataset targets

| Protocol / source | Train → test | ACC | AUC |
|---|---|---:|---:|
| AM, Table V p. 12 | Diffae → LatTrans | 59.98 | 82.31 |
| AM, Table V | Diffae → IAFaces | 69.31 | 91.86 |
| AM, Table V | LatTrans → Diffae | 67.04 | 84.69 |
| AM, Table V | LatTrans → IAFaces | 56.38 | 76.29 |
| FS, Table VII p. 13 | DiffFace → FSLSD | 58.97 | 69.02 |
| FS, Table VII | DiffFace → FaceSwapper | 77.31 | 99.95 |
| FS, Table VII | FSLSD → DiffFace | 58.76 | 60.77 |
| FS, Table VII | FSLSD → FaceSwapper | 60.73 | 96.20 |
| Cross-dataset, Table VIII p. 13 | FF++ → FF++ | 96.37 | 99.69 |
| Cross-dataset, Table VIII | FF++ → Celeb-DF | 66.73 | 73.08 |
| Cross-dataset, Table VIII | FF++ → DFDC | 72.89 | 75.06 |
| Cross-dataset, Table VIII | FF++ → DF-1.0 | 67.68 | 73.97 |

Only L1 and L2 prompts are used for FF++ training (§IV-B, p. 10). Table VIII gives no localization scores. Table V does not report IAFaces as a training source; Table VII does not report FaceSwapper as a training source. Do not invent missing runs as published experiments.

### Ablation inventory and interpretation

**EXPLICIT — Table VI, p. 12:** train DiffFace, evaluate FSLSD and LatDiff. Triples are ACC/AUC/mIoU; `—` means not reported.

| Enabled modules | FSLSD | LatDiff |
|---|---|---|
| IE | 50.00 / 47.72 / — | 58.40 / 70.46 / — |
| IE, AD | 50.00 / 50.86 / — | 59.96 / 73.06 / — |
| IE, MD | 50.00 / 48.62 / 47.96 | 60.32 / 72.58 / 52.94 |
| IE, AD, RE | 52.76 / 59.27 / — | 63.97 / 80.23 / — |
| IE, AD, RE, MD | 53.68 / 60.73 / 58.97 | 65.75 / 83.40 / 63.67 |
| Previous + LE | 54.03 / 62.15 / 61.54 | 70.36 / 86.08 / 65.14 |
| Previous + LD | 58.97 / 69.02 / 65.04 | 86.64 / 93.60 / 71.68 |

**EXPLICIT — Table X, p. 14:** train DiffFace; loss-set results, ACC/AUC.

| Losses | FSLSD | FaceSwapper |
|---|---|---|
| FD | 50.00 / 47.72 | 50.00 / 85.47 |
| FD + CMC | 52.95 / 50.48 | 54.76 / 88.13 |
| FD + AR + FL | 53.68 / 60.73 | 59.66 / 90.48 |
| Previous + CMC | 53.87 / 61.20 | 64.74 / 92.86 |
| Previous + KL | 54.03 / 62.15 | 69.70 / 96.91 |
| Previous + LR | 58.97 / 69.02 | 77.31 / 99.95 |

Architecture retained in each loss ablation is not stated. Similar values in Tables VI and X suggest some architectural coupling, but do not establish it. Removing a loss and removing a module must be separate experiment controls (A28).

| Study | Explicit settings and main observation | Reference / limitation |
|---|---|---|
| Image depth | B=0,1,2,3,4,5; FSLSD AUC 60.97,62.45,62.88,64.96,69.02,65.99; FaceSwapper AUC 94.78,96.09,96.80,97.26,99.95,99.90 | Table XI p. 14; B=0 pooling undefined |
| VIM placement family | No VIM, LE only, LD only, both; MFVLR params 121.07M,133.66M,128.41M,141.00M | Table IX p. 14; train CollDiff |
| VIM within block | LE before/between/after MHA and FF; LD likewise; best LE between, LD after | Table XII p. 14; duplicate `**` configurations disagree |
| Increasing prompts | L1 → L1+L2 → +L3 → +L4; FSLSD AUC 58.80,60.93,64.97,69.02; LatDiff 89.55,91.76,92.89,93.60 | Table XIII p. 15 |
| Missing prompts | Remove L1/L2/L3/L4; FSLSD AUC 68.09/63.74/66.72/65.40; LatDiff 92.09/90.76/91.02/91.98 | Table XIII; overlaps with progressive rows disagree |
| Fusion | Concatenation/summation/cross-attention; FSLSD AUC 64.09/65.78/69.02; LatDiff 92.64/90.79/93.60 | Table XIV p. 15; alternative fusion implementation absent |
| VIM K/V | All / patch / class tokens; 175.71/175.70/173.04G FLOPs, each 141.00M params | Table XV p. 15; token fusion and profiler unspecified |
| LE depth | E=10,11,12,13 with D=6; best E=12 | Table XVI p. 15 |
| LD depth | D=5,6,7,8 with E=12; best D=7 | Table XVI; two distinct rows for E=12,D=6 |
| Language decoder | Compare CLIP with/without TD and MFVLR with/without LD after training CollDiff | Fig. 8 p. 12; curves lack exact labels; no exact digitized values asserted |
| Mask threshold | 0.05,0.10,0.15; train DiffFace; test DDPM,CollDiff,LatDiff,Diffae | §V-J / Fig. 10 p. 15; exact plot values and target-regeneration policy absent |

### Figures and qualitative evidence

| Figure | What is shown (**EXPLICIT**) | Reproduction interpretation |
|---|---|---|
| 1, p. 2 | Baseline versus MFVLR training/inference overview | High-level diagram; Fig. 4 and equations supply details |
| 2, p. 2 | Real, diffusion and GAN residual/Sobel/FFT comparisons; 10k real and 10k fake priors for FID; residual FID 400.31 diffusion / 14.71 GAN, Sobel 25.25/20.32, FFT 32.94/20.28 | Sampling, residual reconstructor, FFT filters and FID implementation not specified; this is not a detector metric |
| 3, p. 2 | MFCLIP t-SNE with/without language; 500 real and 500 fake per diffusion generator | Motivation using another model, not a MFVLR forward-pass specification |
| 4, p. 5 | Detailed train/inference diagram and weight sharing | Resolve apparent feedback using §III-A's sequential data dependencies |
| 5, p. 6 | VIM Q/K/V projections, head splitting, attention, output projection, residual | Reveals singleton key issue when combined with stated tensor shapes |
| 6, p. 6 | L1–L4 prompts for genuine and forged images | Explicit examples; not a full tokenizer or dataset mapping |
| 7, p. 9 | Seven corruption types, five severities plus clean severity 0, AUC and average | Train GenFace/test distorted DF-1.0 (§IV-B p. 10); corruption parameters absent |
| 8, p. 12 | Language-decoder ablation plots | Four test generators are visible; precise unlabelled bar heights are not numerical facts |
| 9, p. 14 | Source, reconstruction, residual examples for real/diffusion/GAN | Display scaling and sample selection unknown |
| 10, p. 15 | Threshold sensitivity | Detection is comparatively stable; localization peaks at 0.1 in plotted examples |
| 11, p. 15 | GT/predicted masks and heatmaps for MVE; MVE+VD; MVE+VD+FLT | Heatmap method/layer not given; mask output must not be replaced by a heatmap |
| 12, p. 16 | MFVLR, HiFi-Net, MSCCNet masks for DiffFace/Diffae | Qualitative examples are not statistical acceptance evidence |

Corruptions are saturation, contrast, block-wise distortions, Gaussian noise, Gaussian blur, pixelation, and video compression (pp. 9–10). Severity parameter values, codec settings, and exact evaluation samples are absent. No invented corruption recipes should be called paper-faithful.

## Numerical and presentation inconsistencies

These are **EXPLICIT conflicting statements**, with **DERIVED** arithmetic where indicated; none is silently reconciled.

| Location | Discrepancy | Treatment |
|---|---|---|
| p. 9 prose vs Table V p. 12 | Prose calls 82.31 an ACC for Diffae → LatTrans; table has ACC 59.98, AUC 82.31 | Use table columns for the reference ledger; preserve prose error |
| §V-A p. 10 vs Table VI p. 12 | AD claim +3.40 LatDiff AUC; 73.06−70.46 = +2.60 | Do not reproduce 3.40 as a computed improvement |
| §V-A vs Table VI | Claimed RE gain +7.17 AUC/+7.65 mIoU lacks one isolating row pair | +7.17 AUC matches adding RE to IE+AD; that pair has no mIoU. +7.65 mIoU does not match a difference between the reported module rows for either test set |
| Table VI caption p. 12 | Calls the model MVLRT | Treat as naming typo, not another architecture |
| Table IV p. 11 vs IX p. 14 | CollDiff → LatDiff AUC 99.99 vs 99.98 | Retain both; no stated run/rounding explanation |
| Table XII p. 14 | Same `VIM**` configuration appears with FSLSD AUC 65.48 vs 67.34 and LatDiff AUC 92.96 vs 90.32 | Distinct reported observations; unspecified seeds/setup |
| Table XIII p. 15 | `L1+L2+L3` versus `without L4` has different values, e.g. FSLSD AUC 64.97 vs 65.40 | Packing/run differences unknown |
| Table XVI p. 15 | E=12,D=6 gives FSLSD AUC/mIoU 63.48/54.11 and 65.80/60.99; LatDiff 92.17/69.61 vs 92.55/66.90 | No unique numerical target for this setting |
| Fig. 8(c) p. 12 vs Tables IV/IX | Plot category order places Diffae before DiffFace, while the bar-height ordering appears reversed relative to the tabulated mIoUs | Visual indication only; no exact numerical correction inferred |
| Eq. (19) prose p. 8 | Temperature 0.5 described as smoothing | Under softmax(z/T), T<1 sharpens; convention must be stated |
| §VI p. 16 | Prose refers to second–fifth rows as predictions in Fig. 12, but row 2 is GT | Use actual figure row labels |

## Reference audit and boundaries of inherited detail

The complete bibliography was read. Reading its entries does not mean all 62 cited works were read in full. The following accounts for every reference and separates methodological dependencies from contextual comparisons.

| References in the supplied PDF | Role in MFVLR | What can be inherited from this citation alone? |
|---|---|---|
| [1] Xception; [2] ResNet; [3] multimodal transformer survey; [4] ViT survey; [5] distilled/local-global transformer; [6] lightweight transformer | Background/backbones/baselines | No MFVLR convolution or transformer configuration |
| [7] GAN; [8] StyleGAN3; [9] StyleGAN2; [10] DDPM | Image generation/background | Generator identity/family; no detector diffusion backbone |
| [11] reliability survey; [12] DADF; [13] digital manipulation detection; [14] M2TR; [15] MSCCNet; [16] HiFi-Net | Detection/localization context and comparisons | No permission to substitute their architectures |
| [17] frequency analysis; [18] spectral up-convolution study | Motivation for forgery priors | No FFT branch in MFVLR |
| [19] VLFFD; [20] DD-VQA; [21] MFCLIP | Language-guided detection; [21] is FTG provenance | Hierarchical prompting is cited; MFCLIP noise branches/losses do not transfer automatically |
| [22] CViT; [23] ViT; [24] composition-search detector; [25] unsupervised contrastive detector; [26] DIRE; [27] FreqNet; [28] SAM; [29] SIDA; [30] DiffForensics | Baselines/related mechanisms | No pretrained reconstruction or segmentation system is required by MFVLR |
| [31] VLM survey; [32] VLM robustness; [33] task residual tuning; [34] CLIP; [35] CoOp; [36] SoftCLIP; [37] CFPL-FAS; [38] C2P-CLIP | VLM background; [34] tokenizer and contrastive precedent | CLIP tokenizer reference is explicit; pretrained CLIP weights and all CLIP block defaults are not |
| [39] accumulated attention; [40] MMDet; [41] M2F2-Det | Multimodal fusion comparisons | No LLM or video temporal module in MFVLR |
| [42] GenFace; [43] high-frequency detector; [44] U-Net | Dataset, related method, encoder/decoder family | U-Net family is explicit; exact adapted 224→14→224 topology is not |
| [45] FatFormer; [46] latent diffusion; [47] collaborative diffusion; [48] Diffusion Autoencoders; [49] latent transformer; [50] IA-FaceS; [51] FSLSD; [52] FaceSwapper | Baseline and generator provenance | Metadata mapping evidence only; DiffFace's generator citation is not separately identifiable in the bibliography |
| [53] FF++; [54] DFDC; [55] Celeb-DF; [56] DeeperForensics-1.0 | Cross-dataset evaluation | Dataset identities; not exact MFVLR splits/frame sampling |
| [57] RECCE; [58] EfficientNet/ViT combination; [59] FoCus; [60] UIA-ViT; [61] uncertainty-aware labeling; [62] gradient regularization | Table VIII comparisons | No extra MFVLR objective or training hyperparameters |

Primary bibliographic records were checked for [MFCLIP](https://arxiv.org/abs/2409.09724), [CLIP](https://proceedings.mlr.press/v139/radford21a.html), and [U-Net](https://arxiv.org/abs/1505.04597). These checks establish the cited works' identities, not missing MFVLR implementation settings. No existing MFVLR source code or reproduction was consulted. Future investigation of cited primary papers or tokenizer artifacts must remain separately attributed and cannot upgrade an MFVLR-specific assumption to EXPLICIT without evidence in the supplied paper.
