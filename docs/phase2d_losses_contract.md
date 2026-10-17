# Phase 2D six-loss contract and equation audit

Status: **DESIGN ONLY; P2D06-P2D18 remain ASSUMED / PROPOSED**. No objective has been implemented or executed. The decision register is in [heads contract](phase2d_heads_contract.md). [Numerical readiness](phase2d_numerical_readiness.md) and [test plan](phase2d_test_plan.md) define future gates. Accepted N2 is not an approval of these loss interpretations.

Primary source: [MFVLR.pdf](../papers/MFVLR.pdf), Section III-E, PDF p. 7 Eq. (17), PDF p. 8 Eqs. (18)-(27), Table I and Section IV-A. The displayed reference equations below preserve the paper's operations, including missing normalizations and reductions. Operational equations use a hat to distinguish proposed executable interpretations from the paper. Notation formatting is normalized; no probability conversion, detach, temperature factor or spatial reduction is silently inserted into a published equation.

## Common tensor and scalar contract

**EXPLICIT:** b is batch size, n=308, s=49,408, d=512 and f=2 (p. 8). **ASSUMED / PROPOSED:** all production feature/logit/loss arithmetic is FP32 on one matching device; analytical tests use FP64 with corresponding copied inputs/weights. No autocast, implicit casting, mixed devices, broadcasting of mismatched targets or partial-batch matching. Natural logarithms are used. All six returns are rank-0 finite scalars. Shapes, class ranges and every numerical intermediate needed by a loss must be validated; finite inputs alone do not guarantee finite dot products, squares or quotients. Fail visibly on nonfinite results; never use `nan_to_num`, hidden epsilon clamps or skipped samples.

| Name | Shape / dtype / meaning | Provenance |
|---|---|---|
| I | `[B,3,224,224]`, FP32 RGB target/input in [0,1] | Input scaling is an accepted engineering boundary, not specified in Eq. (17) |
| Ipre | Same shape; arbitrary finite FP32, **not bounded** | Accepted identity appearance head, VD D01-D08 |
| z_fd | `[B,2]`, raw logits | Proposed one-FC detector; Table I explicitly calls ypre logits |
| c | `[B]`, int64, 0=Real/1=Fake | Proposed compact encoding of paper's one-hot y |
| z_fl | `[B,2,224,224]`, raw logits | Accepted mask head, proposed class interpretation |
| m | `[B,224,224]`, int64, exactly 0 or 1 | Proposed compact one-hot mask encoding; not an image label broadcast |
| z_lr | `[B,308,49408]`, raw tied vocabulary logits | Accepted Eq. (24) implementation |
| t | `[B,308]`, int64 in [0,49407] | Accepted unshifted four-segment tokenizer IDs |
| v, l, a | Each `[B,1,512]`, raw Iv, Tl and T_lpre | Accepted fusion/last-token interface; proposed adapter |
| theta | Scalar FP32 trainable log denominator-temperature | ASSUMED parameterization of explicit trainable tau |

One-hot equivalence is a mathematical reference, not a requirement to allocate dense one-hot targets. Hard class indices are the sole proposed production FD/FL/LR target format. Target IDs and masks cannot require gradients. Images normally do not require gradients; the AR formula does not silently detach I if an analytical input-gradient probe does require it. Do not infer whether a tensor is logits or probabilities from its numerical range: a valid logit can lie in [0,1]. The API fixes that interpretation.

## 1. Forgery detection, L_fd

**EXPLICIT reference, p. 8 Eq. (26):**

\[
\mathcal L_{fd}=\frac1b\sum_{u=1}^{b}-(y^u)^T\log(y^u_{pre}).\tag{26}
\]

The paragraph/Table I call ypre detection **logits**, while this log requires positive normalized probabilities for categorical CE. A linear map need not even produce positive values. This is an actual notation inconsistency, not an instruction to clamp the outputs.

**ASSUMED P2D06:** interpret the expression as categorical negative log likelihood after class softmax:

\[
\widehat L_{fd}=-\frac1B\sum_u\operatorname{logsoftmax}(z_{fd}^u)_{{c_u}}.
\]

Use stable log-softmax along dimension 1 (two classes), class selection, then batch mean; no second softmax in the head, label smoothing, class weights, BCE, focal loss or extra factor. Integer c is equivalent to one-hot y with background class 0 and fake class 1. At zero logits the value is ln(2), independent of class balance and B. Stable CE from logits avoids explicitly logging an underflowed softmax; this is the API semantics of [PyTorch 2.12 CrossEntropyLoss](https://docs.pytorch.org/docs/2.12/generated/torch.nn.CrossEntropyLoss.html), not evidence that MFVLR disclosed this missing operation.

**DERIVED gradients:** `dL/dz_uc=(p_uc-onehot(c_u)_c)/B`. Paths reach detector, fused visual features, both shared MVE calls and AD through residual reconstruction. No direct gradient to FLT, adapter or mask head. Fixed-dimension global Transformer parameters are involved, unlike local-only AR/FL.

**Tests:** independent FP64 logsumexp/one-hot oracle; uniform ln(2); common-logit-offset invariance; analytical derivative including 1/B; wrong-class extreme finite logits; correct Real/Fake ordering; explicit logits boundary; reject invalid labels/shapes. Detached residual controls isolate AD gradients (test-only). See LOSS-FD01-FD04 and JOINT tests.

## 2. Language reconstruction, L_lr

**EXPLICIT reference, p. 8 Eqs. (24)-(25):**

\[
T_{pre}=T^d_{rec}W_{voc}^{T}\in\mathbb R^{n\times s},\tag{24}
\]
\[
\mathcal L_{lr}=\frac1b\sum_{u=1}^{b}\sum_{x=1}^{n}-(T_{gt}^{u,x})^T\log(T_{pre}^{u,x}).\tag{25}
\]

Eq. (24) is a raw linear projection. Eq. (25) omits vocabulary softmax. The displayed token **sum** and batch **mean** are explicit; ignoring pads or replacing the sum with a mean is not explicit.

**ASSUMED P2D07:**

\[
\widehat L_{lr}=-\frac1B\sum_{u=1}^{B}\sum_{x=0}^{307}
\operatorname{logsoftmax}(z_{lr}^{u,x})_{t_{u,x}}.
\]

Vocabulary axis is the last axis, size 49,408. Every position participates; use original unshifted IDs as targets. The accepted FLT already shifts embeddings globally right by one, prepending the BOS embedding; the loss must not shift targets again. Each of four 77-slot segments contains its own BOS=49,406 and EOS=49,407, with accepted zero padding; the pad value 0 remains a valid vocabulary ID. All BOS, EOS and pad positions count, including position 0 and all positions across boundaries 76/77, 153/154, 230/231. Padding is not automatically ignored because the ID is zero. No length normalization, segment averaging, ignore-index, EOS pooling or special-token weighting is introduced.

This is an ASSUMED all-position interpretation of an EXPLICIT n-position sum. Pad-ignored token-sum, nonpad-token mean, all-token mean and segment-wise shifting are distinct alternatives requiring new approval. Log pad/special/ordinary-token contributions separately **without changing the total**. Repetitive pads can dominate counts and supply an easy reconstruction shortcut. Encoder memory includes complete target text, so causal LD self-attention does not make this a target-blind generation objective (pp. 6-7).

**DERIVED:** uniform vocabulary logits give `308 ln(49408) = 3328.823231...`, not `ln(49408)`. Each logit gradient is `(p-onehot)/B`; there is no 1/308 factor. Wvoc receives the sum of output-projection and embedding-input contributions through the **same Parameter**. Input contributions include both the original prompt lookup and the direct Wvoc[49406] access used for the prepended BOS in accepted `shift_low`; detaching the original lookup alone does not isolate all embedding-input paths. Gradients reach LD, encoder memory/LE, learned positions, all connected nondegenerate VIM paths, Iv, MVE and reconstruction-mediated AD. VIM Q/K task gradients remain zero. Adapter, detector and mask head have no direct LR path.

**Stability/tests:** stable last-axis log-softmax, no explicit one-hot full-vocabulary allocation. At finite but extreme logits, underflowed non-target probabilities may be zero while selected log probabilities remain finite; this is different from logging zero probabilities. Test exact reduction factors on small independent fixtures, all 308 positions/full-vocabulary smoke, pad and every special-token contribution, shifted-embedding/unshifted-target ownership, tied-weight gradient addition and finite gradients. Do not require LR decrease at untrained initialization or infer CE magnitudes from N2's squared-logit probes.

## 3. Cross-modal contrastive, L_cmc

**EXPLICIT reference, p. 8 Eqs. (20)-(23):**

\[
S^{um}_{v2l}(I_v,T_l)=
\frac{\exp(sim(I_v^u,T_l^m)/\tau)}{\sum_{m=1}^{b}\exp(sim(I_v^u,T_l^m)/\tau)},\tag{20}
\]
\[
S^{um}_{l2v}(T_l,I_v)=
\frac{\exp(sim(T_l^u,I_v^m)/\tau)}{\sum_{m=1}^{b}\exp(sim(T_l^u,I_v^m)/\tau)},\tag{21}
\]
\[
\mathcal L_{v2l}=\frac1b\sum_{u=1}^{b}-(y^u_{pa})^T\log S^u_{v2l}(I_v,T_l),\tag{22}
\]
\[
\mathcal L_{l2v}=\frac1b\sum_{u=1}^{b}-(y^u_{pa})^T\log S^u_{l2v}(T_l,I_v).\tag{23}
\]

The unnumbered definition is `L_cmc=(L_v2l+L_l2v)/2`, with diagonal positives `y_pa^(uu)=1`, off-diagonal zero. Tau is trainable and initially 0.07. The same paragraph first says **cosine similarity** then defines `sim` as **dot product**. Both statements must be retained in the evidence record.

**ASSUMED P2D08-P2D10, recommended equation-oriented baseline:** remove only token singleton axes, V,L `[B,512]`; use raw dot products `A=V L^T / exp(theta)` `[B,B]`, theta initialized to log(0.07). No normalization, adapter, additional projection, detached bank, all-gather or accumulated negatives. Then

\[
\widehat L_{cmc}=\frac1{2B}\sum_u
\left[-\operatorname{logsoftmax}_{m}(A_{u,:})_u
-\operatorname{logsoftmax}_{m}((A^T)_{u,:})_u\right].
\]

Both softmaxes are over candidate partners, never batch columns accidentally left untransposed. In exact arithmetic raw dot symmetry permits A transpose for reverse direction. Compute both directions from the same matrix to make ownership explicit. Stable row log-softmax avoids unbounded raw exponentials; it does **not** prevent overflow in V L^T or division by a tiny tau. Assert finite positive effective tau, finite A, finite loss and gradients. No lower bound/clamp/inverse-temperature swap is approved. Exp parameterization ensures positivity in real arithmetic but can underflow/overflow in FP32 and must then fail.

Log-temperature is a new optimization assumption: dA/dtheta=-A. Direct tau optimization has derivative -A/tau and a different update trajectory; both have one scalar. Proposed future optimizer excludes theta from decay, but optimizer approval remains separate. Tau=0.07 is an effective temperature, not a logit-scale initialized to 0.07. Numerical representation of exp(float32(log(0.07))) is checked with declared test precision, not literal decimal equality.

**Alternative, not approved:** cosine normalization only for temporary CMC operands before dot product. This would resolve the prose instead of the raw-dot statement. It needs an explicit zero-vector policy/epsilon, changes magnitude gradients and must not normalize Iv globally for VIM/KL/detection. It is not chosen solely to stabilize the unnormalized FLT.

**Duplicate prompts:** retain diagonal instance labels even when metadata/IDs duplicate. Tl is image-conditioned, so equal text does not necessarily imply equal Tl. Nevertheless semantic duplicates are used as negatives and can impose competing objectives. For identical V/L rows all candidate scores tie: CMC=ln(B), not zero. Depending on tied features, gradients can cancel; do not demand nonzero gradients on this fixture. Do not deduplicate the batch, build class positives or add soft targets silently. Record counts of unique prompt hashes and metadata groups. Sampler/class balance is a deferred dataset decision.

**DERIVED gradients:** let R be row softmax(A) and C column softmax(A). Then `G=dL/dA=(R+C-2I)/(2B)`. Partial derivatives treating V,L as leaves are `dL/dV=G L/tau`, `dL/dL=G^T V/tau`, `dL/dtheta=-sum(G*A)`. Actual Tl depends on Iv, so the full Iv gradient also includes the LE/VIM path. CMC intentionally couples different examples through negatives. Accepted exact zero cross-sample **forward-model** gradients remain valid; a zero cross-sample requirement on this contrastive objective would be wrong. B=1 must execute the formula, producing exactly zero task loss and zero task gradients, with connected gradients rather than an arbitrary detached constant. B=1 cannot establish contrastive learning readiness.

**Tests:** nonsymmetric 2x2/3x3 independent analytical oracles expose a missing transpose; row/column normalization, diagonal labels, 1/(2B), temperature direction/derivative, B=1 degeneracy, duplicates, joint pair permutation, controlled nonzero cross-sample loss derivatives, raw-feature scaling sensitivity, rejected invalid tau and extreme finite scores. No N1/N2 tolerance is borrowed for these losses.

## 4. Forgery localization, L_fl

**EXPLICIT reference, p. 8 Eq. (18):**

\[
\mathcal L_{fl}=\frac1b\sum_{u=1}^{b}-(M^u)^T\log(M^u_{pre}).\tag{18}
\]

Paper M is a 224x224 binary spatial mask, whereas Mpre has two channels (p. 6 Section III-C, p. 8 f=2). Neither one-hot class expansion nor a spatial scalar reduction is specified. An ordinary spatial matrix transpose/product is not a scalar pixel classification loss. Taking only foreground `-M log(p_foreground)` would make every real/background pixel contribute zero, an unsuitable silent reinterpretation of the two-class head.

**ASSUMED P2D11/P2D14:** for each pixel use `[1-m,m]` as the mathematical one-hot vector and

\[
\widehat L_{fl}=-\frac1{BHW}\sum_{u,h,w}
\operatorname{logsoftmax}_{class}(z_{fl}^{u,:,h,w})_{m_{u,h,w}},\quad H=W=224.
\]

The selected production representation is integer m; no dense one-hot materialization. Mean pixels then mean batch is equivalent at this fixed resolution. No valid-pixel mask, missing-target sentinel, ignored region, class weighting or Dice term is approved. If a valid target is unavailable, stop target validation rather than substitute a fabricated mask. All-background, all-foreground and mixed masks are valid and retain both class probabilities in the denominator. Two-class softmax is confined to interpretation/presentation, not added to accepted VD.

**DERIVED gradients:** at each pixel `(p-onehot)/(BHW)`. Paths reach mask head, shared decoder trunk, and shared **local** appearance encoder. They do not traverse appearance head, residual-call activations, visual Transformer, FLT or adapter in an isolated FL-only graph. Because weights are shared, updating the local encoder/trunk changes later appearance/residual behavior; that is not a current autograd path through the unused roles.

**Tests:** small independently expanded one-hot reference, both constant masks, rectangular/checkerboard masks, background false-positive penalty, foreground false-negative penalty, exact mean denominator, pixel permutation/spatial tiling invariance within numerical tolerance, analytical derivatives, correct absent gradients for unused branches. Sigmoid per-channel/BCE is not an acceptable replacement.

## 5. Appearance reconstruction, L_ar

**EXPLICIT reference, p. 7 Eq. (17):**

\[
\mathcal L_{ar}=\frac1b\sum_{u=1}^{b}(I^u-I^u_{pre})^2.\tag{17}
\]

The written batch mean leaves channel/spatial axes unreduced under elementwise square. The paper does not identify squared Euclidean norm, pixel mean or channel mean. A scalar implementation must declare its extra reduction.

**ASSUMED P2D12:**

\[
\widehat L_{ar}=\frac1{3BHW}\sum_{u,c,h,w}(I_{u,c,h,w}-Ipre_{u,c,h,w})^2.
\]

Use all real and fake images, including locally edited and fully generated cases. Target is the actual network input I, **not its pristine paired source**; source pairs are only for mask construction. All RGB channels/pixels have unit weight before averaging. Finite unrestricted Ipre is accepted, including <0 or >1. Do not sigmoid, tanh or clip it before AR, residual encoding or reporting numerical extrema. A display-only clipped visualization would be separate and cannot feed the graph. Out-of-range reconstruction is not itself a loss error; nonfinite residuals/squares/reduced results are. Float32 squared differences can overflow even for finite operands, so check intermediates.

**DERIVED gradients:** `dL/dIpre=2(Ipre-I)/(3BHW)`. With independent image-target probes `dL/dI=-dL/dIpre`; a full image-input probe also has network paths. AR reaches appearance head, shared VD trunk and local appearance encoder, not the visual Transformer, residual-call activations, mask head, FLT, detector or adapter. The residual branch receives other objectives' gradients and combines with direct AR at Ipre. The accepted abs derivative is sign(Ipre-I) away from zero and zero at equality; do not replace it or introduce a straight-through gradient.

**Tests:** exact zero at reconstruction identity; constant delta gives delta squared regardless of resolution/B; analytic derivative; out-of-range fixture (e.g. I=0, Ipre=2 gives 4, derivative positive), differentiable image-target control, all-image weighting, no clipping, and isolated residual/direct reconstruction gradient addition. A pixel/channel **sum** is a named alternative scaled by 150,528, not equivalent to this proposal.

## 6. Knowledge distillation / KL, L_kl

**EXPLICIT reference, p. 8 Eq. (19):**

\[
\mathcal L_{kl}=\frac1b\sum_{u=1}^{b}
\delta(T_l^u)^T\log\frac{\delta(T_l^u)}{\delta(T_{lpre}^u)}.\tag{19}
\]

The accompanying prose defines delta as softmax with temperature **0.5**. The probability order is text first, adapted vision second. The paper does not state axis, divide-versus-multiply temperature convention, target detachment or a temperature-squared multiplier. Its description of smoothing is questionable under the conventional divide-by-0.5 interpretation, which sharpens distributions.

**ASSUMED P2D13:** set `logP=logsoftmax(l/0.5, feature axis)` and `logQ=logsoftmax(a/0.5, feature axis)`, P=exp(logP). Treat the 512 coordinates as a categorical feature distribution, not as Real/Fake classes. Then

\[
\widehat L_{kl}=\frac1B\sum_{u=1}^B\sum_{j=1}^{512}
P_{uj}(\log P_{uj}-\log Q_{uj}).
\]

Singleton token axis is harmless; feature axis must be 512, not the singleton (which would give zero). Use both log-probabilities directly; never form P/Q or take log(P) after underflow. P can underflow to zero while logP remains finite; `P*(logP-logQ)` is finite under finite log intermediates. Reject nonfinite intermediates. Do not clamp a small negative roundoff result to zero. Unit tests allow only their specified analytical rounding tolerance for nonnegativity.

No detach on **either** l or a. No frozen/pretrained teacher, EMA or stop-gradient, and no T^2 factor (which would multiply this term by 0.25). Feature **sum**, batch mean; an all-element mean would wrongly divide by 512. If using a framework KL helper later, its argument order must reproduce KL(P_text || Q_adapter) and its reduction must match this explicit formula; default elementwise mean is not equivalent to batchmean. [PyTorch 2.12 KLDivLoss](https://docs.pytorch.org/docs/2.12/generated/torch.nn.KLDivLoss.html) documents that distinction; the written two-sided formula, not an undocumented target-gradient default, is the acceptance oracle.

**DERIVED controlled gradients** (T=0.5): `dL/da=(Q-P)/(BT)`. Let `D_u=sum_j P_uj(logP_uj-logQ_uj)`; `dL/dl_uj=P_uj[(logP_uj-logQ_uj)-D_u]/(BT)`. These derivatives include the target entropy derivative. Differentiating only cross entropy while omitting `P logP` would be wrong when P is trainable. Identical P=Q gives KL=0 and zero gradients; saturation can make some teacher gradients tiny even when the student is wrong.

Full graph: adapter -> Iv directly, and Tl -> LE/Wvoc/encoder positions -> Iv via LE VIM; both reach visual modules and AD through residual reconstruction. No LD/decoder-position/output-projection-only path from KL, despite Wvoc being the same tensor used by LD elsewhere. Detaching Tl in a **test control** removes the LE path but must leave the adapter-to-Iv path. Freezing teacher weights, detaching Iv inside LE and reversing KL are distinct alternatives, none selected.

**Tests:** unequal known probability pair with l=T logP, a=T logQ; KL asymmetry; identical distributions; analytical derivatives on both inputs; feature-axis check; factor-512 reduction check; temperature scaling and absence of T^2; large finite score/log-probability fixture; detached-target control only in tests. The teacher/student names denote distribution direction, not an approved one-way gradient policy.

## Masks and source semantics

**EXPLICIT pp. 7-8:** real mask=0; entire face synthesis (EFS) mask=1; paired source/forgery absolute RGB difference -> grayscale -> divide by 255 -> threshold 0.1 for local manipulation. A source image is not required for the real/EFS constants. The EFS mask is the whole input frame, not an inferred face segmentation. Generator family alone does not prove EFS versus local editing; use the sample's verified manipulation category.

**ASSUMED P2D14 consumer:** real `[B]` label 0 requires all-zero supplied mask; fake label 1 accepts all-one EFS or binary locally manipulated mask. A local manipulation can yield an all-zero mask after thresholding; flag this diagnostic but do not relabel the fake image or force positive pixels. All-one local masks are also mathematically valid. Detection and localization targets are different quantities. No face parsing, synthetic source reconstruction or unknown-region ignore policy is implied.

**ASSUMED P2D15 future proposal, not data-loader approval:** source and fake must be geometrically registered RGB in a known 0..255 representation; subtract in signed/float arithmetic before absolute value, use grayscale `0.299*|deltaR|+0.587*|deltaG|+0.114*|deltaB|`, divide by 255, then strict `>0.1` with no pre-threshold uint8 rounding. Difference-before-grayscale matters: converting images to gray first can cancel channel changes. Do not divide an already normalized [0,1] difference by 255 again. Recommend construction at aligned native resolution, followed by exactly the same approved geometric crop as the image and nearest-neighbor resize of the **binary** mask to 224. Image crop coordinates, alignment policy, RGB conversion, interpolation settings, source compression mismatch and transform order require the future dataset contract; absent registered source evidence blocks local-mask production. Computing masks after image resampling or using `>=0.1` is a separate, reportable alternative. No preprocessing code or dataset loading is authorized now.

## Total objective and relative magnitudes

**EXPLICIT reference, p. 8 Eq. (27):**

\[
\mathcal L=\mathcal L_{fd}+\mathcal L_{lr}+\mathcal L_{cmc}+\mathcal L_{fl}+\mathcal L_{ar}+\mathcal L_{kl}.\tag{27}
\]

**ASSUMED P2D16:** substitute the six separately disclosed operational scalars above, add left-to-right in printed order, all coefficients exactly one. Do not divide the total by six, add smoothing/regularization inside it, rescale LR, average distributed losses without an explicit contract or hide an invalid term. Optimizer weight decay is not a seventh paper loss and remains separately gated. Return raw component scalars, total and diagnostics; do not return normalized reporting values as the training total.

**DERIVED controlled baselines:** uniform FD and mean FL each ln(2)=0.693147; all-position uniform LR=3328.823231; uniform CMC at B=1/2/8 is 0/0.693147/2.079442; identical KL distributions give zero; constant reconstruction error epsilon gives AR=epsilon^2. LR token mean would divide by 308, FL spatial sum would multiply by 50,176, and AR image-element sum by 150,528. Eq. (27)'s equal coefficients do not imply equal gradient influence. Table X (p. 14) supports including all six tasks but does not disclose the missing scalarizations, resolve these scale differences, or prove this engineering baseline will train well.

## Expected isolated gradient ownership

**DERIVED conditional on the accepted graph and proposed no-detach policies.** A path means gradients can flow, not that every parameter must have a nonzero derivative for every fixture. Shared weights accumulate contributions; zero Q/K VIM gradients and saturated/symmetric controls are expected exceptions.

| Loss | New detector | Adapter | Temperature | Local encoder | Visual Transformer | VD trunk / AD head / MD head | LE + Wvoc | LD + Pd |
|---|---|---|---|---|---|---|---|---|
| FD | Yes | No | No | Yes, both roles | Yes, both roles | Yes / Yes via residual / No | No | No |
| LR | No | No | No | Yes, both roles | Yes, both roles | Yes / Yes via residual / No | Yes, tied input/output | Yes |
| CMC | No | No | Yes except B=1/symmetry | Yes, both roles | Yes, both roles | Yes / Yes via residual / No | Yes, LE path | No |
| FL | No | No | No | Yes, appearance-local path | No | Yes / No / Yes | No | No |
| AR | No | No | No | Yes, appearance-local path | No | Yes / Yes / No | No | No |
| KL | No | Yes | No | Yes, both roles | Yes, both roles | Yes / Yes via residual / No | Yes, LE path | No |

CMC/KL do not require evaluating LD to compute their own scalars, although the joint graph includes LD for LR. No loss uses generator strings directly; accepted token construction produces IDs outside the loss boundary. No production detach, pretrained teacher, gradient clipping or architectural repair is part of this proposal.
