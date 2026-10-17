# Phase 2D.2-A1: accepted decoder architecture review

2026-10-12. **Read-only reconstruction of the approved engineering baseline, not author confirmation.** Phase 2D.2 remains BLOCKED. The [paper audit](phase2d2_a1_paper_evidence_audit.md) separates explicit facts from omissions; the [preregistration](phase2d2_a1_ablation_preregistration.md) specifies unimplemented alternatives.

## Actual source and ownership

Inspected sources: [language_decoder.py](../src/mfvlr/language_decoder.py), [language_encoder.py](../src/mfvlr/language_encoder.py), [flt.py](../src/mfvlr/flt.py), [vim.py](../src/mfvlr/vim.py), [objectives.py](../src/mfvlr/objectives.py), and [approved configuration](../configs/flt_reference.yaml). The architecture below is **ASSUMED / APPROVED** under L06, L08, L09, L11 and the relevant L04-L10 boundary decisions. Dimensions/depth have explicit paper support (IV-A p. 8); detailed realization is engineering choice. Counts and gradient consequences are **DERIVED from this realization**.

There are seven distinct `LanguageDecoderBlock` objects, not seven calls to one block. Each owns self-attention, memory-attention, FF and one delta-only VIM. No LD LayerNorm, final LN, RMSNorm, dropout, gating, activation clipping or residual scale exists. `finite` checks raise on nonfinite values without changing them. LE/LD do not own MVE or VD; the fused visual tensor is a differentiable input. Nineteen VIMs are independent across twelve LE and seven LD blocks. The vocabulary is shared by lookup and output, not registered again by the decoder.

## Complete mathematical forward pass

Let N=308, d=512, h=2048, r=8 and head width k=64. Use row-vector math; stored PyTorch linear weights have shape `[out,in]`, hence their transpose appears below. All hidden states are `[B,N,d]`, memory M is `[B,N,d]`, and visual V is `[B,1,d]`.

For an ordinary attention module A with independently owned maps:

\[
Q=\operatorname{heads}(XW_Q^T+b_Q),\quad
K=\operatorname{heads}(MW_K^T+b_K),\quad
U=\operatorname{heads}(MW_V^T+b_V),
\]
\[
P=\operatorname{softmax}_{\rm key}(QK^T/8+C),\qquad
A(X,M;C)=\operatorname{merge}(PU)W_O^T+b_O.
\]

For self-attention M=X and C has zero on/below its token diagonal and negative infinity above it. Memory attention has C=0 everywhere. No operation contracts the batch axis. Q/K/U are `[B,8,308,64]`; scores/probabilities are `[B,8,308,308]`. Causality applies to self-attention, not to LE-memory access.

\[
F(X)=\operatorname{GELU}_{exact}(XW_1^T+b_1)W_2^T+b_2.
\]

The intermediate width is 2048 and exact GELU is \(x\Phi(x)\), not its tanh approximation. For each VIM, Q comes from X and K/U from the single V token, with bias-free maps. Its denominator is **sqrt(8/512)=1/8**, preserving Eq. (10), not ordinary attention's denominator 8. Scores/probabilities are `[B,8,308,1]`.

\[
\Delta(X,V)=\operatorname{merge}\left[
\operatorname{softmax}_{\rm key}\left(QK^T/(1/8)\right)U
\right]W_{O,vim}^{T}.
\]

The full Q/K graph is executed. Conditional on finite singleton attention, its task result is the broadcast of \((VW_{V,vim}^{T})W_{O,vim}^{T}\). Thus \(\partial\Delta/\partial X=0\), and VIM Q/K task gradients are allocated exact zeros when connected. V/O/visual gradients need not vanish. This identity is an analysis, not authorization to simplify production code.

With input token IDs t, W=`vocabulary.weight`, Pe/Pd independent positions:

\[
E=W[t],\quad M=LE(E+P_e,V),\quad T_l=M[:,307:308,:],
\]
\[
X_0=\operatorname{cat}(W[49406],E[:,:307,:])+P_d.
\]

The first existing BOS is itself a target; shifting introduces another BOS at the LD input. The shift crosses segment boundaries globally. No target, padding or special-token position is removed.

For **each** j=1,...,7:

\[
A_j=X_{j-1}+S_j(X_{j-1}),\qquad
B_j=A_j+C_j(A_j,M),
\]
\[
C'_j=B_j+F_j(B_j),\qquad
X_j=C'_j+\Delta_j(C'_j,V).
\]

Here S includes the causal self-attention definition and C_j denotes full memory attention, not the causal-mask symbol above. All four additions use the immediate incoming residual state once. This executes the outer structure of Eqs. (13)-(16), p. 7, under the approved VIM boundary. The output is `decoder_hidden=X7`; raw vocabulary logits are \(Z=X_7W^T\), `[B,308,49408]`, with no output bias/norm/softmax. `return_logits=False` suppresses only vocabulary materialization; it does not bypass LD. Original target IDs are used by LR.

## Shapes, counts and initialization

| Per-block owner | Stored parameter shapes | Unique elements |
|---|---|---:|
| Self-attention | Q/K/V/O: each weight `[512,512]`, bias `[512]` | 1,050,624 |
| Memory attention | independent Q/K/V/O: same shapes | 1,050,624 |
| FF first affine | weight `[2048,512]`, bias `[2048]` | 1,050,624 |
| FF second affine | weight `[512,2048]`, bias `[512]` | 1,049,088 |
| VIM | Q/K/V/O each `[512,512]`, no bias | 1,048,576 |
| Normalization | none | 0 |
| **One LD block** | **24 parameter tensors** | **5,249,536** |

Each row below has the same topology and distinct parameters. Shapes before self-attention, after each residual and on block output are all `[B,308,512]`; each FF hidden tensor is `[B,308,2048]`. The memory and visual input tensors are reused, not their projection parameters.

| Block | Source index | Block count | Cumulative LD parameters | Norms |
|---|---:|---:|---:|---:|
| LD01 | 0 | 5,249,536 | 5,249,536 | 0 |
| LD02 | 1 | 5,249,536 | 10,499,072 | 0 |
| LD03 | 2 | 5,249,536 | 15,748,608 | 0 |
| LD04 | 3 | 5,249,536 | 20,998,144 | 0 |
| LD05 | 4 | 5,249,536 | 26,247,680 | 0 |
| LD06 | 5 | 5,249,536 | 31,497,216 | 0 |
| LD07 | 6 | 5,249,536 | 36,746,752 | 0 |

W has 25,296,896 elements; each position tensor has 157,696. They are FLT-owned rather than counted in the 36,746,752 LD block subtotal. LE blocks total 50,399,232. FLT remains 112,758,272; full visual-language-six-objective ownership remains **154,156,232 / 508 tensors**. The paper's nominal 141M is not matched (Tables IX/XV, pp. 14-15).

**ASSUMED / APPROVED initialization (L11):** private CPU generator, seed 0 by default, construct FP32 and preserve caller RNG. Draw W, Pe, Pd independently from Normal(0,0.02), then twelve LE blocks, then seven LD blocks. Within each LD: self Q/K/V/O; memory Q/K/V/O; FF1/FF2; VIM Q/K/V/O. Every matrix uses separate Xavier-uniform gain 1, and every existing affine bias is zero. There is no tied-output redraw. VIM constructors' private initial draws are overwritten by the full owner's specified draw sequence; they do not make the final nineteen VIMs identical. No pretrained weights are used.

**DERIVED, conditional scaling heuristic:** a square 512 map has variance 1/512 under Xavier; a 512-to-2048 or inverse map has variance 2/2560=1/1280. For independent zero-mean input components with variance q, the first FF preactivation variance is approximately 0.4q. If its preactivations are large symmetric Gaussians, GELU's second moment approaches half that value, and the second affine gives about 1.6 times that moment, or about 0.32q. These independence/Gaussian assumptions do not hold exactly in a residual attention network. They are not a layerwise stability guarantee. In general, `E||x+f||²=E||x||²+E||f||²+2E<x,f>`; dropping the covariance term without evidence is invalid. Memory values and repeated visual additions supply additional scale. Normal(0,.02) lookup plus independent position initially has expected per-coordinate second moment .0008, but this does not bound later blocks.

## Derivative paths and conditioning

For a fixed memory and visual input in exact arithmetic, the state Jacobian of one baseline block is
\[
J_j=(I+J_{F_j})(I+J_{C_j,query})(I+J_{S_j}).
\]
The last residual factor is I because singleton VIM's language derivative vanishes. This does not remove visual derivatives. Seven-block propagation multiplies these nontrivial Jacobians. Shared-memory and visual gradients sum contributions from all depths, including V through LE memory. A cancellation-sensitive derivative may be inaccurate despite finite activations and a small aggregate relative error.

LR reaches: LD parameters; W through both shifted lookup and tied projection; Pe/LE through memory; V through seven LD VIM V/O paths and twelve LE VIM paths; MVE and AD through fused visual/residual reconstruction. CMC and KL depend on LE global output, not LD output. FD, FL and AR have no direct LD dependency. Consequently exact joint-versus-isolated LD parameter gradients can merely reflect the single LR contribution, not establish stable LD VJPs for arbitrary cotangents. N3.1's fixed-cotangent probes test that separate property.

An ordinary key bias shifts all allowed logits equally within a query row, so its exact attention task derivative cancels, including under an inclusive causal mask. Its floating-point cancellation residual is not the structural singleton zero. Ordinary Q/K **weight** derivatives generally remain nonzero and must not be excused as bias cancellation.

## Recorded scale: no new experiment

The following numbers are read from the preserved seed-0, CPU FP32, B=2 `real_efs` [full G2 record](../artifacts/phase2d2_n31/g2/final_real_efs/total.json). Its keys `LD0` through `LD6` are **post-block outputs**, relabeled LD01-LD07 here, not initial inputs.

| State | RMS | Maximum absolute value |
|---|---:|---:|
| LE12 / memory | 44.5647 | 136.4830 |
| LD01 | 51.0959 | 150.6592 |
| LD02 | 98.7089 | 280.6627 |
| LD03 | 172.1520 | 518.6098 |
| LD04 | 307.2766 | 907.8134 |
| LD05 | 492.1476 | 1770.6323 |
| LD06 | 790.1454 | 2392.7666 |
| LD07 | 1252.6392 | 4529.7969 |
| Vocabulary logits | 565.2347 | 2701.6631 |

**DERIVED from recorded diagnostics:** N3.1's unit-scale LD07 local VJP discrepancy for LD1 Q weights was about 0.003965, 0.016772 and 0.083659 relative L2 at common input multipliers .25, 1 and 2, respectively. These are neither full-model error bounds nor training measurements. Local FP64 reduces disagreement substantially but some state/memory and key-bias comparisons still fail unchanged coordinate tolerances. See [held-out validation](phase2d2_n31_heldout_validation.md). No causal attribution to missing LN alone follows from these observations.

Compared with a conventional normalized decoder, this baseline has no normalization around any of its three ordinary sublayers, adds a fourth visual residual, uses exact GELU, and reconstructs targets with access to bidirectional representations of those same targets. The ordinary normalized Transformer comparison is external precedent, not a specification inherited from MFVLR; see the [source audit](phase2d2_a1_paper_evidence_audit.md#external-evidence-has-limited-authority).

## Six-loss interactions remain unchanged

The approved LR is `mean_batch(sum_308(-log_softmax(Z)[target]))`. This is the executable interpretation of Eqs. (24)-(25), p. 8, whose displayed log lacks an explicit probability conversion. The original equation is not rewritten. BOS, EOS, padding and content all contribute. Relative to mean over batch and tokens, this scalar and every mathematical LR gradient are exactly 308 times larger for the same logits; no mean-token objective is proposed here.

At uniform logits LR is `308 log(49408)=3328.823231...`, while uniform FD/FL are `log(2)=.693147...`. These are analytical baselines, not full-model training losses. For a category with C positions over B images, uniform contribution is `(C/B) log(49408)`; the real golden prompt has 24 content, 4 BOS, 4 EOS and 276 padding positions. Padding is tracked by segment lengths, not merely token ID zero. Output-gradient entries are `(p-onehot)/B`, and their magnitude can be bounded even when logits are large; LD/W/visual parameter gradients multiply this by the model Jacobian. A large scalar by itself does not determine a parameter's gradient norm.

The same preserved `real_efs` record has LR **255,077.5625**, FD **3.389758**, CMC **206.232910**, FL **.694495**, AR **.326561**, KL **30.705650**. LR's category counts across two images are padding 542, BOS 8, EOS 8, content 58; batch-mean category NLL contributions are respectively **181,720.40625**, **6,831.86572**, **5,465.76855**, **61,059.51953**. Separate FP32 reductions explain small last-bit differences between category sums and the scalar. These are actual no-update synthetic full-graph observations, not training or convergence results.

Risk sources must remain distinct:

1. **LR reduction and repeated targets:** many padded slots and tied vocabulary rows accumulate correlated gradients; token-sum scaling changes relative objective influence. Loss reweighting/masking is not part of this review.
2. **Teacher-conditioned memory:** a causal decoder does not prevent information about its current/future target from entering via unmasked bidirectional LE memory of the same sequence. That is within-sample reconstruction conditioning, not cross-sample leakage. It may permit copying shortcuts; no shortcut-learning claim is established without an authorized experiment. A causal mask on LD self-attention alone cannot resolve it.
3. **Decoder dynamics:** fixed-cotangent N3.1 sensitivity exists without evaluating LR, so LR's reduction is not its sole cause. The large LE memory is also an input; treating LD as isolated from upstream scale would be misleading.
4. **Other losses:** CMC/KL can saturate before involving LD. Here CMC scores reach about 29,480 at tau=.07 and KL text maximum probabilities exceed .99998. They affect shared LE/visual gradients independently of LD normalization. Duplicate strings may still produce different language features through visual injection; diagonal-negative semantics remain approved and unchanged.

FD, LR, CMC, FL, AR and KL keep their current definitions, reductions, temperatures, differentiability and literal unit-weight order. No balancing, mask changes, detach, cosine substitution or optimizer is proposed as an implicit part of an LD variant.
