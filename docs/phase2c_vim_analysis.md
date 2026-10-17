# Phase 2C: VIM mathematics and reference-equation review

2026-10-11. Analysis only. All implementation choices are **ASSUMED / PROPOSED**, pending the [L01–L13 decision table](phase2c_language_contract.md). The existing MVE and VD are unchanged. No paper loss or language module is implemented.

Source: [MFVLR.pdf](../papers/MFVLR.pdf), §III-D pp. 6–7, Figs. 4–5 pp. 5–6, §III-E p. 8. Equations below preserve the paper's formulation; explanatory batched notation and implementation interpretations are explicitly separate. The original [research equation ledger](paper_analysis.md) remains authoritative historical documentation. The unusual scale is verified from the rendered PDF, not normalized to a familiar convention.

## Published FLT and VIM formulation

**EXPLICIT, p. 6:** lookup vocabulary `W_voc ∈ R^(s×d)` maps tokens to `T_low^e ∈ R^(n×d)`. Positions give `T_1^tra = T_low^e + P_e`. Equation (3) composes E encoder blocks (intermediate composition expansions omitted here, not changed):

\[
\operatorname{LE}(T_1^{\mathrm{tra}})
=\operatorname{TB}^{e}_{E}\circ\operatorname{TB}^{e}_{E-1}\circ\cdots\circ\operatorname{TB}^{e}_{1}(T_1^{\mathrm{tra}})
=T_{\mathrm{hig}}^{e}. \tag{3}
\]

The last token supplies `T_l ∈ R^(1×d)`. LE prose explicitly places LN before self-MHA and adds the original block state:

\[
T_j^{\mathrm{tok}}=\operatorname{MHA}^{e}_j(\operatorname{LN}^{e}_j(T_j^{\mathrm{tra}}))+T_j^{\mathrm{tra}}.
\]

**EXPLICIT, p. 7, Eqs. (4)–(9):** language queries and the single visual class token are projected and partitioned into r heads:

\[
q_j=T_j^{\mathrm{tok}}W_j^{\mathrm{que}}, \tag{4}
\]
\[
k_j=I_vW_j^{\mathrm{key}}, \tag{5}
\]
\[
v_j=I_vW_j^{\mathrm{val}}, \tag{6}
\]
\[
\{Q_{j,i}\in\mathbb R^{n\times d/r}\}_{i=1}^{r}=\operatorname{Pa}(q_j), \tag{7}
\]
\[
\{K_{j,i}\in\mathbb R^{1\times d/r}\}_{i=1}^{r}=\operatorname{Pa}(k_j), \tag{8}
\]
\[
\{V_{j,i}\in\mathbb R^{1\times d/r}\}_{i=1}^{r}=\operatorname{Pa}(v_j). \tag{9}
\]

Each projection matrix is `d×d`. The paper specifically chooses the class token instead of all/patch tokens. The attention equation is:

\[
T_{j,i}^{\mathrm{glo}}=
\delta\left(\frac{Q_{j,i}K_{j,i}^{T}}{\sqrt{r/d}}\right)V_{j,i}. \tag{10}
\]

**The denominator is `sqrt(r/d)`. It is not changed to `sqrt(d/r)`.** The paper identifies δ as softmax, but does not specify its axis. Head outputs are concatenated:

\[
T_j^{\mathrm{glo}}=\operatorname{Cat}(\{T_{j,i}^{\mathrm{glo}}\}_{i=1}^{r})\in\mathbb R^{n\times d}. \tag{11}
\]

The following unnumbered LE expressions occur immediately after Eq. (11), and Fig. 5 draws the first residual:

\[
T_j^{\mathrm{add}}=T_j^{\mathrm{glo}}W_j^{\mathrm{fc}}+T_j^{\mathrm{tok}},\qquad
T_{j+1}^{\mathrm{tra}}=\operatorname{FF}^{e}_j(T_j^{\mathrm{add}})+T_j^{\mathrm{add}}.
\]

**EXPLICIT, p. 7:** LD receives shifted low embeddings with Pd, LE memory and Iv. Equation (12) composes D blocks to give `T_rec^d`; its arguments remain:

\[
\operatorname{LD}(T_{t_1}^{\mathrm{tra}},T_{\mathrm{hig}}^e,I_v)=T_{\mathrm{rec}}^d. \tag{12, input/output equality}
\]

Each LD block follows the printed equations, without inserted normalizations:

\[
T_{t_j}^{\mathrm{mmha}}=\operatorname{MMHA}^{d}_j(T_{t_j}^{\mathrm{tra}})+T_{t_j}^{\mathrm{tra}}, \tag{13}
\]
\[
T_{t_j}^{\mathrm{mha}}=\operatorname{MHA}^{d}_j(T_{t_j}^{\mathrm{mmha}},T_{\mathrm{hig}}^e)+T_{t_j}^{\mathrm{mmha}}, \tag{14}
\]
\[
T_{t_j}^{\mathrm{ff}}=\operatorname{FF}^{d}_j(T_{t_j}^{\mathrm{mha}})+T_{t_j}^{\mathrm{mha}}, \tag{15}
\]
\[
T_{t_{j+1}}^{\mathrm{tra}}=\operatorname{VIM}^{d}_j(T_{t_j}^{\mathrm{ff}},I_v)+T_{t_j}^{\mathrm{ff}}. \tag{16}
\]

The proposed internal definition of VIM in Eq. (16) is the projected attention **delta**, because the equation already adds the input externally. This resolves an ambiguous module boundary; it is not a claim that Fig. 5 omitted a residual or that an author implementation is known.

## Batched attention shapes and scale

**DERIVED, conditional on proposed r=8, d=512, key-axis softmax:**

| Operation | Shape |
|---|---|
| Input language L / visual Iv | `[b,n,512]` / `[b,1,512]` |
| Projected q / k / v | `[b,n,512]` / `[b,1,512]` / `[b,1,512]` |
| Head Q / K / V | `[b,8,n,64]` / `[b,8,1,64]` / `[b,8,1,64]` |
| QK transpose, scaled logits Z, weights A | `[b,8,n,1]` |
| Head result A V | `[b,8,n,64]` |
| Concatenated result and output-projected delta | `[b,n,512]` |

`sqrt(r/d)=sqrt(8/512)=1/8`, so Eq. (10) multiplies raw dot products by 8. Conventional scaled dot-product attention divides them by 8. The published logits are therefore 64 times the conventional logits for this head count. Ordinary LE/LD MHA has no published scale formula; its proposed conventional scale is a separate assumption and must not be copied into VIM.

Softmax over the final/key axis is proposed because each query normally forms a distribution over keys. The paper does not explicitly require it. Softmax over language/query positions would produce token-dependent weights with the same tensor dimensions, but a different normalization meaning. It cannot be adopted silently to rescue the claimed behavior. Softmax across batch would create cross-sample dependence and is prohibited by the proposed contract.

## Singleton-key degeneracy and its precise scope

Let a finite logit at a fixed sample/head/query be z. With one key:

\[
A=\operatorname{softmax}_{\text{key}}([z])=[1],\qquad
\frac{\partial A}{\partial z}=1(1-1)=0.
\]

Thus, for each head, every query receives exactly the same value vector. With head concatenation in the original order and bias-free projections:

\[
\Delta(L,I_v)=\mathbf 1_n\,(I_v W^{\mathrm{val}}W^{\mathrm{fc}}),
\]

where the expression suppresses batch and the singleton visual-position dimension. This is a **DERIVED identity under L09**, not a replacement production algorithm. Its output is independent of language queries, Q/K matrices, attention head count and finite scale. There is no word-selective attention weighting inside this VIM. Language dependence remains through the external identity residual and subsequent self-attention/FF nonlinearities; different language tokens can therefore have different final features. This distinction avoids claiming that the entire language model ignores text.

For a scalar diagnostic J whose only dependence on these parameters is through VIM output, write `g_t = ∂J/∂Delta_t` and `g = sum_t g_t`. With `c=Iv Wval` and `Delta_t=c Wfc`:

\[
\frac{\partial J}{\partial W^{\mathrm{que}}}=0,\quad
\frac{\partial J}{\partial W^{\mathrm{key}}}=0,\quad
\frac{\partial\Delta}{\partial L}=0,
\]
\[
\frac{\partial J}{\partial I_v}=g(W^{\mathrm{fc}})^T(W^{\mathrm{val}})^T,\quad
\frac{\partial J}{\partial W^{\mathrm{fc}}}=c^Tg,\quad
\frac{\partial J}{\partial W^{\mathrm{val}}}=I_v^T\{g(W^{\mathrm{fc}})^T\}.
\]

These are per-sample formulas; parameter gradients sum across batch. Wval/Wfc/Iv gradients are generally nonzero, but a particular probe may cancel. Use a controlled noncanceling probe in tests. For residualized output `R=L+Delta`, `∂R/∂L` has the identity path; zero attention-query gradients do not imply zero language-embedding gradients. Tied vocabulary weights additionally receive gradients through input lookup and output projection.

With 19 independent VIMs, Q/K contain `19×2×512² = 9,961,472` parameters with zero task gradient under this formulation. They still count as registered trainable parameters. Explicit Q/K computation should yield allocated zero gradients rather than missing gradients. Algebraically deleting Q/K would alter parameter identity, graph evidence and optimizer behavior; it is not approved. Weight decay or an explicit parameter regularizer could move Q/K despite zero task gradient; no optimizer is implemented here.

Finite-input qualification matters: NaNs/infinities or overflow in projected/logit tensors can invalidate the singleton-softmax argument. Finite input validation and diagnostics must expose such failures, not replace attention with a constant or clamp values. Full scale correctness cannot be demonstrated using only singleton outputs because both scale conventions yield one. A separate **test-only two-key mathematical fixture** must distinguish the published and conventional scales; the production VIM still rejects multi-key visual input.

## Residual ownership and independent parameters

Three plausible interpretations explain the paper's module-boundary ambiguity:

| Interpretation | LE / LD effect | Assessment |
|---|---|---|
| Delta-only module; caller adds once | LE produces `L+Delta`; Eq. (16) produces `H+Delta` | **Recommended ASSUMED baseline**, consistent with both displayed residual operations |
| Module owns residual; callers do not add | Same mathematical output if applied consistently | Viable API alternative, but Eq. (16)'s caller addition must be understood as part of the module rather than executed again |
| Module owns residual and LD caller also adds | LD produces `2H+Delta` | Literal double ownership interpretation; changes identity gain and stability, not approved |

Production name `VisionInjectionDelta` must clearly indicate its boundary. Zero Wfc should make delta identically zero and the surrounding residualized operation exactly identity. Test LE and LD block ordering separately; do not use all-zero block weights alone to infer which nonzero suboperation ran first.

Per-block subscripts and Fig. 4 motivate independent parameters, but the paper does not expressly discuss sharing across depth. L09 proposes 12 LE VIM instances plus seven LD VIM instances, with separate Q/K/V/O matrices, no bias, dropout, normalization or gating. Shared Iv input does not mean shared VIM weights. The two roles in accepted MVE continue to share their existing encoder; that separate requirement is unchanged.

## Downstream equation ledger: interfaces, not loss implementation

The following **EXPLICIT** formulas appear on p. 8. Preserve them in the reference; executable loss interpretations remain deferred.

\[
\mathcal L_{\mathrm{kl}}=\frac1b\sum_{u=1}^{b}
\delta(T_l^u)^T\log\frac{\delta(T_l^u)}{\delta(T_{l\mathrm{pre}}^u)}. \tag{19}
\]

The prose sets δ temperature to 0.5 and derives `T_lpre` from a fully connected visual adapter. Normalizing the FLT output, detaching a teacher, temperature-squared scaling or adding an adapter inside FLT is not specified here. The language and visual feature dimension for this comparison is 512.

\[
S_{v2l}^{um}(I_v,T_l)=
\frac{\exp(\operatorname{sim}(I_v^u,T_l^m)/\tau)}
{\sum_{m=1}^{b}\exp(\operatorname{sim}(I_v^u,T_l^m)/\tau)}, \tag{20}
\]
\[
S_{l2v}^{um}(T_l,I_v)=
\frac{\exp(\operatorname{sim}(T_l^u,I_v^m)/\tau)}
{\sum_{m=1}^{b}\exp(\operatorname{sim}(T_l^u,I_v^m)/\tau)}. \tag{21}
\]

The dummy denominator m is retained as printed. The text calls the scores cosine similarity and also defines sim as a dot product. A future loss contract must decide whether vectors are normalized there. **Do not normalize Iv or Tl at the FLT interface to settle this silently.** The trainable temperature is initialized at 0.07; its parameterization is not part of this phase. Pair-label `y_pa` has one on the matched pair and zero elsewhere.

\[
\mathcal L_{v2l}=\frac1b\sum_{u=1}^{b}-(y_{\mathrm{pa}}^u)^T\log(S_{v2l}^u(I_v,T_l)), \tag{22}
\]
\[
\mathcal L_{l2v}=\frac1b\sum_{u=1}^{b}-(y_{\mathrm{pa}}^u)^T\log(S_{l2v}^u(T_l,I_v)). \tag{23}
\]

The paper defines `L_cmc = (L_v2l+L_l2v)/2` in prose. These batchwise losses intentionally relate samples; VIM and ordinary language attention do not. Attention sample-independence tests should not mistakenly include a later contrastive loss.

\[
T_{\mathrm{pre}}=T_{\mathrm{rec}}^d W_{\mathrm{voc}}^T\in\mathbb R^{n\times s}, \tag{24}
\]
\[
\mathcal L_{\mathrm{lr}}=\frac1b\sum_{u=1}^{b}\sum_{x=1}^{n}
-(T_{\mathrm{gt}}^{u,x})^T\log(T_{\mathrm{pre}}^{u,x}). \tag{25}
\]

Eq. (24) is a tied raw linear output. Eq. (25) applies log without a displayed softmax; raw logits need not be positive or normalized. This is a mathematical incompleteness in the published loss presentation, not grounds to insert softmax into Eq. (24). A future numerically stable categorical objective would be an explicitly approved interpretation. Whether to exclude padding/BOS or average over tokens is likewise unresolved; Eq. (25) displays a token sum and batch mean. Return unmodified logits and original integer IDs now. Do not create a production CE loss under the guise of a gradient test; test scalars are diagnostic linear/quadratic probes.

## Visual gradient paths and inference boundary

With accepted `Ir=abs(Ipre-I)` and `Iv=Ig+Irg`, a language scalar depending on Iv through VIM value/output maps has paths to:

- appearance global encoding through Ig;
- residual global encoding through Irg and the shared visual encoder;
- reconstruction through `Irg → Ir → Ipre`, then the AD head, shared VD trunk and appearance local encoder.

Away from zero residual, `∂Ir/∂Ipre=sign(Ipre-I)`; accepted framework behavior at zero is preserved. No gradient detachment is proposed at Iv, LE memory or reconstruction. Parameter sharing can hide a missing computational path, so merely observing a nonzero shared-encoder gradient is insufficient. [The integration tests](phase2c_test_plan.md) isolate residual conditioning, retain reconstruction gradients and compare detached controls. The independent mask head has no direct language-only path and should have `grad=None` after a cleared language-only probe. Its trunk is shared with AD and does receive that route's gradient.

At visual inference the language branch and all metadata are absent (§III-A p. 4, Fig. 4 p. 5). Replacing text or FLT weights must not alter the accepted visual forward outputs. This is not enforced by detaching visual tensors during training; it is enforced by omitting the language route at inference. No classifier or full-system inference implementation is included in the current design work.

## Unresolved evidence requests

Author clarification would help determine the intended softmax axis and scale, whether Eq. (16)'s VIM includes the drawn residual, exact FF structure, LD normalization, head count, depth sharing, token packing/shift and last-token pooling. Cited CLIP/MFCLIP tokenizer evidence cannot establish these VIM/LD choices. Until clarified, preserve the singleton key, printed scale and disclosed assumptions; patch-token injection, query-axis normalization, extra visual tokens and frozen/deleted Q/K are separate unapproved variants.
