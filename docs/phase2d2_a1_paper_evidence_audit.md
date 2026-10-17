# Phase 2D.2-A1: paper evidence audit

2026-10-12 (Asia/Saigon). **Architectural research only. Phase 2D.2 remains BLOCKED.** No new architecture, ablation execution, numerical policy or optimization is approved by this document.

## Source and interpretation rules

The primary source is the supplied [MFVLR v1 PDF](../papers/MFVLR.pdf), arXiv:2605.10071v1, 18 pages, SHA-256 `2afed0516c32121513463d326720153887586ceaa328f26b96263bd39da45307`. Page numbers below are one-based PDF pages. This review re-extracted its text and visually inspected pp. 5-8 and 14-15, including Figures 4-6, Eqs. (3)-(16), implementation details and the relevant tables. Text on pp. 4 and 17-18 was also checked for scratch training, inference and references. The supplied version remains authoritative; no existing MFVLR implementation was consulted.

**EXPLICIT** identifies what MFVLR states or visibly prints. An explicit observation that an equation contains no LN symbol is not an explicit architectural prohibition of LN. **DERIVED** identifies a consequence of specified premises, including arithmetic on reported measurements. **ASSUMED** identifies an engineering interpretation, even when approved or common elsewhere. Observations from our implementation are evidence about that baseline, not new explicit MFVLR requirements.

The original [language contract](phase2c_language_contract.md) and [parameter audit](phase2c_parameter_audit.md) retain their historical PROPOSED/unimplemented wording. Later user approval, [implementation report](phase2c2_implementation_report.md), `configs/flt_reference.yaml`, and accepted N2 records establish the current baseline. This review does not rewrite those historical documents.

## Traceable findings

| ID | Paper evidence | Classification and boundary |
|---|---|---|
| E01 | III-D p. 6 explicitly feeds each LE block state through LN before MHA, then adds the incoming state | **EXPLICIT:** one described pre-attention LE LN. **ASSUMED:** epsilon, affine parameters, and exclusion of every other norm |
| E02 | III-D p. 7, Eqs. (12)-(16), prints no LN for LD; IV-A p. 8 supplies no normalization details | **EXPLICIT:** no displayed LD LN. **ASSUMED:** implementing zero LD norms; absence in notation does not establish absence in author code |
| E03 | Fig. 4 p. 5 shows LE MHA-VIM-FF and LD MMHA-MHA-FF-VIM, with residual arrows, but no LN box even for the explicitly normalized LE | **EXPLICIT:** diagram content. **DERIVED:** the diagram is insufficient to settle unprinted normalization or module internals |
| E04 | Eqs. (13)-(16), p. 7: masked self-attention, memory MHA, FF, VIM, each followed by addition of its incoming state | **EXPLICIT:** this external ordering and these additions. Moving/removing them is a departure, except resolving an ambiguous internal module boundary |
| E05 | Eq. (13) prose, p. 7: each word integrates itself and preceding words | **EXPLICIT:** inclusive causal self-attention intent. **ASSUMED:** concrete triangular mask implementation, projection biases, eight heads and ordinary attention scale |
| E06 | Eq. (14), p. 7, takes the post-MMHA state and high-level LE features | **EXPLICIT:** LE-memory communication. **ASSUMED:** state provides Q, full LE provides K/V, no memory/padding mask; no printed Q/K/V formulas resolve this ordinary attention |
| E07 | III-D pp. 6-7 uses FF notation and describes an FF network with a fully connected layer in LE; LD calls its operation an FF layer | **EXPLICIT:** FF operation exists. **ASSUMED:** two affine maps, 2048 hidden units, exact GELU, biases and no dropout. Neither a single affine nor the conventional two-layer FF is uniquely specified |
| E08 | Eqs. (4)-(11), p. 7 and Fig. 5 p. 6: language Q, visual class-token K/V, four d-by-d maps including output projection | **EXPLICIT:** these mappings and singleton visual token. The text specifically excludes visual patch/all-token K/V in the selected method. **ASSUMED:** bias-free implementation and key-axis softmax |
| E09 | Eq. (10), p. 7, divides by sqrt(r/d) | **EXPLICIT:** published denominator. Do not replace it with sqrt(d/r). **DERIVED conditional on key-axis softmax:** a single key gives probability one and zero task Q/K derivatives |
| E10 | Fig. 5 and following prose include the addition producing T_add; Eq. (16) separately adds T_ff outside VIM | **EXPLICIT:** both presentations. **ASSUMED:** delta-only VIM with one caller-owned residual resolves boundary ambiguity. Literal double addition is not established as the intended graph |
| E11 | Subscript j identifies blocks and VIM projections; Fig. 4 marks visual encoder/decoder sharing, not language depth sharing | **EXPLICIT:** notation/marked visual sharing. **ASSUMED:** independent LE/LD blocks and nineteen VIM parameter owners; Table IX offers additional conditional support below |
| E12 | III-D pp. 6-7: vocabulary lookup, encoder positions, last-token global language, low embeddings shifted using a begin token with last removed, separate decoder positions | **EXPLICIT:** stated construction. **ASSUMED:** global right shift with BOS prepended, positions learned independently, physical slot 307, four separately framed/padded 77-slot segments and all-slot masking policy |
| E13 | Eq. (12) passes the same LE output and visual feature through the LD stack; Eq. (24) p. 8 reuses W_voc transpose | **EXPLICIT:** stated inputs and output matrix reuse. **ASSUMED:** exact Python Parameter tying/ownership. **DERIVED under the baseline:** decoding is teacher-conditioned reconstruction rather than unconditioned causal generation |
| E14 | IV-A p. 8: B=4, E=12, D=7; d=512, n=308, s=49408, batch=8; PyTorch/V100 | **EXPLICIT:** dimensions, depths and reported platform. No attention-head count, FF width, norm epsilon or initialization distribution is supplied |
| E15 | II-B p. 4 states scratch training; IV-A p. 8 gives Adam, LR 1e-4, decay 1e-3 and a tenfold LR reduction every 15 epochs | **EXPLICIT:** these reported choices. **ASSUMED:** Xavier/Normal distributions and all initialization gains. Reported optimizer settings neither specify initialization nor authorize optimizer execution here |
| E16 | III-A p. 4 and Fig. 4: language only during training; testing uses MVE, VD and detection head | **EXPLICIT:** no inference language requirement. Training reconstruction's access to target text is separate from forbidden inference label/text leakage |

## Published formulation, retained without corrections

This is a mathematical transcription; composition expansions are condensed without changing operations. Batch dimensions are omitted as in the paper. The reference is separate from the executable baseline in the [decoder review](phase2d2_a1_decoder_architecture_review.md).

\[
LE(T^{tra}_1)=TB^e_E\circ\cdots\circ TB^e_1(T^{tra}_1)=T^e_{hig}.\tag{3}
\]

The unnumbered LE formula on p. 6 is
\(T^{tok}_j=MHA^e_j(LN^e_j(T^{tra}_j))+T^{tra}_j\).

\[
q_j=T^{tok}_jW^{que}_j,\tag{4}
\]
\[
k_j=I_vW^{key}_j,\tag{5}
\]
\[
v_j=I_vW^{val}_j.\tag{6}
\]
\[
\{Q_{j,i}\in\mathbb R^{n\times d/r}\}_{i=1}^{r}=Pa(q_j),\tag{7}
\]
\[
\{K_{j,i}\in\mathbb R^{1\times d/r}\}_{i=1}^{r}=Pa(k_j),\tag{8}
\]
\[
\{V_{j,i}\in\mathbb R^{1\times d/r}\}_{i=1}^{r}=Pa(v_j).\tag{9}
\]
\[
T^{glo}_{j,i}=\delta\left(\frac{Q_{j,i}K_{j,i}^{T}}{\sqrt{r/d}}\right)V_{j,i},\tag{10}
\]
\[
T^{glo}_j=Cat(\{T^{glo}_{j,i}\}_{i=1}^{r})\in\mathbb R^{n\times d}.\tag{11}
\]

The p. 7 unnumbered continuation is
\(T^{add}_j=T^{glo}_jW^{fc}_j+T^{tok}_j\), then
\(T^{tra}_{j+1}=FF^e_j(T^{add}_j)+T^{add}_j\).
This does not silently define whether the LD symbol VIM includes that addition.

\[
LD(T^{tra}_{t1},T^e_{hig},I_v)=TB^d_D\circ\cdots\circ TB^d_1(T^{tra}_{t1},T^e_{hig},I_v)=T^d_{rec}.\tag{12}
\]

The composition carries the same memory/visual inputs into each successive block, as the paper's expanded middle lines specify.

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

No LN, alternative residual coefficient, hidden-state scaling or corrected attention denominator has been inserted in this reference.

## What the ablations do and do not establish

**EXPLICIT:** Table XII p. 14 studies VIM placement. Its selected row uses LE VIM between MHA/FF and LD VIM after MHA/FF. This supports the accepted ordering, not a particular LN layout. Tables XIV/XV p. 15 compare fusion types and all/patch/class-token K/V. Their preferred class-token result does not remove the mathematical singleton issue under the accepted softmax axis. Table X p. 14 reports improved downstream metrics when LR is added to the other losses, but does not disclose token masking, normalization, gradient scales or stability traces.

**EXPLICIT:** Table XVI p. 15 reports the selected 12-LE/7-LD configuration. Its two 12-LE/6-LD rows have different results (FSLSD AUC 63.48 versus 65.80). Table XII's two `**` rows also differ. **DERIVED:** without seed/protocol clarification these are not clean repeatability estimates or architectural recovery evidence. The paper's overfitting interpretation of deeper stacks is not proof of the numerical cause in this untrained baseline.

**EXPLICIT:** Table IX p. 14 lists `Ours w/o VIM` 121.07M, LE-only VIM 133.66M, LD-only VIM 128.41M and both 141.00M. Table XV repeats 141.00M for all K/V alternatives.

| Conditional arithmetic | From reported rounded counts | Four bias-free 512-by-512 maps per independent VIM |
|---|---:|---:|
| LE VIM increment | 12.59M | 12 x 1,048,576 = 12,582,912 |
| LD VIM increment | 7.34M | 7 x 1,048,576 = 7,340,032 |
| Combined increment | 19.93M | 19,922,944 |

**DERIVED:** these differences are compatible with independent VIMs if the rows hold other ownership fixed and each displayed total is rounded to 0.01M (difference uncertainty up to 0.01M). They support, but do not prove, L09. Different unreported changes could also affect counts. Sharing VIM depth purely to match the total would weaken this particular agreement.

Our current full count is 154,156,232, **13,156,232 above nominal 141,000,000**. Removing nineteen VIM allocations arithmetically leaves 134,233,288, still about 13.16M above the reported no-VIM count. **DERIVED:** the discrepancy is not explained by VIM alone under this comparison. Three affine LNs per LD block add only 21,504. The earlier single-affine LD FF counterfactual would give 141,296,840 including current heads/temperature, not 141.00M. Neither proximity nor exact count agreement identifies author architecture. Counting scope, rounding, FF structure and visual topology remain unresolved.

## External evidence has limited authority

MFVLR [21], [MFCLIP III-D](https://arxiv.org/html/2409.09724v3#S3.SS4), describes four separately tokenized 77-position prompts. This is supporting prompt-packing precedent, not evidence of MFVLR LD internals. MFVLR cites [34] for CLIP tokenization; that does not import pretrained CLIP weights or a complete CLIP architecture.

For comparison only, [Vaswani et al., section 3.1](https://arxiv.org/html/1706.03762v7#S3.SS1) describes a decoder with residual-plus-LN sublayers and masked self-attention. [Xiong et al.](https://proceedings.mlr.press/v119/xiong20b.html) analyzes how LN placement affects initialization gradients in the architectures and assumptions it studies. [Glorot and Bengio](https://proceedings.mlr.press/v9/glorot10a.html) connects initialization and activation/gradient propagation. These are primary research precedents for hypotheses, not MFVLR requirements or guarantees for singleton VIM plus a bidirectional reconstruction memory.

**Conclusion (DERIVED, limited):** the accepted code can be correct relative to L06/L08/L09/L11 while decoder fidelity remains underdetermined. The missing operations must remain ASSUMED. Author clarification should request exact block pseudocode, FF internals, norms, VIM residual boundary, initialization, masking and a module-wise parameter inventory.
