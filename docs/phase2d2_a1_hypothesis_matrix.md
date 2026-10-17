# Phase 2D.2-A1: architectural hypothesis matrix

2026-10-12. **All H1-H6 are ASSUMED research hypotheses, not recovered author details.** No hypothesis is marked proven. The current implementation remains the approved engineering reference; Phase 2D.2 remains BLOCKED. Evidence classifications follow the [paper audit](phase2d2_a1_paper_evidence_audit.md).

## H1: no-normalization LD faithfully interprets incomplete equations

- **Paper support (EXPLICIT):** Eqs. (13)-(16), p. 7, show raw sublayer results with external residual additions, and no LN. Unlike LD, the LE text explicitly names LN on p. 6. This makes L06 a defensible literal engineering reading.
- **Contrary evidence/limit:** Fig. 4 omits even LE's known LN, showing that diagram omissions occur. Named MMHA/MHA/FF modules have incompletely specified internals. There is no sentence forbidding LD normalization.
- **Missing evidence:** complete author block pseudocode, module definitions and norm inventory. The paper does not establish that omitted operations are absent.
- **Expected consequence (ASSUMED hypothesis):** raw state/memory amplitudes feed attention and FF at every depth; residual Jacobian products can amplify rounding/cancellation. That prediction does not prove incorrect implementation or imply instability for all weights/inputs.
- **Distinguishing work:** author clarification is the direct fidelity test. A0 versus P3/Q3 numerical experiments can assess consequences of added norms, but cannot establish what the authors implemented.
- **Baseline consequence:** retaining A0 changes nothing. Calling it author-confirmed would be unjustified. N3.1 establishes sensitivity of A0, not falsity of its interpretation.

## H2: conventional normalization is omitted from displayed LD equations

- **Paper support:** **EXPLICIT** omission of LE LN in Fig. 4 despite its presence in p. 6 text makes schematic abbreviation demonstrable. The use of Transformer/MMHA/MHA/FF terminology leaves module internals open.
- **Contrary evidence/limit:** LD equations give residual structure explicitly without LN, and the authors specify LN when discussing LE. Neither p. 7 nor IV-A names pre-LN or post-LN LD. A conventional architecture is not automatically imported by terminology.
- **Missing evidence:** whether norms are inside named functions, outside additions, before vocabulary projection, affine, or shared. Pre/post variants are not equivalent interpretations of a fully specified function.
- **Expected consequence (ASSUMED):** normalization could limit some state scales and alter gradient propagation. It can also change epsilon sensitivity, projected Q/K saturation and gradient cancellation; smaller activations do not guarantee a better-conditioned complete graph.
- **Distinguishing work:** obtain an explicit author ordering. In separate variants compare P3 and Q3 with identical core weights, same number of norms and unchanged memory/VIM. Treat external Transformer literature as motivation only.
- **Baseline consequence:** either variant changes L06 and parameter ownership. Both require new implementation/execution approval; neither replaces the frozen reference by default.

## H3: residual/normalization placement changes conditioning

- **Paper support:** Eqs. (13)-(16) **explicitly** place outer residuals; Fig. 5 and its p. 7 continuation create ambiguity only about VIM's internal residual boundary. Table XII p. 14 reports sensitivity to VIM placement, but not residual multiplicity or normalization.
- **Contrary evidence/limit:** the displayed order is comparatively strong evidence against arbitrarily moving VIM before FF as a baseline repair. No paper result validates two residual additions around one VIM.
- **Missing evidence:** exact VIM callable return, and whether FF includes any normalization/residual itself.
- **Expected consequence (DERIVED conditional):** with singleton VIM, one residual gives state Jacobian I for that operation; literal internal-plus-external addition gives 2I. Pre/post LN modifies different Jacobian factors. Residual handling can matter even with identical parameter counts.
- **Distinguishing work:** symbolic boundary audit and zero-sublayer identities first; proposed R2 is an optional, explicitly nonrecommended double-residual sensitivity/negative control. P3 versus Q3 isolates LN placement with equal count. Moving VIM or ordinary residuals is not included in the minimal matrix.
- **Baseline consequence:** returning delta then adding once and returning the residual-added result without another addition represent the same intended math, not separate architectures. R2 actually changes the math and needs separate approval. No production boundary is changed here.

## H4: scratch initialization interacts with depth and residual accumulation

- **Paper support:** scratch training is **EXPLICIT** on p. 4; 12 LE and 7 LD blocks are explicit in IV-A. The source provides no matrix initializer/gain or residual-specific initialization.
- **Contrary evidence/limit:** the paper reports downstream results for its trained model, including D=7/8 comparisons in Table XVI. It does not report startup activations/gradients; those results cannot identify Xavier gain or disprove a particular numerical concern. The table's deeper-model interpretation is not a controlled initialization experiment.
- **Missing evidence:** matrix distributions, seeds, separate output-projection gains, embedding/position scales, numeric precision and any warm-up details beyond IV-A's epoch scheduler.
- **Expected consequence (ASSUMED):** branch-output magnitude may accumulate through repeated residuals; reducing only initial branch-output weights might reduce initial sensitivity. Independence-based variance heuristics are conditional and do not predict all attention/visual correlations.
- **Distinguishing work:** optional I1 versus A0 using copied identical FP32 weights except the specified LD output-map initialization gain, no norms or forward scale; repeat across predeclared seeds. It can test an initialization mechanism, not infer author initialization or establish repeated-step stability.
- **Baseline consequence:** I1 changes L11's initialized values while retaining shapes/counts. It requires separate approval and must not be combined with P3/Q3 in the first comparison.

## H5: the parameter discrepancy signals unresolved architecture

- **Paper support:** Tables IX/XV **explicitly** report 141.00M; the actual baseline count is 154,156,232. FF internals and other details remain unspecified. Table IX's branch increments conditionally support independent VIMs (see audit arithmetic).
- **Contrary evidence/limit:** reported totals are rounded and their counting scope is incomplete. A discrepancy cannot locate its cause in LD alone. Earlier visual encoder/decoder choices are also engineering assumptions. Matching a total would not distinguish normalization placement or equivalent counts.
- **Missing evidence:** author per-module inventory including train-only modules, shared parameters, heads/adapter/temperature, FF widths/biases and all transformer allocations.
- **Expected consequence:** the count itself has no causal effect on rounding; the unknown structures behind it could affect both count and conditioning. A 21,504-parameter LN addition cannot explain a 13.16M excess.
- **Distinguishing work:** arithmetic sensitivity and author inventory, not dimension fitting. One-affine LD FF is a count counterfactual of 141,296,840 with existing heads, not an implemented/runnable candidate in this matrix. Its remaining mismatch reinforces underdetermination.
- **Baseline consequence:** none from arithmetic alone. No shrinking FFs, deleting zero-gradient VIM Q/K, tying depth or changing MVE/VD is authorized.

## H6: LR reduction and teacher-conditioned memory create risks independently of LN

- **Paper support:** Eq. (25), p. 8, explicitly sums positions and averages batch. Eq. (12)/(14) accesses high-level features of the input prompt while reconstructing that prompt. Eq. (27) gives unit weights. Table X supports LR's reported downstream utility but does not resolve masking or scale.
- **Contrary evidence/limit:** the paper does not explicitly specify BOS/EOS/pad supervision or full bidirectional memory masking. Those are approved assumptions, not facts reconstructed from Table X. Large scalar LR is not proof of large gradients everywhere; reconstructed-target conditioning is compatible with the stated reconstruction task.
- **Missing evidence:** tokenizer padding loss policy, LE/memory masks, whether targets are corrupted, category-wise NLL/gradient distributions and repeated-step trajectories. No target corruption is described or introduced here.
- **Expected consequences (DERIVED/conditional):** all-token sum is 308 times the all-token mean on identical logits; many repeated padding targets can align gradients. Bidirectional target memory makes causal self-attention insufficient to enforce target-autoregressive independence. LR does not cause CMC/KL saturation through LD because those losses do not depend on LD; they can still interact through shared LE/visual parameters.
- **Distinguishing work:** preserve losses and compare fixed-cotangent VJPs with actual LR-induced VJPs at the same recorded states; partition existing LR gradients by padding/BOS/EOS/content and inspect memory/input derivatives. This is diagnostic decomposition, not new loss weights. A new masking, corruption or reduction experiment would need its own design/approval and is excluded here.
- **Baseline consequence:** none in the proposed architecture comparisons. All six objectives, prompt handling, memory masks and temperatures remain identical. Better normalized-LD behavior would not resolve the independent memory-conditioning or loss-scale questions.

## What the evidence currently decides

**DERIVED from preserved N3.1 evidence:** numerical sensitivity is real in the accepted baseline; all-local or aggregate-norm acceptance is not justified. It is not yet possible to choose between H1 and H2 as historical author facts, or to rank an untested variant as a remedy. H3/H4 offer separable mechanisms to investigate. H5 motivates architectural clarification rather than parameter fitting. H6 prevents attributing every future gradient issue to normalization alone.

The minimal recommended future matrix is **A0/P3/Q3**. I1 is a separately optional initialization study; R2 remains deferred unless the VIM boundary question needs a controlled counterexample. None has been implemented or executed. All approvals are enumerated in the [decision record](phase2d2_a1_decision_record.md).
