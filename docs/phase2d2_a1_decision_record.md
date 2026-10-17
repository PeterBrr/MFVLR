# Phase 2D.2-A1: decision and approval record

2026-10-12 (Asia/Saigon). **Design review complete; Phase 2D.2 remains BLOCKED.** This record does not approve an architecture alternative, numerical amendment, optimizer or training. The user authorized architectural research/design only. No decoder variant or ablation was implemented or executed.

## Findings and recommendation

The accepted LD is internally consistent with the approved L06/L08/L09/L11 engineering baseline: seven independent unnormalized blocks, causal attention, full LE-memory attention, two-affine GELU FF, delta-only singleton VIM, one residual per operation, and the specified scratch initialization. That is implementation fidelity to an approved contract, not confirmation of an author implementation.

The paper explicitly describes LE LN, but not LD LN. Figure 4 also omits the known LE norm, so neither figure omission nor Transformer convention settles LD normalization. Table IX's parameter increments support independent VIM allocations conditionally; they do not explain the remaining full-model count discrepancy. No numerical result identifies a unique remedy or justifies accepting the eight failures.

**Recommended next optional handoff:** approve only the separate **A0/P3/Q3** experimental implementation and preregistered no-update comparison, if the user wishes to investigate normalization. Keep I1 separately optional and R2 deferred. P3/Q3 add three ordinary-core norms per LD block with identical parameter counts and copied baseline core weights; they leave VIM, LE-memory values, losses and all visual modules unchanged. Their exact equations, tests and fixture schedule are in the [preregistration](phase2d2_a1_ablation_preregistration.md). No candidate is recommended for production adoption now.

## A. Supported paper facts

These are evidence classifications, not approval of an undocumented implementation.

| ID | Finding | Classification / anchor | Decision status |
|---|---|---|---|
| A1-F01 | LE state is normalized before MHA in the described LE block | EXPLICIT, III-D p. 6 | SUPPORTED PAPER FACT |
| A1-F02 | LD displayed order is MMHA, memory MHA, FF, VIM with external residual additions | EXPLICIT, Eqs. (12)-(16), p. 7 | SUPPORTED PAPER FACT |
| A1-F03 | LD LN is not displayed; Fig. 4 also omits explicitly described LE LN | EXPLICIT observation of pp. 5-7; not an absence guarantee | SUPPORTED PAPER OBSERVATION |
| A1-F04 | VIM uses the visual class token, language query, d-by-d maps and sqrt(r/d) denominator | EXPLICIT, Fig. 5 p. 6, Eqs. (4)-(11) p. 7 | SUPPORTED PAPER FACT |
| A1-F05 | E=12, D=7, d=512, n=308, s=49408; scratch training; train-only language | EXPLICIT, pp. 4, 6-8 | SUPPORTED PAPER FACT |
| A1-F06 | Position-summed LR, batch average, tied output matrix and unit-weight total are displayed | EXPLICIT, Eqs. (24)-(27), p. 8; stable CE interpretation remains ASSUMED | SUPPORTED PAPER FACT |
| A1-F07 | 141.00M and branch VIM increments are reported | EXPLICIT, Tables IX/XV pp. 14-15; independent-VIM support is DERIVED conditional arithmetic | SUPPORTED WITH SCOPE LIMIT |

## B. Existing approved engineering assumptions, unchanged

| ID | Existing decision | Classification | Status in this review |
|---|---|---|---|
| A1-B01 | L06 no LD LN and four once-only residuals | ASSUMED | APPROVED BASELINE / FROZEN |
| A1-B02 | L08 ordinary biased 8-head MHA; FF 512-2048-512, exact GELU, no dropout | ASSUMED | APPROVED BASELINE / FROZEN |
| A1-B03 | L09 nineteen independent bias-free VIMs, key-axis singleton attention, delta-only API | ASSUMED implementation of partly explicit math | APPROVED BASELINE / FROZEN |
| A1-B04 | L11 private scratch initialization and exact draw ordering | ASSUMED | APPROVED BASELINE / FROZEN |
| A1-B05 | L03/L04/L07/L10 packing, bidirectional no-mask LE, global shift, physical last slot, parameter tying | ASSUMED interpretations and ownership | APPROVED BASELINE / FROZEN |
| A1-B06 | P2D six executable losses, all-token LR and literal unit-weight order | ASSUMED executable interpretations | APPROVED BASELINE / FROZEN |
| A1-B07 | Original N1/N2 scopes and N3 supplemental checks | Approved verification policies, not architecture facts | UNCHANGED; not extended to new gradient exemptions |

The original Phase 2C design documents intentionally retain their historical proposal labels. Later approval records and user acceptance govern the baseline; none of those files were retroactively rewritten here.

## C. Proposed experimental alternatives

| ID | Alternative | Classification | Approval status |
|---|---|---|---|
| A1-X01 | P3: pre-LN before each ordinary LD sublayer, VIM residual unchanged | ASSUMED research design; +21,504 parameters DERIVED | PROPOSED / NOT IMPLEMENTED |
| A1-X02 | Q3: post-LN after each ordinary residual, VIM residual unchanged | ASSUMED research design; same count as P3 | PROPOSED / NOT IMPLEMENTED |
| A1-X03 | I1: LD output-map initialization gain 1/sqrt(28), no forward scaling/norms | ASSUMED initialization hypothesis; unchanged counts DERIVED | PROPOSED OPTIONAL / NOT IMPLEMENTED |
| A1-X04 | R2: literal double-VIM-residual sensitivity control | ASSUMED boundary alternative; unchanged counts and factor-2 identity DERIVED | DEFERRED / NOT RECOMMENDED AS REPLACEMENT |
| A1-X05 | Different FF width/topology, depth sharing or visual architecture | Unresolved assumptions; arithmetic counterfactuals only | DEFERRED; not in runnable matrix |

No proposed row is an author-confirmed architecture. P3 versus Q3 is an equal-count placement comparison. A0 versus either changes normalization as one factor, not loss definitions or upstream memory normalization. Initialization and residual multiplicity must not be silently added to a norm variant.

## D. Questions needing author clarification or stronger evidence

| Priority / ID | Exact requested evidence | Why it matters |
|---|---|---|
| 1 / A1-Q01 | Full LD block pseudocode including any pre/post/final LN, affine flags and epsilon | Distinguishes H1/H2; a diagram alone is insufficient |
| 1 / A1-Q02 | Whether VIM returns Delta or residual-added T_add when called in Eq. (16) | Resolves residual ownership without guesswork |
| 1 / A1-Q03 | LE and LD FF affine count, widths, activation and biases | Large parameter/conditioning uncertainty |
| 1 / A1-Q04 | Initializer distributions/gains for embeddings, positions, ordinary projections, FFs and VIM, plus draw/seed policy | Scratch training does not specify initial scale |
| 2 / A1-Q05 | Ordinary attention Q/K/V sources, scale, head count, masks and dropout; VIM softmax axis | Resolves hidden module details and singleton interpretation |
| 2 / A1-Q06 | Exact shifted sequence and memory access, padding/BOS/EOS LR policy and any prompt corruption | Separates reconstruction conditioning from causal language generation |
| 2 / A1-Q07 | Module-wise parameter inventory and counting scope for 141M and each Table IX row | Tests architecture evidence without fitting dimensions to a total |
| 3 / A1-Q08 | Why repeated 12/6 and repeated VIM-placement configurations differ in Tables XVI/XII, including seeds/protocols | Avoids interpreting unmatched experiments as repeatability or normalization evidence |

These are drafted questions only. No author contact/message was sent and no external implementation was used.

## E. New user approval required before any implementation

| ID | Reviewable action | Current status / boundary |
|---|---|---|
| A1-D01 | Implement A0/P3/Q3 research-only variant wrapper, independent tests and separate closed experimental schema exactly as preregistered | **PROPOSED; awaiting explicit approval**. Accepted production/tests/configs remain frozen |
| A1-D02 | Execute the declared no-update CPU B1/B2 schedule, including source/fixture sealing, local oracles, N2, six G2 and new independent fixtures | **PROPOSED; awaiting explicit approval**. Not granted by design approval alone |
| A1-D03 | Add and execute I1 as its own paired initialization experiment | **PROPOSED OPTIONAL; separate selection required** |
| A1-D04 | Add and execute R2 boundary sensitivity control | **DEFERRED; separate selection required** |
| A1-D05 | CPU B8 or GPU capacity experiments | **DEFERRED; resource approval required**, all current alternative results NOT RUN |
| A1-D06 | Promote a variant to production or call it a more faithful reproduction | **NOT APPROVED**; needs separate evidence/contract and acceptance decision |
| A1-D07 | Replace any of the eight failures, introduce tolerances or extend N1/N2 | **NOT PROPOSED / NOT APPROVED** |
| A1-D08 | Modify six objectives, masks, targets, weights, temperature or similarity | **NOT APPROVED**; separate research contract needed |
| A1-D09 | G3/G4, optimizer, scheduler, datasets, training, benchmark evaluation | **NOT APPROVED** |

## Verification performed for this design review

The work consisted of source/config/test reads, paper text extraction and visual inspection, primary-literature review, arithmetic on existing counts, and reading preserved JSON measurements. No model was instantiated for a new forward/backward; no ablation, pytest run, optimizer or new training step was executed. In particular, the legacy MVE-T10 optimizer unit test was not rerun in this design-only phase.

The unchanged prior result remains **328 mandatory cases: 320 passed, 8 failed; 276/276 accepted regressions and seven supplemental cases passed**, with original tolerances intact. This is inherited N3.1 evidence, not a newly executed A1 suite. Full count remains 154,156,232 elements / 508 parameter tensors. Baseline numerical failures, FP32 drift and local FP64 limitations remain documented; all alternative CPU/GPU capacity and numerical results are **NOT RUN**.

A pre-review SHA-256 snapshot protects 1,241 pre-existing files across source, tests, configs, docs, tools, artifacts and the paper (excluding generated `__pycache__`). The final comparison and new-document hashes are recorded in [design_review_verification.json](../tmp/pdfs/phase2d2_a1/design_review_verification.json). The only intended repository additions are these five review documents; read/extraction/preservation scratch records are under `tmp/pdfs/phase2d2_a1`. Existing research, failure logs, manifests and ZIPs are preserved.

## Deliverables and stopping point

1. [Paper evidence audit](phase2d2_a1_paper_evidence_audit.md): original equations, figures, tables, classification and evidence limits.
2. [Decoder architecture review](phase2d2_a1_decoder_architecture_review.md): actual seven-block graph, ownership, initialization, gradient paths and objective interactions.
3. [Hypothesis matrix](phase2d2_a1_hypothesis_matrix.md): support, counterevidence, unknowns and distinguishing experiments for H1-H6.
4. [Ablation preregistration](phase2d2_a1_ablation_preregistration.md): exact alternatives, parameter costs, new tests, deterministic fixtures, metrics and stopping rules.
5. This decision record: facts, approved assumptions, proposed alternatives, author questions and required approvals.

No current evidence selects a production replacement or establishes training readiness. A numerically improved untrained variant would still not establish better detection/localization accuracy, convergence or closer fidelity to MFVLR. Stop at design review and await a scoped implementation/execution handoff.
