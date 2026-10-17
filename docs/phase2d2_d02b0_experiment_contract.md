# A1-D02B0 full six-loss integration ablation contract

2026-10-15, Asia/Saigon. **DESIGN / STATIC PREFLIGHT ONLY. Phase 2D.2 remains BLOCKED.** No D02B experiment is authorized, implemented or executed by this document. The user authorizes inspection, hashes, resource arithmetic and these design documents only. See the [approval record](phase2d2_d02b0_approval_record.md), [fixture matrix](phase2d2_d02b0_fixture_matrix.md), [gates](phase2d2_d02b0_numerical_gates.md) and [resource plan](phase2d2_d02b0_resource_plan.md).

Classification: **EXPLICIT paper** is reserved for statements actually in the paper. The current architecture, objective interpretations and synthetic protocol are approved **ASSUMED engineering choices**, not recovered author code. New orchestration, retention and execution partitions here are **ASSUMED / PROPOSED**. Parameter/storage arithmetic and consequences of graph dependencies are **DERIVED**. Historical measurements are **RECORDED evidence**, not new executions. The paper references remain III-D, p.7, Eqs.(12)–(16) for LD and III-E, pp.7–8, Eqs.(17)–(27) for objectives, as audited in the existing contracts; no published equation is edited here.

## Frozen scientific scope

Use exactly the corrected [A1.1 preregistration](phase2d2_a1_ablation_preregistration_v1_1.md) and accepted research implementation [decoder_a11.py](../src/mfvlr_research/decoder_a11.py). Keep all seven LD blocks, twelve LE blocks, width 512, eight heads, FF width 2048/exact GELU, tied 49,408-word vocabulary, four independent 77-token segments concatenated to 308, global BOS shift, causal LD self-attention, full LE memory attention and physical LE position 307 as global language feature. No pretrained weights, I1, R2, fourth normalization, final normalization, dropout, checkpointing, AMP, clipping or rescaling.

For each block, S/C/F/Delta denote the unchanged causal self-attention, memory attention, FFN and singleton VIM delta. M and V are unchanged memory and fused visual inputs.

| Variant | Exact operation order | Approval basis |
| --- | --- | --- |
| A0 | a=x+S(x); b=a+C(a,M); c=b+F(b); y=c+Delta(c,V) | Accepted unnormalized reference |
| P3 | a=x+S(N1(x)); b=a+C(N2(a),M); c=b+F(N3(b)); y=c+Delta(c,V) | Accepted isolated research topology, not production |
| Q3 | a=N1(x+S(x)); b=N2(a+C(a,M)); c=N3(b+F(b)); y=c+Delta(c,V) | Accepted isolated research topology, not production |

Each N is independent affine LayerNorm over the last 512 features: gamma*(x-mean(x))/sqrt(population_variance(x)+1e-5)+beta. Gamma=1, beta=0, no RNG draws. Memory and visual inputs receive no new normalization. VIM stays outside these three norms, with query from language, singleton K/V from vision, denominator sqrt(8/512), key-axis softmax, four bias-free matrices and caller-owned residual once. Q3 block output is not required to have unit variance after VIM.

The accepted MVE/VD/FLT/objective code and all existing configs/tests remain byte-identical. `full_experiments_enabled: false` in research configs and the disabled legacy launcher remain unchanged. A future separately reviewed research orchestration layer is needed; changing a YAML flag is not authorization.

## Root ownership, initialization and pairing

Each worker owns exactly one working integration graph: MVE; VD shared trunk with separate appearance/mask heads; detector and adapter; FLT; SixObjectives with one CMC log-temperature. Reuse the current `with_language` / `compute_objectives` composition. No new mask or appearance head. Image route remains appearance encoding -> AD/MD -> absolute reconstruction residual -> shared residual encoding -> additive global fusion. FLT and adapter use fused `[B,1,512]`; CMC uses **unadapted Iv**, while KL uses the independent adapter output.

“Frozen” means preserving initial bytes and historical evidence, **not** setting `requires_grad=False`. All applicable parameter/input gradient paths remain live, including both KL distributions and the reconstruction-mediated residual path.

| Owned component | A0 elements | P3 / Q3 elements |
| --- | ---: | ---: |
| MVE | 32,078,912 | 32,078,912 |
| VD | 9,055,365 | 9,055,365 |
| Detector | 1,026 | 1,026 |
| Alignment adapter | 262,656 | 262,656 |
| FLT, including its single tied vocabulary owner | 112,758,272 | 112,779,776 |
| CMC log-temperature | 1 | 1 |
| Full root | **154,156,232 / 508 Parameter tensors** | **154,177,736 / 550 Parameter tensors** |
| LD subset (already inside FLT) | 36,746,752 | 36,768,256 |

P3/Q3 add 21 independent LayerNorm modules, 42 parameters, 21,504 elements. Counts are DERIVED and already recorded by A1.1; they do not establish paper fidelity. Visual+FLT without heads/objective was 153,892,549; the full count includes the additional 263,683. Do not force agreement with the paper's approximately 141M.

For G2, all four accepted component initializers use seed 0; theta is the existing FP32 log(0.07). Construct the exact accepted CPU FP32 reference, then use the frozen `build_variant(reference, variant)` semantics: copy each common name/shape/dtype/byte value, retain disjoint Parameter/storage ownership, and add only constant-initialized norms. A0 is also privately copied. Assert all 508 common parameters byte-identical and independent; all new norms have declared values. Vocabulary output must refer to the actual embedding Parameter, not a separately registered alias. Shared MVE/VD branches retain one owner. Release the temporary reference before any full forward; do not keep three complete variants resident.

For NG2-j, visual/VD/heads seeds remain 0; the complete FLT is initialized with seed 6101+j, then paired the same way. This is the A1.1 recipe, **not** reuse of seed-0 G2 or just swapping in a D02A LD checkpoint. D02A snapshots are local LD states with full-FLT initialization provenance; they are not complete full-model checkpoints. NG2 construction and its new run-manifest schema need separate approval. The fixed `seed: 0` research module config is preserved as a historical architecture contract; no hidden loader override is allowed. A future run manifest must explicitly distinguish its NG2 FLT seed from that fixed module-validation config.

G2 historical root hash is `1865527facba9adc8a14d443c112d492698d275ce4e8beaa784aee61a514895a`, using `tools/phase2d2_diagnostics.py:state_hash` (named-parameter names followed by contiguous bytes). Do not compare it to a different digest algorithm: `a11_evidence.state_hash` additionally serializes name/shape/dtype with per-tensor hashes. Store the algorithm identifier with every digest. Future manifests need full candidate state hashes, tensor maps, object/storage identities before and after every pass. Full new variant/NG2 state hashes are **NOT MATERIALIZED in D02B0**, not guessed from seeds.

## Unchanged six executable objectives

Authority: [loss contract](phase2d_losses_contract.md), [approved config](../configs/heads_losses_reference.yaml), [objective source](../src/mfvlr/objectives.py). The following summarizes approved interpretations, not replacement published equations.

| Objective | Inputs / targets | Exact retained computation |
| --- | --- | --- |
| FD | `[B,2]` raw logits; int64 `[B]`, 0=Real/1=Fake | Stable log-softmax CE, mean over B |
| LR | `[B,308,49408]` logits; original unshifted int64 `[B,308]` | CE at **all** positions; token sum, then batch mean; include BOS/EOS/pad=0, no ignore index |
| CMC | Iv and Tl `[B,1,512]` | Raw dot products divided by exp(theta); theta starts FP32 log(0.07); symmetric row/transpose-column CE, diagonal positives, batch mean; no adapter/cosine normalization, clamp, prompt deduplication or multi-positive substitution |
| FL | `[B,2,224,224]` raw logits; int64 masks `[B,224,224]` | Class CE, mean over batch and pixels; 0 background/1 foreground |
| AR | image and Ipre `[B,3,224,224]` | MSE mean over every RGB element and image; image in [0,1], reconstruction unrestricted finite; no clipping or detach |
| KL | Tl and adapted vision `[B,1,512]` | KL(text||adapted vision); feature-axis softmax at T=0.5; feature sum then batch mean; gradients through both distributions; no T-squared factor |

Total is the literal left-associated `(((((FD+LR)+CMC)+FL)+AR)+KL)` with coefficients one and no division by six. Stable log-softmax operations already approved remain; no new numerical stabilizer. Duplicate prompts keep diagonal CMC targets. B=1 CMC is connected with allocated zero gradients, not omitted. Saturation/symmetry may produce additional zeros; do not force nonzero gradients by changing fixtures.

**DERIVED invariant:** only LR depends on LD. In a matched triple, image outputs, LE/Tl, adapted vision and FD/CMC/FL/AR/KL computations must remain unchanged; their isolated common-parameter gradients should be identical under the same deterministic backend. Record exact equality as a proposed paired-control gate, justified by identical operations/inputs rather than a new tolerance. LR/total may differ. Added LN parameters are connected only to LR and total. A mismatch in a supposedly invariant branch requires investigation, not an explanation based on lower LD activations. LD seed replacement for NG2 changes the whole FLT **between j values**, so this invariant is only **within** each matched A0/P3/Q3 triple, never G2-versus-NG2.

## Proposed no-update measurement procedure per matrix cell

All steps below are **NOT RUN** and require execution approval. No backward, model/fixture construction or launcher invocation occurred during D02B0.

1. A single coordinator takes exclusive ownership of the sealed schedule; verify approved cell ID, source/config/protocol/state/input hashes, paired sample ordering and resource gate. Use a fresh process with explicit CPU FP32, pinned versions, backend and thread controls. Record actual settings and environment. Never infer provenance from exit code.
2. Construct/copy privately as above; verify counts, exact tensor names/shapes and ownership using variant-aware research parity. Do not weaken the production `ownership_report`, which expects the A0 count and cannot validate P3/Q3 unchanged. No source reference remains live during graphs. Keep the root in its original default training flag with gradients enabled (no stochastic layers); changing eval/train mode is not a new factor.
3. Run seven independent full forwards/backwards in order FD, LR, CMC, FL, AR, KL, total. Clear `.grad` to None between graphs; isolated passes compute only their one scalar, never zero-weighted sums of all losses. Each forward still uses full `with_language` dimensions. Verify/reset the same initialized state between passes without parameter replacement. Stream complete parameter gradients and diagnostics, then free the graph. No gradient accumulation into subsequent passes.
4. Sum isolated FP32 gradients into a disk-backed FP64 accumulator in that exact loss order. Preserve unused/None participation separately from allocated zero; compare total against the FP32-rounded sum at the original elementwise tolerance. Also record unrounded FP64-reference statistics with distinct names. Full element coverage is mandatory; selected layers or global norms cannot stand in for it. Streaming never changes the production total or its backward arithmetic.
5. Capture all visual outputs, LE blocks, each of seven LD blocks' self/memory/FF/VIM residual stages, global language, vocabulary logits, six scalars, LR token categories, CMC scores/entropy/temperature, KL probabilities/entropy and reconstruction range. Use stats-only hooks and bounded detached chunks; do not retain graph-owning captures or full layer arrays in a queue. Preserve real computational paths; a detached measurement copy is not the tensor used by a loss.
6. Record per-loss and joint gradients for all 508/550 parameters and Ipre/Iv/Tl/adapter intermediates; all connected VIM Q/K zeros; ordinary Q/K weights and key-bias cancellation residues; new LN derivatives; finite/None masks. Report full per-parameter sums, norms, directions, cosine and sign/zero-coordinate statistics. Loss-gradient interactions and LR-category/tied-vocabulary analysis require an explicitly reviewed research diagnostic implementation; do not silently substitute the old selected-three-parameter cosine report for full coverage.
7. Preserve the first cold pass timing. After the scientific seven-pass measurement, perform the already preregistered one no-grad warm-up and three state-reloaded measured no-update **joint-total** forward/backward repetitions, separately labeled timing repetitions. These do not replace seven-pass evidence or count as three independent fixtures. This exact placement/joint timing interpretation is a proposed orchestration detail requiring approval. No extra LR-category backward sweeps or detached-control sweeps are implicitly included in the matrix; any needed controlled probes must be separately enumerated before execution.
8. Verify state bytes/identities unchanged, hooks removed, graphs released and all record inventories complete. Seal raw data, stdout/stderr, actual exit code, checksums and JUnit identities. Complete all finite diagnostic cells only within the expressly authorized continuation rule; mandatory numerical failures remain FAIL. See [gates](phase2d2_d02b0_numerical_gates.md) for hard stops versus finite failed comparisons.

## Implementation gaps to approve before any execution

The existing G2 worker instantiates A0, the production ownership checker rejects normalized counts, its capture observes 19 block outputs rather than all LD substages, and its base stream retains selected rather than complete loss-pair cosines. `CompleteGradientStream` adds full parameter comparison but is not a complete variant orchestration layer. The existing research launcher refuses execution, and strict configs cannot authorize it. A future isolated runner/record schema needs review for these gaps, NG2 seeds and pre-forward hashes, finite-failure reporting, per-cell resource guards, and no-update sentinels. Nothing is enabled now.

The 57-cell plan is the bounded **full-six-loss portion**, not a claim to complete the entire A1.1 study. Its broader category-wise LR parameter-gradient decomposition, isolated tied-vocabulary path probes and N2/local control executions remain separately scoped, outstanding requirements. Scalar token-category statistics are not represented as that parameter-gradient decomposition. Their omission from the initial cell call budget must remain visible; no completion claim may silently waive them.

Cross-variant gradients are measurements of different functions, **not equality oracles** except the derived unchanged non-LR branches. No production winner follows from fewer FAILs, smaller amplitudes or better whole-model relative L2. Old N1/N2 scope, N3/N3.1 evidence, eight failing A0 assertions, and the VD-T12 incident stay intact. No optimizer-containing test may be executed under the future proposed pilot; MVE-T10's historical exception does not transfer.
