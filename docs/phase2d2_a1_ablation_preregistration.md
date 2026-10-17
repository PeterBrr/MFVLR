# Phase 2D.2-A1: proposed ablation preregistration

Version A1.0, 2026-10-12. **PROPOSED, NOT APPROVED FOR IMPLEMENTATION OR EXECUTION.** This document fixes questions, candidates, fixtures and metrics before any alternative is run. It is not an acceptance-policy amendment. Its content hash is recorded in the design-review verification record under `tmp/pdfs/phase2d2_a1/`. Later changes require a new version and a reason recorded before the affected experiments; no silent post-result fixture selection.

All new normative choices below are **ASSUMED research design** unless explicitly labeled DERIVED arithmetic. None is an explicit MFVLR requirement. Paper facts and existing approved assumptions are separately recorded in the companion audit/decision documents.

## Frozen reference and one-factor comparisons

Reference A0 is the exact accepted architecture in [decoder review](phase2d2_a1_decoder_architecture_review.md). All candidates retain seven independent LD blocks, 12 unchanged LE blocks, 512 dimensions, 8 heads, FF width 2048/exact GELU, causal self-attention, full unmasked LE memory, singleton VIM with the published denominator and four full projection graphs, global BOS shift, tied vocabulary, all-token LR and all six objectives. MVE/VD, tokenizer, heads, adapter and temperatures stay untouched. Ordinary attention remains explicit unfused FP32 unless performing the defined local FP64 control. No dropout/AMP/clipping/checkpointing/loss scaling/optimizer is introduced.

Let S, C, F and Delta mean the exact per-block functions in the decoder review; M/V are fixed memory/visual inputs shared across depth. N_i is a separate affine LayerNorm over the last 512 features:

\[
N_i(x)=\gamma_i\odot\frac{x-\mu(x)}{\sqrt{\operatorname{mean}((x-\mu(x))^2)+10^{-5}}+\beta_i.
\]

Population variance, gamma=1 and beta=0 are **ASSUMED proposed** choices, matching the approved LE norm convention. No norms share parameters. No final LD norm, memory norm or visual norm is added.

| Candidate | Exact equations for every block | Unique LD / FLT / full-model count | Status |
|---|---|---|---|
| **A0** frozen reference | a=x+S(x); b=a+C(a,M); c=b+F(b); y=c+Delta(c,V) | 36,746,752 / 112,758,272 / 154,156,232 | Existing approved baseline; a fresh run still requires future execution approval |
| **P3** pre-normalized ordinary core | a=x+S(N1(x)); b=a+C(N2(a),M); c=b+F(N3(b)); y=c+Delta(c,V) | 36,768,256 / 112,779,776 / 154,177,736 | PROPOSED core comparison |
| **Q3** post-normalized ordinary core | a=N1(x+S(x)); b=N2(a+C(a,M)); c=N3(b+F(b)); y=c+Delta(c,V) | same as P3 | PROPOSED core comparison |
| **I1** initialization only | same equations as A0; initialization change below | same as A0 | PROPOSED optional separate comparison |
| **R2** literal double-VIM-residual sensitivity control | same a,b,c as A0; y=c+[c+Delta(c,V)] | same as A0 | DEFERRED; not recommended as a replacement |

**DERIVED counts:** P3/Q3 add three 1,024-element LNs per block: +3,072 per block, +21,504 total, +42 Parameter tensors (550 full-model versus 508; 429 FLT versus 387). One normalized block is 5,252,608 parameters. No parameter is removed. Count changes do not approach resolution of the nominal 13.16M gap.

P3/Q3 are precisely **three-sublayer normalization experiments**, not an assertion that the entire visual-augmented block is conventionally normalized. Memory K/V remain unnormalized, and the final VIM residual remains outside LN. Thus Q3's final block output need not have unit variance. A fourth *pre*-VIM norm would affect only a query whose singleton attention delta is mathematically query-independent, adding dead task-gradient affine parameters under that interpretation; it is excluded. A *post*-VIM norm would change the output and direct visual gradient and is a different, unproposed factor. Adding a final stack LN to P3 would also confound the first comparison. The conventional pre-LN/post-LN literature motivates this controlled question but does not prove either variant's behavior here.

### Initialization and pairing

For A0/P3/Q3/R2, initialize an exact A0 FP32 state with the approved initializer, then copy **every common tensor by name and shape** into separately owned variant tensors. Record exact byte equality and no storage alias. Initialize only new LN gamma/beta with constants; do not let new layers consume or reorder the original RNG stream. LE, W, Pe/Pd, ordinary matrices, FFs and all VIM tensors must remain pairwise identical at experiment start. No weights are trained or overwritten in the frozen reference.

For **I1 only**, copy that same A0 state, then multiply each LD block's four output matrices (`self_attention.out.weight`, `memory_attention.out.weight`, `ff.fc2.weight`, `vim.out.weight`) once at initialization by `alpha=1/sqrt(4*7)=1/sqrt(28)`, using FP32 arithmetic and recording before/after hashes. Biases remain the original zeros; all other tensors remain exact. There is no runtime residual scaling or gate. This is an explicit **ASSUMED initialization hypothesis**, not L11 and not a proven stability rule. The factor comes from the 28 residual branches and an independent-branch variance heuristic, not the N3.1 failure maxima. Those independence assumptions may fail. I1 must not be combined with LN in the initial study.

R2 preserves all weights and counts and changes only residual multiplicity motivated by the Fig. 5/Eq. (16) boundary ambiguity. In a zero-sublayer, zero-visual-delta fixture it gives 2x per block, hence 128x over seven blocks, versus x for A0. It is useful as a boundary sensitivity/negative control, not as a presumed numerical improvement. A residual-added VIM with its caller residual removed is algebraically the once-addition reference and does not warrant a separate architectural candidate.

## Isolation, APIs and tests

Any authorized future implementation must live in a separate experimental branch or clearly named research variant module/config namespace. Do not modify accepted `src/mfvlr`, loaders, `configs/flt_*`, tests or historical evidence. In particular, do not loosen the approved strict loader to accept variant settings. Variant IDs, norm inventory, initialization transformation and counts must be explicit in a separately reviewed experimental schema. API compatibility means accepting the same `[B,308,512]` state/memory and `[B,1,512]` visual tensors, returning the same shapes, retaining the FLT output fields and supporting tied serialization within the variant. It does not mean an A0 checkpoint can strictly load extra LN keys or that predictions must agree between architectures.

| Candidate | Accepted test expectations that differ **if ported to variant** | New required variant checks |
|---|---|---|
| A0 | None; original 328 cases retain their genuine outcomes | Reference source/config/state hash checks; replay provenance |
| P3/Q3 | FLT-T10 ordering; T14 and FLT-I07 exact counts; T15 initializer inventory; T18/I08 state schema; T19 strict variant schema; old READY-N03 exact historical-output comparisons apply only to A0 | Independent LN forward/VJP oracle; noncommuting ordering oracle; three norms per block; shared-weight parity before forward; new gamma/beta gradients; count/serialization tests |
| I1 | T15 exact initialized values; variant config fingerprint/T19; historical exact forward diagnostics; T18/I08 own initialized-state round trip | Exactly 28 matrices transformed, no extra changed tensor; transformation replay; no forward scaling operation; counts identical |
| R2 | T10 once-only residual expectation; historical exact forward diagnostics; variant config fingerprint/T19 | Zero-sublayer factor-2/factor-128 oracle and visible double-residual metadata; unchanged singleton Q/K zeros; same counts/initialization |

No accepted tests are edited or repurposed. Implement separate variant analogues if later approved. Shape/finiteness, causal self-attention, full-memory connectivity, vocabulary tying, block independence, sample isolation, structural VIM zeros, public visual interfaces and original loss tests remain required for every candidate. T17's FP32 cross-batch behavior remains a diagnostic in its original scope; it is not a gradient waiver. N1 remains MVE-only. A variant must not claim READY-N03 parity with historical A0 numerical values.

Exact sample-isolation checks concern per-sample model outputs, not the intentionally batch-coupled CMC loss. Cross-sample derivatives must use genuine differentiable floating lookup/visual leaves, with a controlled nonzero same-sample derivative; integer IDs or disconnected surrogate tensors cannot establish isolation.

Proposed new tests (all **PROPOSED**, not added now):

| ID | Check and oracle |
|---|---|
| A1-T01 | Exact copied common state, disjoint storage, expected per-block/total tensor inventory |
| A1-T02 | LayerNorm against independent FP64 mean/variance formula and analytic VJP, including constant and near-constant inputs; gamma/beta derivatives |
| A1-T03 | Each block's exact ordering with noncommuting sentinels and independent functional arithmetic; memory remains unchanged |
| A1-T04 | Causal self-attention, full-memory attention, global shift, no batch-axis reduction, exact fixed-batch independence/permutation and zero cross-sample gradients |
| A1-T05 | Full Q/K graph retained; all 38 connected singleton VIM Q/K gradients allocated exact zero, V/O/visual controlled derivatives nonzero |
| A1-T06 | Ordinary Q/K nonzero FP64 manual derivatives; central differences; key-bias rational cancellation and separately recorded floating residuals |
| A1-T07 | Count, independent depth parameters, vocabulary tie and strict variant serialization/config rejection |
| A1-T08 | I1-only transformation/replay or R2-only residual oracle; never mix those factors |
| A1-T09 | Deliberate gradient error detection: exact known vectors including zeros and weak nonzero entries, plus chunk-boundary corruption; presence/nonfinite failures remain hard failures |
| A1-T10 | Complete stage/parameter capture and stale/source/config/input/state/record-integrity rejection; no JSON-loading-as-proof |

## Fixed fixture schedule for a future authorized study

All comparisons are paired, not chosen after viewing results. Generate random tensors first in FP32 on CPU, record their hashes, and promote those exact values for FP64. Initial model states likewise come from the same initialized FP32 tensors. FP64 may be used for the local attention/LD or full language branch; **do not cast the accepted public MVE/VD into unsupported FP64 interfaces or claim full-model FP64 correctness**.

### Existing anchors, unchanged

1. N2 F0/F1/F2: exact fixture definitions in `tools/review_phase2c2_n2.py` and preserved `artifacts/phase2d2_n31/n2/fixture{0,1,2}_{float32,float64}_threads4.json`. Keep original weights/IDs/visual values. Compare each architecture's own B=1/B=2 executions, not a variant to A0 as an equality oracle.
2. All six G2 cases from `tools/phase2d2_fixtures.py`: real, efs, local, real_efs, real_local, local_duplicate. Baseline seed 0; same images, supplied masks, prompt IDs and objective order. All six isolated losses and literal joint backward, full streaming parameter coverage, no updates. Include mixed real_efs with MKLDNN disabled as in N3.1.
3. N3 same-forward controls and FP32/promoted-weight FP64 CMC-chain probe, preserving formulas, fixture hashes and tolerances. Additivity failures remain failures; no automatic exemptions.
4. N3.1 registered local attention, LD and visual probes exactly as in [locked protocol](../artifacts/phase2d2_n31/heldout_preregistration.json), SHA-256 `7328fa21235112ac1dd081062efefefc1ebaa8c96ff9444ebdbedf7e703c7e72`. All predeclared magnitudes, cotangent scales and cuts are retained. Baseline-derived LD leaves are copied to all variants so this comparison does not confound local input changes with architecture changes.

### New independent fixtures, defined now but not generated

**NLD-j, j=0,1,2:** full FLT initialization seeds `6101+j`; use only its LD for local probes, with A0 common tensors copied into each variant. Generate independent CPU FP32 standard-normal B=2 master state, memory and visual arrays with seeds `6201+j`, `6301+j`, `6401+j`, shaped `[2,308,512]`, `[2,308,512]`, `[2,1,512]`; B=1 uses the first sample, cloned contiguously. Reinitialize each private generator for each stated master fixture. Use these exact multiplier triples `(state,memory,visual)`:

`(.02,2,1), (.2,2,1), (2,2,1), (.2,.2,1), (.2,20,1), (.2,2,.1), (.2,2,10), (2,20,10)`.

The first seven vary one factor around `(.2,2,1)`; the eighth is a predeclared joint stress case, reported separately. These values are research choices, not changes to actual model inputs or an acceptance bound.

At LD cuts 1,4,7, u/v are independent FP32 normal cotangents from seeds `6501+j` and `6601+j`, divided by sqrt(B*308*512), then multiplied by s in `{0,1e-4,1,1024}`. Reuse each u/v base array across cuts/variants. Compare VJP(u+v) with the sum of VJP(u), VJP(v) on one fixed forward graph. Record FP32 rounding of u+v separately; in FP64 form the sum after promotion. Also compare direct FP32 VJPs to each variant's own promoted-weight FP64 VJPs. Capture all seven block activations; no result-based early-cut selection. Target all block ordinary Q/K weights/biases, all VIM Q/K/V/O, each new norm parameter, and state/memory/visual leaves. Use four Rademacher directions with seeds 6701-6704 salted by the first four SHA-256 bytes of the parameter name (little endian), normalized by sqrt(numel), independent of observed gradients.

**NATT-j:** reuse initialization seed `6101+j` for an independently instantiated ordinary attention fixture with shape `[B,308,512]`, B=1/2 and input seeds `6801+j`. Input multipliers `.02,.2,2`; cotangent seed `6901+j`, scales `0,1e-4,1,1024`. Q/K directional finite differences at scale 1 use eps=1e-6 and directions from 6701/6702. Compare with the independent local attention formulas and exact rational cancellation. For P3/Q3 composition, include independently evaluated LN forward and Jacobian; do not reuse the production block as its oracle. The constant-input LN fixture is all .25, with a second fixture `.25 + 1e-4 * seeded_normal` (seed 6951+j); no measured near-zero threshold selects it.

**NG2-j:** three new B=2 full-graph pairs, keeping visual/head/adapter initialization seed 0 and FLT seed `6101+j`: j=0 real/EFS; j=1 real/local; j=2 local/local with duplicate prompts. Image seeds for the two rows are `7101+j` and `7201+j`, values `.1+.8*rand([3,224,224])`. Local images invert the fixed rectangle `[64:160,80:176]`; supplied masks are zero, all-one or that rectangle according to kind. Use existing pinned golden prompt indices 0 for real, 1 for EFS and 5 for local, as the accepted fixture utility does. No dataset preprocessing or mask inference is involved. Run all six isolated and joint objectives unchanged and report every parameter, including added LN parameters. These full-graph fixtures are FP32 only.

Do not fold NG2 into old G2 records or overwrite fixture names. New token/image/state hashes must be recorded before any corresponding forward pass. No new fixture was executed during A1.

## Metrics and criteria fixed before results

| Family | Required measurement / rule |
|---|---|
| Activation scale | RMS, L2, max absolute, min/max, finite count at embeddings, shifted input, every LE/LD block and all four LD residual substages, memory/visual, logits; no omitted stage |
| Forward drift | Same-variant B=1/B=2 and FP32/promoted-FP64 differences, max absolute/nonzero-reference relative error, relative L2, old outside-tolerance counts/proportions; backend metadata |
| Derivative agreement | Every local ordinary Q/K parameter versus independent FP64 formulas, norm parameters versus independent LN derivatives, and central differences; exact-zero and nonzero classes separate |
| Fixed-cotangent sensitivity | VJP additivity, absolute error norms, reference norms, relative L2 and cosine where defined, four directional discrepancies; all magnitudes/cuts/scales, no denominator floor |
| Analytical cancellations | Ordinary key-bias versus exact zero and manual FP64 derivative separately; retain small nonzero residuals and failed coordinates |
| Structural zeros | Every connected singleton VIM Q/K task gradient allocated and exactly zero; missing gradient is not a zero substitute |
| Full gradients | All 508 A0/I1/R2 or 550 P3/Q3 tensors, unique counts, per-loss/joint norms, mismatch counts, zero-reference coordinates, sign disagreement, cosine/directions and module summaries |
| Loss interaction | Category-wise LR NLL and gradient decomposition, tied-vocabulary gradient paths, CMC/KL entropy/score scale, per-loss gradient directions on shared parameters; no weight or target changes |
| Resources | Wall time, sampled CPU RSS and available RAM, temporary disk peak, GPU allocated/reserved peak only if actually run; status PASS/FAIL/OOM/NOT RUN per batch/device |

For cancellation, report `sum_i ||g_i||2 / ||sum_i g_i||2` for each parameter, with undefined/infinite cases explicitly tagged if the denominator is zero; retain numerator/denominator and original vectors' sufficient statistics. Do not use this ratio or a whole-model norm to excuse weak Q/K weights. Original elementwise comparison casts the streamed FP64 sum of isolated FP32 gradients back to FP32. Additional norm/direction diagnostics compare to the stored FP64 sum; label the references distinctly. Neither is an independent full-model derivative oracle.

**Unchanged comparison thresholds:** FP32 atol=1e-6, rtol=1e-5; FP64 atol=1e-10, rtol=1e-8; central differences eps=1e-6 with existing atol=1e-5, rtol=1e-3. Exact independence, permutation, structural zeros and cross-sample gradient requirements remain exact. No thresholds are borrowed from MVE N1 or expanded from N2. Conditional gamma-n envelopes remain local diagnostics, not a replacement full-model gate. Local FP64 failures stay visible.

**Hard-stop conditions:** any nonfinite, unexpected disconnected required gradient, incorrect structural zero, ownership/shape/count violation, failure of a mandatory controlled primitive unit oracle, state mutation or incomplete provenance halts dependent runs for that candidate and produces FAIL evidence. The mandatory primitive Q/K oracle uses the existing N3-C02/C03 controlled fixture unchanged; the new LN oracle uses its declared constant/near-constant fixtures. Do not tune epsilon/precision/architecture to continue. The held-out magnitude/scale comparisons are separately labeled exploratory numerical tests: a finite derivative discrepancy outside original tolerances is recorded as a failed comparison, including manual-FP64 or key-bias comparisons; it does not silently remove remaining predeclared finite diagnostic coverage. This distinction does not waive any original mandatory failure or make a variant accepted. A backend OOM is recorded and dependent work stops until an explicit resource decision; do not shrink dimensions or batch labels.

**No automatic acceptance threshold is proposed.** Report paired per-fixture changes, medians/maxima and the number improved/worsened/undefined for each metric; preserve every worsening cell. Do not select a winner from only aggregate RMS or cosine. The strongest permissible conclusion is that a specified variant improved certain measured numerical properties on a specified schedule. Any promotion to a production baseline, waiver of the eight original assertions, or claim of model accuracy/fidelity requires separate evidence and approval.

## Fairness and resource schedule

Recommended first authorization, if desired: A0/P3/Q3 only. Run source/ownership and independent primitive checks, then local held-out comparisons, then full language N2, then all six G2 and NG2. Existing A0 regression assertions remain unedited; no new run of that suite is performed in this design phase. I1 needs a separate optional selection; R2 stays deferred. Do not automatically combine promising variants or tune initializers based on preliminary outcomes.

CPU reference: the pinned [N3.1 environment](../artifacts/phase2d2_n31/environment.json), four threads, deterministic algorithms, highest FP32 matmul precision, no autocast/fused attention. Compare MKLDNN on/off on the same fixtures; use a one-thread replay of NLD nominal and existing mixed real_efs as a separately labeled reduction-order control. Pair each architecture within the same backend/thread setting. Before timing, specify one untimed no-grad warm-up and three measured no-update forward/backward repetitions with identical reloaded state/inputs; also preserve cold-start time. Clear graphs sequentially, and run no concurrent full models. Capture backend/version/device, seed, source/config/state/input hashes and experiment identity before execution; fail on missing/stale/incomplete evidence. Warm-up does not mean optimization.

**DERIVED storage estimates, not capacity measurements:** full baseline FP32 weights alone occupy 588.06 MiB, with an equally sized dense gradient allocation; no optimizer state is budgeted. P3/Q3 add .08203125 MiB of weights and the same gradient size. Below, one hidden/attention/logit tensor is counted once, not as a training-memory total. Twenty-one new LN output tensors are an illustrative saved-activation subtotal; autograd may save/reuse additional tensors and statistics.

| Batch | One hidden `[B,308,512]` MiB | One attention matrix `[B,8,308,308]` MiB | Full vocabulary logits MiB | 21 LN outputs MiB |
|---:|---:|---:|---:|---:|
| 1 | .601563 | 2.895020 | 58.050781 | 12.632813 |
| 2 | 1.203125 | 5.790039 | 116.101563 | 25.265625 |
| 8 | 4.812500 | 23.160156 | 464.406250 | 101.062500 |

Local FP64 doubles these individual floating-tensor sizes; FP32 copies, explicit attention intermediates, vocabulary CE temporaries and backward storage add further costs. Stream isolated-gradient sums rather than retaining six graphs. N3.1 full-G2 sampled RSS was roughly 2.2-3.0 GiB for its B1/B2 workers, but is not a peak prediction for these variants or the whole suite. The observed 96.7% system-memory incident shows why available RAM and competing process conditions must accompany timings.

All proposed variant CPU B1/B2/B8 and GPU results are **NOT RUN**. CPU B1/B2 are the initial proposed measurements. B8/GPU capacity probes require explicit resource authorization after measured B1/B2 footprints; absent hardware or approval remains NOT RUN. Do not report a successful capacity check from these arithmetic estimates. No optimizer, repeated training, G3/G4, dataset ingestion or benchmark evaluation belongs to this schedule.
