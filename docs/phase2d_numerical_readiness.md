# Phase 2D numerical readiness and resource review

Status: **DESIGN ONLY**. Gates and thresholds below are ASSUMED / PROPOSED under P2D17-P2D20 in the [decision register](phase2d_heads_contract.md). No head, loss, optimizer, synthetic training step or new resource benchmark was executed. Parameter/memory arithmetic and inspection of preserved measurements are not loss implementation. [Loss definitions](phase2d_losses_contract.md) must be approved before their proposed checks can run.

## What accepted N2 establishes

The user accepted Phase 2C.2 after N2 as a recorded Windows CPU baseline: 164/164 mandatory cases, including 119 unchanged regressions. Source: [N2 acceptance report](phase2c2_n2_acceptance_report.md), [verification](../artifacts/phase2c2_n2_acceptance/verification.json), [numerical comparison](../artifacts/phase2c2_n2_acceptance/numerical_comparison.json), [parameter/gradient evidence](../artifacts/phase2c2_n2_acceptance/parameter_gradient_preservation.json). Historical records that said acceptance was pending are preserved as records of their date; this design document records the subsequent user acceptance without rewriting them.

**Measured in N2, not newly measured in Phase 2D:** exact fixed-batch independence and permutation; exact zero cross-sample lookup/visual task gradients for per-sample model probes; full promoted-weight FP64 cross-batch agreement at atol=1e-10/rtol=1e-8; finite FP32 B=1/2 forward/backward for the tested probe. FP32 cross-batch discrepancies remain mandatory diagnostics, using unchanged atol=1e-6/rtol=1e-5 to classify mismatches. MVE N1 stays scoped to MVE T12 only.

| Preserved fixture | Final LE max abs | Final LD max abs | Logit max abs | FLT parameter-gradient L2 | Maximum cross-batch logit abs difference across its two samples |
|---|---:|---:|---:|---:|---:|
| F0: normal visual seed 341, golden prompts 1/5 | 27.931324 | 865.817810 | 527.739868 | 3,031,726.345 | 0.005348206 |
| F1: normal visual seed 642, prompts 0/4 | 32.985199 | 980.050354 | 520.926270 | 3,138,625.593 | 0.003684998 |
| F2: accepted MVE+VD feature, image seed 412, prompts 8/9 | 135.378204 | 4411.212891 | 2645.781006 | 62,397,308.906 | 0.178649902 |

Those gradient norms used `mean(logits^2)+mean(LD_hidden^2)+mean(LE_global^2)`, **not any of the six paper losses**. F2 sample 0 had 8,494,886/15,217,664 logits (55.8225%) outside the old FP32 equality tolerance, relative L2 2.69393e-5. Fresh N2 runs reproduced those measurements exactly on that backend. They do not establish accuracy, harmlessness of drift, GPU reproducibility or optimizer stability.

FP64 fixed-batch perturbations with encoder memory/visual held fixed had final LD RMS amplification about 46.79 (F0), 36.53 (F1), 804.53-840.84 (F2), depending on perturbation size. This demonstrates sensitivity in the implemented unnormalized LD; no normalization counterfactual isolated LayerNorm absence as the sole cause. The full architecture remains unchanged: no inserted normalization, clipping, AMP, scaling or smaller vocabulary/sequence.

Mathematically equivalent batched and per-sample floating-point operations need not be bitwise equal; this is also documented by [PyTorch 2.12 numerical accuracy](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html). That general fact supports investigating backend rounding but does not waive this project's exact fixed-batch checks, finite-value requirements or recorded drift review. N2 changes only FLT-T17's specified cross-batch FP32 policy.

## How the proposed objectives interact with these magnitudes

The following are **DERIVED risks**, conditional on the proposed interpretations of Section III-E Eqs. (17)-(27), not measured six-loss behavior.

| Objective | Direct numerical issue | What must be observed before drawing conclusions |
|---|---|---|
| FD | Large Iv can give confident wrong two-class logits. Stable CE avoids log(0), but Jacobians upstream may still amplify bounded logit derivatives | Detector margins, CE value, Iv/AD/visual gradients; no assumed benefit from low training CE |
| LR | Sum over 308 positions gives uniform loss 3328.82. Very large logits can saturate vocabulary probabilities. CE logit derivatives are bounded per coordinate by 1/B, unlike squared-logit probes, but LD/LE Jacobians and tied vocabulary accumulation can still magnify them | All-token/pad/special/content CE subtotals, logits min/max/margins, target log probabilities, entropy, every block/parameter gradient norm |
| CMC | Raw dot grows with both feature norms; division by initially 0.07 amplifies scores. A shrinking trainable tau can intensify this. Stable logsumexp does not prevent the matmul/division overflowing first | Iv/Tl norms, raw dot and scaled score extrema, row/column entropy, diagonal probabilities, theta/tau values/gradients and duplicate counts |
| FL | Mean over 50,176 pixels can have much smaller output-gradient scale than token-summed LR. Class imbalance and constant real/EFS masks can produce misleading pixel accuracy | Separate foreground/background CE and pixel counts, both-class gradients, MD/trunk/local-encoder gradients; no per-class reweighting |
| AR | Identity RGB output permits large errors and unbounded squared loss; loss and residual share Ipre. Other losses can drive reconstruction against AR | Ipre min/max/out-of-range fraction, abs-residual zeros/signs, direct AR and residual-mediated AD gradients separately |
| KL | Dividing feature scores by 0.5 sharpens an already large Tl. Teacher entropy can collapse and teacher gradients can become tiny; student gradient still depends on Q-P. Two-sided optimization can move both distributions or collapse alignment | P/Q entropy, selected feature masses, adapter/Tl gradients, direction checks, teacher-detached control and consistency of finite log probabilities |

A large raw loss is not proof of gradient dominance; measure both. A small loss can coexist with poor task behavior. Shared parameters accumulate gradients from several tasks with opposing directions. Record per-loss module gradient L2, max abs, finite/nonzero counts, and cosine between selected loss-gradient pairs when both norms are nonzero. For zero gradients the cosine is **undefined**, not zero. Report joint norm and cancellation relative to component norms. Numerical reporting may use FP64 norm accumulation; it must not cast the production graph to FP64.

All singleton VIM Q/K **task** gradients remain exactly zero. Proposed future coupled optimizer decay could update these parameters despite that fact; record task gradient separately from optimizer's effective update. New loss gradients through ordinary attention need not resemble N2's diagnostic gradient norms. No task-gradient clipping or loss scaling is justified automatically by a large measured norm.

## Separate claims and gates

| Gate | What a pass establishes | What it does not establish |
|---|---|---|
| G0: preserved baseline and approved contract | Correct source/configuration/identity starting point | Correct new loss interpretation or training |
| G1: independent scalar/gradient oracles | Specified heads/losses and target semantics implemented correctly | Full graph resource capacity or stability |
| G2: finite no-update joint graph | Six losses and gradient paths execute at the tested full shapes | Finite optimizer state/update, repeated-step behavior |
| G3: one finite update | That particular optimizer step and following forward are finite | Repeated training stability or convergence |
| G4: bounded repeated-update smoke | No detected failure over the specified short synthetic trajectory | Long-run stability, data fidelity, generalization or accuracy |
| G5: actual training/evaluation study | Only the observed, pre-registered dataset/protocol results | Recovery of undocumented author choices |

Current evidence: accepted visual/FLT portions of G0 only. New heads/losses G1-G4 are **NOT RUN**. G3/G4 require separate optimizer execution approval under P2D19; G5 remains deferred with no authorization. Approval of a head/loss implementation must not be interpreted as permission to launch training.

### G0-G1: preserved source and controlled tests

Before future implementation, record hashes of every accepted source, golden fixture, config, test and artifact. Keep all 164 cases and the exact MVE N1/FLT N2 policies active. New files carry new scoped approval records. Counts must remain accepted totals plus precisely 263,683 new parameters if all proposed choices are approved. Verify no parameter aliases, no copied vocabulary, no duplicate mask head and no changed initialization. Reference/debug must resolve the same architecture/objective.

Test independent tiny FP64 mathematical fixtures first (test plan lists exact formulas, targets and tolerances). These are scalar algebra oracles, not replacement smaller-model acceptance. Include zero/identical/saturated/unequal distributions, negative logits, duplicate prompts, B=1 CMC, full 308-position reduction and out-of-range reconstruction. Deliberate NaN/Inf and wrong-target inputs must fail validation. Only then proceed to full model dimensions.

### G2: proposed synthetic pre-training gate, without updates

Use accepted module configuration and seed 0; proposed new head seed 0. CPU FP32 with four intra-op threads, recorded inter-op threads, deterministic algorithms and highest FP32 matmul precision. Do not edit accepted profiles. Full image 224x224, 308 tokens, 49,408 output classes, 12 LE/7 LD, full accepted visual graph.

Run B=1 and B=2 separately in fresh processes. For B=1 run each of three sample kinds independently: real, EFS fake, locally modified fake. For B=2 use three documented pair fixtures: real/EFS, real/local, and duplicate-prompt local/local with different image perturbations. Construct synthetic base RGB values uniformly in [0.1,0.9], using deterministic image seeds 17, 412 and 997 respectively. For local examples alter an explicit rectangle `[64:160,80:176]` by complementing its RGB values (1-I), so the input stays in [0,1]; the exact binary rectangle is an **oracle mask**, not a claim to implement the paper's difference-threshold data procedure. Real mask is zero, EFS mask one. Use accepted prompt builder/golden real/DDPM/DiffFace-compatible metadata as appropriate; record literal IDs, image/mask hashes and sample types. B=1 slices are separate fixtures, not assertions of full loss equivalence to B=2; CMC is different by construction.

For each fixture, run each objective in isolation on a fresh deterministic forward and then all six jointly. Record every scalar; logits/features/probabilities; activation RMS/L2/max at all LE/LD blocks; Iv/Tl/adapter norms; temperature; and module/parameter gradient summaries. Retain differentiable tensors only for requested gradient probes, remove hooks afterward, and serialize detached values/statistics rather than graph-owning tensors. Avoid keeping six simultaneous full graphs.

Use explicit test-only detached controls for (a) Ipre before residual construction, (b) Iv supplied to FLT, (c) Tl supplied to KL, and (d) adapter input. Never feed detached controls into the reference production path. Shared parameter gradients alone cannot isolate a role: capture the actual Ipre/Iv/lookup intermediate gradients and compare direct versus indirect paths. Joint gradients should equal sums of individual loss gradients within declared tests' arithmetic tolerance; compute small FP64 oracles first and stream/recompute full-graph measurements to control memory.

Pass G2 only if all specified valid fixtures, scalar/gradient checks and exact ownership checks pass; all outputs/intermediates/gradients are finite; expected controlled nonzero paths exist; expected absent/zero paths agree with the contract; and the complete unchanged baseline suite passes. Saturated/symmetric fixtures have explicitly expected zeros. Count changes, a missing path, a nonfinite value or a tolerance failure stops the gate; diagnose and report, without automatic amendment.

Preserve N2 stage diagnostics on its original F0/F1/F2 fixtures at unchanged seeds/weights. Separately record new head/loss output drift as diagnostic evidence; N2 does not grant a blanket relaxed threshold for new heads, losses or gradients. Exact per-example model independence and permutation remain mandatory where specified. CMC's cross-sample objective gradients are intentionally nonzero, and batch reduction order can affect summed losses. Do not impose exact equality between B=1 CMC and a B=2 CMC slice.

### G3-G4: future optimizer-readiness proposal, not authorized execution

**EXPLICIT paper p. 8 Section IV-A:** Adam, learning rate 1e-4, weight decay 1e-3, tenfold LR decrease every 15 epochs. **ASSUMED candidate P2D19:** Adam (not AdamW), betas=(0.9,0.999), eps=1e-8, amsgrad=False, coupled L2 decay, FP32 states, deterministic unfused/non-foreach path where supported; decay all trainable parameters except CMC theta (decay 0). No gradient clipping, AMP/scaling, warmup or scheduler during this sub-epoch diagnostic. All these optimizer defaults/exclusions require explicit later approval. No optimizer code is created now.

Once separately authorized, G3 starts from a disposable clone of a passed G2 state. Record task gradients before decay, effective gradient/update, parameter norms, update norms, moment min/max/norms, counter and effective tau. Apply exactly one update; require all state/parameters and the subsequent six-loss forward/backward to remain finite. Confirm original accepted checkpoint/hash has not changed. Exactly zero Q/K task gradients must not be confused with zero coupled-decay updates.

G4 candidate: **20 steps**, separately for B=1 and B=2 and data seeds 17/412/997, resetting initial model/optimizer per run. Alternate the fixed real/EFS/local fixtures in a predeclared cycle, preserve sample/metadata alignment and keep all 308 targets. No gradients carried between steps. Record losses, all module activation/gradient maxima, parameter update ratios, tau, entropy, RSS and wall time at every step. B=1 tests finiteness only; CMC carries no learning signal. B=8 is an independent resource-gated execution, not simulated by accumulation.

Hard stop: any NaN/Inf, nonpositive/nonfinite tau, invalid target, missing required gradient, changed topology/tie, count mismatch or optimizer state corruption. **ASSUMED review triggers, not hidden stabilizers:** any nonzero-initial module activation/gradient norm grows over 100x its step-0 value, tau leaves [0.007,0.7], or a matrix update L2 exceeds its pre-step matrix L2. Zero-initial bias/gradient ratios are undefined and reported with absolute norms; do not invent a denominator epsilon. These triggers stop for diagnosis/review before claiming readiness, even if finite; they do not clamp, skip or alter an update. Bounds are engineering triage thresholds, not paper hyperparameters or convergence criteria.

No monotonic decrease requirement: six competing tasks and synthetic targets can produce nonmonotonic loss. Completing 20 steps permits only a statement of observed finite short-run behavior, with all trajectories visible. It does not authorize longer training, establish convergence or waive saturation/scale concerns. G5 requires approved real data, split/geometry manifests, optimizer protocol, epoch budget, checkpoint selection and metrics; those are currently unavailable/deferred.

## Memory arithmetic: B=1, B=2 and B=8

**DERIVED estimates under the proposed 154,156,232-parameter model, FP32, MiB=2^20 bytes.** These are allocation sizes, not measured peaks or GPU capacity claims.

| Parameter-state scenario (batch-independent) | MiB |
|---|---:|
| Full weights only | 588.059 |
| Full weights plus same-dtype gradients | 1176.119 |
| Weights + gradients + two FP32 Adam moments | 2352.237 |
| New heads/temperature only, same four-state scenario | 4.023 |

The four-state scenario assumes every parameter has an allocated gradient and both moments, excludes optimizer step counters/workspace, and is not a current optimizer measurement. Inference graph can omit language/adapter/objective entirely; retaining those modules in a full object still retains their storage.

| Tensor or illustrative allocation | B=1 MiB | B=2 MiB | B=8 MiB |
|---|---:|---:|---:|
| Full vocabulary logits `[B,308,49408]` | 58.051 | 116.102 | 464.406 |
| Logits + a log-probability copy + logits gradient, if all retained | 174.152 | 348.305 | 1393.219 |
| Int64 token targets `[B,308]` | 0.002350 | 0.004700 | 0.018799 |
| One hidden `[B,308,512]` | 0.602 | 1.203 | 4.813 |
| One ordinary attention map `[B,8,308,308]` | 2.895 | 5.790 | 23.160 |
| One mask-logit tensor `[B,2,224,224]` | 0.383 | 0.766 | 3.063 |
| Int64 binary mask `[B,224,224]` | 0.383 | 0.766 | 3.063 |
| One RGB image/reconstruction/residual tensor | 0.574 | 1.148 | 4.594 |
| Adapter output or one KL distribution `[B,1,512]` | 0.001953 | 0.003906 | 0.015625 |
| One CMC score matrix `[B,B]` | 0.0000038 | 0.0000153 | 0.0002441 |

The earlier illustrative FLT retention inventory (52 ordinary attention maps + 19 FF hidden tensors + 19 block outputs) is 207.689 / 415.379 / 1661.516 MiB. It omits visual activations, many saved derivatives, VIM maps, Q/K/V, residuals, allocator/softmax workspace and outputs; it is not a peak estimate. At B=8, adding that illustrative inventory, the three vocabulary tensors and four-state parameter scenario gives about **5406.972 MiB**, still not a full capacity estimate or rigorous lower bound because lifetimes and storage sharing vary. Do not infer that a 6/8/16-GB GPU fits from this sum.

FP64 promoted copies double per-element storage; retaining the original FP32 model adds another 588.059 MiB for the proposed full model (the accepted VD itself only permits its established FP32 API). Do not cast the accepted VD to make a new full FP64 end-to-end promise. Use FP64 mathematical head/loss fixtures and the accepted FLT promoted copy where supported; full visual integration stays accepted FP32. For full per-loss gradient sum diagnostics, a CPU FP64 accumulator alone can add 1176.119 MiB; stream per-module chunks to a temporary evidence file if needed, log the strategy and retain only one loss graph at a time. This changes diagnostic storage, not the objective or production graph.

Do not materialize `[B,308,49408]` one-hot targets, store all logits per step, retain hooks' graphs across steps, or instantiate all precision/fixture workers at once. Chunked vocabulary projection, checkpointing, AMP, microbatch accumulation and distributed negatives are not silently approved memory remedies. Gradient accumulation does not reproduce a larger CMC negative set.

## Preserved measurements versus new capacity status

These are **historical Phase 2C.2 measurements**, from [resources.json](../artifacts/phase2c2/resources.json); they predate six losses, detector/adapter and an optimizer. Windows CPU, Torch 2.12.1+cpu, 4 threads, approximately 16.90 GB host RAM; one warmup then three recorded iterations, memory sampled at 2 ms. Median durations below are calculated from those three recorded samples, not new executions.

| Historical route | B | Sampled peak RSS MiB | Median forward s | Median forward+backward s | Recorded status |
|---|---:|---:|---:|---:|---|
| FLT | 1 | 1780.059 | 0.798 | 1.927 | PASS |
| FLT | 2 | 2082.941 | 1.759 | 4.660 | PASS |
| MVE+VD+FLT | 1 | 2332.801 | 1.617 | 4.452 | PASS |
| MVE+VD+FLT | 2 | 2925.840 | 3.285 | 8.162 | PASS |

Historical B=8 was **NOT RUN**: available RAM 5,372,989,440 bytes was below that diagnostic's conservative 8-GiB available-memory preflight. That was an unexecuted resource decision, not OOM evidence. GPU was **NOT RUN**, CPU-only Torch/CUDA unavailable. N2 subsequently verified numerical behavior but did not add six-loss/optimizer capacity evidence.

| New Phase 2D run | B=1 | B=2 | B=8 |
|---|---|---|---|
| CPU heads + six-loss full forward/backward | NOT RUN, design only | NOT RUN, design only | NOT RUN, design only |
| CPU optimizer/repeated-step smoke | NOT RUN, unapproved | NOT RUN, unapproved | NOT RUN, unapproved/resource-gated |
| GPU heads/losses/updates | NOT RUN, no CUDA evidence | NOT RUN, no CUDA evidence | NOT RUN, no CUDA evidence |

Future resource reports must distinguish PASS, FAIL/OOM, other FAIL and NOT RUN with a reason. For CPU: fresh subprocess, available/total RAM, initial/sampled-peak/lifetime-peak RSS, sampling interval, device/dtype/full shapes, thread/build metadata, forward/backward/step time separately. For GPU: device and memory capacity, CUDA/backend flags, explicit synchronization, peak allocated/reserved memory and optimizer-state creation timing. Measure the first Adam step separately because moments allocate lazily. No hidden skip or smaller-batch substitution qualifies as B=8 acceptance. Resource unavailability may block an optional capacity run but cannot be described as passing it.

## Readiness conclusion

The accepted graph and N2 support proceeding to a separately approved **head/loss implementation and no-update verification**. They do not demonstrate even one finite six-loss optimizer step. The highest-priority risks remain token-sum LR magnitude, raw-dot temperature sensitivity, two-sided sharp KL, duplicate contrastive prompts, unrestricted RGB reconstruction, and the unnormalized LD's measured perturbation amplification. Each must be measured under the actual proposed objectives before any training-readiness claim. No production component or training action is authorized by this document.
