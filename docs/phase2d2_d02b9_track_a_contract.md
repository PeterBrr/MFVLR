# D02B9 Track A: same-forward diagnostic preparation

Status: implementation and synthetic verification only. Real G01-A0 execution is NOT RUN. Phase 2D.2 remains BLOCKED; B0-D08 remains DISABLED.

The user authorized preparation, not a new numerical cell. The original D02B7-R2 result remains FAIL: 293/508 tensors and 8,772,234 coordinates. D02B8 found no definitive cause. This experiment can test sensitivity to separate versus shared forward graphs; it cannot by itself prove training stability or accuracy.

## Frozen scientific contract

Use original G01 real B1 fixture and Windows CPU C4M, unchanged seed-0 reference and A1.1 A0 builder. Required model: 508 tensors, 154,156,232 elements, tied vocabulary, no P3/Q3. Canonical state SHA256: `1865527facba9adc8a14d443c112d492698d275ce4e8beaa784aee61a514895a`. Fixture/input pins are in the preserved R2 `attempt_01/fixture.json`; no new image, token, mask or objective is allowed.

Exactly one `with_language` forward produces all six losses using the production `compute_objectives`. Independently audit actual arithmetic using frozen D02B7 arithmetic routines. Literal loss order is FD, LR, CMC, FL, AR, KL; no normalization, loss weighting, token exclusion or detach. The traversal kernel forms the same left-associated equal-weight total from these six connected scalars.

Six `autograd.grad` calls use the complete parameter tuple, `allow_unused=True`, `materialize_grads=False`, `create_graph=False`, `retain_graph=True`. The seventh joint call uses `retain_graph=False`. Stream each returned tuple to disk before the next call. Never populate parameter `.grad`; preserve None versus allocated zeros. Unexpected disconnection, nonfinite values, VIM Q/K nonzeros, state mutation or missing output hard-stop. No optimizer is constructed.

Reuse frozen D02B1 GradientStore: ordered FP64 accumulation of isolated FP32 gradients; elementwise FP32 comparison `abs(actual-reference) <= 1e-6 + 1e-5*abs(reference)` where reference is the FP32-rounded ordered sum. Keep all 15 loss pairs, four registered directions, cancellation and weak-gradient statistics. This is not FP64 autodifferentiation. N1/N2 exemptions do not apply.

## Implementation and comparison

New namespace: `src/mfvlr_research/d02b9`. `same_forward.py` implements the seven traversals and bounded mock interface; `live_preparation.py` contains the lazy frozen factory and full-roster recipe. Neither factory nor recipe has been executed. Public `execute` and `dispatch_real` always raise, including if a caller supplies an approval flag. No CLI/config switch enables them. Underscore helpers are preparation code, not permission boundaries against arbitrary Python callers.

`comparison.py` pairs all seven raw gradient passes by exact loss/parameter identities, checks shard hashes/shapes, distinguishes None from zero, and computes original-threshold differences, absolute/relative L2 and maxima against R2. These paired differences are diagnostics, not replacement acceptance thresholds. The live recipe writes paired records only after its own full analysis. Actual same-forward comparisons are NOT RUN.

Before execution a separately reviewed wrapper must provide the mandatory `seal` callback: exclusive coordinator PID/creation identity; exact source/config/protocol/input/state/roster/backend pins; actual resource readings; pre-forward, pre-traversal and post-traversal checks. A no-op callback is not an approved live implementation. Integration of that operational wrapper remains part of the next authorization handoff, not a hidden executable switch in this preparation package.

## Memory, graph lifetime and storage

Each full FP32 parameter or gradient vector is 616,624,928 bytes (0.574 GiB). Canonical reference plus private candidate temporarily doubles parameter storage. Full B1 vocabulary logits alone contain 15,217,664 elements / 60,870,656 bytes, excluding saved attention, convolution and FFN activations. Streaming frees each returned gradient tuple before the next pass, but all six losses and their shared saved forward activations remain live until the seventh traversal. First six backward workspaces and graph references are not equivalent to seven independently freed graphs.

Historical separate-forward sampled peak RSS was 2,341,023,744 bytes (~2.18 GiB), not a retained-graph capacity measurement. There is no defensible exact retained-graph peak or runtime yet. Proposed conservative screen: at least 8 GiB available physical RAM and 8 GiB temporary disk before construction and before the sole forward. This operational screen is ASSUMED/PROPOSED, not a scientific threshold or verified capacity guarantee. The former 5 GiB launch floor is insufficient evidence. GPU and B2/B8 capacity are NOT VERIFIED and outside this experiment.

R2 raw gradients/sums occupied 3,790,126,768 bytes; budget at least that plus inventories, comparisons, logs and an interrupted attempt. Existing R2 shards are read in place, never rewritten. No persistent new model checkpoint is required. Measure process RSS/available RAM periodically with timestamps in a future wrapper; stop before further traversal on inadequate resources rather than changing precision or dimensions. Never terminate unrelated applications.

## Interruption and decisions

Use a new exclusive attempt directory, immutable claim and append-only stage records. No overwrite/retry in place. On an exception retain completed shards and partial files with hashes, mark incomplete output HARD STOP; do not fabricate zero gradients. A retained autograd graph cannot resume from disk: replay would require new explicit authorization and a fresh single forward after archiving the entire attempt. No automatic recovery execution is approved.

Finite additivity FAIL means execution COMPLETE, numerical FAIL, dependent_work BLOCKED. It never enables B0-D08. Structural/provenance/nonfinite/completeness failures are HARD STOP, not numerical acceptance. Even all comparisons passing would address this fixture only, not repeated updates, convergence or paper accuracy.

Required next approval: one G01-A0 same-forward attempt, reviewed operational wrapper and proposed resource screen. No such attempt was made in D02B9.
