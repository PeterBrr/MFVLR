# Mask decoder audit — 18/10/2026

Follow-up experiments now include all six loss gradients on two training pairs,
phase-tying controls and bounded decoder-only fits. See the
[root-cause follow-up](decoder_root_cause_vi.md). Findings below preserve the
earlier, more limited no-update audit.

Checkpoint: `artifacts/diffae_pilot/runs/pilot_005/best.pt`, step 2000 (epoch 4).
No optimizer updates or production model changes were performed.
Machine-readable evidence: `artifacts/decoder_diagnostic_pilot005/report.json`.

## Findings

The concern is confirmed. All 400 exported masks are distinct, but agreement with
their per-pixel majority template is 99.4545%. 94.043% of pixel positions never
change across the 400 images. Foreground area is 17.038% on average, with only
0.227 percentage points standard deviation between images.

This is not an observed export duplication or batch-order bug. On one real/fake
pair, PNG masks match freshly computed full visual-route predictions exactly
(zero differing pixels). Full visual-route and direct CNN-local-to-decoder logits
also match exactly. Repeating a decoder call or swapping batch order gives zero
maximum logit difference. These controls cover the checked pair, not every code
path or dataset.

The trained decoder strongly favors a spatially periodic foreground pattern.
Across 16 balanced test examples, a 16x16 phase template explains 98.72% of the
spatial variance of the mean probability map after removing a 16-pixel border.
This quantity describes the mean spatial pattern, not model accuracy or a causal
attribution percentage. A feature tensor made constant over space and samples,
while retaining per-channel means, gives 99.029% binary agreement with the normal
majority template. Thus learned per-pixel decisions depend heavily on upsampling
phase rather than the location of manipulated content.

The current trunk uses four stride-2, kernel-2 transposed convolutions, producing
an overall factor of 16. The observed period is consistent with those stages.
Kernel=stride avoids unequal overlap of neighboring windows but does not force
the four learned subpixel phases to have equal responses. Initialization already
shows a weaker periodic component (82.71% mean-map spatial variance on the
checked pair); training greatly increases its amplitude. This is evidence of a
decoder/optimization artifact, not proof that ConvTranspose alone explains all
failed learning.

Encoder/local and decoder features differ between the real/fake inputs: relative
between-image RMS delta is 19.25% at the local input and 16.15% at U4 for the first
pair. Neither a fully dead encoder nor an exactly constant tensor was observed.
Mask-head weight/bias changes from initialization have L2 norms 0.29425/0.02355.
Setting decoder biases to zero for a temporary diagnostic still preserves
99.516% of the checked binary masks; biases alone do not explain the pattern.

Localization gradients are connected and nonzero, reaching the mask head,
trunk and local input. On the checked real/fake pair:

| Target | FL gradient norm | AR gradient norm |
|---|---:|---:|
| Local input | 0.001437 | 0.353911 |
| U1 upsample weight | 0.369523 | 139.502243 |
| U4 second convolution weight | 7.446582 | 263.523529 |
| Mask head weight | 4.115393 | no path |
| Appearance head weight | no path | 126.861298 |

AR exceeds FL by about 378x at U1 and 35x at U4. These are isolated two-loss
gradients on one pair; LR/CMC/FD/KL were not differentiated in this audit.
Gradient magnitude alone neither establishes Adam update magnitude nor proves
destructive interference. Shared-trunk competition and language-loss effects
remain hypotheses requiring separate measurements.

## Recommended next experiment

Keep the failed checkpoint and reference architecture unchanged. First verify
localization-only overfitting on fixed real/fake examples with known targets,
then measure all six loss contributions at shared parameters. Separately test a
resize-plus-convolution upsampler against the existing transposed-convolution
trunk under the same data/budget. Such an experiment changes the engineering
decoder variant and requires a fresh checkpoint; smoothing already exported
masks would not establish that localization was learned.

Reproduce the no-update decoder audit from the project root:

```powershell
python tools/diagnose_mask_decoder.py --checkpoint artifacts/diffae_pilot/runs/pilot_005/best.pt --exported-masks artifacts/diffae_pilot/runs/pilot_005/masks_test/predicted --output artifacts/decoder_diagnostic_pilot005
```

The audit contains no training or benchmark-based parameter tuning. Runtime
tests consist of the counterfactual input/bias controls, direct/visual-route and
export parity checks above. Existing production source, losses and checkpoints
were not edited.
