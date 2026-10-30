# MFVLR documented reimplementation

Current internal Diffae pilot: see [training commands](docs/diffae_pilot_run.md).
The separately configured [resize-convolution variant](docs/resize_conv_variant_vi.md)
adds language-to-visual stop-gradient and foreground-weighted localization.
The new `train.py` and `evaluate.py` reuse A0 and the six existing objectives;
historical numerical acceptance remains unresolved. The sections below describe
the earlier implementation phases.

Phase 2A implements the Multi-domain Vision Encoder under approved engineering
assumptions M1–M5. Author architectural fidelity is not established by these
choices or by the parameter count. See [the implementation contract](docs/implementation_contract.md)
and [the execution report](docs/phase2a_implementation_report.md).

The package contains the accepted MVE and the Phase 2B Vision Decoder: one
bottleneck-only trunk with separate appearance/mask heads, under approved D01–D08
engineering assumptions. Decoder profiles/schema are separate from accepted MVE
configurations. Language modules, paper losses, classifier, dataset loaders,
training and evaluation remain outside the implementation scope.

From the project root, with PyTorch, PyYAML and pytest installed:

```powershell
python -m pytest -vv -s

$env:PYTHONPATH = Join-Path (Get-Location) 'src'
python -m mfvlr.diagnostics --config configs/debug.yaml --mode smoke --output-dir artifacts/recheck/smoke
python -m mfvlr.diagnostics --config configs/debug.yaml --mode debug --output-dir artifacts/recheck/debug
```

Diagnostics keep the full MVE architecture and use external synthetic
reconstructions. Their scalar is a debugging probe, not a paper objective. They
perform no optimizer updates. Approved CPU FP32 verification uses four threads.

`load_config()` validates both profiles and resolves the debug overlay relative
to its file. Unknown or duplicate keys, topology changes, broadened approval,
and unsupported overlays raise `ConfigError`. `MVE.from_config()` constructs
only MVE and requires explicit scoped approval in every profile in the chain.
The remaining full-reference fields are unimplemented proposals.

Phase 2A acceptance now passes all 14 test IDs (15 cases, including both T01
batch sizes). Approved N1 changes only T12's CPU FP32 cross-batch-size tolerance
to atol=5e-6, rtol=1e-5; fixed-batch sample independence remains exact. The report
preserves the original failure and the evidence supporting the amendment.
The review archive is `artifacts/phase2a/phase2a_review_package.zip`.

Phase 2B passes all 40 cases (14 decoder unit IDs, 8 integration IDs and 14 MVE
regression IDs). Import `VisionDecoder` from `mfvlr.vision_decoder` and
`load_decoder_config` from `mfvlr.decoder_config`. The VD has 9,055,365 unique
parameters and accepts only explicit `bottleneck_only` policy. See the
[Phase 2B report](docs/phase2b_implementation_report.md) and
`artifacts/phase2b/phase2b_review_package.zip` for actual tests, measurements and
source review. Synthetic resource diagnostics additionally require psutil.
