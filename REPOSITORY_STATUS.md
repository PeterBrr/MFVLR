# MFVLR documented reproduction

This repository contains the documented A0 engineering baseline, research-only
verification infrastructure, configurations, unit tests and design/acceptance
reports. The original architecture's undocumented details are not claimed as
recovered author implementation.

The historical README describes early MVE/VD phases. For current readiness use
[the D02B11 checklist](docs/phase2d2_d02b11_authorization_checklist.md) and
[dataset decision](docs/phase2d2_d02b11_dataset_decision.md).

Phase 2D.2 remains BLOCKED. The original G01-A0 numerical result remains FAIL:
293 parameter tensors and 8,772,234 outside-tolerance coordinates. B0-D08 is
disabled. Full-model same-forward execution, optimizer updates, training and
benchmarks require their separately recorded approvals. Creating this repository
does not authorize any experiment.

## Included and external files

Source, tests, configurations, documentation and the small pinned CLIP tokenizer
assets are included. Large historical evidence, gradient/sum shards, review ZIPs,
model checkpoints, datasets, the paper PDF, temporary files and local dependency
installs remain outside Git. They are preserved in the original workspace;
references to `artifacts/` in reports and live-pin inventories require that
external evidence archive. This Git repository alone does not contain every
file needed for a live scientific run.

No CI workflow launches experiments. Do not run the entire historical pytest
collection automatically: it includes legacy optimizer-containing tests whose
execution is restricted. The latest isolated operational controls can be run as:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest tests/research_d02b10 --confcutdir=tests/research_d02b10 -q
```

Use a separately reviewed environment matching the recorded dependency versions.
The existing package metadata is not a complete pinned installation contract.
Git stores files without line-ending normalization to preserve recorded hashes.
