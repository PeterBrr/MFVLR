# A1.1 technical acceptance snapshot

2026-10-12, Asia/Saigon. This freezes the **conditionally reviewed A1-D01 research baseline** and submits the execution-scope reconciliation for user review. It does not approve production replacement, change numerical policy, waive the eight A0 failures, or authorize A1-D02. The VD-T12 deviation remains recorded in the [closure addendum](phase2d2_a11_incident_closure.md).

The [acceptance manifest](../artifacts/phase2d2_a11_reconciliation/acceptance_manifest.json) pins **1,562 existing files**, with sizes and SHA-256 hashes, covering:

- A0/P3/Q3 research source and the accepted production modules it reuses.
- The locked A1.0, corrected A1.1 LayerNorm specification and erratum.
- Strict A0/P3/Q3 configurations, independent tests and mathematical oracles.
- Detailed parameter inventories/common-state parity, original pytest/JUnit logs, historical numerical-policy evidence and incident/approval records.
- Original review archives and dependency/tokenizer artifacts in the submitted inventories.

All pinned files were rehashed and matched. Nothing was overwritten or made read-only; the manifest defines the immutable content snapshot. Future changes require a new version. New reconciliation documents and guard evidence are separately inventoried in `reconciliation_evidence_manifest.json`, avoiding a self-referential hash.

**Acceptance manifest SHA-256:**

`8ae96c21facf3c4cf3e11d0ccf21176c4c9ee005f265e19d3de890a6b913f566`

**Canonical source-inventory SHA-256:**

`bbab51e5715ea1759f9c4004b5204c9c3e81cdb04f36d9e1e586cd4aba0f19ff`

The latter hashes the sorted compact JSON file mapping in [source_hashes.json](../artifacts/phase2d2_a11_reconciliation/source_hashes.json); it is an inventory digest, not the hash of that entire JSON document.

| Frozen item | Exact SHA-256 |
|---|---|
| `src/mfvlr_research/decoder_a11.py` | `eed5c43d3a39ab00d452bbbf114d3b7dd51a34331243b3f2fe22c3841030e898` |
| `src/mfvlr_research/a11_config.py` | `dd6a5deb1df4d3fa942ca87997e4f53b1f025bf563b01f77660f6ee46e1445f4` |
| `src/mfvlr_research/a11_evidence.py` | `e41188315b57a80b8a6bb945944e1e267ba2997e00e9cf7afb07fa47430d655a` |
| Corrected A1.1 preregistration | `3577e592a3a4013d136c840093d1d843ac92481b2f4088e4caf66fed7b3f92a3` |
| `configs/research_a11/A0.yaml` | `29fe970a478d7c66942281c455274ae4b301173e4d669f22d932f028d81bfa0c` |
| `configs/research_a11/P3.yaml` | `d40cb7cc36b289dc1443ac3251f57264da2fdf25692326082f6c41a7f521b561` |
| `configs/research_a11/Q3.yaml` | `e86cdcc06ea25b63d32f10e52cb13b25c479174ac060add43268cfa046b99654` |
| Collection runner | `8af82bfe462cd6b85098f06ce24b0df58fa6f5f66e9d8cf2c9d6119f0f5bc93d` |
| Disabled ablation launcher | `322dd1a89d56585829a47c04f69a7d9d2eef82959fa4188500c41d7713b99b5c` |

The corrected norm remains `gamma * (x - mean(x)) / sqrt(variance(x) + 1e-5) + beta`, using final-512-feature population variance and independent gamma=1/beta=0 initialization. P3/Q3 norms remain **ASSUMED research choices**, not recovered author architecture.

## Verification basis

| Finding | Basis in this reconciliation |
|---|---|
| 25/25 no-update cases PASS, exit 0, no skips/xfails | **Directly executed now:** 23 unchanged guard/registry cases plus two new collection/CLI probes |
| VD-T12 rejected before execution; CLI blocks A1-D02 | **Directly executed now:** actual collection and actual blocked entry points; zero optimizer/model attempts in instrumented pytest processes |
| Frozen source/config/test/assets/logs unchanged | **Directly executed now:** file hashing and comparison with submitted manifests |
| 54/54 research cases PASS | **Submitted JUnit verified**, not a new complete numerical-suite run; only the 23 guard cases overlap the fresh selection |
| 276/276 historical regressions PASS across two runs | **Submitted JUnit identities/outcomes verified**, including the preserved scope deviation; neither legacy update test rerun |
| Eight original A0 failures | **Submitted failing JUnit and identity set verified**; assertions/source unchanged, not executed here or reclassified |
| Parameters/common-state parity | **Submitted detailed reports verified by hash**, not new model construction or recount |
| 51 independently selected Linux cases PASS | **User-reported independent verification**; no Linux execution or locally supplied Linux JUnit inspected here |

Recorded inventories remain A0 **154,156,232 / 508 tensors**, P3/Q3 **154,177,736 / 550 tensors**; each normalized variant adds 21 LayerNorm modules, 42 Parameter tensors and 21,504 elements. Nineteen VIM roles, tied vocabulary, common byte-equal initialization and independent storage retain their submitted evidence. These are derived counts under engineering assumptions, not evidence of fidelity to the paper's approximately 141M count.

Fresh evidence is under `artifacts/phase2d2_a11_reconciliation/`: [stdout](../artifacts/phase2d2_a11_reconciliation/stdout.log), [stderr](../artifacts/phase2d2_a11_reconciliation/stderr.log), [JUnit](../artifacts/phase2d2_a11_reconciliation/junit.xml), [guard verification](../artifacts/phase2d2_a11_reconciliation/guard_verification.json) and [verification basis](../artifacts/phase2d2_a11_reconciliation/verification_basis.json). The run took 17.564 seconds including startup; pytest reported 11.66 seconds. These are guard-test timings, not model benchmarks.

Both optimizer-containing nodes were collected only. The existing runner's MVE exception parser was inspected without executing that test; a separate reconciliation sentinel forbade either test entering execution. The ablation CLI returned exit 1 with the expected `PermissionError`, which passed its refusal test. Registry resolution/preparation generated metadata only, including one new request beneath the allowed prepared tree; no model, fixture tensor or expensive experiment ran.

Accepted source, independent oracles, all numerical tolerances and N1/N2/N3/N3.1 policies remain unchanged. No original evidence or incident record was superseded. **A1-D02 remains UNAPPROVED / NOT RUN; Phase 2D.2 remains BLOCKED.** Stop for user review of this acceptance reconciliation and incident closure.
