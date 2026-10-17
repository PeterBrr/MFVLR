# A1-D02B0 scoped approval and evidence-freeze record

2026-10-15, Asia/Saigon. **D02B0 design/static preflight authorized by the current user request. D02B implementation/execution is NOT authorized. Phase 2D.2 remains BLOCKED.**

Documents submitted: [experiment contract](phase2d2_d02b0_experiment_contract.md), [fixture matrix](phase2d2_d02b0_fixture_matrix.md), [numerical gates](phase2d2_d02b0_numerical_gates.md), [resource plan](phase2d2_d02b0_resource_plan.md). New choices remain ASSUMED / PROPOSED until explicitly approved. Earlier A1.1 module approval and D02A local execution approval are not full-model execution permission or confirmation of author architecture.

## Decision register

| ID | Decision | Status / authority |
| --- | --- | --- |
| B0-D00 | Read-only source/evidence inspection, hashes, historical arithmetic and five design docs | **APPROVED**, this D02B0 request only |
| B0-D01 | Preserve exact A0/P3/Q3 topology, MVE/VD/FLT/objectives, frozen configs/fixtures/thresholds and historical failures | **APPROVED preservation requirement**; no architectural change |
| B0-D02 | New isolated, variant-aware full-integration orchestration/record schema; do not unlock old launcher or loosen loaders | **PROPOSED** implementation contract; no code implemented |
| B0-D03 | First execution partition G01–G06: 18 matched G2 cells, seven loss passes and specified timing passes | **PROPOSED / BLOCKED pending separate execution approval** |
| B0-D04 | C01–C07: 21 backend/reduction-order control cells | **PROPOSED**, separate scope selection after primary review |
| B0-D05 | NG2 materialization and run manifest with FLT seeds 6101–6103, pre-forward tensor/state hashes | **PROPOSED / BLOCKED**; no new fixtures generated |
| B0-D06 | N01–N06: 18 paired NG2 cells | **BLOCKED pending G2 evidence review and explicit NG2 execution gate approval** |
| B0-D07 | Exact paired invariants for unchanged non-LR branches, full parameter coverage and all loss-pair diagnostics | **PROPOSED verification additions**, no changed existing numerical tolerance |
| B0-D08 | Continue explicitly authorized finite diagnostic cells after recording a genuine mandatory numerical FAIL; preserve failing assertions/exit status | **PROPOSED execution handling**, not a numerical acceptance amendment |
| B0-D09 | RAM/disk reservations, sequential coordinator and strict recovery plan | **PROPOSED operational safeguards**; existing 3-GiB floor retained; current RAM below it |
| B0-D10 | Retain canonical full gradient shards/FP64 sum; state checkpoint per seed/variant; stats-only activations and timing repeats | **PROPOSED retention/resource contract** |
| B0-D11 | New detached controls, extra category-backward sweeps, fresh N2/local oracle sweeps, regression execution | **DEFERRED**, must be separately enumerated/authorized; none bundled now |
| B0-D12 | Any numerical tolerance/acceptance-policy change | **NOT PROPOSED / UNAUTHORIZED**; N1/N2 remain narrowly scoped |
| B0-D13 | I1, R2, production promotion, full FP64 visual route, B8/GPU experiments, G3/G4, optimizer/scheduler, data preprocessing/training/evaluation | **UNAUTHORIZED / NOT RUN** |

The two legacy optimizer-containing tests **MVE-T10 and VD-T12 must not run** in this design phase or an unqualified no-update handoff. The historical MVE-T10 exception does not transfer. The VD-T12 incident is preserved and is not retroactively approved because its weights were restored.

## What was directly checked now

Static preflight used standard-library file reads/SHA-256, Windows memory information and disk-space inspection. **1,594/1,594 files in D02A `preservation_before.json` matched their recorded hashes**, including accepted source/config/tests and earlier evidence. A further **43 artifact/manifest/state file fingerprints** were captured, including the final large/lite ZIPs, all D02A manifests found recursively, the final lite manifest and its archived first-candidate manifest, corrected preregistration, fixtures/golden metadata and all nine D02A checkpoint files plus their JSON records. Every file among those belonging to the final D02A manifest matched it. Both final ZIP hashes matched their sealed verification records.

The exact read-only record is [static_preflight.json](../artifacts/phase2d2_d02b0/static_preflight.json). It contains full digests/byte sizes, the resource timestamp and copied historical G2 provenance. It is new D02B0 metadata, not regenerated numerical results. Cache/complete raw-row provenance remains in unchanged original manifests; the 30-GiB activation cache was not reread and the pilot's scientific comparisons were not recalculated. ZIP CRC/per-entry verification is the preserved packaging evidence, not a fresh D02B0 CRC run. No tests or model code were imported/executed.

| Fact | Verification basis in D02B0 |
| --- | --- |
| Accepted baseline source/config/tests and historical 1,594-file inventory unchanged | **Fresh file SHA-256 comparisons** |
| Final D02A and lite ZIPs, manifest files and nine saved local initial-state files unchanged | **Fresh SHA-256**, matched prior recorded digests where available |
| D02A 414 jobs, 399,522 raw FAILs, 4,907 cross-batch FAILs, 58,752 structural checks | **Previously sealed numerical records / compact reconciliation**, not re-executed |
| 2,736 FP64 ordinary-attention oracle comparisons | Existing full role taxonomy; **864 Q/K-weight-only**, 432 key-bias, 864 input/query-bias and 576 added-LN comparisons; no new oracle run |
| G2 fixture image/mask/token hashes, initialized A0 root hash, historical memory/time | **Read from historical records**; fixture tensors and full model were not regenerated |
| P3/Q3 new full-six-loss state/output/gradient/capacity and NG2 materialization | **NOT RUN / NOT VERIFIED** |
| New full numerical acceptance, stable optimizer updates, convergence/accuracy | **Not established; unauthorized** |

The original eight A0 assertions and all finite local failures remain genuine. D02A's two Q3 FP64 additivity failures refer to the same registered scientific fixture under two backend settings; lower activation magnitudes/FAIL totals do not select a production winner. Full six-loss evidence is still needed and, even if finite, would not establish training stability.

## Fingerprints and preservation boundaries

The appendices below record exact digests read during this design review. “All manifest fingerprints” here means the recursively found D02A manifest files, final and archived lite manifests, and accepted A1.1 freeze manifest. The old full-package manifest itself enumerates its complete payload; cache/excluded inventories retain omitted evidence paths/hashes. None of those files was edited. No new final archive replaces either accepted archive.

The saved D02A `.pt` files are local LD initialization evidence, not complete G2/NG2 roots. Identical P3/Q3 checkpoint SHA values do not identify identical functions: architecture/config identity is separate. New full candidate checkpoints and NG2 byte hashes must be separately sealed after their materialization is authorized and before the first corresponding forward.

## Handoff requirements

Approval must name the future implementation scope, cell partition, operational reservations/retention and finite-failure continuation rule. A code review must verify the isolated runner, full-gradient/stage coverage, no-update/collection guards, variant-aware ownership, exact model/fixture seeds and fail-closed provenance. Resource availability must be rechecked; the present snapshot is not reusable launch permission. All NG2 hashes remain a mandatory pre-forward dependency under a separate gate.

**Stop for design review.** No production module/config/test was changed. No experiment driver or enabling flag was added. No forward, backward, optimizer step, dataset operation, test suite or benchmark ran. A1-D02B, G3/G4 and training remain unapproved.

## Static fingerprint appendix

| Frozen file | Bytes | SHA-256 |
| --- | ---: | --- |
| `artifacts/phase2d2_a11_reconciliation/acceptance_manifest.json` | 278840 | `8ae96c21facf3c4cf3e11d0ccf21176c4c9ee005f265e19d3de890a6b913f566` |
| `artifacts/phase2d2_d02a/cache_manifest.json` | 175408 | `a7d7e82168863f708c76a55436a5e95e0c525b64bfe44bf4d426bebbc16b423b` |
| `artifacts/phase2d2_d02a/driver_manifest.json` | 567 | `dadc0683f986cce2b476d6086f44d64e3f97935d3276624e4d638e136fba7d4f` |
| `artifacts/phase2d2_d02a/driver_manifest_v2.json` | 567 | `64c8b0589e96ec5eeb76cd4f2c8328400140791517fd897c8c43821dbb3f0c41` |
| `artifacts/phase2d2_d02a/history/interruption_0121_20261013/interrupted_attempt_manifest.json` | 5375 | `3fd40f02b79ca0f2179f0dd474a1ae8d8a4305890791d896ae8c9c17e33bb4bf` |
| `artifacts/phase2d2_d02a/history/poweroff_0407_20261013/interrupted_attempt_manifest.json` | 8173 | `363c428b2bead058ea74b633d548f4d314423d5e111952ce675e387be61b0038` |
| `artifacts/phase2d2_d02a/history/poweroff_20261013/interrupted_attempt_manifest.json` | 3581 | `54690f03d20b23046cac84503c0622efbdc685fdeb0db2a375fa960c8d9bf8e3` |
| `artifacts/phase2d2_d02a/LN_preexecution_input_manifest.json` | 2682 | `4a3a5215116d3573ca553d6446799f647cff89292aee6a2db9c97e63a29b252f` |
| `artifacts/phase2d2_d02a/phase2d2_d02a_review.zip` | 3884864027 | `d5ce9bc27b95b41b980b8583965a131158642b7126d5edde5d172180297c5a27` |
| `artifacts/phase2d2_d02a/pilot_protocol.json` | 107557 | `cdc6bd1c9bd940e61b6f5fbbff7d8b7e2cedbb66a52e5ff1b5a7a3a5b3ae8908` |
| `artifacts/phase2d2_d02a/preservation_after.json` | 270186 | `34a859a3dc8e21808d0ac6fbdbd2e7554bf13b6dff7a5dde1b3476635945fe7c` |
| `artifacts/phase2d2_d02a/preservation_before.json` | 270186 | `34a859a3dc8e21808d0ac6fbdbd2e7554bf13b6dff7a5dde1b3476635945fe7c` |
| `artifacts/phase2d2_d02a/preservation_verification.json` | 478 | `ab413844eb103f242f17408d5a8b7df3a3d78c255c12771817dae5cc1285b3d7` |
| `artifacts/phase2d2_d02a/progress_snapshots/20261013_204519/manifest.json` | 596 | `40486ac2342e2bbd5581935567ba79d3722c7618d6edf0925e3c3205711574da` |
| `artifacts/phase2d2_d02a/progress_snapshots/20261014_131145/manifest.json` | 584 | `ac5a75c7456672d3708db707a15cdce20dfbf2fec980b892935817dbbc7f31e9` |
| `artifacts/phase2d2_d02a/review_manifest.json` | 929204 | `90dab0e20d7e904509e94404ce00b53b8ce9fca1020b1da59031b47f3ed14aeb` |
| `artifacts/phase2d2_d02a/review_package_record.json` | 421 | `4bacc00085595891a31a5bf966de3f4e122999f50995f6b354ffc321518ca725` |
| `artifacts/phase2d2_d02a/states/j0_A0.json` | 109273 | `9c46c94e84822b3e7e0b3052b9c93c07741fbfbc5df2f1559c963fc642f5e2ca` |
| `artifacts/phase2d2_d02a/states/j0_A0.pt` | 147044655 | `e6c5948caccc8e6142b84b6672433ec2c53b3f96902e52a8feb068b4f6aaa71b` |
| `artifacts/phase2d2_d02a/states/j0_P3.json` | 110833 | `4c7b82cb0c81db2a06b7fb64a88aafbb7e3e2dabae4bd89cc7f4539fb0b2398e` |
| `artifacts/phase2d2_d02a/states/j0_P3.pt` | 147143707 | `f52023ae219426c78d003ba341f2195dbb2ead5ca1751ddc96e5f009eedff915` |
| `artifacts/phase2d2_d02a/states/j0_Q3.json` | 110833 | `6b75f3144b938d50fd8460a559adac49964d7e56748c1d4520b3a1d8e656f056` |
| `artifacts/phase2d2_d02a/states/j0_Q3.pt` | 147143707 | `f52023ae219426c78d003ba341f2195dbb2ead5ca1751ddc96e5f009eedff915` |
| `artifacts/phase2d2_d02a/states/j1_A0.json` | 109273 | `d22b677dccee8f211fc1d1765df31431762e017930c26de9db4a8e32a7f76948` |
| `artifacts/phase2d2_d02a/states/j1_A0.pt` | 147044655 | `cfb8d2aa15eee6f30e54e8feebca37064a0520f784113f3ebfe3e0b06b908f56` |
| `artifacts/phase2d2_d02a/states/j1_P3.json` | 110833 | `89298074e1526363275c3e79bf11dac1141961e983d0ec42cfc8137fe3116e46` |
| `artifacts/phase2d2_d02a/states/j1_P3.pt` | 147143707 | `a16bbf51014755a4b68bdbaa123e8bb106ba5ed002b583cff5887efe45afbcc5` |
| `artifacts/phase2d2_d02a/states/j1_Q3.json` | 110833 | `1710a5023135d570cc810db365b9b5ff525f11b1de13a9a0ec3a581b7b127e19` |
| `artifacts/phase2d2_d02a/states/j1_Q3.pt` | 147143707 | `a16bbf51014755a4b68bdbaa123e8bb106ba5ed002b583cff5887efe45afbcc5` |
| `artifacts/phase2d2_d02a/states/j2_A0.json` | 109273 | `929e163027bdd09306f2f0c1c7afd3f8e52abbf86d979d3db201c6b52e2c2148` |
| `artifacts/phase2d2_d02a/states/j2_A0.pt` | 147044655 | `45e858a4c40ab0be3ffc059277ee52f796bfcb6e76664b3d7d5b7798cb089980` |
| `artifacts/phase2d2_d02a/states/j2_P3.json` | 110833 | `fa9670486daf3476fc1aaaa4e7c2014a6c620b4388f8b7035350202af6965e01` |
| `artifacts/phase2d2_d02a/states/j2_P3.pt` | 147143707 | `dd944a2fd1ab88ba33156eb7093c339f4bd8638198ebabac82695ef6fd284bf9` |
| `artifacts/phase2d2_d02a/states/j2_Q3.json` | 110833 | `74632d6bd47951b5d46236a9abf957933892c054d31ccd93a4ac15f4666bf0d2` |
| `artifacts/phase2d2_d02a/states/j2_Q3.pt` | 147143707 | `dd944a2fd1ab88ba33156eb7093c339f4bd8638198ebabac82695ef6fd284bf9` |
| `artifacts/phase2d2_d02a_review_lite/attempts/with_paper/review_lite_manifest.json` | 777547 | `6f32c289207cc6deddb7215fe7577e12094b3483fd2ed3b219412581e37a7741` |
| `artifacts/phase2d2_d02a_review_lite/phase2d2_d02a_review_lite.zip` | 68758321 | `362efeffd231e8d62fbad6ef18dfafab41d9d516410172026dcbd99dd142eded` |
| `artifacts/phase2d2_d02a_review_lite/review_lite_manifest.json` | 793762 | `8d8b62fadbebf11e0b4ff3d8bffe2ea40c8b7cf652ea7bffa13bb7cdc62edb09` |
| `artifacts/phase2d2_d02a_review_lite/review_lite_verification.json` | 678684 | `18ff611d9cf578df5f0a1920d043b118d28ea8aef00a3836e6bdfeb92e56c2a8` |
| `artifacts/phase2d2_n31/heldout_preregistration.json` | 4320 | `7328fa21235112ac1dd081062efefefc1ebaa8c96ff9444ebdbedf7e703c7e72` |
| `docs/phase2d2_a1_ablation_preregistration_v1_1.md` | 23755 | `3577e592a3a4013d136c840093d1d843ac92481b2f4088e4caf66fed7b3f92a3` |
| `tests/foundation/fixtures/golden_tokens.json` | 114572 | `1e8950d386d765c5cd13e628e751356afe4ac11e35482701eadc7a696761d0e6` |
| `tools/phase2d2_fixtures.py` | 2796 | `c431242aa2fe96672082e42f2a5887db30798510029b2aa20538ddb003257d02` |

Static preflight metadata SHA-256: `7f28b1dba62b833e6a20494c85838f527b4a73bf6dc22942b7bf91a9beab2011`. This digest is of the new read-only record; it does not substitute for any frozen numerical-result hash.
