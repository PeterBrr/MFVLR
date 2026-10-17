# A1-D02B0 fixture and execution matrix

2026-10-15, Asia/Saigon. **PROPOSED / NOT RUN.** Design is authorized; execution is not. This is a partition of the corrected A1.1 schedule for separate approvals, not an extension of D02A execution authority. All numerical definitions remain frozen. [Contract](phase2d2_d02b0_experiment_contract.md), [gates](phase2d2_d02b0_numerical_gates.md), [approval record](phase2d2_d02b0_approval_record.md).

## Existing G2 anchors

Use [phase2d2_fixtures.py](../tools/phase2d2_fixtures.py) exactly, SHA-256 `c431242aa2fe96672082e42f2a5887db30798510029b2aa20538ddb003257d02`. Golden token JSON SHA-256 `1e8950d386d765c5cd13e628e751356afe4ac11e35482701eadc7a696761d0e6`. Hashes below are copied from historical N3.1 `g2/final_<case>/summary.json` records, whose file bytes were inspected/hashed in static preflight. No image/token/mask tensor was regenerated now.

| Fixture | B | Ordered kinds | Image seeds | Golden indices | Detection labels | Mask foreground pixels per sample |
| --- | ---: | --- | --- | --- | --- | --- |
| real | 1 | real | 17 | 0 | 0 | 0 |
| efs | 1 | efs | 412 | 1 | 1 | 50,176 |
| local | 1 | local | 997 | 5 | 1 | 9,216 |
| real_efs | 2 | real, efs | 17, 412 | 0, 1 | 0, 1 | 0, 50,176 |
| real_local | 2 | real, local | 17, 997 | 0, 5 | 0, 1 | 0, 9,216 |
| local_duplicate | 2 | local, local | 997, 412 | 5, 5 | 1, 1 | 9,216, 9,216 |

Images are FP32 `[B,3,224,224]`, generated per sample as `.1+.8*rand` with private stated seeds. Local images invert RGB in `[64:160,80:176]`; their oracle masks cover the same 96x96 rectangle. Real masks are zero; EFS masks are all one. These are supplied synthetic targets, not dataset-mask preprocessing. IDs/masks/labels are int64, token IDs `[B,308]`, masks `[B,224,224]`, labels `[B]`. Sample IDs retain `<case>:<row>:<kind>:<seed>` and must align with targets. Batch-independent image bytes do not imply B1/B2 loss equality: CMC changes with B. Duplicate prompt tokens do not ensure identical encoder features because each sample receives its own visual injection.

## Registered NG2 recipes: held behind a separate execution gate

| Fixture | B | Visual / VD / heads seeds | FLT seed | Image seeds in order | Kinds | Golden indices | Labels |
| --- | ---: | --- | ---: | --- | --- | --- | --- |
| NG2-0 | 2 | 0 / 0 / 0 | 6101 | 7101, 7201 | real, efs | 0, 1 | 0, 1 |
| NG2-1 | 2 | 0 / 0 / 0 | 6102 | 7102, 7202 | real, local | 0, 5 | 0, 1 |
| NG2-2 | 2 | 0 / 0 / 0 | 6103 | 7103, 7203 | local, local | 5, 5 | 1, 1 |

The image formula, rectangle, masks, loss reductions and temperatures remain those of A1.1. Source: corrected preregistration, “New independent fixtures”; static `fixture_spec` in [research launcher](../tools/research_phase2d2_a11.py). Do not invoke that launcher in D02B0. Suggested new sample IDs `NG2-j:<row>:<kind>:<seed>` are **PROPOSED bookkeeping**, not a changed tensor recipe.

**NG2 image/whole-model state hashes: NOT MATERIALIZED / BLOCKED before execution.** This phase neither generates new fixtures nor claims that a seed is a tensor hash. Before the separately authorized NG2 gate, a reviewed materialization-only stage must pin exact image, labels, masks, token IDs, sample order, full FP32 state and every parameter hash before any corresponding forward. Existing token/mask values are prescribed by kinds/indices, but their future record must still contain actual byte hashes. P3 and Q3 can have identical serialized parameter bytes and different functions; variant/config identity must be hashed separately. The nine D02A local checkpoint hashes remain in the approval record; they cannot supply a missing full-model hash.

## Backend identities and fixed ordering

All proposed cells use CPU FP32, deterministic algorithms, `matmul_precision=highest`, explicit unfused attention, no autocast/TF32/GPU substitution. Pin Python 3.14.3, torch 2.12.1+cpu and the recorded Windows environment; dependencies/tokenizer versions remain pinned. Record actual inter-op threads=4, ordinary BLAS/OpenMP settings and builds. Set the backend before any model operation. A changed version/host is a different unapproved cohort, not a quiet replacement of a failed cell.

| Backend ID | Intra-op | Inter-op | MKLDNN | Use |
| --- | ---: | ---: | --- | --- |
| C4M | 4 | 4 | on | Primary G2 and first NG2 cohort |
| C4N | 4 | 4 | off | Paired backend controls; includes historical real_efs native anchor |
| C1M | 1 | 4 | on | Preregistered real_efs reduction-order control only |

Each table row below expands into **three distinct, sequential cells**, variant order **A0, P3, Q3**, no concurrent full models. Cell ID is `D02B-<row>-<variant>`. Frozen cell identity includes fixture, complete seeds, B, dtype, backend, seven-objective order, protocol/config/source/input/state hashes. Repetition suffixes are scientific `FD/LR/CMC/FL/AR/KL/total`, warm-up `W`, timing `T1/T2/T3`; timing repeats never inflate fixture counts.

| Row ID | Fixture | B | Backend | Expanded cells | Execution gate |
| --- | --- | ---: | --- | ---: | --- |
| G01 | real | 1 | C4M | 3 | G2-primary |
| G02 | efs | 1 | C4M | 3 | G2-primary |
| G03 | local | 1 | C4M | 3 | G2-primary |
| G04 | real_efs | 2 | C4M | 3 | G2-primary |
| G05 | real_local | 2 | C4M | 3 | G2-primary |
| G06 | local_duplicate | 2 | C4M | 3 | G2-primary |
| C01 | real | 1 | C4N | 3 | G2-controls |
| C02 | efs | 1 | C4N | 3 | G2-controls |
| C03 | local | 1 | C4N | 3 | G2-controls |
| C04 | real_efs | 2 | C4N | 3 | G2-controls |
| C05 | real_local | 2 | C4N | 3 | G2-controls |
| C06 | local_duplicate | 2 | C4N | 3 | G2-controls |
| C07 | real_efs | 2 | C1M | 3 | G2-controls |
| N01 | NG2-0 | 2 | C4M | 3 | NG2, separate approval after G2 review |
| N02 | NG2-1 | 2 | C4M | 3 | NG2, separate approval after G2 review |
| N03 | NG2-2 | 2 | C4M | 3 | NG2, separate approval after G2 review |
| N04 | NG2-0 | 2 | C4N | 3 | NG2, separate approval after G2 review |
| N05 | NG2-1 | 2 | C4N | 3 | NG2, separate approval after G2 review |
| N06 | NG2-2 | 2 | C4N | 3 | NG2, separate approval after G2 review |

Recommended first future authorization: **G01–G06 only, 18 cells**. Backend controls are another 21 cells, NG2 another 18: **57 cells total if every partition is later explicitly authorized**. The broader on/off pairing and partition boundaries are proposed scheduling choices consistent with A1.1, not an already authorized 57-cell run. Do not expand C1M to all fixtures or add seeds/cuts/stress factors. N2/local/N3 reruns are not bundled into this full-graph matrix; their original requirements/evidence remain separate, not waived.

Each cell entails seven scientific backward passes, one no-grad warm-up, and three measured joint backward passes under the contract: **10 backward / 11 forward calls**. Therefore the primary partition is 126 scientific +54 timing backwards and 198 forwards; all partitions would be 399 scientific +171 timing backwards and 627 forwards. Counts exclude construction, metadata, guard checks and any separately authorized isolated primitive verification. No such call has run in D02B0.

G2 must precede NG2. NG2 needs a submitted G2 table with every failure/worsening cell, complete provenance and resource records, successful mandatory structural/execution checks, explicit acknowledgment of unresolved numerical failures, and **new user approval**. A finite failed G2 comparison does not automatically authorize NG2 or become a numerical PASS. Hard stops block dependent cells; no replacing a difficult fixture with another kind or seed.

## Historical G2 tensor fingerprints

The following appendix is generated by reading existing JSON metadata, not by invoking fixture generation. Byte order is contiguous CPU little-endian raw array bytes; shape/dtype/field name are recorded separately. Labels and sample order are fixed in the table above; historical provenance did not provide a separate labels digest, so a future pre-forward manifest must add it without changing target values.

| Fixture | Tensor | SHA-256 |
| --- | --- | --- |
| real | image_sha256 | `40e42f2ce46907f2a78718d33e6b8a4c37b9d667760e837b006457fc95f1bde8` |
| real | token_ids_sha256 | `27ed7daf953d91a50b04f64ad8bf1307ec184ed73a2dcdd793d9fd90936e0afa` |
| real | mask_sha256 | `daede069d703460dc1275e43c772caa2a215db963efcbfa2dc4eef610aab241d` |
| efs | image_sha256 | `ed4a448429d28ea6f379fa4a2f7a23bfb046f1f3cc8437eef7354bd84b04e76b` |
| efs | token_ids_sha256 | `cec14db7480d205fb4d34cb6d5d4393d8a548543acf5389a46f716f3fc519149` |
| efs | mask_sha256 | `1022233b804b94240d9431605f020464820f5919e6c7fc44d966a1311e6bab89` |
| local | image_sha256 | `9c60121c60f218b04d5835c71a6ec1c3eb730152f414eeefcfab4b2241b74f19` |
| local | token_ids_sha256 | `feb5d891525bf4f6f67a4fd773fc8ff860a6af94cd8a7052280cd29b823ab481` |
| local | mask_sha256 | `a0d0a26b8b697d4fa90d78780d2ef263d33a6b459296261e6a2399c5c123fcfa` |
| real_efs | image_sha256 | `7fb3134a2753af5afb9841c916d562eb6f2506773a06fc6661573ca21b75fa43` |
| real_efs | token_ids_sha256 | `8edd4b52d12e58b21312ac892f411236553aee040d134c4be73d94e3eb281612` |
| real_efs | mask_sha256 | `c2184962a3fb30879c3baf061f32ab3840716e954c64fda7da45738365de24b0` |
| real_local | image_sha256 | `af25fa0053bafd2f5ef276157bf9d252a94dd055d90b7ac6e20364eb64ae837f` |
| real_local | token_ids_sha256 | `bb89db14e855b75662a0452ef6cb547310f4c438833a890f466c64885ba34871` |
| real_local | mask_sha256 | `186898db30057ab316ee96847c92891a1fffe778bcac3d7aaa7568b3ba602e3c` |
| local_duplicate | image_sha256 | `fa0e71c2ebf648d7fb858cb98cdb969af2dbe98e96ac4b83d3695f35a889d23d` |
| local_duplicate | token_ids_sha256 | `c70168135dd01498264be8218a32ff1fbe2c2aa7d771003c8805fe6fb8268457` |
| local_duplicate | mask_sha256 | `a4c4d5a4fe9c8ddf4544f739938c687eaaa35314c55b7f7cc7910ff4c9af0630` |

All three variants and backend controls for a fixture must match these exact bytes before forward. `real_efs_native` reuses the same real_efs hashes, not a separate randomized fixture. Historical record file hashes:

| Historical summary path | SHA-256 |
| --- | --- |
| `artifacts/phase2d2_n31/g2/final_real/summary.json` | `360eec4a082ec1e4e00c7bb76a6b023f67a140dd6bcfb3b8acb63316bc141dea` |
| `artifacts/phase2d2_n31/g2/final_efs/summary.json` | `3eb4ba8e79284278fc17ef158eca29f007e6488d05e9181984075287e764ecf6` |
| `artifacts/phase2d2_n31/g2/final_local/summary.json` | `db82b74875089285cd74157119288ed36049a13bdb8fcbd9afb964a527c0ca92` |
| `artifacts/phase2d2_n31/g2/final_real_efs/summary.json` | `19b9f4b37800b3769381fa5695f1c2d2fa8afd5fd5f0c491ad4e154246247a13` |
| `artifacts/phase2d2_n31/g2/final_real_local/summary.json` | `f17dd960730f06e200ca9c866aae221e0e6b5d324750d67db63b6d2522ba566d` |
| `artifacts/phase2d2_n31/g2/final_local_duplicate/summary.json` | `edf46f2004f98634c986456ab2fa3813e51ddac73cf2b3b1a0e871cad0b354b6` |
| `artifacts/phase2d2_n31/g2/final_real_efs_native/summary.json` | `e15c8f97c3ca1afb5042846a2f84edaab15a5a8e39983fdf789f90900dfe678b` |
