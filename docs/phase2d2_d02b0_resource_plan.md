# A1-D02B0 historical resource preflight and sequential plan

2026-10-15, Asia/Saigon. **STATIC PREFLIGHT ONLY. No model construction, fixture generation, forward/backward, test collection or experiment execution.** Resource arithmetic is DERIVED; additional reservations/retention are ASSUMED / PROPOSED operational choices requiring approval. No numerical tolerance changes. [Contract](phase2d2_d02b0_experiment_contract.md), [matrix](phase2d2_d02b0_fixture_matrix.md), [approval](phase2d2_d02b0_approval_record.md).

## Current read-only observation

At **2026-10-15 10:23:12 UTC+07**, Windows GlobalMemoryStatusEx reported 16,901,771,264 bytes physical RAM (15.741 GiB), **2,146,336,768 available bytes (1.999 GiB)**, 87% load. D: had **663,935,868,928 bytes free (618.338 GiB)**. See [static_preflight.json](../artifacts/phase2d2_d02b0/static_preflight.json). This is a single system observation, not a capacity experiment or reservation. Available RAM is below the existing **3 GiB** launch floor; no numerical worker should launch even if otherwise authorized. No request to free RAM is needed for this documentation-only phase.

Historical CPU-only package/environment records identify Windows 11, Python 3.14.3, torch 2.12.1+cpu, NumPy 2.4.4, psutil 7.2.2, pytest 9.1.1, PyYAML 6.0.3 and the pinned tokenizer dependencies. This phase read their records and source hashes; it did not import torch, reinstall packages or probe CUDA. GPU and B8 capacity remain **NOT VERIFIED / NOT RUN**, not PASS or invented OOM.

## Historical full-G2 evidence, not new measurements

Source: [complete_G2_summary.json](../artifacts/phase2d2_n31/complete_G2_summary.json), individual `g2/final_<case>/resources.json` and corresponding `logs/full_G2_<case>.execution.json`. All seven cells below are historical **A0**, all finite, all **FAIL_GRADIENT_ADDITIVITY**, worker exit **1**. The historical available-RAM field is ambiguously named `start_or_end` and was sampled at teardown in the old runner; it must not be represented as a launch gate. New records must timestamp start/end separately.

| Historical fixture | B | MKLDNN | Sampled peak RSS MiB | Worker seconds | Outer process seconds |
| --- | ---: | --- | ---: | ---: | ---: |
| real | 1 | on | 2,271.242 | 118.404 | 123.192 |
| efs | 1 | on | 2,280.844 | 152.891 | 157.941 |
| local | 1 | on | 2,267.281 | 121.591 | 126.056 |
| real_efs | 2 | on | 2,885.180 | 148.741 | 153.214 |
| real_local | 2 | on | 2,848.867 | 176.582 | 183.361 |
| local_duplicate | 2 | on | 2,878.789 | 153.576 | 159.709 |
| real_efs | 2 | off | 3,045.824 | 164.099 | 169.168 |

RSS sampling nominally every 5 ms is not a guaranteed peak. These times include model construction, seven full graphs, scalar/activation/gradient statistics, hashes and streaming; they are not pure model throughput. The original stream did not retain every isolated full gradient file or calculate every loss-pair cosine. D02B's proposed complete retention/interaction audit may be significantly slower. No P3/Q3 full-six-loss or NG2 capacity/timing has been measured.

D02A local workers peaked at 2.183 GiB sampled RSS and totaled 46,187.552 worker seconds across 414 jobs. Those are local LD/attention graphs with different copies/temporaries. They are **not** a full-model footprint or a way to estimate G2/NG2 throughput. Original interruption/RAM-guard incidents remain recorded; unknown termination causes remain unknown.

## Storage and activation arithmetic

| Quantity | A0 | P3 / Q3 |
| --- | ---: | ---: |
| One FP32 full parameter copy | 588.059357 MiB | 588.141388 MiB |
| One dense FP32 full gradient | 588.059357 MiB | 588.141388 MiB |
| Disk-backed FP64 sum | 1.148553 GiB | 1.148714 GiB |
| Six isolated FP32 gradients + joint FP32 gradient + FP64 sum (36 bytes per parameter) | 5.168490 GiB | 5.169211 GiB |
| Previous row plus one FP32 checkpoint | 5.742767 GiB | 5.743568 GiB |

These are raw element bytes, excluding headers, metadata, compressed-file overhead, memory allocator/workspace, OS cache and file-system overhead. No optimizer, momentum or moment-state allocation is authorized. LayerNorm adds 0.08203125 MiB of weights and the same gradient size; its full activation/autograd effect is not captured by that small parameter delta.

| One FP32 tensor / subtotal | B1 MiB | B2 MiB | B8 MiB (arithmetic only) |
| --- | ---: | ---: | ---: |
| Hidden `[B,308,512]` | 0.601563 | 1.203125 | 4.812500 |
| Attention `[B,8,308,308]` | 2.895020 | 5.790039 | 23.160156 |
| Vocabulary logits `[B,308,49408]` | 58.050781 | 116.101563 | 464.406250 |
| 21 additional LN outputs, counted once | 12.632813 | 25.265625 | 101.062500 |

Softmax/log-softmax, LR validation/diagnostics, saved attention activations, backward buffers and visual feature maps may create several copies. Do not multiply this table into a claimed peak. Full visual FP64 is unsupported; doubling numbers is not authorization to cast that graph. B8/GPU results remain NOT VERIFIED regardless of arithmetic.

## Proposed safe worker policy

Retain the existing **at least 3 GiB available RAM** floor. For full integration, that floor alone does not cover the observed near-3-GiB B2 process plus construction, diagnostics and host headroom. **Proposed capacity reservations**, not scientific acceptance thresholds: budget at least **5 GiB available for B1** and **6 GiB for B2**, and use the stricter applicable condition. These reservations conservatively allow historical RSS, temporary reference-copy/capture overhead and approximately 2 GiB host/measurement headroom; they are not measured bounds. They require approval as operational safeguards. Current 1.999 GiB fails even the old floor.

One coordinator, one child worker, one full working graph. Release the initialization source after exact parity checks and before forward. A short construction-time reference plus copy is permitted only under the reservation; never concurrently run three full variants. Stream gradient chunks of 262,144 elements; avoid simultaneously loading six full gradient vectors, full activation archives or vocabulary logits for all cases. Do not use lower precision, smaller batches, shortened token length or partial parameters to pass the resource gate.

Before each worker, record available/total RAM, disk free, current coordinator/worker identity, complete cell provenance and reservation result. Insufficient resources yields WAIT/NOT RUN, no scientific restart. Measure child and process-tree RSS plus system available RAM throughout the worker with actual sampling intervals; record baseline, start/end, sampled peak, process exit and time waiting separately. A resource collapse stops subsequent launches. An observed OOM or abnormal exit preserves partial files and blocks continuation pending diagnosis; do not automatically change backend or rerun under a smaller batch.

The scheduling design must avoid the D02A guard's process-startup race. Prefer a coordinator that waits **before child creation**, rather than suspending it while a child's main thread is not yet running. An exclusive lock must bind PID plus process creation identity, schedule hash and active cell. A stale PID alone is not proof of a live owner. On resume, verify completed cell seals; archive partial attempts byte-for-byte with checksums; only replay the interrupted identical cell under an explicitly approved recovery protocol. Compare original complete scientific prefix fields, excluding timing only; unexpected divergence blocks continuation. No automatic deletion of interrupted evidence.

## Retention and temporary disk plan (proposed)

Keep the canonical seven-pass isolated/joint FP32 parameter-gradient shards and the FP64 sum, with separate None/presence maps, shapes, offsets and SHA-256. This permits complete per-parameter and all-15-pair interaction review without keeping all arrays in RAM. Preserve a state checkpoint once per seed/variant, not once per backend cell; record strict state hashes for each use. Record timing-repeat gradient statistics/hashes without multiplying canonical vector retention threefold. Large activation arrays are unnecessary for the proposed stats-only capture; no post-hoc replacement of mandatory full gradient coverage by sampled layers.

Reserve **8 GiB temporary/free workspace per active cell** as a proposed operational allowance above the approximately 5.744 GiB raw checkpoint/gradient maximum, with larger reservation if actual measured growth requires it. Retention across cells matters separately: 18 primary cells need about 93.05 GiB raw gradient data plus approximately 1.72 GiB for three initial states; budget **110 GiB** including logs/headers. All 39 G2 cells need roughly 203.3 GiB plus overhead; budget **230 GiB**. All 57 cells with up to twelve seed/variant states need about 301.6 GiB plus overhead; budget **350 GiB**. No compression ratio is assumed. The observed 618.338 GiB disk free is arithmetically sufficient for those reservations at observation, **not a guarantee of later capacity**.

Existing D02A archives, caches and histories remain untouched. Disk pressure never authorizes deleting them. Any alternative retention policy, cleanup of newly generated gradient data, staging to another drive or reduced interaction coverage requires explicit review before execution. Do not silently remove raw files after reading them as the older temporary stream did. No data is allocated or cleaned in D02B0.

## Time estimate and gate result

The six historical C4M worker times sum to **871.784 seconds**. Multiplying by three variants gives **2,615.352 seconds / 43.59 minutes** for a naive seven-pass primary estimate. Scaling by 10/7 for three additional timing backwards gives **62.27 minutes**, before warm-up, new full-vector I/O, all-pair diagnostics, construction differences, host waits or failures. This is transparent arithmetic, not a performance prediction. A provisional **1–3 hour planning window** for the 18-cell primary partition may be exceeded and is not a timeout or acceptance condition. C1M speed is unknown; no 4x speed law is assumed. NG2 differs in initialized FLT and inputs, so no measured NG2 duration exists. Report actual durations later, preserving all slower cells.

| Device / batch / scope | Current finding |
| --- | --- |
| Historical A0 CPU B1/B2 | Executed historically, finite; mandatory numerical failure remains |
| New A0/P3/Q3 full integration CPU B1/B2 | **NOT RUN / capacity NOT VERIFIED; launch blocked by authorization and present RAM** |
| NG2 CPU B2 | **NOT RUN; separate execution approval and pre-forward hashes missing** |
| CPU B8 | **NOT VERIFIED / NOT RUN; outside proposed matrix** |
| GPU B1/B2/B8 | **NOT VERIFIED / NOT RUN; no new hardware probe or capacity claim** |
| Optimizer, G3/G4, training, benchmark evaluation | **UNAUTHORIZED / NOT RUN** |

Recommendation: review/approve the isolated runner design and primary 18-cell scope first; materialize/seal only that authorized schedule and re-observe RAM before any subsequent execution. Review results before backend controls, and again before NG2. No automatic promotion follows any resource or finite-output success.
