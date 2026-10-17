# D02B9 Track B: usable reproduction interfaces

Status: research-only implementation, tested with synthetic in-memory data and tiny models. Production A0, six losses, initialization, existing tests and configurations are unchanged. Architecture choices remain engineering assumptions; P3/Q3 are not promoted. No training or accuracy claim follows from interface tests.

## Dataset and loader

`data.py` validates the complete manifest before selecting train/validation/test. Require stable sample IDs, subject-group IDs, source tensor hashes, mask hashes, prompt metadata hash, preprocessing contract hash, declared Real/Fake class, manipulation kind and supplied mask semantics. Reject duplicated sample IDs, cross-split subject groups and duplicate source bytes across splits. Subject labels are asserted provenance: the validator cannot discover mislabeled identities or detect all semantically duplicate images. A real data protocol must establish those labels externally.

Consume already prepared CPU float32 `[3,224,224]` RGB in [0,1] and int64 `[224,224]` binary masks. No image file decoding, resizing, face detection, normalization, augmentation, mask synthesis or prompt regeneration. Hash source tensor bytes using contiguous native float32 bytes, and mask bytes as int64; this schema does not claim those are original encoded image hashes. The explicit manifest JSON digest uses sorted keys and compact separators. The preprocessing and prompt hashes must later resolve to independently reviewed artifacts; mock hash strings are not real dataset approval.

Reuse D02B8 metadata semantics: Real=0/background, fake=1, supplied fully generated foreground or supplied local binary localization. These checks only reject contradictory supplied declarations; dataset mask creation/preprocessing remains unauthorized. Copy input tensors on ingestion and return copies to prevent accidental mutation. `loader` uses no workers, no shuffle, explicit generator seed and batches 1/2. This is a reproducible interface default, not an approved scientific sampling protocol. Loader outputs image, mask, label, sample ID and subject group; a future approved language-data adapter must supply corresponding validated prompts/tokens.

## Checkpoints

`checkpoint.py` uses a directory of numbered `.npy` tensor files plus JSON, never pickle, optimizer state or executable model objects. Record schema, A0 variant, mock/research scope, update_count=0, numerical FAIL/NOT RUN, and source/config/dataset/tokenizer/loss-contract hashes. Include every state key, tensor shape/dtype, file size and SHA256. Full-model roster validation remains required by a future caller; tiny model checkpoints do not pretend to contain 508 tensors.

Loading requires a separately trusted manifest hash and positive byte budget. Validate paths, reject symlinks, excessive manifest size, unknown fields, missing/extra keys, size/hash/layout mismatches, object arrays and nonfinite values before modifying the model. `np.load(allow_pickle=False, mmap_mode='r')` prevents pickle execution. Preserve Parameter objects with `load_state_dict(assign=False)` and reject contradictory tied values. No unsafe fallback. Cryptographic hashes establish integrity only when the expected digest comes from a trusted source; they do not authenticate an untrusted author. Load untrusted files in a resource-limited isolated process after review; no format provides a denial-of-service guarantee. Concurrent file mutation during loading is outside this local immutable-directory contract.

## Trainer and inference

`trainer.py` reuses the frozen strict D02B8 metadata schema in separately scoped D02B9 examples. Unknown fields, enabled execution, optimizers, changed loss order/weights or unsupported modes fail. `batches` binds the actual dataset manifest digest to config; `fit`, `step` and `backward` always raise. No optimizer, scheduler, training graph or loss reimplementation exists here. All-zero hashes in example configs are deliberate mock placeholders and do not authorize real data use.

`inference.py` accepts a caller-owned visual module and image tensor, calls `visual_only` when available or its visual forward otherwise, and does not accept tokens. Returns raw class `[B,2]` logits and softmax over Real=0/Fake=1; mask `[B,2,224,224]` logits and channel softmax background=0/foreground=1. No threshold, morphology or benchmark metric. Use inference mode and temporarily eval; restore each module's prior training flag even on error. Tests use a tiny output-compatible module only. The real ImageOnlyMFVLR route has not been executed in this phase. CPU B1/B2 interface coverage is not full-model capacity evidence.

## Boundaries

No real dataset, preprocessing, optimizer update, training, MFVLR forward/backward or accuracy evaluation was executed. Practical next work requires dataset protocol approval, a separately authorized bounded optimizer smoke test after numerical readiness review, and a reviewed training handoff. Six original losses and historical failures remain intact.
