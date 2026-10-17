# Phase 2C: FLT and VIM proposed implementation contract

Design review, 2026-10-11. **No Phase 2C implementation is authorized. All new engineering decisions below are ASSUMED / PROPOSED.** Approval of research documentation, M1–M5, N1, or D01–D08 does not approve these choices. This document proposes a scratch-initialized language branch; it does not recover the authors' unpublished architecture.

The accepted visual boundary remains MVE 32,078,912 parameters plus bottleneck-only VD 9,055,365 parameters, totaling 41,134,277. The user reports independent acceptance of all 40 existing cases. No accepted source, tests, configurations, or numerical tolerances are changed here. In particular, N1 remains exclusive to the existing MVE-T12 cross-batch-size comparison.

## Evidence and classification

Primary source: [MFVLR.pdf](../papers/MFVLR.pdf), arXiv:2605.10071v1, SHA-256 `2afed0516c32121513463d326720153887586ceaa328f26b96263bd39da45307`. Page references are one-based PDF pages. The review revisited pp. 4–8, visually checked Figs. 5–6 and Eqs. (4)–(25) on pp. 6–8, and retains the full [Phase 1 equation ledger](paper_analysis.md). See also [VIM mathematical review](phase2c_vim_analysis.md) and [verification plan](phase2c_test_plan.md).

**EXPLICIT** means specified by MFVLR. **DERIVED** means a mathematical consequence, with any assumed premises identified. **ASSUMED** means an implementation selection absent from or ambiguous in MFVLR. External precedent does not turn an assumption into an explicit MFVLR requirement. Unless separately labeled EXPLICIT or DERIVED, every normative implementation detail in the proposed specifications below is ASSUMED / PROPOSED.

| Component | Explicit paper anchor | Missing implementation details |
|---|---|---|
| Training-only language | §III-A p. 4; Fig. 4 p. 5 | Module API and orchestration policy |
| Hierarchical prompts | §III-D p. 6; Fig. 6 | Complete generator spelling registry, punctuation, missing metadata policy |
| Token lookup and positions | §III-D p. 6; §IV-A p. 8 | Tokenizer artifact/version, packing, padding, truncation, initialization |
| LE | Eq. (3), p. 6 and block prose pp. 6–7 | Heads, FF internals, biases, dropout, LN epsilon, position initialization |
| VIM | Fig. 5 p. 6; Eqs. (4)–(11), p. 7 | Softmax axis, head count, biases, precise module residual boundary |
| LD | Eq. (12), Eqs. (13)–(16), p. 7 | Exact shift, attention masks and projections, FF internals, any unprinted norms |
| Vocabulary output | Eq. (24), p. 8 | Physical ownership/API; formula specifies reuse of the vocabulary matrix |
| Alignment/reconstruction consumers | Eqs. (19)–(25), p. 8 | Loss implementation ambiguities; outside this phase |

Relevant external primary evidence, checked during this review:

- MFVLR [21], [MFCLIP §III-D](https://arxiv.org/html/2409.09724v3#S3.SS4), explicitly concatenates four separately tokenized 77-position sentence embeddings into 308 positions. This supports the packing recommendation, but does not establish MFVLR masks, shifting, pooling, or weight initialization.
- MFVLR [34], official CLIP [tokenizer source](https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py) and [tokenize function](https://github.com/openai/CLIP/blob/main/clip/clip.py), provide the tokenization convention. These are tokenizer evidence only; no model weights, CLIP model architecture, or MFVLR implementation were used. The linked `main` sources are mutable; a commit and asset hashes remain an implementation prerequisite.

## Decision table: approval required

All rows are engineering assumptions even where part of the bundle implements an explicit paper operation. No row is APPROVED. “Blocked” below describes implementation readiness, not a substitute approval status.

| ID | Proposed baseline and approval scope | Alternatives and expected consequence | Prior assumptions | Status |
|---|---|---|---|---|
| L01 | Exact L1–L4 templates, explicit generator registry, strict metadata checks | Free-form names risk prompt drift; automatically guessing missing levels creates unsupported labels | A08 | PROPOSED |
| L02 | CLIP-compatible byte BPE, 49,408 entries, tokenizer-only pinned assets, no pretrained learned tensors | Another tokenizer changes IDs/word boundaries; copying CLIP weights violates scratch training | A08/A22 | PROPOSED |
| L03 | Four independently BOS/EOS-framed, right-padded 77-token segments concatenated to 308; reject overflow | One joined string with a single BOS/EOS uses different context and many fewer special/pad positions | A08 | PROPOSED |
| L04 | LE bidirectional across all 308 slots, no padding mask; global feature physical slot 307 | EOS pooling or masked padding is plausible, but changes the last-token interpretation and aggregation | A09/A11 | PROPOSED |
| L05 | LE: 12 independent blocks, LN → self-MHA + residual → VIM + residual → FF + residual; no extra norms | Stock pre-LN blocks add norms; one-affine FF is closer to a literal reading of “a fully connected layer” but less expressive | A03/A04 | PROPOSED |
| L06 | LD: seven independent blocks, causal self-MHA → full LE-memory MHA → FF → VIM, residual at each operation; no norms | Adding LD norms changes printed ordering; masking encoder memory does not remove leakage already mixed by a bidirectional LE | A03/A11 | PROPOSED |
| L07 | Global one-slot right shift of raw lookup embeddings; prepend existing BOS embedding; original IDs remain targets | Shifting each segment separately resets context at levels; dropping BOS targets changes task alignment | A10 | PROPOSED |
| L08 | Ordinary MHA: 8 heads, biased projections, conventional scale; FF 512→2048→512 exact GELU; dropout zero | Other heads/widths/nonlinearities or a single affine FF are unresolved author choices | A03/A16 | PROPOSED |
| L09 | 19 independent, bias-free VIMs; singleton visual K/V; published scale; key-axis softmax; delta-only API | Depth sharing lowers capacity/count; query-axis softmax or patch keys changes the mathematical model | A12/A13/A27/A29 | PROPOSED |
| L10 | One trainable vocabulary matrix shared by LE/LD lookups and bias-free output projection; independent learned Pe/Pd | Untied output adds 25,296,896 parameters and contradicts Eq. (24)'s reuse; freezing zero-padding row changes training | A04/A16 | PROPOSED |
| L11 | Exact scratch initialization and FP32 validation described below; no extra norms/clipping/normalization | Initialization affects an unnormalized deep LD; AMP/fused execution requires separate numerical verification | A22/A31 | PROPOSED |
| L12 | Tensor-only FLT interfaces, direct differentiable Iv input, no visual ownership or mutation, train-only orchestration | Silent detach breaks language-to-visual learning; automatic `.eval()` no-op prevents isolated language validation | A07/A35 | PROPOSED |
| L13 | Separately scoped strict language reference/debug configuration and full test/memory gates | Reusing unapproved language fields in the accepted MVE config could accidentally broaden approval | A30/A31/A35 | PROPOSED |

Work status: **BLOCKED pending explicit L01–L13 approval**. Tokenizer artifact pinning and independently generated token fixtures are also prerequisites for tokenizer acceptance; no hash/version is fabricated in this review. Full losses, adapter, classifier, dataset mapping/loading, optimizer, benchmark evaluation and training remain **DEFERRED**. Two-level FF++ packing, EOS pooling, extra norms, one-layer FF, shared-depth VIM, joined prompts and multi-token VIM remain unapproved ablation proposals. The existing visual approvals remain APPROVED only in their original scope.

## Prompt and tokenizer specification

### Prompt construction

**EXPLICIT:** §III-D and Fig. 6 p. 6 use authenticity, manipulation type, generator family and generator identity. Genuine faces repeat the same real-face sentence at all four levels. Proposed canonical strings below transcribe Fig. 6 with single ASCII spaces and no trailing full stop; precise whitespace/punctuation normalization is ASSUMED.

| Level | Genuine | Forged |
|---|---|---|
| L1 | `A photo of a real face` | `A photo of a fake face` |
| L2 | `A photo of a real face` | EFS: `A photo of an entire synthesized face`; FS: `A photo of an identity swapped face`; AM: `A photo of an attribute manipulated face` |
| L3 | `A photo of a real face` | `A photo generated by the diffusion-based model` or `A photo generated by the GAN-based model` |
| L4 | `A photo of a real face` | `The source generative model of this photo is {generator}` |

Fig. 6 supplies the following exact generator spellings and depicted combinations. The initial proposed registry admits these nine names only; validation against the table is an engineering policy, while the depicted labels themselves are EXPLICIT examples.

| Generator strings | L2 category | L3 family |
|---|---|---|
| DDPM, LatentDiffusion, CollaborativeDiffusion | EFS | diffusion |
| StyleGAN3 | EFS | GAN |
| DiffFace | FS | diffusion |
| FSLSD, FaceSwapper | FS | GAN |
| Diffae | AM | diffusion |
| LatentTransformer | AM | GAN |

Use this explicitly versioned registry rather than expanding paper table abbreviations heuristically. New generators require registry review. A missing/unknown fake type, family or generator raises an informative error. Real examples need only authenticity; unused fake metadata must not alter the four real strings. No LLM-generated prompts, model predictions, source-image processing or dataset loader belongs to this module. Dataset-specific aliases beyond Fig. 6 remain unapproved.

Proposed input is a sequence of typed per-image metadata records; output is an ordered tuple `(L1,L2,L3,L4)` per image. Preserve image order and a caller-provided sample identifier outside the learned tensor graph. The model cannot infer whether otherwise valid token IDs have been paired with the wrong image; that is a future data-pipeline invariant (§III-A p. 4).

### Tokenization, packing and truncation

The official tokenizer uses byte BPE, text repair, double HTML unescaping, whitespace cleanup and lowercasing. Its vocabulary construction gives 49,408 entries; the final two are BOS 49406 and EOS 49407. The official wrapper inserts them, zero-pads, and by default raises on overflow. Zero is an ordinary vocabulary entry used as padding by convention, not a separate PAD vocabulary symbol. See [SimpleTokenizer](https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py) and [tokenize](https://github.com/openai/CLIP/blob/main/clip/clip.py). Transferring these conventions to MFVLR is **ASSUMED**, even though MFVLR cites CLIP's tokenizer and explicitly states vocabulary size 49,408 (§III-D p. 6; §IV-A p. 8).

The proposed production boundary is CPU tokenization to `int64 [b,4,77]`, flattened in L1→L4 order to `int64 [b,308]`. Each segment is `[BOS, payload..., EOS, 0, ...]`; payload limit is 75 BPE tokens, not 75 words or characters. Return payload lengths, EOS indices and a padding-position mask as diagnostic metadata. Do not infer padding solely from token value zero, because that value is also a vocabulary entry. Special-marker literals in user-supplied generator strings are rejected by the registry.

**Truncation rule:** reference and debug both reject an overlength segment before model execution. No silent truncation is allowed. A future explicitly approved truncation variant could retain the first 75 payload tokens and then EOS, but is not part of this baseline. EOS is never dropped or moved by this baseline. Empty raw text, if tested at the tokenizer level, produces BOS/EOS followed by zeros; the prompt builder itself never emits an empty level.

| Packing choice | Representation | Consequences / recommendation |
|---|---|---|
| Four independent 77-token segments | Four BOS/EOS pairs, four padding regions, fixed level offsets 0/77/154/231 | **Recommended**, supported by MFCLIP precedent; does not mean four independent LE calls or block-diagonal attention |
| Joined text, one 308-token segment | Explicit separator needed; one BOS/EOS pair; payload allowance 306 | Different prompt boundaries, padding distribution, shift targets and absolute positions; still an engineering alternative, not proved wrong by `n=308` |

After concatenation, the LE attends across the complete 308-position sequence. Pe and Pd use absolute positions 0…307; no position reset at level boundaries, segment embedding, learned CLS or CLIP positional tensor is added. No key-padding masks are applied anywhere. The last physical LE slot is selected even if its input ID is zero. With unmasked bidirectional attention that slot can aggregate every level; it is not an EOS-pooling implementation. Extra masks, EOS selection or normalized pooling need separate approval.

Tokenizer dependencies must be pinned during an approved implementation: immutable official source commit, BPE gzip asset SHA-256 (and decompressed-content hash), byte-map/vocabulary hash, text-cleaning dependency versions and license/provenance. Validate 49,408 unique entries and special IDs before use. Obtain independent golden token arrays from the pinned official tokenizer without loading a CLIP model. Tokenizer vocabulary/merge data are allowed non-neural tokenization assets; all learned embeddings/positions/projections start from scratch. No automatic network access during a model forward.

## Exact proposed language architecture

### Dimensions, attention and ownership

**EXPLICIT (§IV-A p. 8):** `E=12`, `D=7`, `d=512`, `n=308`, `s=49408`, reference training batch 8. Batch is dynamic and positive; validation targets b=1,2,8. A tensor's leading dimension never participates in attention softmax.

**ASSUMED common primitives:** separate Q/K/V/output linear maps for ordinary MHA, each 512→512 with bias. Eight heads of width 64. Split `[b,n,512]` to `[b,8,n,64]`, compute logits `Q K^T / sqrt(64)`, apply any self-attention causal mask, softmax over final/key axis, multiply V, concatenate heads in their original order, project back to 512. No dropout, rotary positions, relative bias or learned attention scaling. The ordinary-attention scale must not replace VIM's different published scale. Reference explicit matrix operations precede any future fused-kernel option.

FF is two biased affine maps 512→2048→512 with exact GELU (`approximate='none'`) between them, no activation after the second and no dropout. MFVLR only specifies an FF network “with a fully connected layer”; this two-layer interpretation follows the earlier A03 proposal and common Transformer practice, **not an explicit MFVLR requirement**. A single 512→512 affine is a meaningful alternative, reducing the proposed language count by 34,904,064 parameters. Approval must cover this substantial choice; parameter-count matching cannot decide it.

Only the LE's explicitly described pre-self-attention LN is included: affine LayerNorm over the last 512 dimensions, epsilon `1e-5`, population variance. No pre-VIM/pre-FF LN, LD LN, post-block LN or final LN is introduced. An off-the-shelf TransformerEncoder/Decoder layer with default norms is not equivalent. Normalization omissions can amplify activation magnitude; measure them before proposing any revision.

### Vocabulary and positions

One owner registers `Wvoc [49408,512]`; lookup returns `Tlow [b,308,512]`. `padding_idx=None`: row zero is trainable. The LE lookup and shifted LD lookup refer to the same matrix. The output computes the linear map with that same parameter's transpose, with no copied parameter, independent output matrix, output bias, normalization or softmax (Eq. (24), p. 8). No embedding scale by `sqrt(d)` is applied. Pe and Pd are separate trainable `[1,308,512]` tensors broadcast over batch. BOS uses row 49406, not another learned parameter.

### Language Encoder: twelve blocks

The following is a batched executable interpretation of Eq. (3), the LE prose pp. 6–7 and Eqs. (4)–(11), with the delta-only VIM boundary justified in the companion analysis. `Delta_j` owns no residual addition.

1. `X_1 = lookup(ids) + Pe` → `[b,308,512]`.
2. For each independent block j=1…12:
   - `N_j = LN_j(X_j)` → `[b,308,512]`.
   - `S_j = X_j + SelfMHA_j(N_j,N_j,N_j)` → `[b,308,512]`; unmasked weights `[b,8,308,308]`.
   - `A_j = S_j + Delta_j(S_j,Iv)` → `[b,308,512]`; singleton weights `[b,8,308,1]`.
   - `F_j = GELU(FC1_j(A_j))` → `[b,308,2048]`.
   - `X_(j+1) = A_j + FC2_j(F_j)` → `[b,308,512]`.
3. `Thig_e = X_13 [b,308,512]`; `Tl = Thig_e[:,307:308,:] [b,1,512]`, without another projection, norm or EOS lookup.

There are 12 different LN/MHA/FF/VIM parameter sets. They are shared across samples and positions, not depths. Every VIM consumes the same differentiable Iv tensor; each learns its own projections. There is no connection to MVE Transformer weights or VD weights.

### Language Decoder: seven blocks

**EXPLICIT:** p. 7 describes shifting low-level embeddings with a begin token and preserving length; Eqs. (13)–(16) order masked MHA, LE-memory MHA, FF and VIM, each with residual addition. **ASSUMED:** interpret the ambiguous “appended” begin token as prepending at position zero for next-token prediction.

Define shifted IDs `z[0]=49406`, `z[t]=ids[t-1]` for `1≤t<308`; lookup is mathematically the shift of raw `Tlow`. Encoder positions must not be shifted/reused. `Y_1 = lookup(z) + Pd [b,308,512]`. Targets remain the original `ids [b,308]`; this review authorizes no loss or target-mask reduction.

The initial two LD inputs can both be BOS because each original sequence already starts with BOS. The shift is across the entire concatenation: LD input at index 77 is original index 76, not a newly inserted BOS. Analogously test 154 and 231. Do not remove intermediate EOS/BOS or padding. Original index 307 is omitted from inputs and remains a target. These consequences distinguish this proposal from segment-wise shifting.

For each independent block j=1…7:

1. `U_j = Y_j + CausalSelfMHA_j(Y_j,Y_j,Y_j)` → `[b,308,512]`. Score/probability `[b,8,308,308]`; permitted keys satisfy `k≤q`. Add zero on allowed entries and negative infinity on future entries before softmax. No row is entirely masked.
2. `V_j = U_j + MemoryMHA_j(U_j,Thig_e,Thig_e)` → `[b,308,512]`. Full unmasked memory length 308; separately projected queries, keys and values. No cached/detached memory.
3. `G_j = GELU(FC1_j(V_j))` → `[b,308,2048]`; `H_j = V_j + FC2_j(G_j)` → `[b,308,512]`.
4. `Y_(j+1) = H_j + Delta_j(H_j,Iv)` → `[b,308,512]`.

`Trec_d = Y_8`; tied projection produces raw logits `[b,308,49408]`. A seven-block composition is Eq. (12)'s result; it does not require recursive token generation. There are seven different pairs of ordinary attention modules, seven FFs and seven VIMs. LE and LD weights are independent except Wvoc. No LD normalization is specified by the paper's equations; none is proposed here.

**Important limitation, DERIVED under L04/L06:** causal decoder self-attention alone does not make this a causal language model. Thig_e contains the entire unshifted prompt, and decoder cross-attention sees all of it. Changing a future source token can change an earlier decoder output through LE memory. Do not claim leakage-free autoregressive generation, impose a full-model causality test, or silently causalize LE. Language reconstruction is a training-side, image-conditioned reconstruction task. This training access to labels is separate from the prohibited use of text at visual inference (§III-A p. 4).

## Initialization and interfaces

All choices in this section are **ASSUMED / PROPOSED**. Use a private CPU generator seed 0, independent of global RNG and visual-module initialization. Construct scratch weights in FP32; transfer devices only afterward. Wvoc, Pe and Pd use Normal(0, 0.02). Each ordinary-attention, FF and VIM matrix uses Xavier uniform gain 1 separately (not one concatenated QKV initializer); all existing linear biases are zero. VIM has no biases. LE LN weight is one and bias zero.

Deterministic initialization order: Wvoc, Pe, Pd; LE blocks 1…12, each LN constants, ordinary Q/K/V/O, VIM Q/K/V/O, FF1/FF2; LD blocks 1…7, each self Q/K/V/O, memory Q/K/V/O, FF1/FF2, VIM Q/K/V/O. Tied output is never reinitialized. Constructor internals must preserve caller RNG state, including any default module initialization. No pretrained tensor, residual scaling, clipping, norm stabilization or hidden reinitialization is permitted.

Proposed public interfaces (signatures are design notation, not implemented code):

| Interface | Inputs → output / invariant |
|---|---|
| `PromptBuilder.build(metadata)` | Typed records → four strings per image; no learned tensors |
| `TokenizerAdapter.encode_levels(prompts)` | `[b,4]` strings → immutable token record: IDs `[b,308]`, segment lengths/EOS positions/padding metadata |
| `VisionInjectionDelta.forward(language, visual)` | FP32 `[b,n,512]`, `[b,1,512]` → delta `[b,n,512]`; production FLT fixes n=308, isolated VIM can test other positive n |
| `FineGrainedLanguageTransformer.encode(ids, visual)` | `int64 [b,308]`, FP32 `[b,1,512]` → LE record (`low`, `encoder_hidden`, `global_language`) |
| `FineGrainedLanguageTransformer.decode(low, encoder_hidden, visual)` | Same-call tensors `[b,308,512]`, `[b,308,512]`, `[b,1,512]` → decoder hidden `[b,308,512]`; shift raw low embeddings and use this owner's BOS row |
| `FineGrainedLanguageTransformer.project_vocabulary(hidden)` | `[b,308,512]` → raw logits `[b,308,49408]` with Wvoc transpose |
| `FineGrainedLanguageTransformer.forward(ids, visual, *, return_logits=True)` | Record with encoder hidden, global language, decoder hidden, logits or explicit `None`; default computes complete Eq. (24) output |

Validate positive batch, exact dimensions, ID range 0…49407, required dtypes, matching batch/device and finite floating inputs before attention. Reject `[b,512]` or `[b,197,512]` as visual input instead of silently reshaping/pooling. Do not clamp finite Iv, L2-normalize it or copy it into a fresh detached tensor. Caller is responsible for same-sample/same-forward records; low-level tensor APIs cannot prove provenance. Records are per call; no persistent hidden-state cache or graph retention. FP64 copies are permitted only for mathematical unit tests, with matching inputs; production reference/debug remain FP32.

Integration uses the existing accepted interfaces unchanged:

1. `appearance = mve.encode_appearance(image)`.
2. `decoded = vd.decode_appearance(appearance)` (accepted Option B consumes bottleneck only).
3. `visual = mve(image, decoded.reconstruction, appearance_encoding=appearance)`.
4. On the **training-side language route only**, pass `visual.fused_global` directly to FLT with paired IDs.

No new visual adapter precedes VIM. Iv is the accepted sum `[b,1,512]`. Language gradients can reach both MVE roles and AD through the absolute reconstruction residual; no language path directly traverses the independent mask head. Per-module `.eval()` must still allow deterministic standalone FLT evaluation for tests; the future task-level inference path explicitly omits tokenization/FLT entirely. This phase will test that boundary with a test harness, not implement the classifier or a full training/inference system. Raw mask logits and fused visual features can be checked now; full detection performance cannot.

## Conditional parameter budget

**DERIVED from L03–L11, not an author checksum.** Count unique registered trainable elements; tied Wvoc once. Tokenization, masks, residual sums, activations and attention scaling add no parameters.

| Item | Derivation | Parameters |
|---|---|---:|
| Wvoc | 49408×512 | 25,296,896 |
| Pe and Pd | 2×308×512 | 315,392 |
| Ordinary MHA | 4×(512²+512) | 1,050,624 |
| FF | 512×2048+2048+2048×512+512 | 2,099,712 |
| VIM | 4×512² | 1,048,576 |
| One LE LN | 2×512 | 1,024 |
| One LE block | MHA+FF+VIM+LN | 4,199,936 |
| Twelve LE blocks | 12×4,199,936 | 50,399,232 |
| One LD block | 2×MHA+FF+VIM | 5,249,536 |
| Seven LD blocks | 7×5,249,536 | 36,746,752 |
| **FLT unique total** | Wvoc+Pe+Pd+LE+LD | **112,758,272** |
| All VIMs, already included | 19×1,048,576 | 19,922,944 |
| Singleton Q/K, already included | 19×2×512² | 9,961,472 |
| Accepted MVE+VD | Fixed accepted count | 41,134,277 |
| **Visual+proposed FLT** | Excludes adapter/classifier/loss parameters | **153,892,549** |

MFVLR reports 141.00M for its model (Tables IX/XV, pp. 14–15); this proposal already exceeds that before the omitted components. This is an unresolved fidelity risk, not permission to change accepted visual widths or tune hidden language widths to force a match. Likely-sensitive unknowns include FF depth/width, projection bias and VIM sharing; exact author architecture would help. Extra norms alone cannot explain the entire gap. Do not claim recovered architecture from any eventual count agreement.

## Memory and execution planning

**DERIVED estimates, not measured execution.** FP32 elements occupy four bytes; MiB = 2²⁰ bytes. Shapes follow §§III-D/IV-A and the proposed eight-head choices. Main dense attention includes 12 LE self maps + 7 LD self maps + 7 LD memory maps = 26; VIM has 19 singleton maps.

| One tensor or illustrative retained set | b=1 MiB | b=2 MiB | b=8 MiB |
|---|---:|---:|---:|
| Hidden `[b,308,512]` | 0.602 | 1.203 | 4.813 |
| FF hidden `[b,308,2048]` | 2.406 | 4.813 | 19.250 |
| Ordinary attention map `[b,8,308,308]` | 2.895 | 5.790 | 23.160 |
| VIM map `[b,8,308,1]` | 0.0094 | 0.0188 | 0.0752 |
| Full vocabulary logits `[b,308,49408]` | 58.051 | 116.102 | 464.406 |
| Illustrative activation subtotal: 52 ordinary maps + 19 FF hiddens + 19 block outputs | 207.689 | 415.379 | 1,661.516 |
| That subtotal + one logits tensor | 265.740 | 531.480 | 2,125.922 |

The illustrative subtotal is exactly `4b×(52×8×308² + 19×308×2048 + 19×308×512)` bytes: two ordinary maps (logits and probabilities) per attention, one FF hidden and one output per block. **VIM maps are not included**; both scores and probabilities for all 19 add 0.357/0.714/2.857 MiB at b=1/2/8. This subtotal is an allocation inventory under an explicit retention scenario, not a peak prediction, lower bound or exhaustive autograd count. Some listed buffers may be released or shared; Q/K/V, LN/GELU saved tensors, residual intermediates, visual activations, gradients, softmax workspaces and allocator fragmentation are omitted. No claim of fitting a GPU follows from it.

FLT weights alone occupy 430.139 MiB; weights plus same-dtype gradients 860.277 MiB. Hypothetical FP32 Adam weights+gradients+two moments occupy 1,720.555 MiB (no optimizer is implemented or authorized). Visual+FLT equivalents are 587.053 / 1,174.107 / 2,348.214 MiB. These parameter-state costs do not depend on batch. Full logits and a same-sized logits gradient can together require 116.102 / 232.203 / 928.813 MiB. Materializing one-hot floating targets would add another logits-sized tensor; propose integer target IDs for a future loss design, without implementing that loss here.

CPU b=1 and b=2 full-width/depth forward/backward tests must run in fresh processes to measure RSS and runtime separately. b=8 is a scheduled resource diagnostic, with explicit capacity reporting if unsupported, never a hidden unit-test skip. GPU diagnostics require a CUDA-enabled host: synchronize timing, reset/report peak allocated/reserved memory, record device, dtype and actual shape. Current accepted Phase 2B logs used a CPU-only PyTorch build; no new CPU/GPU execution or capacity measurement was performed for this design. Those visual-only measurements cannot predict a combined FLT peak.

Use `return_logits=False` only for clearly labeled feature-only diagnostics, never to claim the full vocabulary path passed. Do not instantiate an optimizer, use AMP/checkpointing/chunked vocabulary projection, reduce layers/width/sequence length or shrink b=8 to conceal memory failure. Such optimizations can be separately proposed after the reference path has numerical evidence.

## Configuration and loss boundaries

Proposed future files `language_reference.yaml` and `language_debug.yaml` have their own closed schema and scoped L01–L13 approval records; no files are created now. Reference/debug must resolve identical architecture, initialization and tokenizer assets. Only diagnostic batch/device/thread/log-output settings may differ. Reject duplicate/unknown keys, wrong scalar types (including booleans as integers), missing required fields, altered immutable topology, unsafe/recursive includes and stale source/asset hashes. Resolve approved one-level debug overlays relative to their own file; allow only documented diagnostic overrides. A debug overlay cannot self-grant approvals, relax tolerances, modify MVE/VD config or substitute another tokenizer. Frozen resolved values should be included in later evidence packages. The accepted visual loaders and snapshots remain untouched.

Eq. (19) consumes Tl and an independently adapted visual feature; Eqs. (20)–(23) consume Iv and Tl; Eq. (24) requires raw tied logits; Eq. (25) consumes token targets. Preserve those interfaces now. The exact unchanged equations and inconsistencies are documented in [the companion analysis](phase2c_vim_analysis.md). Do not implement an adapter, similarity normalization, learned temperature, KL, contrastive objective, language CE, classifier or full training pipeline under this contract.

## Approval summary and remaining risks

The recommendation is four independently framed 77-token segments processed jointly; 12 LE and seven LD blocks at width 512; eight heads; 2048-hidden GELU FF; only the explicitly placed LE LN; 19 independent singleton VIMs with one external residual each; and a tied scratch vocabulary matrix. Approval is requested for L01–L13 as disclosed engineering assumptions.

Unresolved risks are the parameter-count gap; singleton attention's zero Q/K task gradients; lack of genuinely word-selective VIM attention; potential activation growth without LD norms; last padded-token pooling; prompt/shift and padding-target ambiguity; future-token access through LE memory; and missing tokenizer asset pinning. Author clarification could settle topology/sharing/masks/residual ownership. MFCLIP and CLIP narrow packing/tokenizer choices but cannot settle the authors' learned architecture. No implementation should begin before explicit approval, and no later loss or training decision is implicitly approved by accepting these interfaces.
