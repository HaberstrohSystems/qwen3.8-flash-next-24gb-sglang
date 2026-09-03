# [Feature] Serve Qwen3.8-Flash-Next (176B / 6B-active MoE) at 2.57 bpw on one 24 GB GPU with 32 GB host RAM

## Motivation

### What this is

Qwen3.8-Flash-Next (architecture `Qwen4-Exp`: 176B total / 6B active parameters, 48 layers =
36 GatedDeltaNet + 12 Qwen sparse attention (QSA), 512 experts with top-10 routing, a 51 GB PLE
n-gram embedding table) was quantized with AutoRound to 2-bit experts with group size 128, INT8
dense projections and 16-bit routers and norms, 2.572 bits per weight overall (model card,
"Quantization"). It is served through SGLang on one RTX PRO 4000 Blackwell (24 GB, compute
capability 12.0) with 32 GB of host RAM at the full 262,144-token context.

The SGLang side is a series of five commits against the branch `qwen4-main-squashed` (PR #36497,
"Introduce Qwen 3.8 Flash Next"). The base commit of that branch cannot load a 2-bit MoE at all
(`moe_wna16` accepts 4 and 8 bits only) and its `--cpu-offload-gb` path fails on Qwen4-Exp at
weight loading, in `functional_call` and with a silent NaN in the GDN convolution (companion
repository, `docs/WRITEUP.md` section 3, "Nine findings in SGLang"). The first working path
decoded at 15.5 tok/s (`docs/WRITEUP.md` section 5); the series brings the same machine to the
numbers below.

This issue asks the maintainers (a) whether the series is wanted on `qwen4-main-squashed`, and
(b) for decisions on the open questions at the end, before the PRs are polished further.

### What exists and is measured

All speed numbers are single-request streaming measurements of 200 generated tokens at about 10k
tokens of context (`tools/bench_speed.py` in the companion repository), taken with the flat patch
on base commit `73a255206f`; the series is a cleaned split of that patch. Sources: dated entries
of the append-only engineering log `docs/CAMPAIGN.md` in the companion repository ("CAMPAIGN
<date> <time>"), `docs/logs/tiers-validate.log` there, and the model card on the Hub ("card,
<section>").

| Quantity | Value | Source |
|---|---|---|
| Decode at 101 / 421 / 1,701 / 6,821 / 10,001 tokens of context | 56.2 / 54.3 / 56.8 / 55.7 / 54.5 tok/s | CAMPAIGN 2026-09-02 21:05; `tiers-validate.log` |
| Prefill, 10,001-token prompt | 2,271 tok/s | same |
| Context | 262,144 tokens admitted (366,976 profiled by the lazy backing); a 257,905-token prompt prefills in 165.3 s (1,560 tok/s) and decodes at 52.3 tok/s | CAMPAIGN 2026-09-02 21:05 and 2026-09-03 00:55 |
| Needle retrieval | 5/5 codes at 41,370 tokens and 5/5 at 247,629 tokens | CAMPAIGN 2026-09-02 21:05 |
| KV cache, default mode `int8ring_int4` | 7,308 B per token at 256k over the 12 QSA layers (bf16: 24 KB) | `tiered_kv_pool.py` docstring; CAMPAIGN 2026-09-02 21:05 |
| Long-text quality of the default KV mode vs a bf16 KV cache | NLL delta -0.0001 nats/token, mean abs. dlogprob 0.074 over 8,561 cache-reading positions; bf16-vs-bf16 noise on the same test +0.0002 / 0.059 | CAMPAIGN 2026-09-02 21:05 and 18:48 |
| Expert cache at its floor | S = 184 experts per layer resident, arena 11.06 GB, 84.3 % of routing mass served from VRAM | CAMPAIGN 2026-09-02 14:20 |
| PCIe traffic per decoded token | 26 GB (stock offloader) -> 0.31 GB (expert streaming) | `docs/WRITEUP.md` section 5 |
| Speed ladder on the same machine | 15.4 -> 19.4 (offload fixes) -> 21.8 (N-contiguous GEMV) -> 40.0 (breakable decode graphs) -> 48.4 (routing-mass placement) -> 56.2 tok/s (elastic expert cache) | CAMPAIGN 2026-09-02 09:30, 11:30, 11:48, 12:10, 13:17 |

Quality evidence is an internal protocol (short held-out NLL, long-text NLL ladder against a bf16
KV cache, teacher-forced logprob oracle on a 10k prompt with a noise floor of mean 0.002, needle
retrieval; card, "Serving fidelity"). No standard benchmark score is published for the 2-bit
checkpoint; the series does not claim one.

### Hardware and software

One RTX PRO 4000 Blackwell (24 GB, sm_120), 32 GB host RAM, NVMe for the PLE table. torch
2.13.0+cu130, triton 3.7.1, flash-attn 2.8.3.post1 built for sm_120, transformers 5.12.1 (pinned
by SGLang at `73a255206f`), nvcc 13.3 from the `nvidia-cuda-nvcc` wheel inside the virtualenv
(companion repository, `README.md`, "Quick start"); flashinfer 0.6.17 is the version installed in
that virtualenv, relevant only on the rebased branch (open question 3). Host RAM is the binding
constraint: 31 offloaded layers pin 24-26 GB of the 32 GB, so nothing may pin memory at runtime,
and the published configuration serves one request at a time (`--max-running-requests 1
--max-mamba-cache-size 1`; CAMPAIGN 2026-09-02 12:49; card, "Scope and limitations").

### The five PRs

The series is `upstream/series-q4head/0001..0005` in the companion repository (rebased on the
head of `qwen4-main-squashed`, `78c5024e9d`; 39 files, +8,042 / -94). Each part carries a commit
body with the PR template's sections and sourced numbers; the PR descriptions are
`upstream/PR-1.md` to `PR-5.md`.

**PR 1 -- `fix(qwen4): CPU-offload correctness, breakable graphs, mmap PLE table`** (9 files,
+423 / -33). General: `GemmaRMSNorm` re-derives `gemma_weight` on the parameter's device;
`functional_call(..., tie_weights=False)` in the offloader, which also keeps a hard reference to
the pinned storage and can restrict itself to expert tensors; `RadixLinearAttention.conv_weights`
becomes a property so the GDN convolution never reads a stale `.view()` of a parameter whose
`.data` the offloader replaced (the silent-NaN finding); the breakable CUDA graph backend
understands `LogitsProcessorOutput`; the fp8 KV write path saturates to +-448 before the e4m3
cast. Qwen4-Exp-specific: `qwen_sparse_attention` as a layer type in the model,
`packed_modules_mapping` propagated to the quant config, `Qwen4ExpForConditionalGeneration` in
`LANGUAGE_MODEL_ONLY_ARCHITECTURES`, the PLE table served from a memory-mapped file with parallel
`pread`, `MADV_RANDOM` / `POSIX_FADV_RANDOM` / `POSIX_FADV_DONTNEED`, and the PLE lookup wrapped
as the eager break of breakable decode graphs; a hoist of the `positions.max().item()` sync in the
QSA indexer. Decode 21.8 -> 40.0 tok/s from the breakable graphs alone (CAMPAIGN 2026-09-02
11:48). No unit tests in this part.

**PR 2 -- `feat(moe_wna16): 2-bit experts, expert streaming, N-contiguous GEMV`** (9 files,
+1,583 / -33). General for any symmetric 2-bit GPTQ/AutoRound MoE checkpoint: the 2-bit unpack in
`fused_moe_kernel_gptq_awq` (zero point 2), a word-load variant of the kernel for an N-contiguous
`[E, K/16, N]` int32 layout selected by tensor dtype, `use_int2_w2a16` plumbing, the 2-bit loader
(asymmetric `qzeros` refused by design), MoE-aware expert streaming (only the routed experts cross
PCIe, one shared staging buffer), a batch-1 in-place GEMV through int64 address tables (320 GB/s
from device memory, 51 GB/s from pinned host memory; CAMPAIGN 2026-09-02 08:29-08:31, 09:45),
and a routing-mass placement pass (hottest S experts of every layer resident; the top 171 of 512
experts per layer cover 82 % of routing mass on a 3-domain probe, CAMPAIGN 2026-09-02
09:48-09:53). Nothing in it is Qwen4-specific. Registered unit test with 7 tests / 34 subtests.
Decode 19.4 -> 21.8 tok/s and prefill 1,443 -> 2,249 tok/s from the layout and GEMV; 40.0 -> 48.4
tok/s from placement (CAMPAIGN 2026-09-02 11:30, 12:10).

**PR 3 -- `feat(mem_cache): elastic VMM expert row arenas and lazy KV backing`** (8 files,
+1,070). General: `RowArena` (CUDA VMM address space reserved once, 4 MiB chunks aligned to the
2 MiB device granularity, mapped for a rank-ordered prefix, tail unmap returns VRAM to the driver
while every row address stays fixed, so CUDA graphs need no recapture), `ExpertElastic` (one
arena per layer and tensor kind in
routing-mass order, grow and shrink as table rewrites, a control file for live resizing), and lazy
KV backing on the existing `KvVmmBufferOwner` (address space reserved for `max_total_tokens`,
backed in 2,048-token steps as pages are handed out, released to a 4,096-token floor at pool idle,
a driver-free headroom watermark that shrinks the expert cache first, an admission cap). Wired to
the `moe_wna16` word layout of PR 2, otherwise model-agnostic. Registered unit test for
`RowArena` (3 tests). Exact by construction (oracle max 0.060 / mean 0.0016, CAMPAIGN 2026-09-02
13:17); 56.2 tok/s decode; KV committed at startup 0.8 GB -> 0.1 GB (`docs/ELASTIC_MEMORY.md`,
"Result").

**PR 4 -- `feat(qsa): quantized KV pools with dequant-on-gather (fp8/int8/int4)`** (14 files,
+4,424 / -25). General: three `--kv-cache-dtype` values `int8_g64` (12.4 KB/token), `int4_g32`
(6.75 KB/token) and the tiered `int8ring_int4` (an int8 ring of 8,192 slots with an owner table
over the int4 pool; 7,308 B/token at 256k), implemented as `MHATokenToKVPool` subclasses whose
payload and fp16 group scales are extra `KvBufferDesc`s on the lazy VMM owner, with fused
quantize + scatter at write. Qwen4-Exp-specific: the read path. QSA has exactly two places where
every K/V row is gathered anyway (the decode/verify compaction into the attention scratch and the
prefix-chunk row gather), so dequantization happens there and no attention kernel is touched; the
fp8 read path uses the same sites. Formats chosen from measured K/V statistics of the model
(simulated relative RMS error e4m3 2.66 %, int8 per-token 1.3 %, int8-g64 0.9 %; CAMPAIGN
2026-09-02 15:50). Four registered unit tests (41 tests, all kernels bit-exact against torch
references). Long-text NLL vs bf16: int8-g64 +0.001, int4-g32 +0.0088, tiered -0.0001 (CAMPAIGN
2026-09-02 21:05, 19:40).

**PR 5 -- `fix(spec): commit PLE state after ReplaySSM verify, NGRAM on Qwen4-Exp`** (4 files,
+542 / -3). General bug fix: `commit_mamba_states_after_verify` returns early in both ReplaySSM
branches before the PLE n-gram history and short-conv state are rolled to the last accepted node,
so the PLE history freezes after a verify step; this affects MTP with top-k 1 under
`--enable-linear-replayssm-spec` as well (CAMPAIGN 2026-09-02 17:08). General for hybrid
backends: `_linearize_chain` in the NGRAM worker collapses the corpus's star into a chain.
Qwen4-Exp-specific: the NGRAM guard in `_prepare_ple_batch` is dropped and a startup self-check
added. Registered CPU unit test (8 tests). Lossless within the decode-vs-prefill floor, but no
speed-up on this model (mean accept length 1.11-1.27; CAMPAIGN 2026-09-02 18:25, 18:50), hence
opt-in.

### Open questions for the maintainers

1. **Configuration surface.** Every switch of the series is an environment variable
   (`SGLANG_MOE_EXPERT_STREAM`, `SGLANG_QWEN4_PLE_MMAP`, `SGLANG_MOE_NCONTIG`, `SGLANG_MOE_GEMV`,
   `SGLANG_MOE_PLACEMENT`, `SGLANG_MOE_PLACEMENT_S`, `SGLANG_MOE_ELASTIC*`, `SGLANG_KV_LAZY*`,
   `SGLANG_KV_TIERS_W`), except the three new `--kv-cache-dtype` values. Which of these should
   become server arguments before review, and is a `--ple-mmap-dir` style argument acceptable for
   the PLE file? The control file `SGLANG_MOE_ELASTIC_CTL` would become an HTTP endpoint.
2. **Generic dispatch for quantized pools.** The int8 / int4 / tiered read path exists only in
   `qwen_sparse_attn_backend`; any other GPU backend that meets one of these pools would read raw
   bytes, and the CPU fallback raises `NotImplementedError`. Preferred shape: a `kv_bits`-aware
   dequant-to-bf16 materialisation on the pool interface that every backend can call, or a guard
   in the KV cache configurator that rejects the new dtypes for backends without a read path?
   Also, `kv_bits` / `kv_tiered` are class attributes read through `getattr` today, and
   `TORCH_DTYPE_TO_KV_CACHE_STR` maps `torch.uint8` to `int4_g32`, which the tiered mode shares.
3. **SM120 decode routing on the current branch head.** Since #36806 (`is_sm120()` gate) and
   #36845, `_forward_paged_attention` on exact SM120 takes FlashInfer's
   `trtllm_batch_decode_with_kv_cache` (the XQA kernel, JIT-compiled at the first sparse decode)
   instead of FA2's `flash_attn_varlen_func`. Both routes compact the selected rows through
   `qwen_sparse_kv_extraction_compact_triton`, which PR 4 extends, so the quantized pools are
   dequantized into the bf16 scratch before either kernel runs and no device guard was added.
   Two consequences to be aware of: the XQA JIT needs an nvcc >= 12.9 at `CUDA_HOME`
   (FlashInfer's bound, `flashinfer/compilation_context.py`, `_normalize_cuda_arch`; with nvcc
   12.0 it aborts with `RuntimeError: No supported CUDA architectures found`, a loud failure, not
   corruption), and the published speed numbers were measured on `73a255206f` with the FA2
   route and have not been re-measured on `78c5024e9d`. A standalone call of the FlashInfer
   kernel with the backend's arguments (24 query heads, 2 KV heads, head_dim 256, page 64, topk
   2,051 = `indexer_budget` 2048 + `indexer_compress_ratio` 4 - 1, the index-row width of
   `qsa_indexer.py`; batch 1 and 4, plus topk 130 at batch 3; bf16) with flashinfer 0.6.17 and
   nvcc 13.3 matches a torch softmax reference within bf16 precision (max relative error 3.9e-3,
   2.5e-3, 2.5e-3; no NaN) on the RTX PRO 4000; probe and output: `tools/probe_trtllm_sm120.py`
   and `docs/logs/probe_trtllm_sm120.log` in the companion repository. Is the FlashInfer route
   the intended default on SM120, and
   should PR 4 keep the FA2 route selectable for quantized pools?
4. **Overlap with draft PR #36787** ("[Qwen 3.8 Flash Next] Add sm120 (RTX PRO 6000 Blackwell)
   support"). It touches 11 of the files of this series (`configs/qwen4_exp.py`,
   `qsa_indexer.py`, `sparse_attn.py`, `qwen_sparse_attn_backend.py`, `kv_cache_configurator.py`,
   `memory_pool.py`, `pool_configurator.py`, `models/qwen3_5.py`, `models/qwen4_exp.py`,
   `server_args.py`, `spec_utils.py`), adds `SGLANG_QSA_DECODE_BACKEND` /
   `SGLANG_QSA_MQA_BACKEND` / `SGLANG_PLE_OFFLOAD_NUMA_INTERLEAVE`, and targets 96 GB cards. This
   series targets 24 GB cards and does not merge #36787 in. Which should land first, and should
   the QSA backend selection of #36787 be the mechanism behind question 3?
5. **Host-RAM constraints and CI.** The reference machine has 32 GB of host RAM, of which 24-26 GB
   are pinned by the offloaded layers, so the elastic cache never pins at runtime and the
   published configuration is single-request. Some behaviours are validated only in that regime:
   `lazy_release` fires when the whole pool is idle, the scheduler does not survive
   `alloc_extend` returning `None` mid-prefill (the admission cap refuses such prompts earlier),
   and the radix cache was disabled throughout. The registered unit tests need 30-150 MB of VRAM
   and no checkpoint; an end-to-end test would need a synthetic 2-bit MoE (a generator exists in
   the companion repository). Is that acceptable as CI coverage for a first version?

### How the series was verified

- The flat patch (34 files, +4,155 / -89) reproduces the served tree byte for byte on
  `73a255206f`; the five commits on that base differ from it only by the removed measurement and
  debug hooks (`SGLANG_KV_STATS`, `SGLANG_KV_FAKEQ`, `SGLANG_NAN_TRACE`,
  `SGLANG_MOE_NCONTIG_LAYERS`, `SGLANG_NGRAM_CHECK`, `SGLANG_NGRAM_FORCE_REJECT`) and the added
  registered tests.
- Rebased onto `78c5024e9d`: cherry-picks 2, 3 and 5 applied cleanly; part 1 conflicted in
  `qsa_indexer.py` and part 4 in `qwen_sparse_attn_backend.py` against the reformatting of
  #36667 and the routing changes of #36649 / #36806 / #36845; resolved keeping upstream formatting
  and the series' logic (verified by diffing against the original commits: only upstream hunks
  remain). The `layers_block_type` alias hunk of `configs/qwen4_exp.py` is already on the branch
  since #36772 and dropped out of part 1.
- Per commit: black 26.1.0, isort 7.0.0, ruff 0.15.1 (`F401,F821,UP037`), codespell and
  `py_compile` clean.
- All seven registered test files pass on the finished branch: 59 tests, 40 subtests, 0 failures
  (with ninja and the CUDA 13.3 nvcc on `PATH`; the Triton and `load_inline` JITs need them).

## Related resources

- Weights and model card: https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang
  (text-only checkpoint, 222,856 tensors, 38,755,352,600 bytes; PLE table `ple/ple.f8_e4m3.bin`,
  51,200,245,760 bytes, plus `ple/ple.json`; Qwen Community License 1.0)
- Companion repository (flat patch, the two patch series, PR texts, measurement tools, logs,
  design documents): https://github.com/HaberstrohSystems/qwen3.8-flash-next-24gb-sglang
- Review target: PR #36497 "Introduce Qwen 3.8 Flash Next", branch `qwen4-main-squashed`
  (https://github.com/sgl-project/sglang/pull/36497)
- Related draft: PR #36787 "[Qwen 3.8 Flash Next] Add sm120 (RTX PRO 6000 Blackwell) support"
  (https://github.com/sgl-project/sglang/pull/36787)
- Base model: https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8
