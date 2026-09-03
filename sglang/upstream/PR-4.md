Base branch: qwen4-main-squashed (PR #36497). Stacked on PR-3.

# feat(qsa): quantized KV pools with dequant-on-gather (fp8/int8/int4)

Part 4/5 of the Qwen3.8-Flash-Next 24 GB serving series (RFC issue: *link*). Patch:
`upstream/series-q4head/0004-feat-qsa-quantized-KV-pools-with-dequant-on-gather-f.patch`
in the companion repository (14 files, +4,424 / -25). The pool classes are general; the read
path is specific to the Qwen sparse attention backend.

## Motivation

The Qwen sparse attention (QSA) kernels read bf16/fp16 K/V only, so the stock `fp8_e4m3` KV
option cannot be consumed by Qwen4-Exp, and a bf16 KV cache (24 KB per token over the 12 QSA
layers) stops a 24 GB card at about 80k tokens once the expert cache is at its floor (CAMPAIGN
2026-09-02 13:55). QSA has exactly two places where every K/V row is gathered anyway (the
decode/verify compaction into the attention scratch and the prefix-chunk row gather), so
dequantization is free there and the attention kernels stay untouched. The formats were chosen
from measured K/V statistics of this model (~19k tokens): simulated relative RMS error e4m3
2.66 %, int8 per-token 1.3 %, int8 with 64-channel groups 0.9 % (CAMPAIGN 2026-09-02 15:50;
`docs/ELASTIC_MEMORY.md`, "The three mechanisms", 3).

Sources: dated entries of `docs/CAMPAIGN.md`, `docs/ELASTIC_MEMORY.md` and
`docs/logs/tiers-validate.log` in https://github.com/HaberstrohSystems/qwen3.8-flash-next-24gb-sglang
and the model card
https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang.
Reference machine: one RTX PRO 4000 Blackwell (24 GB, sm_120) with 32 GB host RAM.

## Modifications

New `--kv-cache-dtype` values `int8_g64`, `int4_g32`, `int8ring_int4` (`server_args.py` with
help text, `mem_cache/kv_cache_dtype.py`, `mem_cache/kv_cache_configurator.py` pool-class
selection, `model_executor/pool_configurator.py` cell sizes). Pool classes are subclasses of
`MHATokenToKVPool`; the payload and the fp16 group scales are extra `KvBufferDesc`s on the same
(lazy, part 3) VMM owner, and writing is one fused quantize + scatter Triton launch.

- `mem_cache/int8_kv_pool.py` (new): int8 K/V `[rows, H, D]` plus one fp16 absmax/127 scale per
  (token, kv-head, 64-channel group), the first group being the rotary dimensions; 12.4 KB per
  token over the 12 QSA layers. `kv_bits = 8` is the backend's dispatch key.
- `mem_cache/int4_kv_pool.py` (new): nibble-packed K/V `[rows, H, D/2]` (low nibble = even
  channel, offset-binary q + 8, q in [-7, 7]) plus fp16 absmax/7 scales per 32-channel group;
  6.75 KB per token. `kv_bits = 4`.
- `mem_cache/tiered_kv_pool.py` (new, `int8ring_int4`): every token is written twice, int8-g64
  into a ring of `SGLANG_KV_TIERS_W` (8192) slots with an int32 owner table, and int4-g32 into
  the full-context pool. Readers test `owner[slot & (R-1)] == slot` on the device and read the
  int8 ring row (hot) or the int4 row (cold) with `tl.where`: CUDA-graph safe, no compactor, no
  unmapping mid-request. 7,308 B per token at 256k (6,912 int4 + 12,672 x 8192/262,144), ring
  103.8 MB + 32 KB owner table.
- `layers/attention/qsa/sparse_attn.py`: `_compact_kv_fp8` (uint8 -> fp8 bitcast -> bf16 into
  the attention scratch), `_quant_store_kv_int8` / `_compact_kv_int8` /
  `_gather_dequant_rows_int8` (the last replaces `index_select + cat` for prefix chunks), the
  int4 siblings, `_stamp_ring_owner`, `_quant_store_kv_tiered`, `_compact_kv_tiered`,
  `_gather_dequant_rows_tiered` and their wrappers; fp16 scale clamp to 65504 in every write
  kernel. On-disk layouts (nibble order, scale index arithmetic, owner-table semantics) are in
  the module docstrings.
- `layers/attention/qwen_sparse_attn_backend.py`: dispatch on `pool.kv_bits` / `pool.kv_tiered`
  (`_kv_bits`, `_kv_head_dim`, `_kv_scratch_dtype`, `_int8_gather_kwargs`, `_kv_tier_kwargs`),
  bf16 scratch keyed by the query dtype, prefix-chunk gather on the uint8 view for fp8; the CPU
  fallback raises `NotImplementedError` for int8/int4 pools.
- `mem_cache/memory_pool.py`: `HybridLinearKVPool.get_kv_smooth_buffer`, `get_kv_ring_buffer`,
  `get_kv_ring_owner` forwarders. Per-channel smoothing constants are plumbed but identity: a
  static smoothing A/B on the fake-quantized int4-g32 pool was mixed and was not adopted
  (CAMPAIGN 2026-09-03 00:40).
- Registered unit tests (`register_cuda_ci`, stage `base-b`, runner `1-gpu-small`, 30-50 MB of
  VRAM each, torch references, no server):
  `test/registered/unit/layers/attention/qsa/test_sparse_attn_fp8_gather.py` (`est_time=10`),
  `test/registered/unit/mem_cache/test_int8_kv_pool.py` (20), `test_int4_kv_pool.py` (20),
  `test_tiered_kv_pool.py` (40); contents under Accuracy Tests.

Decode routing on `qwen4-main-squashed` (#36806, #36845): `_forward_paged_attention` first tries
`_resolve_trtllm_sparse_decode()`, which returns FlashInfer's `trtllm_batch_decode_with_kv_cache`
on exact SM120 as well as SM100, and otherwise uses `_resolve_flash_attn_varlen_func()` (the KDA
kernel on SM121, else FA2, else FA4). Both routes gather the selected rows through
`qwen_sparse_kv_extraction_compact_triton`, which this PR extends with the scale / smoothing /
ring arguments, so a quantized pool is dequantized into the bf16 scratch before either attention
kernel runs, and the prefix-chunk gather in `forward_extend` is not routed by device at all; no
device-specific guard is added. What does change on SM120 relative to the base the numbers below
were measured on (`73a255206f`, FlashInfer route gated on SM100 only): the attention over the
gathered scratch runs FlashInfer's XQA kernel (JIT-compiled at the first sparse decode; it needs
an nvcc >= 12.9 at `CUDA_HOME`, FlashInfer's bound in `flashinfer/compilation_context.py`
(`_normalize_cuda_arch`: SM 12.x requires CUDA >= 12.9); otherwise FlashInfer raises
`RuntimeError: No supported CUDA architectures found`) instead of FA2 `flash_attn_varlen_func`.
A standalone call of that kernel with the backend's arguments (24 query heads, 2 KV heads,
head_dim 256, page 64, topk 2,051 = `indexer_budget` 2048 + `indexer_compress_ratio` 4 - 1, the
index-row width `token_topk + compress_ratio - 1` of `qsa_indexer.py`; batch 1 and 4, plus topk
130 at batch 3; bf16) on the RTX PRO 4000 with flashinfer 0.6.17 and nvcc 13.3 matches a torch
softmax reference within bf16 precision (max relative error 3.9e-3, 2.5e-3, 2.5e-3; no NaN):
`tools/probe_trtllm_sm120.py` and its output `docs/logs/probe_trtllm_sm120.log` in the companion
repository. See the RFC, open question 3.

Scope and known limits, disclosed: the read path exists for `qwen_sparse_attn_backend` only; any
other GPU backend that meets one of these pools today would read raw bytes, so a generic guard or
a `kv_bits`-aware materialisation is the follow-up (RFC, open question 2).
`TORCH_DTYPE_TO_KV_CACHE_STR` maps `torch.uint8` to `int4_g32`, which the tiered mode shares.
`fp8_e4m3` mode still crashes on 1-token prompts (CAMPAIGN 2026-09-02 15:26). Prefix-chunk
prefill materialises the prefix as bf16 per layer and chunk; a paged prefix kernel reading the
pools directly was built, verified within 2 row-ulps and rejected as 4-5x slower
(`docs/ELASTIC_MEMORY.md`, "Where the long-context prefill time goes"). `SGLANG_KV_TIERS_W`
should become a server argument.

## Accuracy Tests

Protocol (card, "Serving fidelity"; `docs/ELASTIC_MEMORY.md`, "Quality protocol"): prompts
shorter than one prefill chunk never read the KV cache, so the decisive test scores every
position from the second chunk on (8,561 positions of a 9.6k-token text, bf16 pool as reference,
run-to-run noise NLL +0.0002 / mean |dlogprob| 0.059), plus the 512-window test (noise 0.099 /
+-0.008 NLL), the 10k logprob oracle, and needle retrieval with `ignore_eos`.

- Registered tests, all pass (41 tests, 6 subtests; also on the finished 5-commit branch):
  - `test_sparse_attn_fp8_gather.py` (2): the fp8 compaction gather equals `pool.to(fp8).to(bf16)`
    bit for bit; e4m3 relative RMS error in (0.01, 0.04) on N(0, 3).
  - `test_int8_kv_pool.py` (10 + 2 subtests): int8-g64 quantize + scatter (int32 and int64
    loc), scale index arithmetic, compaction gather-dequant (3 requests, permuted `req_to_token`,
    invalid positions untouched), prefix row gather-dequant with interior gaps, all bit-exact;
    relative RMS error < 1.2 %; `MHATokenToKVPoolInt8` eager and lazy-VMM paths.
  - `test_int4_kv_pool.py` (14 + 2): nibble packing and unpack round trip, quantize + scatter
    with `rint` tie rounding, scale index and nibble order, compaction gather including the
    trtllm strided layout (a packed-width scratch is rejected), prefix gather with gaps,
    fp16-max scale clamp, all bit-exact; relative RMS error 8-11.5 %; `MHATokenToKVPoolInt4`
    eager and lazy-VMM paths; `kv_bits` keys.
  - `test_tiered_kv_pool.py` (15 + 2): ring owner stamping (R = 64 over 4,096 slots, ring wrap),
    hot/cold boundary and same-launch ring-row collisions (x20), hot/cold selection in the
    compaction and prefix gathers (stale owner and no owner -> int4 path), fp16-max clamp on the
    ring, all-cold == int4 and all-hot == int8 kernels; `MHATokenToKVPoolTiered` eager and
    lazy-VMM paths, ring reset on `lazy_release`, a bad `SGLANG_KV_TIERS_W` rejected.
- Fake-quant ladder on the bf16 pool, 512 window, mean |dlogprob|: noise 0.099, int8_g64 0.099,
  int8 per-token 0.100, int8_g32 0.106, e4m3 0.110 (CAMPAIGN 2026-09-02 16:17).
- INT8-G64 server: short NLL -0.001, 512-window 0.094 vs noise 0.099, NLL +0.010 (noise
  +-0.008), oracle 0.0019; all-position +0.001 / 0.059; needle 41k 5/5 (CAMPAIGN 2026-09-02
  16:45 and 21:05).
- INT4-G32 server: all-position NLL +0.0088 / 0.138 (fake-quant int4_g32 +0.0078 / 0.139),
  oracle 0.0015; needle 5/5 at 247,629 tokens (CAMPAIGN 2026-09-02 19:40 and 20:00).
- Tiered default: short NLL -0.0002, 512-window 0.118, all-position NLL -0.0001 / 0.074,
  oracle 0.0019; needles 5/5 at 41,370 and 5/5 at 247,629 tokens (CAMPAIGN 2026-09-02 21:05).
- Controls: fake int2_g16 +0.30 / 0.62 (the cliff), K/V := 0 +3.70 / 3.92. The ladder was
  re-measured after the write-path fake-quant hook was found to be bypassed by the int8 pool
  subclass; the K/V := 0 control caught it (CAMPAIGN 2026-09-02 19:40).

Reproduction (needs `ninja` on `PATH` (otherwise `FileNotFoundError: ninja`) and an nvcc that
supports the GPU architecture at `CUDA_HOME` (compute_120a on the reference machine; a system
nvcc 12.0 does not, the virtualenv's 13.3 does); the lazy-VMM cases build a `load_inline` stub):

```
git clone https://github.com/sgl-project/sglang.git && cd sglang
git fetch origin qwen4-main-squashed && git checkout 78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385
git am /path/to/upstream/series-q4head/000{1,2,3,4}-*.patch
pip install -e python
export CUDA_HOME=<toolkit whose nvcc supports the GPU architecture>; export PATH="$CUDA_HOME/bin:$PATH"
python -m pytest -v test/registered/unit/layers/attention/qsa/test_sparse_attn_fp8_gather.py \
    test/registered/unit/mem_cache/test_int8_kv_pool.py \
    test/registered/unit/mem_cache/test_int4_kv_pool.py \
    test/registered/unit/mem_cache/test_tiered_kv_pool.py
# server level (companion repository): tools/nll_long.py with NLL_LONG_ALL=1, tools/logprob_diff.py,
# tools/needle_test.py against a server started with --kv-cache-dtype int8ring_int4
```

## Speed Tests and Profiling

Streaming bench, single request, ~10k context (`tools/bench_speed.py`; card, "Performance"),
measured on `73a255206f` with the flat patch and the FA2 decode route:

- fp8 read path: decode 55.9-57.2 / prefill 2,303-2,339 tok/s, parity with bf16 (CAMPAIGN
  2026-09-02 15:12).
- INT8-G64: decode 56.1-57.4 / prefill 2,301-2,316; 115,560-token prompt prefill 71.8 s
  (1,610 tok/s), decode 50.8 (16:45); the longest prompt admitted on the reference machine
  162,215 tokens at prefill 95.8 s (1,694 tok/s), decode 51.2, VRAM free bottomed at 1.18 GB, a
  ~179k prompt refused at admission (17:00).
- INT4-G32: decode 55-57, prefill 1,493 at 10k (int8 2,316: the unpack gather); 257,905-token
  prompt prefill 182.2 s (1,415 tok/s), decode 51.9 (19:40 and 19:34); needle haystack of
  247,629 random-word tokens prefill 269 s (920 tok/s, PLE rows from NVMe), decode 51.4 (20:00).
- Tiered default: decode 54-57 (56.2 / 54.3 / 56.8 / 55.7 / 54.5 at context 101 / 421 / 1,701 /
  6,821 / 10,001), prefill 2,271 at 10k; 257,905-token prompt prefill 171 s (1,508 tok/s), decode
  51.8; lazy backing 366,976 tokens profiled -> 262,144 admitted (21:05; `tiers-validate.log`);
  with the rate-limited `empty_cache` of part 3: 165.3 s (1,560 tok/s), decode 52.3 (2026-09-03
  00:55). Per-chunk prefill time at 258k fits 615 ms + 0.49 us x prefix tokens; the O(prefix)
  term is the QSA indexer, not the KV gather (`docs/ELASTIC_MEMORY.md`, "Where the long-context
  prefill time goes").

Not re-measured on `78c5024e9d`, where SM120 decode runs FlashInfer XQA after the gather (see
Modifications).

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit). (black 26.1.0, isort 7.0.0, ruff 0.15.1 `F401,F821,UP037`, codespell and `py_compile` clean on the commit.)
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests). (The four registered tests above, 41 tests / 6 subtests, `register_cuda_ci`. Missing: an end-to-end test with a model, which needs a checkpoint.)
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations). (Missing: the three new `--kv-cache-dtype` values in the server-arguments documentation, `SGLANG_KV_TIERS_W`, the on-disk layouts outside the docstrings.)
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed). (Long-text NLL ladder against a bf16 KV cache with controls, logprob oracle, needle retrieval, streaming bench and long-prompt runs above; no standard benchmark score.)
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance). (Known deviations: `kv_bits` / `kv_tiered` as `getattr` class attributes, `SGLANG_KV_TIERS_W` as an environment variable.)

## Suggested reviewers

- `python/sglang/srt/layers/attention` (QSA kernels and backend): @Fridge003, @ishandhanani,
  @Qiaolin-Yu (NV and model-specific optimizations); @JustinTong0323 as author of #36497 and of
  the SM120 routing changes (#36806, #36845).
- `python/sglang/srt/mem_cache` (pools, dtype, configurator): @ispobock, @xiezhq-hermann
  (KV Cache).
- `python/sglang/srt/model_executor` (`pool_configurator.py`): @merrymercy, @hnyls2002, @cctry
  (Scheduler).
