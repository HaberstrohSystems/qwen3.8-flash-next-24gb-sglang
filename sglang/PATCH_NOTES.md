# SGLang serving patch for Qwen3.8-Flash-Next at 2.572 bpw -- patch notes

File: `qwen4exp-serving-73a255206f.patch` (this directory), the verbatim diff of the tree that
served the published numbers. The reviewable form of the same change is the five-commit series
under `upstream/` (`UPSTREAM.md`); these notes describe the flat patch, which remains the
reproduction artifact. Written 2026-09-03. Every number below names its source (file and line).
Paths are relative to the root of this repository: `CAMPAIGN.md` means `docs/CAMPAIGN.md` (the
dated, append-only engineering log; `:N` is line N), `state` means `assets/phase1_state.json`,
`WRITEUP.md` means `docs/WRITEUP.md`.

## 1. What the patch is

The complete difference between the SGLang tree that served the accepted state (the served
checkout on 2026-09-03, at restart #23) and its base commit:

| | |
|---|---|
| Base commit | `73a255206f916366c8d26d4022f82ddfb0ab558d` "Introduce Qwen 3.8 Flash Next" (Qiaolin-Yu, 2026-08-26 01:36:27 -0700); the first commit of the branch `qwen4-main-squashed` of the open PR #36497 in `sgl-project/sglang`. It is not on `main`. |
| Size | 34 files changed, 4155 insertions(+), 89 deletions(-) (`git diff --stat HEAD`); 5439 lines, 251,796 bytes |
| New files | 7 (`new file mode` headers): `expert_stream.py` plus the six modules listed in section 3 |
| SHA-256 | `92f669b2525f9c86190825390fafc2b41a28071c398fcb7fd95716fbce744bb5` |
| Supersedes | `patches/base/sglang-qwen4exp-2bit.patch` (the pre-campaign baseline: 15 files, 604 lines) |
| Review form | `upstream/series-q4head/` (on `78c5024e9d`, the head of `qwen4-main-squashed`) and `upstream/series-base/` (on `73a255206f`): the same change split into five commits without the measurement and debug hooks, with registered unit tests |

It contains everything in the served tree, i.e. also two measurement-only hooks and the opt-in
NGRAM edits. Section 5 lists their hunks (the series drops them); section 2 shows that the
patch reproduces the served tree byte for byte.

## 2. How the patch was produced and verified

Produced on 2026-09-03 from the served working tree, without changing any file content:

```
cd <served SGLang checkout>                         # HEAD 73a255206f, 28 modified + 7 untracked
git ls-files --others --exclude-standard python/    # 7 untracked files
#   python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py.ncontig.orig   <- backup written by
#                                                     patches/ncontig_gemv.py, NOT source, excluded
git add -N python/sglang/srt/layers/moe/expert_elastic.py \
           python/sglang/srt/layers/moe/expert_gemv.py \
           python/sglang/srt/layers/moe/row_arena.py \
           python/sglang/srt/mem_cache/int4_kv_pool.py \
           python/sglang/srt/mem_cache/int8_kv_pool.py \
           python/sglang/srt/mem_cache/tiered_kv_pool.py      # intent-to-add only (index entries)
git diff HEAD > sglang/qwen4exp-serving-73a255206f.patch
git reset -q -- <the same six files>                # index restored; git status --short identical
                                                    # to before (28 " M"/"A", 7 "??")
```

Verified against a clean checkout of the base commit in a temporary worktree (git 2.43.0):

```
git worktree add --detach <scratch>/sgl-check HEAD
git -C sgl-check apply --check qwen4exp-serving-73a255206f.patch    # exit 0, no output
git -C sgl-check apply qwen4exp-serving-73a255206f.patch            # 34 files changed, 4155 insertions(+), 89 deletions(-)
cmp of all 34 patched files against the served checkout             # compared 34 files, 0 differ
git worktree remove --force sgl-check && git worktree prune          # `git worktree list` = main tree only
```

So `base commit + this patch == the served tree` (excluding the `.ncontig.orig` backup).

## 3. How to apply

```
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout 73a255206f916366c8d26d4022f82ddfb0ab558d
git apply --check /path/to/qwen4exp-serving-73a255206f.patch
git apply         /path/to/qwen4exp-serving-73a255206f.patch
```

To remove it: `git apply -R` the same file (or `git checkout -- . && git clean -f python/`).
The patch is plain `git diff` output (no leading text), so `patch -p1` works as well.

Relationship to `patches/*.py`: those scripts are exact-string edit layers that were
applied one after another to the tree (their target is the checkout named by `$SGLANG`, default
`~/quant/sglang`) and each supports `--check | apply | revert`. This `.patch` is the flattened result of
all of them plus the base 2-bit patch; it is the portable form. The layer order, needed only if
you want to peel layers off with the scripts, is:

```
base 2-bit/Qwen4-Exp patch  <  host_fixes items (hook, skipgather, memo, rope, ple, bcg, bcg2)
  <  ncontig_gemv  <  placement  <  elastic  <  kv_lazy  <  ple_random
  <  kv_fp8  <  kv_int8  <  kv_int4  <  kv_tiers        (revert in reverse order)
```
(state `patches` list; layering rule from the docstrings of `kv_int8.py`, `kv_int4.py`,
`kv_tiers.py`). Environment- and flag-inert layers that are ALSO in the tree and therefore in
the patch: `ngram_ple` (opt-in), `kv_stats` edit 0 and 1, `kv_fakeq` edits 0-1
(measurement-only) -- see section 5. `kv_paged_prefix` and the `dump`/`moedump` items are NOT in
the tree (their `--check` reads `clean`), so they are not in the patch.

New modules installed by the layers (all six are in the patch as new files):

| module (python/sglang/srt/...) | lines | from | layer |
|---|---|---|---|
| `layers/moe/expert_stream.py` | 182 | base patch | MoE-aware expert streamer |
| `layers/moe/expert_gemv.py` | 109 | `patches/ncontig_gemv.py` | in-place int2 GEMV, address tables |
| `layers/moe/row_arena.py` | 215 | `gemv/row_arena.py` | CUDA VMM row arena |
| `layers/moe/expert_elastic.py` | 376 | `gemv/expert_elastic.py` | elastic expert residency |
| `mem_cache/int8_kv_pool.py` | 159 | `patches/kv_int8.py` (+ `kv_bits` from kv_int4) | INT8-G64 pool |
| `mem_cache/int4_kv_pool.py` | 173 | `patches/kv_int4.py` | INT4-G32 pool |
| `mem_cache/tiered_kv_pool.py` | 169 | `patches/kv_tiers.py` | int8 ring over int4 pool |

## 4. Per-file summary, grouped by feature

Line counts are `git diff --stat HEAD` insertions/deletions. "Source" is where the change is
explained.

### 4.1 Base: 2-bit MoE path and Qwen4-Exp / CPU-offload correctness (WRITEUP.md sections 3-5, findings 1-9)

| file | +/- | change |
|---|---|---|
| `kernels/ops/moe/fused_moe_triton_kernels.py` | +362 (part) | 2-bit unpack in `fused_moe_kernel_gptq_awq` (`offs_k // 4`, `& 0x3`, zero point 2; WRITEUP.md:369-378); the rest of the +362 is the word-load kernel of 4.3 |
| `layers/moe/moe_runner/triton_utils/fused_moe.py` | +58 (part) | `use_int2_w2a16` plumbing through `fused_experts` / `_fused_moe_kernel_sequence`; N-contiguous handling of 4.3 |
| `layers/moe/moe_runner/triton.py` | +4 | `use_int2_w2a16` on `TritonMoeQuantInfo` and its three call sites |
| `layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py` | +6 | config dtype name `int2_w2a16` so a tuned int4 config is never picked up by accident (the `SGLANG_MOE_CONFIG_DIR` lookup itself is upstream code, `fused_moe_triton_config.py:92`, unchanged) |
| `layers/quantization/moe_wna16.py` | +281 (part) | 2-bit loader (asymmetric `qzeros` raise `NotImplementedError`, WRITEUP.md:396-398), `ExpertStreamer` hook; also carries 4.2-4.4 |
| `layers/moe/expert_stream.py` | new, 182 | MoE-aware expert streaming: only the routed experts cross PCIe (finding 9: 26 GB -> 0.63 GB per forward, WRITEUP.md:210-228) |
| `utils/offloader.py` | +70 (part) | `tie_weights=False` for `functional_call` (finding 5), stale `param.data` views, expert-only offload; the `hook` item of 4.2 |
| `layers/layernorm.py` | +26 | `gemma_weight` as a persistent buffer so CPU offload moves it (finding 4) |
| `models/qwen3_5.py`, `layers/radix_linear_attention.py` | +12, +23 | pass the `nn.Conv1d` module instead of a `.view()` of its weight and derive the view on each access (finding 8: silent NaN under `--cpu-offload-gb`, WRITEUP.md:171-207). `radix_linear_attention.py` also carries a debug print gated by `SGLANG_NAN_TRACE=1` (patch lines 3538-3542) that can be dropped |
| `layers/attention/linear/gdn_backend.py` | +1 | `qwen_sparse_attention` in `ALL_DECODER_LAYER_TYPES` (finding 2) |
| `configs/qwen4_exp.py` | +6 | `packed_modules_mapping` propagated to the quant config (finding 3: `in_proj_ba` would land at 8 bits) |
| `server_args.py` | +13 (part) | `Qwen4ExpForConditionalGeneration` in `LANGUAGE_MODEL_ONLY_ARCHITECTURES` (finding 7); KV dtype values of 4.7-4.9 |
| `models/qwen4_exp.py` | +215 (part) | `Qwen4ExpMmapEmbedding`: the 51.2 GB PLE n-gram table read from an mmap'd file instead of pinned host memory (`SGLANG_QWEN4_PLE_MMAP`), table created on `meta` so nothing is allocated first (patch hunks at lines 4730 and 4762) |
| `test/manual/test_triton_moe_wna16.py` | +40 | 2-bit cases; note that the unmodified 8-bit path of this test fails 66 of 96 large shapes upstream (WRITEUP.md:381-384) |

### 4.2 Host fixes (`patches/host_fixes.py`, items with their line numbers in that file)

| item | file | change |
|---|---|---|
| `hook` (:14-47) | `utils/offloader.py` | do not install the offloader forward hook when every offloaded parameter of the module is a streamed expert param |
| `skipgather` (:49-75) | `layers/quantization/moe_wna16.py` | fully GPU-resident layers skip the gather+renumber and run the kernel on original expert ids |
| `memo` (:76-102) | `layers/moe/expert_stream.py` | memoised `arange`/cast tensors (`_ARANGE_CACHE`) |
| `rope` (:103-152) | `layers/attention/qsa/qsa_indexer.py` (+23) | hoist `positions.max().item()` (a device sync, 12x per token) and pre-size the rotary cos/sin cache |
| `ple` (:201-231) | `models/qwen4_exp.py` | parallel `os.pread` row fetch for decode (16-thread pool, GIL released; the numpy memmap fancy-index held the GIL 2.5-3.1 ms per token) |
| `bcg` (:232-249) | `models/qwen4_exp.py` | `Qwen4ExpMmapEmbedding.forward` wrapped with `eager_on_graph(True)` so decode runs under `--cuda-graph-backend-decode breakable` with the PLE lookup as the eager break (patch lines 4971-4975) |
| `bcg2` (:250-328) | `model_executor/runner_backend/breakable_cuda_graph_backend.py` (+35) | the four structure helpers learn the `LogitsProcessorOutput` dataclass (capture died with "Unsupported BCG output type", CAMPAIGN.md:59-61) |

### 4.3 N-contiguous layout and in-place GEMV (`patches/ncontig_gemv.py`)

| file | change |
|---|---|
| `layers/quantization/moe_wna16.py` | after loading, re-lay every 2-bit expert tensor from `[E, N, K/4]` uint8 to `[E, K/16, N]` int32 (same bytes, N contiguous); GEMV dispatch for M <= 16 in `apply()`; `SGLANG_MOE_NCONTIG=0` / `SGLANG_MOE_GEMV=0` switches, `SGLANG_MOE_NCONTIG_LAYERS` bisection aid |
| `kernels/ops/moe/fused_moe_triton_kernels.py` | word-load int2 branch `fused_moe_kernel_gptq_awq_word` for the tiled prefill kernel (reads coalesced 128-B lines instead of 32 distinct lines per warp load); layout derived from the tensor dtype (int2 + int32 = word layout) so no flag crosses the custom-op schema |
| `layers/moe/moe_runner/triton_utils/fused_moe.py` | grid/stride/N handling for the word layout |
| `layers/moe/expert_gemv.py` (new) | batch-1 int2 GEMV that reads experts in place through int64 address tables (device or pinned host), `to_word_ncontig`, `make_tables` |

### 4.4 Frequency placement and elastic expert cache (`patches/placement.py`, `patches/elastic.py`)

| file | change |
|---|---|
| `layers/quantization/moe_wna16.py` | deferred, interleaved placement pass at the last layer: hottest `S` experts of every layer on the GPU (`SGLANG_MOE_PLACEMENT=expert_freq.pt`, `SGLANG_MOE_PLACEMENT_S`), cold rows in donated pinned slots, int64 address tables per (layer, kind); with `SGLANG_MOE_ELASTIC=1` an `ExpertElastic` is built instead and `apply()` polls the control file (`SGLANG_MOE_ELASTIC_CTL`) |
| `layers/moe/expert_stream.py` | shape-keyed staging, table-driven prefill gather |
| `layers/moe/row_arena.py` (new) | `RowArena`: VA reserved once (`cuMemAddressReserve`), physical 2 MiB granules mapped for a rank-ordered prefix, tail unmap returns VRAM to the driver while every row address stays fixed (CUDA graphs need no recapture) |
| `layers/moe/expert_elastic.py` (new) | per-(layer, kind) arenas in routing-mass rank order; grow = table-driven host->arena gather + table rewrite, shrink = D2H copy of tail ranks into pool slots + table rewrite + unmap; host slot pool; `free()` never pins at runtime (`SGLANG_MOE_ELASTIC_PIN_MB`, `_FILL_MB`, `_RESERVE_ROWS`) |

### 4.5 Lazy VMM KV backing (`patches/kv_lazy.py`)

| file | +/- | change |
|---|---|---|
| `mem_cache/kv_vmm_backing.py` | +43 | `ensure_prefix`, `uncommit_beyond`, `release_beyond`, `backed_tokens`/`bytes_per_token` on the VMM owner (local driver import) |
| `mem_cache/memory_pool.py` | +195 (part) | with `SGLANG_KV_LAZY=1` the full-attention pool is allocated through the VMM owner in the classic flow too, `SGLANG_KV_LAZY_FLOOR` (4096) tokens backed at start; `lazy_ensure` commits in `SGLANG_KV_LAZY_MARGIN` (2048) steps with a `SGLANG_KV_LAZY_HEADROOM_MB` (1536) driver-free watermark that shrinks the expert cache first and rate-limits `empty_cache` to once per 30 s; `lazy_release` unmaps beyond the floor at pool idle and regrows the expert cache there (patch lines 4191-4304 minus the `_kv_stats` block, see 5) |
| `mem_cache/kv_cache_configurator.py` | +23 (part) | virtual capacity `SGLANG_KV_LAZY_TOKENS` above the profiled value, admission cap `min(requested, SGLANG_KV_LAZY_SAFETY x profiled)` |
| `mem_cache/allocator/paged.py` | +28 | `_lazy_hook` before pages are consumed, allocation refused (returns `None`) when the commit fails, `_lazy_idle_check` in `_release_page_ids` |
| `mem_cache/allocator/token.py` | +15 | alloc-time backing and idle reset for the page-1 allocator |

Note: `kv_lazy.py`'s docstring assumes `--max-running-requests 1` ("the live KV prefix is exactly
the current request's length"), which is the published configuration (`scripts/serve.sh`). The
concurrent-benchmark restart #23 ran with 4 (CAMPAIGN.md:454-459); the idle release then happens
when *all* requests have finished. Not separately measured.

### 4.6 PLE mmap fixes (`patches/ple_random.py`)

`models/qwen4_exp.py`: `MADV_RANDOM` on the memmap and `POSIX_FADV_RANDOM` on the pread fd
(page-cache read-around of 128 KB+ per 160-byte row drove host pressure past systemd-oomd at
~55k-token prompts, CAMPAIGN.md:341), `POSIX_FADV_DONTNEED` after bulk gathers of >= 512 ids
(random-word text churned ~1M pages and killed a 250k needle run, CAMPAIGN.md:408). No numerics.

### 4.7 fp8_e4m3 KV read path for QSA (`patches/kv_fp8.py`)

| file | change |
|---|---|
| `layers/attention/qsa/sparse_attn.py` (+1083 total for 4.7-4.10) | `_compact_kv_fp8`: gather-dequant (uint8 -> fp8 bitcast -> bf16) into the bf16 FA2 scratch |
| `layers/attention/qwen_sparse_attn_backend.py` (+133 total) | bf16 scratch keyed by q dtype, prefix-chunk gather on the uint8 view with bitcast |
| `mem_cache/memory_pool.py` | write path: skip the no-op `div_` by a unit scale, saturate to +-448 before the e4m3 cast (the cast returns NaN beyond the range) -- both `set_kv_buffer` variants, patch lines 4318-4329 and 4331-4346 |

Mode `--kv-cache-dtype fp8_e4m3`; it is a prerequisite layer of 4.8 and stays optional.

### 4.8 INT8-G64 KV cache (`patches/kv_int8.py`, new `mem_cache/int8_kv_pool.py`)

int8 K/V `[rows, 2, 256]` plus one fp16 absmax/127 scale per (token, kv-head, 64-channel group) as
extra `KvBufferDesc`s on the same lazy VMM owner (12.4 KB/token over the 12 QSA layers vs 24 bf16;
docstring). Kernels in `sparse_attn.py`: `_quant_store_kv_int8` (fused quantize + scatter at
write, fp16 scale clamp added 2026-09-02 22:04, CAMPAIGN.md:415), `_compact_kv_int8` (decode /
verify gather-dequant), `_gather_dequant_rows_int8` (prefix chunk, replaces `index_select + cat`).
Backend dispatch on `pool.kv_bits == 8`; `memory_pool.py` / `HybridLinearKVPool` gain
`get_kv_scale_buffer` / `get_kv_smooth_buffer`; `server_args.py`, `kv_cache_dtype.py`,
`kv_cache_configurator.py`, `pool_configurator.py` learn `int8_g64`.

### 4.9 INT4-G32 KV cache (`patches/kv_int4.py`, new `mem_cache/int4_kv_pool.py`)

Nibble-packed K/V `[rows, 2, 128]` uint8 (low nibble = even channel, offset-binary q + 8,
q in [-7, 7]) plus fp16 absmax/7 scales per 32-channel group, 6.75 KB/token (docstring).
Kernels `_quant_store_kv_int4`, `_compact_kv_int4`, `_gather_dequant_rows_int4`; backend derives
the logical head_dim through `_kv_head_dim`; `kv_bits = 4`. Mode `--kv-cache-dtype int4_g32`.

### 4.10 Tiered "int8 ring over int4" KV cache (`patches/kv_tiers.py`, new `mem_cache/tiered_kv_pool.py`) -- the default

Every token is written twice: int8-g64 into a ring of `R = SGLANG_KV_TIERS_W` (8192) slots with
an int32 owner table (`_stamp_ring_owner`), and int4-g32 into the full-context pool
(`_quant_store_kv_tiered`). Readers test `owner[slot & (R-1)] == slot` on the device and read the
int8 ring row (hot) or the int4 row (cold) with `tl.where` -- CUDA-graph safe, no compactor.
7,308 B/token at 256k (6,912 int4 + 12,672 x 8192/262,144), ring 103.8 MB + 32 KB owner
(docstring; CAMPAIGN.md:413). `HybridLinearKVPool.get_kv_ring_buffer` / `get_kv_ring_owner`.
Mode `--kv-cache-dtype int8ring_int4`.

### 4.11 NGRAM speculation with the PLE -- opt-in (`patches/ngram_ple.py`)

| file | +/- | change |
|---|---|---|
| `models/qwen4_exp.py` | (part) | drop the "Qwen4 PLE does not support NGRAM speculation" guard (patch hunk 4691-4701); startup self-check that both PLE intermediates are allocated on the first target-verify forward (hunk 4702-4729) |
| `speculative/ngram_worker.py` | +116 | force a LINEAR draft chain (`_linearize_chain`: the corpus fans out into a star even at bfs breadth 1; the GDN ReplaySSM fold, the QSA pending ring and the KV move assume a chain); constructor asserts `max_bfs_breadth == 1`; debug checks under `SGLANG_NGRAM_CHECK=1`, `SGLANG_NGRAM_FORCE_REJECT=1` |
| `speculative/spec_utils.py` | +31 | commit the PLE n-gram history + short-conv state after verify in both ReplaySSM branches of `commit_mamba_states_after_verify` (they returned before the generic update; virtual -> physical slot translation). This is a genuine bug fix that also affects MTP topk=1 under `--enable-linear-replayssm-spec` (CAMPAIGN.md:379) |

Inert unless `--speculative-algorithm NGRAM ...` is passed (flags in the docstring). Not in the
accepted set.

### 4.12 Measurement-only hooks that are in the tree (drop for a release build)

| layer | file | change |
|---|---|---|
| `kv_stats` (`patches/kv_stats.py`) | `mem_cache/memory_pool.py` | `_kv_stats`: per-layer/head K/V absmax, sum-of-squares and histograms accumulated at `set_kv_buffer` when `SGLANG_KV_STATS=/path.pt` (design data for 4.8, `kv_stats_fp8run.pt` on the measuring host, not included in this repository; CAMPAIGN.md:356) |
| `kv_fakeq` (`patches/kv_fakeq.py`) | `mem_cache/memory_pool.py` | `_fake_quant_kv` / `_fq_smooth`: quantize -> dequantize on the base pool's write path per `SGLANG_KV_FAKEQ=<scheme>` (int8_tok, int8_g64, int8_g32, e4m3, int4_g32, int3_g16, int2_g16, int2_g8, `*_sm`, zero, noise). Bypassed by the int8/int4/tiered subclasses (ERRATUM CAMPAIGN.md:394). The flat patch carries the measuring host's default path for the statistics file (`~/Downloads/.../kv_stats_fp8run.pt`); `patches/kv_fakeq.py` in this repository reads `SGLANG_KV_STATS_FILE` only |

## 5. Hunks an upstreamer should drop or split (exact locations in the patch file)

Line numbers refer to `qwen4exp-serving-73a255206f.patch`. Verified with the
`patches/*.py --check` runs of 2026-09-03 (appendix A).

Measurement-only (`kv_stats`, `kv_fakeq`) -- all in `python/sglang/srt/mem_cache/memory_pool.py`:

| patch lines | hunk | content | layer |
|---|---|---|---|
| 4128-4190 | `@@ -1735,6 +1735,62 @@ class KvBufferDesc` | `_FQ_SMOOTH`, `_fq_smooth()`, `_fake_quant_kv()` -- the whole hunk | kv_fakeq edit 1 |
| 4210-4240 | inside `@@ -2262,6 +2326,101 @@ class MHATokenToKVPool` | `_KVSTATS = None` and `def _kv_stats(...)` -- only these 31 lines; the rest of the hunk (4241-4304: `lazy_ensure`, `lazy_release`) is kv_lazy and must stay | kv_stats edit 0 |
| 4312-4313 | inside `@@ -2472,11 +2631,20 @@` | `if os.environ.get("SGLANG_KV_STATS"): self._kv_stats(...)` | kv_stats edit 1 (reads `clean` in `--check` only because kv_fakeq edit 0 was inserted right after it) |
| 4314-4317 | same hunk | `_fq = os.environ.get("SGLANG_KV_FAKEQ") ...` (4 lines); lines 4318-4329 of the hunk are kv_fp8's unit-scale skip + saturation and must stay | kv_fakeq edit 0 |

Guarded two-line calls `if os.environ.get("SGLANG_KV_STATS") and hasattr(self, "_kv_stats"): self._kv_stats(...)`
in the new pool files -- harmless without the hook (the `hasattr` guard) but they go with it:
`int4_kv_pool.py` patch lines 3800-3801, `int8_kv_pool.py` 3965-3966, `tiered_kv_pool.py` 4521-4522.

Debug aid: `SGLANG_NAN_TRACE` print in `radix_linear_attention.py`, patch lines 3538-3542
(inside `@@ -153,6 +169,11 @@ def unified_linear_attention_with_output`).

Opt-in NGRAM (`ngram_ple`): the entire diffs of `speculative/ngram_worker.py` (patch lines
5009-5175) and `speculative/spec_utils.py` (5176-5224), plus in `models/qwen4_exp.py` the hunks
`@@ -116,9 +119,7 @@` (4691-4701, guard drop) and `@@ -129,6 +130,27 @@` (4702-4729, self-check).
Recommendation: keep the `spec_utils.py` hunk as a separate bug-fix PR (see 4.11).

Everything else in the patch is part of the accepted serving state (state `patches` list:
hook, skipgather, memo, rope, ple, ncontig, bcg, bcg2, placement, elastic, kv_lazy, ple_random,
kv_fp8, kv_int8, kv_int4, kv_tiers).

## 6. Server flags and environment

**Published launch line: [`../scripts/serve.sh`](../scripts/serve.sh)** (`--max-running-requests 1
--max-mamba-cache-size 1`). Its flag set is `scripts/sweep.sh`'s base set minus the state `drop`
list plus the state `add` list, with single-request concurrency; its environment is the three fixed
variables of `sweep.sh` plus the twelve variables of the state `env` string, with the asset paths
pointing at `assets/` of this repository. Every headline log (`docs/logs/tiers-validate.log`,
`night.log`) was recorded with that flag set at concurrency 1 / 1; only `night4.log` and
`elastic.ctl.status` were recorded on restart #23 below.

Flags of the published configuration (as in `scripts/serve.sh`; the harness `scripts/phase1.py`
passes `--max-mamba-cache-size` twice, 2 from step S4 and 1 from step S21, and argparse keeps the
last):

```
--host 127.0.0.1 --port 30000 --tp-size 1 --cpu-offload-gb 19 --no-ple-offload-embedding
--mem-fraction-static 0.95 --language-model-only --page-size 1 --disable-overlap-schedule
--disable-radix-cache --weight-loader-drop-cache-after-load
--chunked-prefill-size 1024 --max-prefill-tokens 32768 --cuda-graph-backend-decode breakable
--model-path <checkpoint directory> --max-total-tokens 262144
--context-length 262144 --kv-cache-dtype int8ring_int4 --attention-backend triton
--max-mamba-cache-size 1 --reasoning-parser qwen3 --tool-call-parser qwen3_coder
--max-running-requests 1
```

Notes: `--page-size 1` is requested but QSA forces page size 64 (CAMPAIGN.md:325), which is why
the paged allocator hooks of 4.5 exist. The checkpoint is the INT8-dense re-pack of the sealed
2-bit quant (`scripts/requant_int8.py`; S11b, CAMPAIGN.md:299-300), i.e. the Hub download, not an
SGLang change.

Wrapper (`scripts/serve.sh`, `scripts/sweep.sh:55-60`, `:73-78`): `systemd-run --user --scope
-p MemoryMax=30G env ... python3 -m sglang.launch_server <flags>`, then one warm-up request and
`POST /freeze_gc` after `/health` is up. Never pass `-p ManagedOOMPreference=omit` on that scope:
under pressure systemd-oomd then kills other units of the session instead of the server
(CAMPAIGN.md:460-464; `docs/ELASTIC_MEMORY.md`, host-RAM section).

Environment. Fixed part (`scripts/sweep.sh:56-58`; `VENV=~/quant/venv-sglang`,
`CU=$VENV/lib/python3.12/site-packages/nvidia/cu13`):

```
PATH="$CU/bin:$VENV/bin:$PATH" CUDA_HOME="$CU"
SGLANG_QWEN4_PLE_MMAP=<ple directory> SGLANG_VLM_CACHE_SIZE_MB=0 SGLANG_MOE_EXPERT_STREAM=1
```

Accepted `EXTRA_ENV` (state `env`; the original values pointed at the measuring host's copy of
`assets/` and a control file next to it, written here as `<repo>` = this repository):

```
SGLANG_MOE_CONFIG_DIR=<repo>/assets/moe_configs
SGLANG_MOE_PLACEMENT=<repo>/assets/expert_freq.pt
SGLANG_MOE_PLACEMENT_S=184 SGLANG_MOE_ELASTIC=1 SGLANG_MOE_ELASTIC_PIN_MB=512
SGLANG_MOE_ELASTIC_CTL=<writable control file containing "S 184">
SGLANG_KV_LAZY=1 SGLANG_MOE_ELASTIC_FILL_MB=2048 SGLANG_MOE_ELASTIC_RESERVE_ROWS=0
SGLANG_KV_LAZY_TOKENS=262144 SGLANG_KV_LAZY_SAFETY=0.77 SGLANG_KV_TIERS_W=8192
```

Historical note, not the published configuration: restart #23 (2026-09-03 00:33, the tree state
this patch was taken from) passed the same flags with `--max-mamba-cache-size 8
--max-running-requests 4` and the wrapper carried `-p ManagedOOMPreference=omit`, for a concurrent
benchmark attempt. Concurrency history: 1 (base set) -> 8 (restart #22, CAMPAIGN.md:452) -> 4
after systemd-oomd killed the 8-concurrent run 90 s in (CAMPAIGN.md:454); the `omit` exemption
made systemd-oomd end the desktop session (CAMPAIGN.md:460-464) and was withdrawn. The published
configuration is single-request; higher concurrency needs more host RAM.

All environment variables read by the patched code (`grep environ` over the added lines):

| variable | default in code | accepted value | reader |
|---|---|---|---|
| `SGLANG_QWEN4_PLE_MMAP` | unset = pinned-host PLE | the `ple/` directory of the Hub download (`ple.f8_e4m3.bin`, `ple.json`) | `qwen4_exp.py` |
| `SGLANG_MOE_EXPERT_STREAM` | unset = off | 1 | `moe_wna16.py`, `offloader.py` |
| `SGLANG_MOE_CONFIG_DIR` | package dir | `assets/moe_configs` | upstream `fused_moe_triton_config.py:92` |
| `SGLANG_MOE_NCONTIG` | "1" | (default) | `moe_wna16.py` |
| `SGLANG_MOE_GEMV` | "1" | (default) | `moe_wna16.py` |
| `SGLANG_MOE_NCONTIG_LAYERS` | unset = all | (unset) | `moe_wna16.py` (bisection aid) |
| `SGLANG_MOE_PLACEMENT` | unset = no placement | `assets/expert_freq.pt` | `moe_wna16.py` |
| `SGLANG_MOE_PLACEMENT_S` | "184" | 184 | `moe_wna16.py` |
| `SGLANG_MOE_ELASTIC` | unset = static v3 placement | 1 | `moe_wna16.py` |
| `SGLANG_MOE_ELASTIC_PIN_MB` | "0" | 512 | `expert_elastic.py` |
| `SGLANG_MOE_ELASTIC_FILL_MB` | unset | 2048 | `expert_elastic.py` |
| `SGLANG_MOE_ELASTIC_RESERVE_ROWS` | "0" | 0 | `expert_elastic.py` |
| `SGLANG_MOE_ELASTIC_CTL` | unset = no control file | a writable file containing `S 184` | `moe_wna16.py` / `expert_elastic.py` |
| `SGLANG_KV_LAZY` | unset = off | 1 | `memory_pool.py`, allocators, configurator |
| `SGLANG_KV_LAZY_FLOOR` | "4096" | (default) | `memory_pool.py` |
| `SGLANG_KV_LAZY_MARGIN` | "2048" | (default) | `memory_pool.py` |
| `SGLANG_KV_LAZY_HEADROOM_MB` | "1536" | (default) | `memory_pool.py` |
| `SGLANG_KV_LAZY_TOKENS` | "0" = profiled capacity (`kv_cache_configurator.py:1913`) | 262144 | `kv_cache_configurator.py` |
| `SGLANG_KV_LAZY_SAFETY` | "0.85" (`kv_cache_configurator.py:1919`; CAMPAIGN.md:350) | 0.77 (CAMPAIGN.md:372) | `kv_cache_configurator.py` |
| `SGLANG_KV_TIERS_W` | "8192" | 8192 | `tiered_kv_pool.py` |
| `SGLANG_KV_STATS`, `SGLANG_KV_STATS_FILE`, `SGLANG_KV_FAKEQ` | unset | (unset) | measurement-only, section 4.12 |
| `SGLANG_NGRAM_CHECK`, `SGLANG_NGRAM_FORCE_REJECT` | "0" | (unset) | NGRAM debug, section 4.11 |
| `SGLANG_NAN_TRACE` | unset | (unset) | debug print, section 5 |

Discrepancy: CAMPAIGN.md:343 says `SGLANG_LOG_GC=1` was "added to the accepted env"; it is in
neither the state file nor `sweep.sh`, and the patched code does not read it.

Runtime assets the environment points at (all in `assets/` of this repository):
`assets/moe_configs/configs/triton_3_7_1/` (four files
`E={10,512},N=160,device_name=NVIDIA_RTX_PRO_4000_Blackwell,dtype=int2_w2a16{,_down}.json`, tuned
for this GPU only), `assets/expert_freq.pt` (296,837 B; `{"mass": [48, 512], "count": [48, 512]}`
built by `tools/expert_freq.py` from a `SGLANG_ROUTE_DUMP` routing probe), and a writable control
file for `SGLANG_MOE_ELASTIC_CTL` (`scripts/serve.sh` creates one).

KV modes available after the patch (`--kv-cache-dtype`):

| value | storage | bytes/token (12 QSA layers) | evidence |
|---|---|---|---|
| `auto` | bf16 | 24 KB | 68,905-token prompt proven (CAMPAIGN.md:348); estimated ceiling 64-80k on the reference host (CAMPAIGN.md:339) |
| `fp8_e4m3` | fp8, unit scale | 12 KB | speed parity (CAMPAIGN.md:353); 1-token prompt crash open (CAMPAIGN.md:355) |
| `int8_g64` | INT8-G64 | 12.4 KB | 162,215-token prompt (CAMPAIGN.md:372) |
| `int4_g32` | INT4-G32 | 6.75 KB | 257,905-token prompt (CAMPAIGN.md:407), needle 5/5 at 248k (:409) |
| `int8ring_int4` (default) | int8 ring (8192) over int4 | 7,308 B at 256k | S21 (CAMPAIGN.md:413) |

## 7. Validation evidence per feature

Speed = streaming bench (`tools/bench_speed.py`, 200 streamed tokens; introduced
CAMPAIGN.md:306). Exactness oracle = teacher-forced logprobs on fixed continuations
(`tools/logprob_diff.py`; noise floor MAX 0.09 / MEAN 0.002 over 450 tokens, thresholds
MEAN <= 0.01 host-class, <= 0.05 kernel-class, MAX <= 0.5; CAMPAIGN.md:247, :97-100).
Long-text KV quality = `tools/nll_long.py` (short prompts never read the cache: CAMPAIGN.md:353).

| feature | evidence | source |
|---|---|---|
| Base 2-bit kernel | packing order `torch.equal` vs uint8 view; Triton expressions bit-identical vs torch; end-to-end vs bf16 dequant bit-identical on small shapes, 1e-5 relative on large (incl. e=64, n=640, k=2560, g128) | WRITEUP.md:356-370 |
| Base launch state | 15.5 tok/s decode, ~1158 prefill, 26 GB/token -> 0.31 GB PCIe | CAMPAIGN.md:20-22; WRITEUP.md:11-13 |
| Campaign baseline (old two-request bench) | S0 13.4 / 1189 | CAMPAIGN.md:237 |
| Tuned int2 configs (asset) | S0b KEPT 15.2 / 1190; oracle max 0.086 mean 0.0018 | CAMPAIGN.md:251-252 |
| Host fixes | S5 KEPT 19.4 / 1443; oracle 0.078 / 0.0019 | CAMPAIGN.md:259-260 |
| ncontig + GEMV | S9 KEPT 21.8 / 2249; oracle max 0.275 mean 0.0125 (kernel-class); per-layer A/B with identical inputs 5.1e-5 (M=6) / 1.7e-4 (M=1) relative = bf16 rounding, verdict "numerically equivalent"; GEMV 320 GB/s device / 51 GB/s pinned host | CAMPAIGN.md:285-286, :92-97, :218-219 |
| bcg + bcg2 | S6c KEPT 40.0 / 2249; capture 3.97 s, 0.12 GB; oracle 0.079 / 0.0022 | CAMPAIGN.md:290-291, :102-103 |
| placement v3 | S10 KEPT 48.4 / 2334; oracle 0.057 / 0.0013; top-171 experts/layer cover 82 % of routing mass | CAMPAIGN.md:295-296, :210-212 |
| (model side) INT8-dense re-pack | S11b 48.8 / 2319; NLL overall -0.001; +1.6 GB VRAM back | CAMPAIGN.md:299-300 |
| streaming-bench re-base | 56.0 / 2362 on the same state | CAMPAIGN.md:306 |
| elastic expert cache | S13 KEPT 56.2 / 2335; oracle 0.060 / 0.0016; `row_arena.py` self-test (2 MiB granules, graph replay valid after shrink/regrow); live S sweep 184/200/216: +2-3 % decode per ~1.7 GB, mass covered 84.3 % at S=184 | CAMPAIGN.md:318-319, :304, :320; `docs/logs/elastic.ctl.status` |
| lazy VMM KV | S14 KEPT 55.4 / 2323, oracle 0.057 / 0.0014; S15 (128k) 56.9 / 2340 re-measured; headroom watermark after the 60k crash; bf16 68,905-token prompt 48.9 s (1408 tok/s), decode 53.3; admission cap; rate-limited `empty_cache`: 258k prefill 171.0 -> 165.3 s (1508 -> 1560 tok/s) | CAMPAIGN.md:328-330, :336, :337, :348, :350, :418, :448 |
| ple_random | root cause and fix of the ~55k oomd deaths; 250k needle run survived after the page-drop | CAMPAIGN.md:341, :408-409 |
| kv_fp8 | mode: decode 55.9-57.2 / prefill 2303-2339 (parity); reverted as a speed step (-3 % startup noise); fake e4m3 512-window NLL -0.0133 / mean dlogprob 0.1099 (noise 0.099) | CAMPAIGN.md:353, :352, :368 |
| kv_int8 | unit tests bit-exact (quant+scatter, compact gather, prefix gather; relative RMS 0.62 %); S17 VALIDATED: 12.4 KB/token, short NLL -0.001, 512-window mean dlogprob 0.094 vs noise 0.099, NLL +0.010 (+-0.008), oracle 0.0019, bench 56.1-57.4 / 2301-2316; prompts 115,560 (71.8 s, 1610 tok/s, decode 50.8), 126,060 (78.1 s), 150,560 (92.8 s), 162,215 (95.8 s, 1694 tok/s, decode 51.2, VRAM free min 1.18 GB); ~179k refused at admission. No valid all-position int8-vs-bf16 figure exists (ERRATUM) | CAMPAIGN.md:365, :370, :371, :372, :394 |
| kv_int4 | S19: all-position NLL +0.0088 / 0.138 vs bf16 pool (noise +0.0002 / 0.059), oracle 0.0015, decode 55-57, prefill 1493 at 10k (int8 2316); 162k prompt 112.6 s (1441 tok/s); 257,905-token prompt 182.2 s (1415 tok/s), decode 51.9; needle 248k 5/5 (269 s, 920 tok/s on random text, decode 51.4) | CAMPAIGN.md:406, :407, :409 |
| kv_tiers (default) | S21 VALIDATED: short NLL -0.0002; 512-window 0.118; all-position NLL -0.0001 / 0.0743 over 8561 positions; oracle 0.0019; decode 56.2/54.3/56.8/55.7/54.5 at ctx 101/421/1701/6821/10001, prefill 2271 at 10k; 257,905-token prompt 171.0 s (1508 tok/s), decode 51.8; needles 5/5 at 41,370 tokens (41.2 s, 52.7) and 5/5 at 247,629 (257.2 s, 50.8); lazy backing 366,976 profiled -> 262,144 admitted | CAMPAIGN.md:413; `docs/logs/tiers-validate.log:3-4, 6, 9, 21, 26, 30-34`; state `accepted` |
| post-restart 10k benches of the default | restart #20 2291 / 56.5; #22 1669 / 55.7; #23 (running 4, mamba 8) 2225 / 54.5 (startup transient caveat CAMPAIGN.md:306, :336) | `docs/logs/night.log`, `night3.log`, `night4.log` |
| Corrected KV precision ladder (all-position, bf16 pool ref) | noise +0.0002 / 0.059; fake int4_g32 +0.0078 / 0.139 (rerun +0.0059 / 0.1369); real int4 +0.0088 / 0.138; tiered -0.0001 / 0.074; fake int2_g16 +0.30 / 0.62 (cliff); K/V := 0 +3.70 / 3.92 | CAMPAIGN.md:406, :413, :444 |
| ngram_ple (opt-in) | lossless within the decode-vs-prefill floor (0 mismatches / 600 tokens, mean dlogprob 0.009-0.019 = non-spec baseline); accept length 1.11-1.27, rate 7-9 %; 22-25 tok/s eager vs 56; MTP head bf16 4.7 GiB does not fit | CAMPAIGN.md:386-387, :379 |
| kv_paged_prefix (NOT in the patch) | built, correct within 2 row-ulps, rejected: 6.1 ms (tiered) / 3.19 (int8) / 4.21 (int4) vs 1.4 ms materialised per head-layer at prefix 60k | CAMPAIGN.md:439; `docs/KV_PAGED_PREFIX_PLAN.md:180-186` |

## 8. Benchmarks

No standard-benchmark score is published for the 2-bit model (`docs/HISTORY.md`, "Benchmarks").
The quality evidence behind this patch is the internal protocol of section 7: the short held-out
NLL, the long-text NLL ladder against a bf16 KV cache, the teacher-forced logprob oracle and the
needle tests, with the thresholds in `../CONTRIBUTING.md`.

## 9. Known issues in the patched tree

- `fp8_e4m3` mode: a 1-token prompt crashes with an illegal memory access in the QSA indexer
  prefill; >= 101-token prompts work; bf16 handles 1-token prompts (CAMPAIGN.md:355).
- The scheduler did not survive `alloc_extend` returning `None` mid-prefill (CAMPAIGN.md:349);
  the admission cap (CAMPAIGN.md:350) refuses such prompts earlier, the scheduler path itself is
  not fixed.
- `kv_int8`'s kernel got the fp16 scale clamp later than int4/tiers (CAMPAIGN.md:415); before
  that, groups with absmax > 127 x 65504 produced inf scales.
- Host RAM is the wall on the reference host (24 GB pinned by 31 offloaded layers, ~2 GB free,
  CAMPAIGN.md:305): the elastic cache never pins at runtime (a 4 MB `cudaHostAlloc` took 22 s and
  triggered oomd, CAMPAIGN.md:346; a 315 MB startup reserve killed the load, :347), and the
  published configuration serves one request at a time (:454).
- `kv_lazy` documents a single-request assumption (section 4.5).
- `patches/ngram_ple.py --check` needs an interpreter with numpy (it `exec`s the chain
  linearizer at import); with the venv interpreter all 11 edits read APPLIED (appendix A).

## 10. Tests that exist (not in the patch; `gemv/`, `tools/`)

GPU tests, run with the corresponding layer applied: `test_kv_fp8.py` (41 lines; gather-dequant
bit-exact vs `pool.to(fp8).to(bf16)`, saturation), `test_kv_int8.py` (228; six checks, bit-exact
kernels, eager + lazy-VMM pool), `test_kv_int4.py` (352; pack/unpack, trtllm strided layout,
fp16-max clamp, pool), `test_kv_tiers.py` (604; ring aliasing, same-launch collisions, stale
owner, pool, microbench), `test_kv_paged_prefix.py` (629; the rejected kernel), `test_gemv.py`
(127), `test_gemv_ncontig.py` (102), `test_gemv_tab.py` (101; bit-exact layout conversion +
full-layer decode MoE vs fp32 reference), `test_tiled_word.py` (95; A/B through the real
`invoke_fused_moe_kernel`). CPU: `test_ngram_chain.py` (412). Pre-campaign: `tools/test_unpack_2bit.py`,
`test_vs_bf16.py`, `test_group_layout.py`; upstream `test/manual/test_triton_moe_wna16.py` (+40).
Server-level protocols (all in `tools/`): `int8_validate.sh` (the KV-mode validation sequence; the
`tiers_validate.sh` wrapper it was run through for S21 is not included), `nll_series.sh`,
`needle_test.py`, `longctx_test.py`, `spec_lossless.py`, `elastic_sweep.py`.

## Appendix A. `patches/*.py --check` on the served tree, 2026-09-03 (read-only)

Interpreter `/usr/bin/python3` (the scripts import only `os`/`sys` at module level; the `torch`
imports seen in `kv_int8.py:57` etc. are inside the string literals of the new modules they
install). MISMATCH lines under an overlay are expected and documented in each script's docstring.

```
host_fixes.py    hook APPLIED  skipgather MISMATCH(overlaid by ncontig edit 9 / elastic edit 1)
                 memo APPLIED  rope APPLIED  dump clean  ple MISMATCH(overlaid by ple_random 1-2)
                 bcg APPLIED  bcg2 APPLIED  moedump clean
ncontig_gemv.py  edits 0-8 APPLIED, 9 MISMATCH (moe_wna16.apply overlaid by elastic);
                 kernel word-load branch APPLIED; gemv file present; kernel backup present
placement.py     0,1,3,4 APPLIED; 2 clean ("if was_pinned": superseded by elastic)
elastic.py       0,1 APPLIED; row_arena.py and expert_elastic.py installed
kv_lazy.py       0-11 APPLIED
ple_random.py    0-2 APPLIED
kv_fp8.py        0 clean, 1-3 MISMATCH (overlaid by kv_int8), 4 APPLIED
kv_int8.py       P: kv_fp8 overlaid by kv_int8; _compact_kv_fp8 present;
                 2,3,5,6,7,13,14 APPLIED; 0,1,4,9,12,15,17 MISMATCH; 8,10,11,16 clean; F int8_kv_pool.py MISMATCH
                 (all overlays of kv_int4, as its docstring states)
kv_int4.py       P: kv_int8 overlaid by kv_int4; kv_int8 edits [2,3,5,6,7,13,14] intact;
                 0,1,2,3,6,9,11,12,13,14 APPLIED; 4,8,10,15 MISMATCH; 5,7 clean; F int4_kv_pool.py APPLIED
                 (overlays of kv_tiers, as its docstring states)
kv_tiers.py      P: kv_int4 overlaid by kv_tiers; kv_int4 edits [0,1,2,3,6,9,11,12,13,14] + file intact;
                 0-11 APPLIED; F tiered_kv_pool.py APPLIED
kv_stats.py      0 APPLIED, 1 clean (its text is present but followed by kv_fakeq's insert)
kv_fakeq.py      0,1 APPLIED
ngram_ple.py     0-10 APPLIED (venv interpreter; system python3 lacks numpy)
kv_paged_prefix.py  P kv_tiers prerequisite APPLIED; 0-5 clean (not applied)
```
