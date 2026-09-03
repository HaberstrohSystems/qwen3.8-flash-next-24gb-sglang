# Tiered ('compost') KV cache plan (2026-09-02)

> **Status:** implemented as `patches/kv_tiers.py` (layout (B), dual write); validated and accepted as the
> default CAMPAIGN.md:413 (all-position NLL -0.0001 / 0.0743, 257,905-token prompt, needles 5/5).

## kv_tiers.py implementation plan (compost: int8 recent window over an int4 full-context pool)

Anchors below are in `$SGLANG/python/sglang/srt/` (tree with kv_fp8 -> kv_int8 -> kv_int4 -> kv_lazy applied) and `perf/`. Nothing was run; all numbers are analytic from the measured per-token constants.

### 1. Layout: (B) int4 full-context + int8 ring + owner table (dual-write) — 7,308 B/token at 256k

Bytes/token at N = 262,144, W = 8192 (12 QSA layers, 2 KV heads x 256; int8-g64 row = 1056 B/layer -> 12,672 B/token, int4-g32 row = 576 B/layer -> 6,912 B/token: `int8_kv_pool.py:1-8`, `int4_kv_pool.py:1-7`, KV_INT4_PLAN.md:8-13):

| layout | B/token | GB at 256k | new VMM code | mid-request unmap/sync |
|---|---|---|---|---|
| int8 only (S17) | 12,672 | 3.32 | 0 | no |
| int4 only (S19, measured) | 6,912 | 1.81 | 0 | no |
| ideal tiered (no int4 under the window) | 7,092 | 1.86 | - | - |
| (A) two full-context buffer families, int8 backing unmapped per demoted range | 7,680-8,060 | 2.01-2.11 (+3.3 GB VA) | ~50 lines: interval `[lo,hi)` bookkeeping replacing the prefix watermark (`commit_range` kv_vmm_backing.py:155-183, `uncommit_beyond` :190-214 is tail-only, `_back_spans` :344-358 skips below `backed_to`), `uncommit_before` | yes: `cuMemUnmap` + `torch.cuda.synchronize()` (:371) every tick; int8 scale rows are 16 B -> one 2 MiB granule = 131,072 rows, so scale backing frees nothing before row 131k; payload straddle granules are pure waste |
| (B) int4 full-context on the lazy owner (unchanged) + int8 ring of R = W = 8192 slots as plain tensors + owner table int32[R] | 6,912 + 12,672*8192/262,144 + 0.1 = **7,308** | **1.916** (ring 103.8 MB + owner 32 KB) | 0 | no |
| (B) with compactor lag (R = W + C + 64 = 9,280) | 7,360 | 1.93 | 0 | no |
| (B) with a bf16 ring (exact recent window) | 7,680 | 2.01 | 0 | no |
| (C1) int8-sized buffer, int4 packed at slot//2 with a hole | = (A) | = (A) | = (A) + second scale family | yes |

Chosen: (B) with **dual-write** (each fresh token is written int8 into the ring AND int4 into the full-context row in one launch; the ring row expires by itself when slot s+R is written). Reasons: zero lines in kv_vmm_backing.py (the ring is off the owner, so `bytes_per_token()` :380-381 and the `lazy_ensure` headroom estimate memory_pool.py:2366-2379 stay exact — a full-span ring desc on the owner would double the estimated delta and refuse commits before 256k); no mid-request unmap/synchronize; the int4 row is quantized from bf16 (exactly the measured S19 error, CAMPAIGN.md:406, no int8->int4 double rounding); the int4 rows always exist, so the hot/cold decision is a pure device-side ownership test with no flip ordering to get right; robust to slot-0 dummy writes, speculative rows, radix and batching (none of which the compactor variant tolerates). The 57 MB of int4 rows under the window (0.22 KB/token) is the price; recovering it needs per-desc token offsets in the owner (`ensure_prefix` backs every spec to the same token count, kv_vmm_backing.py:360-365) and is not worth it. The compactor (section 4b) is specified as the optional `SGLANG_KV_TIERS_MODE=compact` path because the brief asked for it; implement dual-write first.

### 2. Pool subclass: `MHATokenToKVPoolTiered(MHATokenToKVPoolInt4)` in new file `srt/mem_cache/tiered_kv_pool.py`

- Class attrs: `kv_bits = 4` (inherited: keeps `_kv_head_dim` = 2*128, `_kv_scratch_dtype` bf16, prefix pack shape — backend :1585-1593, :1487), `kv_tiered = True` (new dispatch key), `ring_slots = R`, `ring_dtype` (int8 default, bf16 optional).
- `__init__`: `super().__init__()` then, inside `self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE)` (like `_create_buffers_normal`, int4_kv_pool.py:78-89): `self.ring_k/ring_v = [torch.empty((R, H, D), int8) x L]`, `self.ring_ks/ring_vs = [torch.empty((R, H, D//64), fp16) x L]`, `self.ring_owner = torch.full((R,), -1, int32)`. R from `SGLANG_KV_TIERS_W` (power of two required unless `SGLANG_KV_TIERS_MOD=1`), assert `R >= 2*chunked_prefill_size` is not needed for dual-write; assert `R <= size`. These are allocated once, before graph capture (the pool is built in model_runner init; capture calls the compact wrapper with the same tensors, backend :1798-1815 / :1666-1683), and never reallocated. They sit in the torch allocator, so no 2 MiB granule rounding (48 tensors = 103.8 MB exact). Descs/`_build_kv_buffer_descs`/`_assign_post_capture_tensors` (int4_kv_pool.py:91-114) untouched — the lazy VMM owner, `lazy_ensure`/`lazy_release` (memory_pool.py:2358-2415) and kv_lazy.py stay byte-for-byte as today.
- Accessors: `get_kv_ring_buffer(layer_id) -> (ring_k[idx], ring_v[idx], ring_ks[idx], ring_vs[idx])`, `get_kv_ring_owner() -> ring_owner`, property `ring_mask = R-1`. `HybridLinearKVPool` forwarders next to memory_pool.py:4091-4101: `get_kv_ring_buffer` with `_wait_for_layer` + `_transfer_full_attention_id` (mirror `get_kv_scale_buffer`), `get_kv_ring_owner` plain passthrough, `kv_tiered`/`ring_mask` passthrough properties.
- `set_kv_buffer` (override of int4_kv_pool.py:131-164): same unwrap/oob/stats prologue, then `quant_store_kv_tiered(cache_k, cache_v, loc, k4, v4, ks4, vs4, rk, rv, rks, rvs, ring_owner, sm_k_inv, sm_v_inv, ring_mask)`; host assert `loc.numel() <= R` (two programs of one launch must never share a ring row; chunk 1024 <= 8192).
- `lazy_release` hook: after `super().lazy_release()` do `ring_owner.fill_(-1)` (hygiene only; correctness never depends on it because every slot is re-stamped by its own write before any read; `_lazy_idle_check` paged.py:162-168 calls `clear()` then `release()`).
- `get_kv_size_bytes` (memory_pool.py:2446-2456): add ring bytes to the K/V sums (logging); `_pd_registerable_tensors` (int4_kv_pool.py:116-117) append ring tensors (PD is off; keeps the registrar honest).
- Capacity accounting: pool_configurator's int4 cell size (kv_int4.py:270-278 edit) stays 6,912 B/token; the ring is a fixed 104 MB. `--max-total-tokens 262144` is pinned by flag (phase1_state.json), and `lazy_ensure` reads live `mem_get_info` (memory_pool.py:2371), so the ring is seen automatically; the 258k run bottomed at 1.46 GB free (CAMPAIGN.md:407) -> expect ~1.36 GB, i.e. the elastic expert cache gives up ~1 more slab at the end (that is its purpose). Log one line at construction: `KV tiers: ring R=%d slots (%s), %.0f MB, owner %d KB`.

### 3. Kernels (all in `sparse_attn.py`, inserted after `qwen_sparse_prefix_gather_dequant_int4` :914-952, before `qwen_sparse_valid_counts_triton` :954 — the same anchor kv_int4.py used)

Common device-side tier test used by every reader (no host branch; owner is a static tensor read at replay):
```
r    = slots & RING_MASK                         # (or slots % RING if MOD)
o    = tl.load(owner + r, mask=valid, other=-1).to(tl.int64)
hot  = valid & (o == slots)                       # ring row belongs to this slot -> int8
cold = valid & (o != slots)                       # -> int4 row (always exists under dual-write)
```

a) Write: `_quant_store_kv_tiered[(N, H)]` = body of `_quant_store_kv_int8` (:455-503) with row/srow computed from `r = slot & RING_MASK` (`row8 = (r*H+h)*D`, `srow8 = (r*H+h)*NG8`), **plus** the fp16 clamp `s = tl.minimum(..., 65504.0)` the int8 kernel lacks (:488/:498 vs :728; CAMPAIGN.md:392), followed by the body of `_quant_store_kv_int4` (:684-753) verbatim at `row4 = (slot*H+h)*DH` (loads of k/v are re-issued even/odd from L1/L2 — 3 x 512 B per head, irrelevant), then `tl.store(owner + r, slot.to(tl.int32))` (both heads store the same value; benign). Wrapper `quant_store_kv_tiered(...)` = the asserts of both wrappers (:505-537, :755-789) plus `rk.shape == (R, H, D)`, `rks.shape == (R, H, D//64)`, `owner.numel() == R`, `RING_MASK = R-1`. Static grid (N, H) -> capture-safe like today (`set_kv_buffer` runs inside the captured forward, backend :1726-1728). bf16 ring option: skip the int8 half and `tl.store(rk_bf16 + row, x.to(bf16))`.

b) Decode/verify compact gather (captured): `_compact_kv_tiered` = `_compact_kv_int4` (:791-857) with signature + `k8, v8, k8_scale, v8_scale, owner, GROUP8: tl.constexpr, RING_MASK: tl.constexpr`. Per tile (BLOCK_TOPK=16 slots x 1 head): positions/valid/slots exactly :830-836; the tier test above; cold path = :837-848 with `pmask = cold[:, None] & (pairs < DH)` -> `k4 = tl.interleave(lo, hi)` [16,256] fp32; hot path = :583-591 with `src8 = (r[:, None]*heads + head)*dim + dims`, `ssrc8 = (r[:, None]*heads + head)*NG8 + dims // GROUP8`, `mask8 = hot[:, None] & (dims < dim)` -> `k8 = q*sc*sm` [16,256] fp32; `kk = tl.where(hot[:, None], k8, k4)`; **store mask unchanged** `valid[:, None] & (dims < dim)` (:848; the trtllm strided tables backend :1623-1636 rely on unused columns never being written). Same for V. Masked-off lanes issue no transactions, so "load both, select" costs no bandwidth; two [16,256] fp32 tiles at num_warps=8 is ~32 regs/thread. Do not branch on `tl.max(hot)`. `dim` is asserted power-of-two (:981) so no BLOCK_D. Wrapper branch: first `if` in `qwen_sparse_kv_extraction_compact_triton` (:970-976): `if kv_bits == 4 and owner is not None: _compact_kv_tiered[...]; return` (new kwargs `ring_k=, ring_v=, ring_ks=, ring_vs=, owner=, ring_mask=` default None). Two-pass fallback if Triton misbehaves: existing `_compact_kv_int4` over all valid rows, then `_compact_kv_int8` with `valid &= hot` over the ring (+1 captured launch/layer ~36 us/step = 0.2 %).

c) Prefix-chunk row gather (eager): `_gather_dequant_rows_tiered` = `_gather_dequant_rows_int4` (:858-913) with the same fusion (rows t < length -> slots :890 -> tier test -> both dequants -> `tl.where` -> one masked store, masks :895-897). Wrapper `qwen_sparse_prefix_gather_dequant_tiered(...)` = :914-952 + ring/owner args; backend :1490-1494 `gather_dequant = (tiered if kv_tiered else int4 if kv_bits == 4 else int8)` and pass the ring tensors. Side effect: rows inside the window use the int8 path (no nibble unpack), so 10k-prompt prefill should move back from int4's 1493 toward int8's 2316 tok/s (CAMPAIGN.md:406); prompts >> W stay int4-bound.

d) int8->int4 conversion (compactor mode only): `_requant_kv_int8_to_int4[(n, H)]`: program (t, h), `slot = start + t`, `r = slot & RING_MASK`; load 256 int8 + 4 fp16 from the ring row (:583-591 math), `x = q8 * s8` (sm identity), then the `_quant_store_kv_int4` body from `ge/go = tl.reshape(...)` (:719-752) to the int4 row at `slot` (fresh 32-channel scales; do NOT reuse s8*127/7, that would be int4-g64). Torch reference: `(q8.float() * s8.repeat_interleave(64, -1))` then `ref_quant` (gemv/test_kv_int4.py:55-61). Error adds in quadrature: sqrt(9.7^2 + 0.62^2) = 9.72 % RMS vs 9.7 %.

Backend glue: `_int8_gather_kwargs` (backend :1595-1604) returns the extra ring/owner kwargs when `getattr(full_pool, "kv_tiered", False)`; both compact call sites (:1666-1683, :1798-1815) and the prefix block (:1480-1510) are covered through it. `_kv_bits()`/`_kv_head_dim`/`_kv_scratch_dtype` unchanged. CPU-fallback guards (:1437, :1745) already raise for `kv_bits == 4`.

Cost model (decode, per step): gather per layer between 1.18 MB (all cold = int4 today) and 2.16 MB (all hot = int8 today) + 32 B owner reads per tile; both endpoints measured at parity (int8 56.1-57.4, int4 55-57 tok/s, CAMPAIGN.md:370/406); worst case +12 MB/step ~ +20 us = 0.1 % of 17.6 ms. Write per token +12 x 576 B int4 (negligible). Prefill per 1024 chunk: +7 MB int4 writes vs ~437 ms/chunk.

### 4. Demotion

4a) Dual-write (default, `SGLANG_KV_TIERS_MODE=dual`): there is no compactor and no tier flip. Invariant: for every slot s, the int4 row at s is written in the same launch as its ring row; ring row `s & MASK` is owned by s until slot s+R is written (owner overwritten in that later launch). Any reader sees either (owner == s: ring row complete for that layer, because layer l's write :1726 precedes layer l's gather :1730 on the same stream) or (owner != s: int4 row, which exists). No dependence on the scheduler, overlap mode, page size, idle hooks, slot ordering, radix, batching, or speculative padding. Slot-0 dummy writes (capture, backend :892/:989; paged.py:360 reserves slot 0) set `owner[0] = 0` and clobber ring row 0: a later reader of slot k*R sees `owner != slot` and correctly falls back to its int4 row — self-healing, no trash row needed. Window semantics: exactly the last R written slots are int8 (for one request = the last R tokens).

4b) Compactor (optional `SGLANG_KV_TIERS_MODE=compact`, if int4-from-int8 is ever wanted; needs `tier_map` uint8[rows] instead of the owner table, `tier[slot] = 1` stored by the write kernel, and a trash ring row `r = where(slot == 0, R, slot & MASK)` with R+1 ring rows):
- Trigger: `compost_tick(frontier)` from `_lazy_hook` (paged.py:149-160) right after `ensure(...)` at :156 with `frontier = (pages.max()+1)*page_size`. With page 64 this fires once per 64 decode tokens (`get_num_new_pages(decode=True)` counts `seq_lens % 64 == 1`, utils/common.py:4475-4478, via `alloc_decode` :282-283) and once per prefill chunk (`alloc_extend` :241). With the page-1 token allocator (kv_lazy.py:283-294 hook) it fires every step: throttle inside the tick to `cut - _demoted_to >= 64`.
- Algorithm: `cut = ((frontier - W) // 64) * 64`; if `cut > _demoted_to`: for l in 12 layers launch `_requant_kv_int8_to_int4(start=_demoted_to, n=cut-_demoted_to)`; then `tier_map[_demoted_to:cut].zero_()`; `_demoted_to = cut`. Slots are a contiguous ascending range per request (`clear()` paged.py:359-363 hands out pages 1.. in order; only the single request allocates), so the range is two scalars. Add the invariant check `int(pages.min()) * 64 >= _demoted_to` (the hook already synced at :156). Ring bound: the occupant of ring row `s & MASK` is s-R, which must be demoted before s is written: `R > W + 62` for decode, `R >= W + C + 64` across a prefill chunk (tick precedes the chunk with frontier = chunk end) -> R = 9,280 for C = 1024 (or 16,384 for a power-of-two mask).
- Safety vs in-flight forwards and CUDA graphs: forwards run on the scheduler thread in non-overlap mode (`--disable-overlap-schedule`, sweep.log:2, kept per CAMPAIGN.md:254), `run_batch` -> `forward_batch_generation` -> `process_batch_result` (host sync `next_token_ids.tolist()` in batch_result_processor.py) -> `prepare_for_decode` -> `alloc_decode`. So the tick is issued (a) with the GPU drained by `.item()` at paged.py:156 and (b) on the same stream as the graph replays (decode_cuda_graph_runner dispatched from model_runner on the current stream). Stream order alone guarantees the 12 requant launches and the tier zero-fill complete before the next replay reads them; the tier map and ring are static pointers baked into the graph, only their contents change. No VMM call, no synchronize. If overlap scheduling is ever enabled (forwards on `forward_stream`, scheduler.py:3670-3671; tp_worker.py:540) the tick must move to the backend's `init_forward_metadata` (backend :814-849) or wait on an event.
- Ordering of tier updates: requant launches for all 12 layers first, then one `tier_map[range].zero_()`; readers therefore see (tier 1, ring rows valid for every layer) or (tier 0, int4 rows valid for every layer). Reset at idle in `lazy_release` (memory_pool.py:2402-2415): `tier_map.zero_(); _demoted_to = 0` — required here (unlike dual-write) because tier 1 for a stale slot would read a clobbered ring row.

### 5. Selector

- Keep `--kv-cache-dtype int4_g32` (reuses kv_int4's server_args.py:699, kv_cache_dtype.py:20/64, pool_configurator cell-size edits unchanged). Add env `SGLANG_KV_TIERS_W=<ring slots>` (0/unset = plain int4 pool); `SGLANG_KV_TIERS_RING=int8|bf16` (default int8); `SGLANG_KV_TIERS_MODE=dual|compact` (default dual). Configurator edit at kv_cache_configurator.py:1547-1549: `MHATokenToKVPoolTiered if self.kv_cache_dtype_str == "int4_g32" and int(os.environ.get("SGLANG_KV_TIERS_W", "0")) > 0 else MHATokenToKVPoolInt4 if ...`; import next to :65.
- Optional alias `--kv-cache-dtype int8_int4_w8192`: would need server_args choices + kv_cache_dtype.py parse (`int8_int4_w(\d+)` -> uint8 + set the env) + the pool_configurator `elif` — three more edits for no functional gain; not recommended for the first cut. phase1.py step `S20_tiers`: add `SGLANG_KV_TIERS_W=8192` to the env string, `--kv-cache-dtype int4_g32`; `S20b_tiers_w2048` with W=2048 for the quality test.
- Patch file `patches/kv_tiers.py` in the kv_int4.py style (NEW_FILES = [tiered_kv_pool.py], EDITS anchored on kv_int4-inserted text: the `if k.dtype == torch.uint8 and kv_bits == 4:` branch :976, the `gather_dequant = (` block :1490-1494, `_int8_gather_kwargs` :1595-1604, the configurator `full_pool_class` :1547, the Hybrid forwarders :4097-4101, `get_kv_size_bytes` :2446); `apply()` refuses unless `kv_int4.int4_applied()`; revert kv_tiers before kv_int4.

### 6. Tests and validation protocol

Unit (`gemv/test_kv_tiers.py`, style of test_kv_int4.py:1-25, ~40 MB VRAM):
1. `_quant_store_kv_tiered`: int4 rows/scales bit-exact vs `quant_store_kv_int4` output; ring rows/scales bit-exact vs `quant_store_kv_int8` output at `slot & MASK` (plus the new fp16 clamp: absmax > 127*65504 group finite); owner == slot; int32 and int64 loc; slot 0; wrap: write s then s+R -> owner flips, int4 row of s intact.
2. `_compact_kv_tiered`: 3 requests (300/1200/4000 tokens, permuted req_to_token), random owner pattern -> rows bit-exact vs per-tier references (hot vs `_compact_kv_int8` reference dequant, cold vs int4 reference); rows beyond the packed range and invalid columns untouched; trtllm strided case (cu_k = arange*64, topk 40, -1 padding, pos >= seq_len); stale owner (owner = slot+R) -> int4 path; poisoned ring rows for cold slots never influence output.
3. `_gather_dequant_rows_tiered`: 3 lengths, 32-row gaps, mixed tiers, bit-exact, gaps untouched.
4. Pool: eager + `SGLANG_KV_LAZY=1` paths; ring shapes/dtypes; owner reset on `lazy_release`; `kv_tiered`, `kv_bits == 4`; `get_kv_size_bytes` includes the ring; `bytes_per_token()` of the owner still 1152 for 2 layers (ring not on the owner); `set_kv_buffer` with N > R raises.
5. Compactor mode (if built): requant bit-exact vs torch path; nibble mismatch fraction vs int4-from-bf16 < 2 %; `compost_tick` watermark math on a synthetic ascending allocation (never demotes within the last W; chunk sizes 1 and 1024; reset at idle).
6. Microbench: tiered compact gather time <= 1.10 x `_compact_kv_int4` for all-cold and <= 1.10 x `_compact_kv_int8` for all-hot at topk 2048, bs 1.

End-to-end (nll_series.sh / needle_series.sh; steps S20_tiers W=8192 and S20b_tiers_w2048), pass thresholds:
- nll_long ALL positions vs `nll/long_bf16kv.json` (noise +0.0002 / 0.059; int8 +0.0010 / 0.059; int4 +0.0088 / 0.138): W=8192 on the 9.6k text: NLL delta <= +0.003 and mean|dlogprob| <= 0.08 (nearly all read rows are hot); W=2048: delta in [+0.000, +0.0088] and mean|dlogprob| <= 0.138 (must not exceed int4). A W=8192 result at the int4 number means the hot path is not being taken (check owner/tier dispatch).
- needle 41k: 5/5 for both W (CAMPAIGN.md:403-404 protocol, 400 new tokens).
- decode bench: >= 55 tok/s (accepted 56.4; int4 55-57), all five contexts.
- prefill 10k: >= 1493 tok/s (no regression vs int4), target >= 1900 (hot rows skip the unpack); 162k/258k prompt: prefill within 5 % of 1441/1415 tok/s, decode >= 50, correct answer, server alive, VRAM free bottom >= 1.2 GB, `KV lazy commit` lines show the same commit cadence as S19.
- oracle 10k mean 0.0015 (unchanged).

### 7. Risks and pre-checks

1. Fused-kernel codegen (two differently shaped predicated loads + `tl.interleave` + `tl.where` on [16,256] tiles): pre-check with the unit test compile + microbench; fallback = two-pass (int4 all rows, int8 over hot rows).
2. Memory headroom: ring 104 MB fixed and outside the owner; 258k run bottomed at 1.46 GB free. Pre-check: run the 258k prompt and watch for `KV lazy backing: no headroom` (memory_pool.py:2378); if hit, raise `SGLANG_MOE_ELASTIC_FILL_MB` give-back or lower `SGLANG_KV_LAZY_HEADROOM_MB` by 128.
3. Ring row aliasing within one launch: assert `loc.numel() <= R` in `set_kv_buffer` (chunked prefill 1024, `--max-prefill-tokens 32768` with one request never exceeds the chunk; the assert catches any future change).
4. Owner width: int32 holds slots < 2^31 (262,208 rows); compare after `.to(tl.int64)`.
5. Speculative draft (`QwenSparseMultiStepDraftBackend._step_out_cache_loc`, backend :1856+): writes go through `set_kv_buffer` and are self-describing; verify rows use the same compact wrapper. Off in phase1_state.json; no design dependence, but untested.
6. Page size / allocator flavor only matter for compactor mode: sweep.log:2 carries `--page-size 1` while the QSA override sets 64 (arg_groups/overrides.py:671-672 logs "Setting page size to 64"); confirm the resolved `page_size=` line (:832) in the server log before relying on the 64-step cadence; the tick's `>= 64 slots` throttle covers page 1.
7. Overlap scheduling: dual-write is immune; compactor mode requires `--disable-overlap-schedule` (sweep.log:2) — add an assert at pool construction (`server_args.disable_overlap_schedule`) in compact mode.
8. Quality gain is unmeasured: the ladder gives the endpoints (+0.001 int8, +0.0088 int4); the value of an int8 recent window depends on how recency-weighted the QSA selection is. The W=2048 run is the discriminating measurement; if it lands at the int4 number, the tiering buys nothing and the plain int4 pool should stay.
9. VMM granularity assumed 2 MiB for the (A) figures only (the `KvVmmArena[...] ready: granularity=` line, kv_vmm_backing.py:108-116, is not in docs/logs); irrelevant to (B).
10. The int8 write kernel's missing fp16 clamp (:488/:498) is fixed in the tiered writer only; add it to `_quant_store_kv_int8` in kv_int8.py separately.

Key files: `$SGLANG/python/sglang/srt/layers/attention/qsa/sparse_attn.py`, `$SGLANG/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`, `$SGLANG/python/sglang/srt/mem_cache/int4_kv_pool.py`, `$SGLANG/python/sglang/srt/mem_cache/memory_pool.py`, `$SGLANG/python/sglang/srt/mem_cache/kv_vmm_backing.py`, `$SGLANG/python/sglang/srt/mem_cache/allocator/paged.py`, `$SGLANG/python/sglang/srt/mem_cache/kv_cache_configurator.py`, `patches/kv_int4.py` (patch template), new: `patches/kv_tiers.py`, `$SGLANG/python/sglang/srt/mem_cache/tiered_kv_pool.py`, `gemv/test_kv_tiers.py`.

## Reader summaries
[
 {
  "key": "layout",
  "confidence": "medium",
  "blockers": [
   "bytes_per_token() sums row_bytes over ALL specs (kv_vmm_backing.py:380-381); a full-span ring desc would add 12,672 B/token to lazy_ensure's headroom estimate (memory_pool.py:2366-2379), doubling the estimated delta and refusing commits long before 256k -> the compost pool must exclude ring descs from bytes_per_token (or lazy_ensure must use its own per-token figure) before any 256k run",
   "VMM granularity is assumed 2 MiB; the 'KvVmmArena[...] ready: ... granularity=' log line (kv_vmm_backing.py:108-116) is not present in docs/logs, so the granule-rounding figures for (A) and the ring packing must be confirmed on the live server",
   "Page size discrepancy: the base launch flags carry --page-size 1 (sweep.log:2) while the brief states page 64; with page 1 the token allocator hook (kv_lazy.py:283-294) calls lazy_ensure every decode step (no natural 64-step cadence), so compost_tick must throttle itself (cut - _demoted_to >= 64). Confirm which allocator is live (qsa_kv_pool.py:25-35 states the full-KV allocator is paged with a multiple of the compress ratio)",
   "Stream safety of the allocator-hook tick relies on --disable-overlap-schedule (sweep.log:2; CAMPAIGN.md:254 kept it): forwards and the hook share schedule_stream (scheduler.py:1707-1708, 1746-1747). If overlap is ever enabled, forwards move to forward_stream (scheduler.py:3670-3671) and the tick must be issued from the backend's init_forward_metadata / init_forward_metadata_out_graph (qwen_sparse_attn_backend.py:814-849) on that stream",
   "Compost does not recover prefill speed: rows older than W are int4 during prefix-chunk prefill (tick precedes each chunk), so the -15 % int4 prefill cost (CAMPAIGN.md:406: 1493 vs 2316 tok/s) remains for long prompts; the gain is decode-side quality for the recent window only",
   "Demotion re-quantizes from the int8 rows (int8 -> bf16 temp -> int4) rather than from bf16; the extra error is bounded by the int8 step (measured +0.001 nats) but the combined path is unmeasured -- validate with the all-position NLL series and the needle test (CAMPAIGN.md:385-406 protocol)",
   "Tier-map reset depends on the idle hook firing between requests (paged.py:162-168 / kv_lazy.py:304-309): holds with --max-running-requests 1 and no radix cache; any future batching/radix would need the write path to store tier_map[loc] = 0 (one extra tl.store in _quant_store_kv_int8)",
   "Speculative/MTP write paths (QwenSparseMultiStepDraftBackend._step_out_cache_loc, backend 1856+) are not covered by this design; phase1_state.json shows no speculative flags, so assumed off",
   "The tier map and ring tensors are baked into the captured decode graph (compact wrapper called under capture, backend 1798-1815 / 1666-1683): they must be allocated once before capture and never reallocated; the int4 main buffers are already fixed VA under the owner"
  ],
  "numbers": [
   "int8 pool: 12,672 B/token (payload 12,288 + fp16 G64 scales 384) -- KV_INT8_PLAN.md:9-20",
   "int4 pool: 6,912 B/token (payload 6,144 + fp16 G32 scales 768) -- KV_INT4_PLAN.md:8-13, int4_kv_pool.py:62-76",
   "256k (N = 262,144): int4-only 1.81 GB; int8-only 3.32 GB; ideal tiered (W=8192) 1.859 GB = 7,092 B/token",
   "(B) ring R = 9,280 slots (W 8192 + chunk 1024 + page 64): 117.6 MB resident; total 1.93 GB = 7,360 B/token at 256k; R = 16,384: 207.6 MB, 2.02 GB = 7,700 B/token",
   "(B) overhead vs ideal: ~72 MB = 57 MB int4 rows under the window + 14 MB lag slack (0.27 KB/token)",
   "(B) ring desc packing: 4 descs (rk/rv (12*R, 512), rks/rvs (12*R, 16)) = 112 MiB vs 48 per-layer descs = 192 MiB under a 2 MiB granule",
   "(A) at 256k: 2.01-2.11 GB = 7,680-8,060 B/token (int8 window 151-201 MB incl. straddle granules + 50-100 MB int8 scale granules), plus 3.32 GB extra VA (int8 full-context) and a torch.cuda.synchronize per tick (kv_vmm_backing.py:371)",
   "granule geometry: int8 payload row 512 B -> 4,096 rows per 2 MiB granule; int8 scale row 16 B -> 131,072 rows per granule; int4 scale row 32 B -> 65,536 rows per granule",
   "demotion cadence: allocator hook fires once per new page = every 64 decode steps at page 64 (paged.py:149-160, common.py:4475-4478) and once per prefill chunk (paged.py:241); per tick 64 rows x 12 layers, or C rows per chunk",
   "ring overwrite bound: R > W + 62 (decode) / R >= W + C + 64 (guaranteed window >= W across a prefill chunk)",
   "new VMM code: (A) ~50 lines touching commit_range/_back_spans/uncommit_beyond/release_beyond + interval bookkeeping; (B) 0 lines (one ~6-line KvRingBufferDesc subclass in the pool file)",
   "measured ladder: int8 +0.001 nats, int4 +0.0078-0.0088, int2 +0.30; needle 41k int8 5/5, int4 5/5; int4 prefill 1493 vs int8 2316 tok/s at 10k -- CAMPAIGN.md:406"
  ]
 },
 {
  "key": "kernels",
  "confidence": "high",
  "blockers": [
   "Option A (range unmap) is not implementable on the current arena without redesign: per-buffer backing is a single monotonic high-water mark (kv_vmm_backing.py:167-182, early return :168-169; _back_spans skips below backed_to :356), uncommit_beyond strips only the tail (:190-215), and release_beyond needs a device-wide torch.cuda.synchronize (:371). A hole would never be re-backed on slot reuse -> illegal access. Go with (B).",
   "Slot 0 dummy writes alias ring row 0: the graph capture writes to out_cache_loc = zeros (qwen_sparse_attn_backend.py:892, :989; paged.py:360 reserves slot 0), and ring index slot&MASK maps slot 0 and slot W_RING to the same row. Reserve a trash ring row (r = where(slot==0, W_RING, slot&MASK)) in both the write kernel and the gathers.",
   "Ring + tier are fixed always-resident tensors (130-208 MB) not counted by pool_configurator's per-token cell size (kv_int4.py:270-278 edits) nor by the lazy headroom rule (kv_lazy.py:125-136): capacity profiling/admission must subtract them explicitly, otherwise the headroom check sees less free memory than it accounted for.",
   "Demotion must also run per prefill chunk (alloc_extend hook, kv_lazy.py:240-248), not only every N decode steps, and W_RING must exceed W + chunk size + N + page + draft tokens; otherwise a long prompt overwrites ring rows before they are demoted (silent corruption, no fault).",
   "Tier flip ordering: the tier[start:end] zero-fill must be enqueued on the same stream after all 12 per-layer demotion launches; with overlap scheduling re-enabled (currently off, sweep.sh:25) the allocator-hook launches would land on the scheduler stream while the forward runs on model_runner.forward_stream (tp_worker.py:540) -> add an event wait or move the compactor to the forward stream before ever enabling overlap.",
   "The int8 write kernel lacks the fp16 scale clamp (sparse_attn.py:488/:498 vs the int4 kernel :728; CAMPAIGN.md:392) -> add it to the ring writer or a >7*65504 group yields inf scale and NaN on demotion.",
   "Numerics unmeasured: int8->int4 double rounding at demotion (expected +0.02 % RMS) and the actual quality gain of a recent-int8 window (depends on how recency-weighted the QSA selection is). Validate with the all-position NLL vs bf16kv (must be <= int4's +0.0088, CAMPAIGN.md:406) and needle 5/5 (CAMPAIGN.md:403-404) before accepting; the bf16-ring variant (C) is the fallback if the int8 chain shows anything.",
   "Triton codegen risk in the fused kernel (two differently shaped loads, tl.interleave then tl.where on [16,256] tiles): if register pressure or a layout conversion misbehaves, use the two-pass scheme (int4 all rows, then int8 over tier==1 rows) at +1 launch/layer.",
   "trtllm strided path (backend :1623-1636) and speculative verify rows rely on the valid-columns-only store mask (sparse_attn.py:814-818); the tiered kernel must keep the store mask = valid (not hot|cold computed separately) so no unused column is written."
  ],
  "numbers": [
   "int8 row/token/layer 1056 B -> 12,672 B/token; int4 576 B -> 6,912 B/token (int8_kv_pool.py:1-8, int4_kv_pool.py:1-7, KV_INT4_PLAN.md:8-12)",
   "256k pure int8 3.32 GB; pure int4 1.81 GB; (A) 7,308 B/token = 1.92 GB physical + 3.3 GB extra VA; (B) 7,408 B/token = 1.94 GB (ring 10240 slots, 130 MB) or 7,705 B/token = 2.02 GB (ring 16384, 208 MB); (C) bf16 ring 7,873 B/token = 2.06 GB",
   "tier map uint8 [size+page_size] = 256 KB at 262,208 rows (memory_pool.py:2228)",
   "decode gather per layer: 1.18 MB (all cold) .. 2.16 MB (all hot) + 2 KB tier reads; 4.2 MB bf16 scratch write unchanged; worst-case +12 MB/step ~ +20 us = 0.1 % of 17.6 ms",
   "measured endpoints at parity: int8 56.1-57.4 tok/s (CAMPAIGN.md:370), int4 55-57 tok/s (CAMPAIGN.md:406); int4 prefill 1493 vs int8 2316 tok/s at 10k (CAMPAIGN.md:406)",
   "compactor: 64 slots x 12 layers = 1.25 MB moved + 13 launches per 64 steps ~ 1 us/step; per 1024-token prefill chunk ~20 MB ~ 30 us vs ~437 ms/chunk (KV_INT8_PLAN.md:225)",
   "two-pass fallback: +1 captured launch/layer ~ 36 us/step = 0.2 %",
   "ring slack: W_RING >= W + chunked_prefill_size (512, sweep.sh:23) + N + 64 + <=4 draft tokens; topk 2048 (config.py:54); BLOCK_TOPK 16 (sparse_attn.py:975)",
   "double-quantization int8->int4: sqrt(9.7^2+0.6^2) - 9.7 = +0.02 % RMS (unmeasured)",
   "VMM granule 2 MiB (CAMPAIGN.md:304); int8 scale rows 16 B -> 131k rows per granule, so option A frees no scale backing below 131k demoted rows"
  ]
 },
 {
  "key": "compactor",
  "confidence": "medium",
  "blockers": [
   "B1 (layout A only): kv_vmm_backing.py assumes prefix-contiguous backing (commit_range :167-183 `prev`, uncommit_beyond :190-214 tail-only, :212 decrements the high-water mark); head/range unmap needs a per-buffer [backed_from, backed_to) window model and re-map of the head hole after idle release; int8 scale granules (131072 tokens each) cannot be released before the window passes 131k.",
   "B2: new pool class + third dispatch branch in the compact wrapper (sparse_attn.py:970-1076) and backend helpers (qwen_sparse_attn_backend.py:1579-1600, scratch head_dim :1585-1587 must report 256 for the [rows,2,128] int4 payload); the trtllm strided path (:1653-1700) uses the same kernel.",
   "B3: fused tiered gather kernels (compact + prefix row gather) are new Triton code with two predicated sources per row; register pressure/num_warps untested; decode speed (int4 55-57 tok/s) must be re-measured.",
   "B4: no fake-quant preview possible \u2014 kv_fakeq hooks MHATokenToKVPool.set_kv_buffer, which the quantized pools override (CAMPAIGN.md:394); quality evidence comes only from the real pool, and nll_long's 9.6k text needs a small W (2048) to exercise the int4 tier at all.",
   "B5: the ~250k needle is blocked by host oomd on random-word prompts until the ple_random POSIX_FADV_DONTNEED fix is validated (CAMPAIGN.md:408).",
   "B6: ring buffers live outside the VMM owner: not in bytes_per_token()/lazy headroom or the pool_configurator cell size; must be allocated before the elastic fill and subtracted from capacity (104 MB at W=8192).",
   "B7: the ascending-slot invariant depends on max_running_requests 1 + no radix + idle clear(); retraction (scheduler.py:3496-3557) re-prefills and resets; add the pages.min() >= demoted_upto assert in the tick, and require W >= 2 x chunked_prefill_size.",
   "B8: read-only task, no GPU runs \u2014 all numbers analytic; the requant nibble-mismatch fraction, fused-kernel speed, and e2e NLL are predictions to be measured.",
   "B9: stage-B smoothing (currently identity in both pools) would require the requant kernel to apply sm8 * sm4_inv unless both pools share sm."
  ],
  "numbers": [
   "int8-g64: 12288 + 384 = 12672 B/token (12.4 KB); int4-g32: 6144 + 768 = 6912 B/token (6.75 KB)",
   "256k int8-only 3.32 GB; int4-only 1.81 GB",
   "(A) two full VMM buffers + range unmap: ~2.05 GB at 256k = 7.8 KB/token (int4 1.812 GB + int8 window 104 MB rounded to 2 MiB granules: 144 MiB payload + up to 96 MiB scales)",
   "(B)/(C) int4 full + int8 ring W=8192: 1.812 GB + 103.8 MB ring + 393 KB owner = 1.916 GB = 7.31 KB/token at 256k; 390 MB at 41k",
   "int4 copy of the window (duplicated) = 57 MB; ring is fixed-size, independent of context",
   "demotion trigger cadence: once per 64 decode tokens (page allocation, get_num_new_pages decode=True) and once per 1024-token prefill chunk; per-tick work 1 page (decode) / <=17 pages (prefill)",
   "requant error: sqrt(9.7^2 + 0.62^2) = 9.72 % RMS vs 9.7 % for int4-from-bf16 (expected nibble mismatch < 2 %, to be measured)",
   "VMM granularity 2048 KiB: payload granule = 4096 tokens (512 B rows), int8 scale granule = 131072 tokens (16 B rows)",
   "expected e2e: W=8192 nll_long ALL ~ int8 (+0.0010/0.059); W=2048 between +0.001 and +0.0088; needle 41k 5/5; prefill 10k toward 2316 tok/s, 256k ~1415 tok/s"
  ]
 }
]