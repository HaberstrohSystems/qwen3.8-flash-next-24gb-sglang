# Paged prefix-chunk prefill kernel: plan (2026-09-02)

> **Status:** built, verified within 2 row-ulps, rejected on timing — 6.1 ms vs 1.4 ms per head-layer at
> prefix 60k (CAMPAIGN.md:439); see the Outcome section at the end. The reader summary below
> ("+5-13 %, deferred") is superseded by that outcome. `patches/kv_paged_prefix.py` and
> `gemv/test_kv_paged_prefix.py` are kept, not applied.

Plan: `patches/kv_paged_prefix.py` — paged prefix-chunk attention for the int8 / int4 / tiered QSA pools

Verified on the reference host while reading: GPU = NVIDIA RTX PRO 4000 Blackwell, 24 GB, L2 = 48 MB (torch device props: 50331648 B), 70 SMs, torch 2.11.0+cu128, Triton 3.6.0; measured dense-read bandwidth 494-509 GB/s (from a pre-campaign idea list that is not part of the repository). Layering: kv_fp8 < kv_int8 < kv_int4 < kv_tiers < kv_paged_prefix (new).

## 0. Premise corrections (they change the numbers, not the design)

1. Budget: `indexer_budget` = 2048 TOKENS = 512 compressed blocks x 4 (qsa_indexer.py:77-79, :127-131); the index row is `token_topk + compress_ratio - 1` = 2051 wide (qsa_indexer.py:462-466; kernel.py:200-224 tail). The kernel's `topk = indices.shape[-1]` (sparse_attn.py:289) is 2051, not 2048 x 4.
2. Selection is per query TOKEN: grid `(max_q, batch * num_kv_heads)` = one query per program (sparse_attn.py:202-205, :279; indexer rows = tokens, qsa_indexer.py:461-505). There is no cross-query reuse inside the kernel; 1024 x 2 x 2051 = 4.2 M row reads per chunk-layer is the floor of any direct-paged kernel.
3. The 2271 -> 1508 tok/s headline is mostly NOT prefix length. Per-chunk fit on the 257,905-token tiered run (~/quant/logs/server-S21_tiers.log, 251 "Prefill batch" lines): `chunk_ms = 615 + 0.487 us x prefix_tokens`. At the SAME 1-9k prefixes the 10k bench prompt costs 414-440 ms/chunk vs 551-593 for the 258k prompt: ~145 ms of the 228 ms/chunk gap (64%) is prompt-content base cost (MoE expert movement / PLE), ~63 ms mean (28%; 124 ms at the last chunk) is the O(prefix) slope, ~20 ms (8%) are periodic ~1 s spikes every 16-18 chunks (unexplained; candidate memory_pool.py:2360-2388 lazy_ensure/empty_cache). Only the slope (and a bounded L2 effect, section 5) is addressable by this kernel.

## 1. What is replaced (anchors)

- qwen_sparse_attn_backend.py:1480-1512: for `k_buffer.dtype == torch.int8 or kv_bits == 4` allocate `packed_k/packed_v = torch.empty((sum(seq_lens), H, D), bf16)` (:1486-1489; 2 KB/token = 2 x 528 MB at 258k, per layer, per chunk) and launch `qwen_sparse_prefix_gather_dequant_{tiered,int4,int8}` over rows [0, seq_len) (:1490-1512; kernels sparse_attn.py:597-641 int8, :858-912 int4, :1224-1302 tiered, grid (batch, cdiv(max_len,16), heads)).
- :1538-1547: `sparse_gqa_fwd_interface_triton_ck(q, packed_k, packed_v, topk_indices, cu_seqlens_q, cu_seqlens_k, sequence_lens_tensor, scaling)` -> `_sparse_gqa_chunk_prefill` (sparse_attn.py:170-266): per program one query x one kv-head group; `visible/row_topk/row_limit` (:214-216); loop over BLOCK_N selected positions: `token` (:230), `valid = token >= 0` (:231), K loaded transposed [D, N] (:232-236), V [N, D] (:237-241), `tl.dot(q, keys)` (:242), online softmax (:243-249), epilogue (:250-258). `token` is the logical position inside the request; the packed buffer holds rows cu_k[b] + t (:227-228, gather dst sparse_attn.py:1265/1283).
- Note the current chunk's own rows (positions >= prefix_len) are committed to the pool BEFORE the gather (backend :1413-1416) and read back quantized by the gather (:1497-1512 covers the whole seq_len): today's path already attends to quantized current-chunk rows, so a fully paged kernel is numerically the same thing.
- Untouched: no-prefix first chunk (:1456-1465 -> `sparse_gqa_fwd_interface_triton`, sparse_attn.py:32-166), the fp8/bf16 `else` branch (:1513-1537, keeps index_select + cat + the ck kernel), decode/verify (`_compact_kv_*` + FA2/trtllm), the writers (kv_tiers.py:176-220).

## 2. Kernel spec: `_sparse_gqa_chunk_prefill_paged` (new, in sparse_attn.py, inserted before `def _check_tier_args(` :1305)

One kernel body, three pools via constexpr `MODE` (0 = INT8 pool, 1 = INT4 pool, 2 = TIERED). "Hot" lanes read an int8 row (ring row for TIERED, pool row for INT8), "cold" lanes read an int4 row; MODE fixes hot/cold at compile time for the pure pools so the other tier's loads and the owner load are compiled out.

Signature (replaces `k, v, cu_k` and the k/v strides of :171-201):
```
_sparse_gqa_chunk_prefill_paged(
    q, out, indices, cu_q, kv_lens,
    req_to_token, req_indices,
    k4, v4, ks4, vs4,          # INT4/TIERED: uint8 [rows,H,128] + fp16 [rows,H,8] (kv_int4.py:4-10); INT8: unused (pass k8)
    rk, rv, rks, rvs, owner,   # TIERED: ring int8 [R,H,256] + fp16 [R,H,4] + owner int32 [R] (kv_tiers.py:123-131)
                               # INT8: the int8 pool itself (k_buffer int8 [rows,H,256] + fp16 [rows,H,4]); owner unused
    sm_k, sm_v, scale, topk,
    sq_m, sq_h, sq_d, so_m, so_h, so_d, si_m, si_g, si_n: tl.constexpr,
    NUM_KV_HEADS, GROUP_SIZE, BLOCK_M, BLOCK_N, HEAD_DIM, REQ_STRIDE,
    GROUP4, GROUP8, RING_MASK, MODE: tl.constexpr)
```
Verbatim from the current kernel: prologue :202-209 (batch/group/query, early return), :212-226 (`kv_len`, `visible`, `row_topk`, `row_limit`, q tile load, `q * scale * 1.4426950408` pre-scale), index load :229-231, online softmax :242-249, epilogue :250-258. Removed: `cu_k`/`k_start` (:211, :227-228). Grid stays `(max_q, batch * num_kv_heads)`; BLOCK_M = 16 for GROUP_SIZE 12 (:274).

New loop-invariant prologue (as sparse_attn.py:1170-1171, :1195-1197): `req = tl.load(req_indices + batch)`; `slot_row = req_to_token + req * REQ_STRIDE`; `head = group`; `pairs = arange(128)`, `dims = arange(256)`; `sm_e_k, sm_o_k` ([128] f32, `sm_k + head*dim + 2*pairs (+1)`), `smr_k` ([256] f32), same for V. Smoothing is identity today (kv_int4.py:~64, kv_int8.py) but stays in the expression.

Per tile (BLOCK_N positions), in order:
1. `token = tl.load(idx_row + current*si_n, mask=current < topk, other=-1)`; `valid = token >= 0` (:230-231). Defensive addition: `valid &= token < kv_len` (an out-of-range index would otherwise walk req_to_token into a foreign request; today's kernel would read a foreign packed row instead). No numeric effect on valid inputs (kernel.py:206-214 bounds `tail < sequence_length`).
2. Gather A (slot): `slots = tl.load(slot_row + tl.where(valid, token, 0), mask=valid, other=0).to(tl.int64)` (copy of :1174-1178). 4 B/lane, random within one 1 MB row at 258k (L2-resident).
3. Tier test: TIERED: `r = slots & RING_MASK; o = tl.load(owner + r, mask=valid, other=-1).to(tl.int64); hot = valid & (o == slots); cold = valid & (o != slots)` (copy of :1180-1183; owner is 32 KB, L2-resident). INT8: `r = slots; hot = valid; cold = valid & False` (no loads). INT4: `hot = false; cold = valid`.
4. Gather K as [BLOCK_N, 256] (row-major, NOT the transposed load of :232-236):
   - cold: `src4 = (slots[:,None]*H + head)*128 + pairs`, `ssrc4 = (slots[:,None]*H + head)*8 + pairs // 16`, `pmask = cold[:,None] & (pairs < 128)`; `b = tl.load(k4 + src4, mask=pmask, other=0).to(int32)`; `sc = tl.load(ks4 + ssrc4, mask=pmask, other=0).to(f32)`; `lo = ((b & 15) - 8).to(f32) * sc * sm_e[None,:]`; `hi = ((b >> 4) - 8).to(f32) * sc * sm_o[None,:]`; `k4t = tl.interleave(lo, hi)` (:1187-1188, :1198-1202 character for character).
   - hot: `src8 = (r[:,None]*H + head)*256 + dims`, `ssrc8 = (r[:,None]*H + head)*4 + dims // 64`, `mask8 = hot[:,None] & (dims < 256)`; `sc8 = tl.load(rks + ssrc8, mask=mask8, other=0).to(f32)`; `k8t = tl.load(rk + src8, mask=mask8, other=0).to(f32) * sc8 * smr[None,:]` (:1191-1192, :1204-1205; for MODE=INT8 this is exactly `_gather_dequant_rows_int8`'s expression :631-633, same op order).
   - `keys_nd = tl.where(hot[:,None], k8t, k4t).to(tl.bfloat16)` (:1206) — this is the bf16 value the scratch holds today.
   - `scores = tl.where(valid[None,:], tl.dot(q_values, tl.trans(keys_nd)), -inf)` replaces :242. The `tl.trans` of a register-computed [16,256] bf16 tile is a shared-memory layout conversion (8 KB per tile). It cannot be avoided by loading [D, N]: the nibble unpack `tl.interleave` works along the last axis only (sparse_attn.py:1202), and for the int8 pool a [D, N] load of 1-byte elements is uncoalesced anyway. Variant T2 (perf fallback if T1's trans is expensive): hoist `qT = tl.trans(q_values)` out of the loop, `scores = tl.trans(tl.dot(keys_nd, qT))` — transposes a [N,16] f32 tile instead of [16,256] bf16, but swaps MMA operand roles (bit-identity must be re-checked; section 4).
5. Gather V as [BLOCK_N, 256]: identical to 4 with `v4/vs4/rv/rvs/sm_v` (:1209-1220); orientation already matches the current `values` load (:237-241), so `tl.dot(probabilities.to(bf16), values, acc*alpha)` (:245-247) is unchanged.
Masking: masked lanes load nothing (`other=0`), scores of invalid lanes are forced to -inf by `valid` exactly as today (:242); tail -1 entries and `current >= topk` lanes behave identically. `row_limit`'s rounding (:216) and the `visible` causality (:214) are unchanged.

Scale-load optimization (step 2, after the bit-exact baseline): `ks4 + ssrc4` issues 128 2-byte loads per row for 8 distinct values (16 with K+V); load `[N, 8]` once and expand by `sc[:, :, None]` broadcast + `tl.reshape` to `[N, 128]` (same values, so still bit-identical). Same for `rks` ([N, 4] -> [N, 256]).

Config: do NOT reuse `_get_best_config` (:12-29: non-H20 table, total_q 1024 -> BLOCK_N=16, num_warps=1, num_stages=2). One warp holding two [16,256] f32 dequant temporaries + the [16,256] f32 accumulator is ~384 f32/thread -> spills. Own table `_PAGED_CONFIG = (BLOCK_N=16, num_warps=4, num_stages=2)` first (peak ~190 regs/thread by count: b4/sc/lo/hi [16,128] + k4/k8/sc8 [16,256] transient, acc 32, q 16), with (16, 8, 2) and (32, 4, 2) as tuning alternatives. The address chain index -> slot -> owner -> row is 3 dependent loads per tile; add a source-level one-tile-ahead prefetch of `token`/`slots`/`o` (loads of iteration i+1 issued at the end of iteration i) if the microbench shows latency-bound behaviour rather than relying on the pipeliner. Also pass `max_q = max(extend_lens)` from the backend's CPU list (backend :1450) instead of the `.item()` sync of :274 (free; no numeric effect).

Wrapper `sparse_gqa_fwd_interface_paged(q, indices, cu_q, kv_lens, req_to_token, req_indices, k_buf, v_buf, k_scale, v_scale, sm_k, sm_v, scale, max_q, kv_bits, ring_k=None, ring_v=None, ring_ks=None, ring_vs=None, owner=None, ring_mask=None)`: asserts = `_check_tier_args` (:1305-1319) for tiered, the int8/int4 wrapper asserts (:646-660 / :915-930 style: dtype, scale shapes `[rows,H,D//64]` / `[rows,H,D//32]`, contiguity, `dim == next_pow2(dim)`); MODE from `(kv_bits, owner is not None)`; `REQ_STRIDE = req_to_token.stride(0)` (as :1341-1346); returns `out = torch.empty_like(q)`.

## 3. Backend wiring (qwen_sparse_attn_backend.py, edits expressed as kv_tiers-style `(path, old, new)` anchors)

- Import (BE:36-38 block; anchor = kv_tiers.py:739-745 text): add `sparse_gqa_fwd_interface_paged,` after `qwen_sparse_prefix_gather_dequant_tiered,`.
- Branch (anchor = kv_tiers.py:763-775 text `tier_kwargs = self._kv_tier_kwargs(layer)\n            gather_dequant = (`): 
```
if k_buffer.dtype == torch.int8 or kv_bits == 4:
    assert q.dtype == torch.bfloat16, ...
    k_sf, v_sf = pool.get_kv_scale_buffer(layer.layer_id)
    sm_k, sm_v = pool.get_kv_smooth_buffer(layer.layer_id)
    tier_kwargs = self._kv_tier_kwargs(layer)
    if _PAGED_PREFIX:                                   # module constant: os.environ.get("SGLANG_QSA_PAGED_PREFIX", "1") != "0"
        output = sparse_gqa_fwd_interface_paged(
            q.contiguous(), topk_indices, cu_seqlens_q, sequence_lens_tensor,
            req_to_token, forward_batch.req_pool_indices,
            k_buffer, v_buffer, k_sf, v_sf, sm_k, sm_v, layer.scaling,
            max(extend_lens), kv_bits if kv_bits == 4 else 8, **tier_kwargs)
        return self._pad_extend_output(output, num_output_rows)
    packed_shape = ...   # existing materialized path (:1486-1512) unchanged below this line
```
`cu_seqlens_k` (:1478) is only needed by the materialized/else path; leave it (one small cumsum) or move it under the fallback. `sequence_lens_tensor` (:1475-1477) stays (kv_lens). `SGLANG_QSA_PAGED_PREFIX=0` selects the materialized path (fallback env, default on); read once at import into a module constant so the per-call cost is a Python bool. Never graph-captured (prefix path is eager, :1484 comment).
- Pool tensors are the same objects the gather kernels already take (`get_key_buffer`/`get_value_buffer` memory_pool.py:1873-1877 forwarders, scale/ring/owner accessors kv_int4.py:175-183, kv_tiers.py:167-172, backend `_kv_tier_kwargs` :1611-1622), so the lazy-VMM base-pointer stability assumption is exactly today's.
- Patch mechanics: `kv_paged_prefix.py` imports `kv_tiers` (as kv_tiers.py:48 imports kv_int4), `apply()` refuses unless `kv_tiers.tiers_applied()` (kv_tiers.py:840) and every kv_tiers edit is APPLIED; `--check|apply|revert`; docstring states the order and that kv_tiers must be reverted AFTER it (phase1.py:53-58 `revert_patches` already reverses order). Add `"kv_paged_prefix": "kv_paged_prefix.py"` to phase1.py:38-43 and a step `("S22_paged", <S21 drops>, <S21 adds>, ["kv_int4","kv_tiers","kv_paged_prefix"], "SGLANG_KV_TIERS_W=8192", False)` next to S21_tiers (phase1.py:226-229). Optional hardening: a `paged_applied()` check in kv_tiers.revert() mirroring its int4 guard (:913-924) — or leave kv_tiers.py untouched and rely on the reversed order.

## 4. Unit test: `gemv/test_kv_paged_prefix.py` (pattern: test_kv_tiers.py sections 3-4, :277-316)

Setup (GPU, ~100 MB): pool of 4096 slots, H=2, D=256; ring R=64 (`SGLANG_KV_TIERS_W` semantics) so ring wrap/aliasing is exercised; pools filled by the REAL writers (`quant_store_kv_tiered` for tiered, `quant_store_kv_int8` / `quant_store_kv_int4` for the pure modes) from random bf16 K/V; owner pattern per ring row: -1 (post-lazy_release), the true owner, or a stale/stolen member of the slot class (as test_kv_tiers.py:3); `req_to_token` = random permutation per request (test_kv_tiers.py:284; NOT identity — this is the logical-position vs slot confusion test); batch 3 with seq_lens (300, 1200, 4000) and extend_lens (200, 700, 1100) -> prefix_lens (100, 500, 2900), cu_q ragged; `q` random bf16 [2000, 24, 256]; `indices` int32 [2000, 2051] built like the indexer's output (per query: sorted random positions < visible, valid-first then -1 padding; some rows with visible < topk, some rows with all -1, one row entirely from the current chunk).
Reference = the materialized path executed exactly as backend :1486-1547: `qwen_sparse_prefix_gather_dequant_{mode}` into fresh bf16 packed buffers + `sparse_gqa_fwd_interface_triton_ck`.
Assertions:
1. Bit-exact at matched tile config: launch `_sparse_gqa_chunk_prefill` (packed) and `_sparse_gqa_chunk_prefill_paged` DIRECTLY with the same (BLOCK_N, num_warps, num_stages) for (16,1,2), (16,4,2), (32,4,2): `torch.equal(out_paged, out_packed)` required for all three MODEs (isolates dequant/dot identity from reduction order).
2. Production config vs the table config (paged (16,4,2) vs packed (16,1,2), i.e. what the server will compare): report max |diff| in bf16 ulps and the count of differing elements; acceptance <= 1 ulp, differing fraction reported (reduction-order effect of tl.max/tl.sum :243/:248 across warps). If step 1 passes but step 2 shows diffs, they are reduction-order only.
3. Variant T2 (if built): same two checks; expect step 1 to fail on MMA operand-role swap -> T2 is acceptable only under the <= 1 ulp criterion.
4. Robustness vs a torch reference (fp32 attention over the per-tier torch-dequantized rows, `allclose` bf16 tolerance): indices >= seq_len are ignored (defensive mask), all-cold (owner = -1), all-hot (ring 4096 with owner = arange, as test_kv_tiers.py:7), stolen ring rows -> int4 taken, batch with one request having prefix > R (ring wrap).
5. Microbench (informational, also the pre-check of section 6): at L = 10k / 64k / 258k rows with a 1024-query chunk: (a) time `qwen_sparse_prefix_gather_dequant_tiered` alone (= term A per layer), (b) packed kernel over a 2 KB x L scratch with random vs "local" (adjacent-query-overlapping) indices (= term C's L2 effect: 20 MB scratch at 10k fits the 48 MB L2, 528 MB at 258k does not), (c) the paged kernel at each config. Report ms and effective GB/s vs 509 GB/s.

## 5. Cost model and expected speedup at 258k (tiered default)

Per chunk-layer (1024 queries x 2 kv-heads x 2051 rows = 4.2 M row selections):
- Today: (a) whole-prefix gather: read 576 B/row int4 (+1056 B/row hot) + 8 B tables, write 2048 B/row bf16 -> ~2.6 KB per prefix token per layer = 31 KB per prefix token per chunk (0.68-0.80 GB per chunk-layer at 258k; 0.047 us/prefix-token at 672 GB/s spec, ~0.06 at 509 measured); (b) the ck kernel then requests 4.2 M x 1 KB = 4.3 GB of 512 B random rows from a scratch that is 20 MB at 10k (L2-resident) and 1.06 GB at 258k (48 MB L2 misses; real DRAM traffic depends on adjacent-query top-k overlap).
- Paged: (a) = 0 (no 2 x 528 MB temporaries, no gather launch); (b) 4.2 M x 288 B (int4, cold) = 1.21 GB + hot_fraction x 1.01 GB (hot rows = the last 8192 slots; at most 8.7 MB distinct ring bytes per layer, L2-resident) + 50 MB index/slot/owner. Per chunk-layer cost becomes O(1) in prefix length. For the same top-k overlap, distinct row bytes are 3.5x smaller than bf16, so the 48 MB L2 covers 3.5x more distinct rows -> the L2 hit rate of (b) improves too. Whole 258k prompt (252 chunks x 12 layers): ~13-14 TB of row requests + ~1.0-1.2 TB gather -> ~3.3-6.2 TB, zero gather.
- Time: the measured slope is 0.487 us/prefix-token/chunk = 63 ms mean per chunk (124 ms at the last chunk). Term A at roofline is 10% of it; if the 16-row-program gather with per-element fp16 scale gathers (sparse_attn.py:1275-1279) runs at 1/5 of bandwidth, A is ~50-70%. The indexer's O(prefix) fp32 logits/mask/topk passes (qsa_indexer.py:47-58, :485-496; mqa.py:304-332; kernel.py:23-39; >= 3-4 KB per prefix token per layer + 262 kFLOP) are NOT touched and remain (>= 0.08-0.12 us/token).
- Expectation for the 257,905-token prompt (base 615 ms/chunk unchanged): removing A alone: 1520 (A at roofline) to 1580 tok/s (A = 50% of slope); plus the ck kernel's byte reduction if it is bandwidth-bound (up to ~4.5 ms/layer = ~50 ms/chunk): up to ~1700 tok/s. Hard ceiling from the entire slope = 1665 + C. Realistic: 1508 -> 1580-1700 tok/s (+5-13%). 10k bench: < 1% from the slope (2.4 ms of 451), plus whatever (b) gains from the 4.3 GB -> 1.2 GB request volume (the KV_TIERS_PLAN.md:86 target 1900-2316 at 10k is NOT reachable from this change alone unless the ck kernel is bandwidth-bound at 10k; measure). Reaching 2271 on the 258k prompt is impossible: its base is 570-615 ms/chunk vs 414-440 for the bench prompt.
- Indirect, unquantified: 528 MB x 2 of transient allocator footprint per layer-call disappears, which competes with the elastic expert cache at 258k (CAMPAIGN.md:407/413: cache 184 -> 192 only after the request; VRAM free bottomed at 1.46 GB). Measure separately (proxy: run the 258k prompt on the materialized path with `SGLANG_MOE_ELASTIC_FILL_MB` raised by 512).

## 6. Risks and pre-checks (do these before writing the kernel)

1. No per-kernel profile exists (no nsys/torch.profiler artefacts in perf/). Gate: the section-4.5 microbench, or a torch.profiler trace of 2-3 chunks at ~200k prefix, must attribute the 0.487 us/token slope among A (gather), B (indexer), C (ck L2 misses). If A + C < ~25 ms/chunk at 258k, the kernel buys < 4% and the indexer's fused score+topk (never materialising [rows, keys] fp32) is the better next step.
2. Bit-exactness: guaranteed for the bf16 operands (dequant copied from :1198-1206 / :631-633); NOT guaranteed for the dot: `tl.dot(q, tl.trans(keys_nd))` on a register-computed tile may pick a different MMA k-order than the transposed load of :232-236, and num_warps != 1 changes the tl.max/tl.sum order (:243, :248). Decide the acceptance criterion up front: bit-exact at matched config (test 4.1) AND <= 1 bf16 ulp at production config (test 4.2); end-to-end acceptance via nll_long (section 7), which is what CAMPAIGN.md:413 accepted S21 on.
3. Register pressure / codegen: (16,1,2) will spill; (16,4,2) is the first candidate; `tl.interleave` + `tl.where` on [16,256] tiles + `tl.trans` in one loop body is new codegen for Triton 3.6 — compile + microbench first; fallback = T2 (trans q once, trans scores), then BLOCK_N=32.
4. Memory-level parallelism: 2048 programs x 4 warps on 70 SMs with a 3-deep dependent address chain per tile is latency-exposed; the one-tile-ahead index/slot/owner prefetch is the lever; keep `num_stages` at 2-3 and check achieved GB/s in the microbench.
5. Correctness hazards specific to paging: logical position vs slot (req_to_token indirection; test with a permuted table), owner int32 vs slot int64 compare after `.to(tl.int64)` (:1181), `owner` shared by all 12 layers (kv_tiers.py:131) is consistent because layer l's set_kv_buffer (stamp + dual-write, kv_tiers.py:176-219) precedes layer l's attention on the same stream and stamping is idempotent across layers; same-launch collisions leave losers cold (kv_tiers.py:193-197); `lazy_release` fills owner with -1 (kv_tiers.py:224-229) -> all cold, fine. Out-of-range indices: the defensive `token < kv_len` mask.
6. Scope: int8 / int4 / tiered pools only; fp8/bf16 pools keep :1513-1537. The int4 payload + scales live on the lazy-VMM owner (kv_int4.py:8-10); the paged kernel reads through the same base pointers as the gather kernels (expected stable; verify with a 258k run that `KV lazy commit` lines keep S19/S21 cadence and no `KV lazy backing: no headroom`, memory_pool.py:2378).
7. Hot fraction (tiered traffic) depends on the indexer's selection recency and is unknown; the kernel inherits exactly today's tier decision, so numerics do not depend on it.
8. The periodic ~1 s spikes (8% of the gap) and the prompt-content base cost (64%) are outside this kernel's reach; do not attribute their movement to it.

## 7. Validation protocol (server work, after the unit test passes)

Bring-up as `tiers_validate.sh S22_paged` (a wrapper around tools/int8_validate.sh that is not part of the repository; the sequence is int8_validate.sh's: short NLL, 512-window nll_long vs bf16, oracle lp2 10k, bench_speed 200, longctx 222000 (~258k tokens), all-position NLL, needles) with these acceptance criteria:
1. nll_long ALL positions vs the MATERIALIZED path of the same tree: first `SGLANG_QSA_PAGED_PREFIX=0 ... nll_long.py save S22_mat` (NLL_LONG_ALL=1, cache at S 184 as tiers_validate.sh does), then with the paged path `NLL_LONG_ALL=1 NLL_LONG_TAG=S22_paged nll_long.py check S22_mat`: |NLL delta| < 0.001 (required) and mean|dlogprob| <= 0.06 (run-to-run noise floor is +0.0002 / 0.059, CAMPAIGN.md:407); also `check bf16kv` must reproduce S21's -0.0001 / 0.074 within noise (CAMPAIGN.md:413). Greedy 300-token continuation: divergence position reported, not gated (server is not run-to-run bitwise reproducible, logprob_diff.py docstring).
2. Oracle lp2 10k mean <= 0.002 (S21: 0.0019).
3. Prefill: 10k bench >= 2271 tok/s (no regression); 257,905-token prompt (longctx_test.py 222000) prefill >= 1580 tok/s (+5%, the "worth keeping" bar), target 1650-1700; re-fit `chunk_ms = B + s x prefix` from the "Prefill batch" lines of ~/quant/logs/server-S22_paged.log: expect s from 0.487 to <= 0.30 us/token with B unchanged (615 +- 20 ms); decode >= 50 tok/s during/after; correct answer; server alive; VRAM free bottom >= 1.2 GB.
4. Needles: 41k (`needle_series.sh 60000 cur`) 5/5 and 248k (`needle_test.py 360000 S22_paged_248k`, keepalive 1800, ple_random page-drop in place) 5/5 with prefill >= 257 s-equivalent rate (S21: 257 s).
5. Fallback check: the same server restarted with `SGLANG_QSA_PAGED_PREFIX=0` must reproduce S21's numbers (proves the env switch and that no other edit changed behaviour).
6. Record in CAMPAIGN.md with the per-chunk slope before/after; if s does not drop by at least the microbench-predicted A term, the kernel is latency-bound -> tune (prefetch, warps, BLOCK_N) before accepting.

Files: $SGLANG/python/sglang/srt/layers/attention/qsa/sparse_attn.py (:12-29, :170-266, :269-312, :597-641, :858-912, :1129-1221, :1224-1302, :1305-1319), $SGLANG/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py (:33-42, :1413-1416, :1449-1481, :1480-1512, :1513-1547, :1584-1588, :1611-1622), $SGLANG/python/sglang/srt/layers/attention/qsa/qsa_indexer.py (:47-58, :77-79, :461-505), $SGLANG/python/sglang/srt/layers/attention/qsa/kernel.py (:200-224, :237), patches/kv_tiers.py (:1-45, :91-131, :167-229, :235-810, :840-941), patches/kv_int4.py (:1-30, :175-183), gemv/test_kv_tiers.py (:1-30, :277-316), scripts/phase1.py (:38-43, :53-58, :226-229), tools/int8_validate.sh (and its tiers_validate.sh wrapper, not included), tools/nll_long.py (:22, :52), CAMPAIGN.md (:407, :413), KV_TIERS_PLAN.md (:86, :90-96), the pre-campaign idea list (not included), the S21 server log (not included).

## Reader summaries
[
 {
  "key": "kernel",
  "confidence": "high",
  "blockers": [
   "Premise correction: the budget is 2048 TOKENS = 512 compressed blocks x 4 (config.json:21-22, qsa_indexer.py:79), and the index row is 2051 wide (kernel.py:237); the task's '2048 compressed positions x 4 tokens' would be 4x the real width. All traffic numbers above use 2051.",
   "Top-k is per query TOKEN (indexer rows = tokens, qsa_indexer.py:473-505; grid sparse_attn.py:202,:279), not per query block: there is no cross-query reuse, so 4.2 M row reads per chunk-layer is the floor of any direct-paged kernel; a 'union of selected rows' compaction is the only way below that.",
   "Bit-exactness vs today cannot be asserted from reading: identical bf16 operands are guaranteed by copying sparse_attn.py:1198-1206, but tl.dot on a register-computed [N,D] tile (needs tl.trans) may pick a different MMA k-order than the transposed load of :232-236 -> requires a GPU bit-compare test of paged vs packed interface (not runnable here: read-only, no GPU).",
   "The existing config (BLOCK_N=16, 1 warp, 2 stages; sparse_attn.py:18-29) will spill with fused f32 dequant tiles; the paged kernel needs its own tuned config (suggest num_warps=4, BLOCK_N 16-32, explicit one-tile-ahead prefetch of index/slot/owner) -- untuned until measured.",
   "No kernel-level profile of the 258k prefill exists in the perf dir (no nsys/torch.profiler artefacts found): the estimate is analytical. Part of the 2271->1508 tok/s drop may come from the indexer's O(prefix) MQA scoring per chunk (qsa_indexer.py:485-496 qsa_mqa_prefill over all compressed keys), which this kernel does not touch.",
   "L2 size / DRAM bandwidth of the RTX PRO 4000 Blackwell were not verified from files; the '10k scratch is L2-resident, 258k is not' reasoning is an assumption.",
   "Scope: the paged kernel covers the int8 / int4 / tiered pools only; the fp8 and bf16 pools keep the index_select+cat path (BE:1519-1541). The int4/scale pool lives on the lazy-VMM owner (kv_int4.py:8-10); confirm the base pointers passed to the kernel are stable across lazy commits (the existing gather kernels already rely on this, so it is expected to hold).",
   "Rows within the ring are hot only while owner[slot & 8191] == slot (kv_tiers.py:15-18); a prefill chunk's own rows and the last ~8k written slots are int8, older prefix rows int4 -- the paged kernel inherits exactly today's tier decision (same test as sparse_attn.py:1180-1183), so numerics match, but the hot fraction (and thus the tiered traffic) depends on the indexer's selection recency and is not known."
  ],
  "numbers": [
   "Top-k width in the kernel: topk = indices.shape[-1] = 2051 = 512 blocks x 4 + 3 tail (config.json:21-22 budget 2048 / ratio 4; qsa_indexer.py:77-79; kernel.py:237; sparse_attn.py:289) -- NOT 2048 compressed x 4",
   "Grid: (max_q=1024, batch x 2 kv heads) = 2048 programs per chunk, one query TOKEN each (sparse_attn.py:202-205, :279); BLOCK_M=16 for GROUP_SIZE=12 (:274; config.json:100,104); <=129 tiles of BLOCK_N=16 per program",
   "Current tile config on 'NVIDIA RTX PRO 4000 Blackwell' (nvidia-smi): non-H20 table, total_q=1024 > 512 -> BLOCK_N=16, num_warps=1, num_stages=2 (sparse_attn.py:18-29)",
   "Bytes per selected row (K+V): bf16 scratch 1024 B; int8 ring 2x(256+8)=528 B; int4 row 2x(128+16)=288 B; plus 12 B index/slot/owner (kv_int4.py:4-6, kv_tiers.py:123-131)",
   "Per chunk-layer selected-row requests: 1024 x 2 x 2051 = 4.2 M rows -> today 4.30 GB (bf16 scratch); paged int8 2.22 GB, int4 1.21 GB, tiered 1.21 GB + hot_frac x 1.01 GB (+50 MB metadata)",
   "Whole-prefix gather today per chunk-layer at L=258k: int8 0.80 GB (272 MB read + 528 MB write), int4/tiered ~0.68 GB; 2 x 528 MB bf16 temporaries per call (BE:1487-1489; KV_INT4_PLAN.md:52-53) -> 0 in the paged variant",
   "258k prompt totals (252 chunks x 12 layers): today ~13 TB scratch-row requests + ~1.0-1.2 TB gather/materialize -> paged ~3.3-6.2 TB; per-chunk cost becomes O(1) in prefix length",
   "Measured today: 2271 tok/s at 10k; 258k prompt 1508 tiered / 1415 int4 / 1694 int8 (CAMPAIGN.md:372, :407, :413; KV_TIERS_PLAN.md:86)",
   "Register estimate: two [16,256] f32 dequant tiles + [16,256] f32 accumulator = 12288 f32 -> 384/thread at 1 warp (spills) vs 96/thread at num_warps=4"
  ]
 },
 {
  "key": "cost",
  "confidence": "medium",
  "blockers": [
   "No per-kernel profile exists: the split of the 0.487 us/prefix-token slope between the whole-prefix gather (A), the indexer's fp32 logits/mask/topk passes (B) and the chunk kernel's L2 misses (C) is roofline-estimated only (A = 10% of the slope at roofline, up to ~70% if it is the inefficient kernel). A torch.profiler/nsys trace of 2-3 chunks at ~200k prefix is required before committing to the kernel; not done here (read-only, no GPU work).",
   "The headline comparison (2271 tok/s at 10k vs 1508 at 258k) conflates prompt content with prefix length: ~64% of the per-chunk gap is base cost specific to the 258k prompt (570-615 ms/chunk at 1-9k prefix vs 414-440 for the bench prompt on the same server). No attention kernel recovers that; only the O(prefix) 28% (+ a bounded chunk-kernel byte saving) is addressable.",
   "GPU roofline numbers (RTX PRO 4000 Blackwell ~672 GB/s, L2 size) are from memory, not measured or found in the repo; device name comes from the server logs (device_name=NVIDIA_RTX_PRO_4000_Blackwell).",
   "Periodic ~1.0-1.3 s chunk spikes every ~16-18 chunks in the 258k run (chunks 10, 28, 43, ...) are unexplained (candidate: lazy VMM commit / elastic give-back, memory_pool.py:2361-2383); ~8% of the gap, unaffected by the kernel.",
   "The int4-mode 257,905-token run's per-chunk lines are not in the surviving server-S19_int4kv.log (log restarted 19:41); the int4 slope (0.79 us/token) is from the random-text needle prompt and is noisy.",
   "Bit-exactness with the current path is guaranteed only at the identical Triton config (16,1,2); the dequant-in-kernel variant will almost certainly need num_warps=4 / larger BLOCK_N for register pressure, which changes tl.max/tl.sum reduction order -> the acceptance criterion (bit-exact vs <= 1 bf16 ulp) must be decided up front.",
   "Indirect benefit (528 MB scratch freed -> more resident experts -> lower base cost) is real (CAMPAIGN.md:407/413 show the elastic cache shrinking during the run) but unquantified; it may be the larger win and should be measured separately (e.g. run the 258k prompt with SGLANG_MOE_ELASTIC_FILL_MB raised by 512 as a proxy)."
  ],
  "numbers": [
   "topk per query = 2051 token indices = 512 compressed blocks x 4 + 3 (config.json indexer_budget 2048, ratio 4; qsa_indexer.py:462-466, :77-79) -- not 2048 x 4",
   "Tiered 258k run fit: chunk_ms = 615 + 0.487 us/prefix-token (40.6 ns per prefix token per QSA layer); mean chunk 679 ms = 1516 tok/s (server-S21_tiers.log 20:35:15 run; CAMPAIGN.md:413 says 1508)",
   "Same server, 10k bench prompt: chunks 2-9 at prefix 1-8k = 414-440 ms; 258k prompt chunks 2-10 at the same prefix = 551-593 ms -> ~145 ms/chunk (64% of the 228 ms gap) is prompt-content base cost, not prefix length",
   "O(prefix) slope share of the gap: ~63 ms mean per chunk (28%), 124 ms at the last chunk; periodic ~1 s spikes ~20 ms mean (8%)",
   "Gather traffic today: ~2.6 KB per prefix token per layer (576 B int4 read + 2048 B bf16 write + 8 B tables) = 31 KB per prefix token per chunk -> 0.047 us/token at 672 GB/s = ~10% of the measured slope if at roofline",
   "Indexer O(prefix) traffic: >= 3-4 KB per prefix token per layer (fp32 logits write, mask rewrite, topk read; mqa.py:304-332, kernel.py:39) + 262 kFLOP per prefix token per layer -> ~0.08-0.12 us/token at roofline; remains after the paged kernel",
   "Scratch removed: 2 x 2 KB x L_total per layer per chunk = 528 MB at 258k (backend :1477-1481)",
   "Paged kernel bytes per query-group: 2051 x 528 B = 1.08 MB (int8) / 2051 x 288 B = 0.59 MB (int4/cold tiered) vs 2.0 MB bf16 today; per layer per chunk 2.2 GB / 1.2 GB vs 4.2 GB (pre-L2)",
   "Chunk kernel config on this GPU: BLOCK_N=16, num_warps=1, num_stages=2 (sparse_attn.py:12-29, L20 table, total_q=1024); grid (1024, 2) = 2048 one-warp programs (:279)",
   "Achievable at 258k for this prompt: 1508 -> ~1580-1700 tok/s (+5-13%); hard ceiling from removing the entire slope = 1665 tok/s; 10k figure moves < 1% (2.4 ms of 451 ms/chunk)",
   "Base cost references: 437 ms/chunk PCIe-expert-gather bound (KV_INT8_PLAN.md:225); random-text needle prompt base 1001 ms/chunk with the same slope 0.498 us/token (S21 log 20:39:29 run)",
   "Owner table: one int32 [8192] for all 12 layers (kv_tiers.py:131); ring 104 MB; int4 row 288 B per head incl. 16 B scales, int8 ring row 528 B per head (kv_int4.py:58-60, sparse_attn.py:1274-1279)"
  ]
 }
]
## Outcome (2026-09-03, workflow wsdi1xxzv)
Implemented as patches/kv_paged_prefix.py + gemv/test_kv_paged_prefix.py (correct within 2 row-ulps of the
packed kernel, all three pool modes). REJECTED on timing (prefix 60000, chunk 1024, one KV head, cold L2): materialised
gather 0.18 ms + packed kernel 1.14 ms = 1.4 ms vs paged 6.1 ms (int8 3.2 ms, int4 4.2 ms). The in-kernel dequant is
repeated for every query tile, and the gather it removes is only ~2.5 % of the chunk time at 258k (about 19 ms of
741 ms); the O(prefix) slope is the QSA indexer. Not applied; kv_tiers stays the accepted tree. A future variant would
have to dequantize each selected row once per chunk (i.e. a gather), which is the materialised path.
