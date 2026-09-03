#!/usr/bin/env python3
"""Paged prefix-chunk prefill kernel for the int8 / int4 / tiered QSA KV pools -- perf/KV_PAGED_PREFIX_PLAN.md.

Today (kv_int8 / kv_int4 / kv_tiers) every prefix-chunk prefill of a QSA layer first materialises the WHOLE
context of every request in the batch as bf16 (`qwen_sparse_prefix_gather_dequant_{int8,int4,tiered}` into
two per-call `torch.empty` temporaries of 2 KB/token, qwen_sparse_attn_backend.py "packed_k/packed_v") and
then runs `_sparse_gqa_chunk_prefill` over that scratch: O(prefix) bytes per chunk per layer (0.487 us per
prefix token per chunk measured on the 258k-token S21 run, 2 x 528 MB temporaries per layer call).

This patch adds `_sparse_gqa_chunk_prefill_paged` (sparse_attn.py, before `_check_tier_args`): the same
per-query online-softmax kernel, but each BLOCK_N tile of selected positions is resolved on the device
(index -> slot via req_to_token -> tier via the owner table) and the K/V rows are gathered + dequantized
from the pool in registers (the dequant arithmetic of `_gather_dequant_rows_{int8,int4,tiered}` per element, so
the bf16 operands are the ones the scratch would hold; K is computed straight in the [D, N] orientation the packed
kernel loads keys in and int4 nibbles are picked by channel parity: on Triton 3.7.1 `tl.trans` of a register tile
changed the MMA accumulation order and a `tl.interleave` tile was mis-lowered as a dot operand -- see the unit test).  Constexpr MODE selects the pool:
0 = int8_g64 pool, 1 = int4_g32 pool, 2 = tiered (int8 ring over int4; hot = owner[slot & MASK] == slot).
No prefix-sized temporaries, no gather launch, per-chunk cost O(1) in the prefix length.  The defensive
`token < kv_len` mask keeps an out-of-range index from walking req_to_token into a foreign request.

Backend (qwen_sparse_attn_backend.py, the `k_buffer.dtype == torch.int8 or kv_bits == 4` prefix branch):
`sparse_gqa_fwd_interface_paged(...)` replaces gather + `sparse_gqa_fwd_interface_triton_ck` when the module
constant `_PAGED_PREFIX` is true.  DEFAULT = PAGED ON once this patch is applied (env SGLANG_KV_PAGED_PREFIX unset
or anything but "0"); SGLANG_KV_PAGED_PREFIX=0 is the fallback to the materialised path (byte-identical to kv_tiers
behaviour; SGLANG_QSA_PAGED_PREFIX is accepted as an alias).  The backend logs once at import which path is
selected ("QSA prefix-chunk attention (int8/int4/tiered pools): paged kernel" / "materialised gather + packed
kernel").  fp8 / bf16 pools, the no-prefix first chunk and decode/verify are untouched.  Tile config
`_PAGED_CONFIG = (16, 4, 2)` (BLOCK_N, num_warps, num_stages): the ck table's (16, 1, 2) would spill with the
fused f32 dequant tiles.  The wrapper takes cu_q as int32 or int64 (the backend's cu_seqlens_q is
`F.pad(int32.cumsum(0), (1, 0))` = int64: torch.cumsum promotes integer inputs).

Numerics (perf/gemv/test_kv_paged_prefix.py, Triton 3.7.1 / RTX PRO 4000 Blackwell): NOT bit-exact vs the
materialised path, at any tile config: the bf16 K/V operands are identical (rows with exactly one valid position
return the gather's bf16 V row bit-exactly, and the operands were verified tile by tile), but a computed K operand
changes the fp32 MMA accumulation order of the QK dot (a loaded [D, N] tile is bit-exact, a computed one is not,
V computed is bit-exact) -> 0.04-0.08 % of the outputs differ.  Both kernels cast the probabilities to bf16 for the
PV dot, so a score difference that flips one bf16 rounding of a dominant p_i moves the output by ~2^-8 |p_i v_i|:
the differences scale with the ROW (measured max 1.0 ulp of the (query, head) row's absmax; per element up to ~15
ulps on individual elements, a per-element bound is not attainable -- the production kernel against itself at
another tile config, (16,1,2) vs (32,4,2), shows 3.7 % differing elements with the same profile), and both kernels
are equally far from an fp32 torch reference.  Bit-exact by construction and asserted: the K/V operand tiles vs the
gather rows, and single-valid-position rows vs the gather's V row.  MEASURED (same test, the "TIMING" line; prefix 60k,
chunk 1024 = the production chunk, 1 kv head, 64 MB memset before every "cold" launch): materialised 1.31 warm /
1.40 ms cold (gather 0.18 + packed kernel (16,1,2) 1.10-1.14) vs paged tiered 6.10 ms (16,4,2), int8 3.19, int4 4.21
-> the paged kernel is 4.4x (tiered) / 2.3x (int8) / 3.0x (int4) SLOWER than the whole materialised path, cold or
warm alike (both kernels are insensitive to L2 state at this size): the dequant is redone per (query, selected row) = 2051 x queries row-dequants per
layer-chunk (2.1 M at 1024 queries, more than any prefix in scope) and the kernel is instruction-bound (scalar 1-2 B
loads + f32 dequant ALU per element vs 16 B bf16 vector loads), while the whole-prefix gather is cheap per token.
The 60k / 1-head microbench cannot reproduce the 258k regime (2 x 528 MB scratch) under the 200 MB test budget; the
258k slope-fit of plan section 6.1 remains the decision criterion.  This patch is kept as the measured result and
for further tuning; do not expect S22_paged to beat S21 unless the per-tile instruction count is cut ~5x.

ORDERING: layered on perf/patches/kv_tiers.py (kv_fp8 < kv_int8 < kv_int4 < kv_tiers < kv_paged_prefix).
apply() refuses (exit 1) unless every kv_tiers edit is APPLIED.  While this patch is applied, kv_tiers' sparse_attn
edit 5 reads MISMATCH (edit 1 here marks its anchor line, so `kv_tiers.py apply` cannot re-insert the tiered block)
and kv_tiers.py's apply()/revert() refuse (exit 1) while the paged kernel is in the tree.  Revert this patch BEFORE
kv_tiers (phase1.py's revert_patches already reverses the order); revert() here refuses (exit 1, --force overrides)
when kv_tiers' non-overlaid edits are no longer intact underneath.

  python3 kv_paged_prefix.py --check | apply | revert [--force]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_tiers                                            # prerequisite patch (same directory)
SG = os.path.join(os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")), "python/sglang")
SA = f"{SG}/srt/layers/attention/qsa/sparse_attn.py"
BK = f"{SG}/srt/layers/attention/qwen_sparse_attn_backend.py"

EDITS = [
  # ------------------------------------------------------------- sparse_attn: the paged chunk kernel + wrapper
  (SA, """def _check_tier_args(heads, dim, k_scale, v_scale, ring_k, ring_v, ring_ks, ring_vs, owner, ring_mask):
""", """_PAGED_CONFIG = (16, 4, 2)      # (BLOCK_N, num_warps, num_stages) of the paged chunk kernel (own table: the ck
                                # table's 1-warp config spills with the fused f32 dequant tiles)


@triton.jit
def _paged_tile(
    p4,
    ps4,
    p8,
    ps8,
    slots,
    r,
    hot,
    cold,
    head,
    dims,
    sm,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP4: tl.constexpr,
    GROUP8: tl.constexpr,
    MODE: tl.constexpr,
    TRANSPOSED: tl.constexpr,
):
    \"\"\"One f32 tile of dequantized K or V rows for the paged chunk kernel: [BLOCK_N, HEAD_DIM] (V, TRANSPOSED=False)
    or [HEAD_DIM, BLOCK_N] (K, TRANSPOSED=True: computed directly in the orientation the packed kernel loads keys in,
    so no tl.trans of a register tile -- on Triton 3.7 that changed the MMA accumulation order).
    MODE 0: int8 row r (= slot) of p8/ps8 -- the _gather_dequant_rows_int8 expression `q.to(f32) * scale * sm`;
    MODE 1: int4 row `slots` of p4/ps4 -- the _gather_dequant_rows_int4 expression `((nibble) - 8).to(f32) * scale * sm`
    with the nibble picked by channel parity (byte dims // 2, low nibble for even channels) instead of tl.interleave
    (whose join+reshape tile is mis-lowered as a dot operand on Triton 3.7: wrong values);
    MODE 2: int8 ring row r for hot lanes, int4 row for cold lanes, tl.where -- _gather_dequant_rows_tiered.
    Masked-off lanes issue no loads (other=0); the caller casts to bf16 (= what the scratch holds today).\"\"\"
    DH: tl.constexpr = HEAD_DIM // 2
    NG4: tl.constexpr = HEAD_DIM // GROUP4
    NG8: tl.constexpr = HEAD_DIM // GROUP8
    if TRANSPOSED:
        s2 = slots[None, :]
        r2 = r[None, :]
        hot2 = hot[None, :]
        cold2 = cold[None, :]
        d2 = dims[:, None]
        sm2 = sm[:, None]
    else:
        s2 = slots[:, None]
        r2 = r[:, None]
        hot2 = hot[:, None]
        cold2 = cold[:, None]
        d2 = dims[None, :]
        sm2 = sm[None, :]
    inb = d2 < HEAD_DIM
    if MODE == 0:
        mask8 = hot2 & inb
        sc8 = tl.load(ps8 + (r2 * HEADS + head) * NG8 + d2 // GROUP8, mask=mask8, other=0).to(tl.float32)
        x = tl.load(p8 + (r2 * HEADS + head) * HEAD_DIM + d2, mask=mask8, other=0).to(tl.float32) * sc8 * sm2
    else:
        pmask = cold2 & inb
        b = tl.load(p4 + (s2 * HEADS + head) * DH + d2 // 2, mask=pmask, other=0).to(tl.int32)
        sc = tl.load(ps4 + (s2 * HEADS + head) * NG4 + d2 // GROUP4, mask=pmask, other=0).to(tl.float32)
        nib = tl.where((d2 & 1) == 0, b & 15, b >> 4)
        x = (nib - 8).to(tl.float32) * sc * sm2
        if MODE == 2:
            mask8 = hot2 & inb
            sc8 = tl.load(ps8 + (r2 * HEADS + head) * NG8 + d2 // GROUP8, mask=mask8, other=0).to(tl.float32)
            x8 = tl.load(p8 + (r2 * HEADS + head) * HEAD_DIM + d2, mask=mask8, other=0).to(tl.float32) * sc8 * sm2
            x = tl.where(hot2, x8, x)
    return x


@triton.jit
def _sparse_gqa_chunk_prefill_paged(
    q,
    out,
    indices,
    cu_q,
    kv_lens,
    req_to_token,
    req_indices,
    k4,
    v4,
    ks4,
    vs4,
    rk,
    rv,
    rks,
    rvs,
    owner,
    sm_k,
    sm_v,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    GROUP4: tl.constexpr,
    GROUP8: tl.constexpr,
    RING_MASK: tl.constexpr,
    MODE: tl.constexpr,
):
    \"\"\"_sparse_gqa_chunk_prefill reading the quantized pool directly (no packed bf16 scratch).  Program =
    one query x one kv-head group, same prologue / online softmax / epilogue; per BLOCK_N tile the selected
    logical positions are mapped to slots through req_to_token[req_indices[batch]], the tier is decided on
    the device (MODE 2), and the K/V rows are dequantized in registers (_paged_tile: K straight in [D, N]).  Positions >= kv_len
    are treated as invalid (defensive: they would otherwise index a foreign request's slots).
    MODE 0: k4/v4/ks4/vs4 unused, rk/rv/rks/rvs = the int8 pool + scales, owner unused;
    MODE 1: rk/rv/rks/rvs/owner unused; MODE 2: int4 pool + int8 ring + owner (RING_MASK = R - 1).\"\"\"
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q + batch + 1)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q
        + query * sq_m
        + (group * GROUP_SIZE + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    idx_row = indices + query * si_m + group * si_g
    req = tl.load(req_indices + batch).to(tl.int64)
    slot_row = req_to_token + req * REQ_STRIDE
    head = group
    dims = offs_d
    smr_k = tl.load(sm_k + head * HEAD_DIM + dims).to(tl.float32)
    smr_v = tl.load(sm_v + head * HEAD_DIM + dims).to(tl.float32)
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = (token >= 0) & (token < kv_len)
        slots = tl.load(slot_row + tl.where(valid, token, 0), mask=valid, other=0).to(tl.int64)
        if MODE == 2:
            r = slots & RING_MASK
            o = tl.load(owner + r, mask=valid, other=-1).to(tl.int64)
            hot = valid & (o == slots)
            cold = valid & (o != slots)
        else:
            r = slots
            hot = valid
            cold = valid
        keys = _paged_tile(
            k4, ks4, rk, rks, slots, r, hot, cold, head, dims, smr_k,
            NUM_KV_HEADS, HEAD_DIM, GROUP4, GROUP8, MODE, True,
        ).to(tl.bfloat16)
        values = _paged_tile(
            v4, vs4, rv, rvs, slots, r, hot, cold, head, dims, smr_v,
            NUM_KV_HEADS, HEAD_DIM, GROUP4, GROUP8, MODE, False,
        ).to(tl.bfloat16)
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (group * GROUP_SIZE + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_paged(
    q, indices, cu_q, kv_lens, req_to_token, req_indices, k_buf, v_buf, k_scale, v_scale, sm_k, sm_v, scale,
    max_q, kv_bits, ring_k=None, ring_v=None, ring_ks=None, ring_vs=None, owner=None, ring_mask=None, config=None,
):
    \"\"\"Prefix-chunk sparse attention straight from a quantized pool (replaces gather-dequant + the packed
    _sparse_gqa_chunk_prefill).  q: bf16 [total_q, QH, D]; indices: int32 [total_q, topk] (or [total_q, H, topk]);
    cu_q: int32 or int64 [batch + 1] (the backend's cu_seqlens_q is int64); kv_lens: int32 [batch] (full context
    lengths); req_to_token: int32 [reqs, ctx];
    req_indices: [batch]; k_buf/v_buf + k_scale/v_scale: the pool (int8 [rows, H, D] + fp16 [rows, H, D // 64]
    for kv_bits 8; uint8 [rows, H, D // 2] + fp16 [rows, H, D // 32] for kv_bits 4); sm_k/sm_v: [H, D];
    max_q = max chunk length (CPU int); ring_*/owner/ring_mask: the tiered pool (kv_bits 4).  config =
    (BLOCK_N, num_warps, num_stages) overrides _PAGED_CONFIG (tests).  Returns out = torch.empty_like(q).\"\"\"
    total_q, num_q_heads, head_dim = q.shape
    _, heads, last = k_buf.shape
    assert q.dtype == torch.bfloat16 and q.stride(2) == 1, "paged prefix attention needs bf16 queries"
    assert k_buf.is_contiguous() and v_buf.is_contiguous() and v_buf.shape == k_buf.shape
    assert indices.dtype == torch.int32 and kv_lens.dtype == torch.int32
    assert cu_q.dtype in (torch.int32, torch.int64) and cu_q.is_contiguous(), cu_q.dtype
    assert req_to_token.dtype == torch.int32 and req_to_token.ndim == 2 and req_to_token.stride(1) == 1
    if owner is not None:                                     # tiered: int4 rows + int8 ring
        dim = 2 * last
        assert kv_bits == 4 and k_buf.dtype == torch.uint8 and v_buf.dtype == torch.uint8
        R = _check_tier_args(heads, dim, k_scale, v_scale, ring_k, ring_v, ring_ks, ring_vs, owner, ring_mask)
        mode, mask = 2, R - 1
        k4, v4, ks4, vs4 = k_buf, v_buf, k_scale, v_scale
        rk, rv, rks, rvs, own = ring_k, ring_v, ring_ks, ring_vs, owner
    elif kv_bits == 4:                                        # int4_g32 pool
        dim = 2 * last
        assert k_buf.dtype == torch.uint8 and v_buf.dtype == torch.uint8
        assert dim % KV_INT4_GROUP == 0 and dim == triton.next_power_of_2(dim)
        assert k_scale.shape[1:] == (heads, dim // KV_INT4_GROUP) and k_scale.is_contiguous()
        assert v_scale.shape[1:] == (heads, dim // KV_INT4_GROUP) and v_scale.is_contiguous()
        assert k_scale.dtype == torch.float16 and v_scale.dtype == torch.float16
        mode, mask = 1, 0
        k4, v4, ks4, vs4 = k_buf, v_buf, k_scale, v_scale
        rk, rv, rks, rvs, own = k_buf, v_buf, k_scale, v_scale, req_indices      # unused in MODE 1
    else:                                                     # int8_g64 pool
        dim = last
        assert k_buf.dtype == torch.int8 and v_buf.dtype == torch.int8
        assert dim % KV_INT8_GROUP == 0 and dim == triton.next_power_of_2(dim)
        assert k_scale.shape[1:] == (heads, dim // KV_INT8_GROUP) and k_scale.is_contiguous()
        assert v_scale.shape[1:] == (heads, dim // KV_INT8_GROUP) and v_scale.is_contiguous()
        assert k_scale.dtype == torch.float16 and v_scale.dtype == torch.float16
        mode, mask = 0, 0
        rk, rv, rks, rvs = k_buf, v_buf, k_scale, v_scale
        k4, v4, ks4, vs4, own = k_buf, v_buf, k_scale, v_scale, req_indices      # unused in MODE 0
    assert dim == head_dim, f"pool head_dim {dim} != query head_dim {head_dim}"
    assert sm_k.shape == (heads, dim) and sm_v.shape == (heads, dim)
    group_size = num_q_heads // heads
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = config or _PAGED_CONFIG
    batch = cu_q.shape[0] - 1
    out = torch.empty_like(q)
    if batch == 0 or int(max_q) == 0:
        return out
    _sparse_gqa_chunk_prefill_paged[(int(max_q), batch * heads)](
        q,
        out,
        indices,
        cu_q,
        kv_lens,
        req_to_token,
        req_indices,
        k4,
        v4,
        ks4,
        vs4,
        rk,
        rv,
        rks,
        rvs,
        own,
        sm_k,
        sm_v,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=dim,
        REQ_STRIDE=req_to_token.stride(0),
        GROUP4=KV_INT4_GROUP,
        GROUP8=KV_INT8_GROUP,
        RING_MASK=mask,
        MODE=mode,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def _check_tier_args(heads, dim, k_scale, v_scale, ring_k, ring_v, ring_ks, ring_vs, owner, ring_mask):
"""),
  # ------------------------------------------------------------- sparse_attn: mark kv_tiers' edit-5 anchor line
  # (kv_tiers edit 5 then reads MISMATCH instead of clean while this patch is applied: `kv_tiers.py apply` refuses
  # instead of re-inserting its whole tiered block a second time)
  (SA, """def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
""", """def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):   # kv_paged_prefix applied
"""),
  # ------------------------------------------------------------- backend: os import (env switch)
  (BK, """import logging
import math
""", """import logging
import math
import os
"""),
  # ------------------------------------------------------------- backend: import the paged wrapper
  (BK, """    qwen_sparse_prefix_gather_dequant_tiered,
    qwen_sparse_valid_counts_triton,
""", """    qwen_sparse_prefix_gather_dequant_tiered,
    qwen_sparse_valid_counts_triton,
    sparse_gqa_fwd_interface_paged,
"""),
  # ------------------------------------------------------------- backend: module constant (read once at import)
  (BK, """logger = logging.getLogger(__name__)
""", """logger = logging.getLogger(__name__)

# Paged prefix-chunk attention for the int8 / int4 / tiered pools (perf/patches/kv_paged_prefix.py): DEFAULT ON;
# SGLANG_KV_PAGED_PREFIX=0 selects the materialised gather + packed-kernel path.  Read once: the prefix path is
# eager, never graph-captured.
_PAGED_PREFIX = os.environ.get("SGLANG_KV_PAGED_PREFIX", os.environ.get("SGLANG_QSA_PAGED_PREFIX", "1")) != "0"
logger.info(
    "QSA prefix-chunk attention (int8/int4/tiered pools): %s",
    "paged kernel (SGLANG_KV_PAGED_PREFIX=0 selects the materialised path)" if _PAGED_PREFIX
    else "materialised gather + packed kernel (SGLANG_KV_PAGED_PREFIX=0)",
)
"""),
  # ------------------------------------------------------------- backend: the prefix branch takes the paged kernel
  (BK, """            sm_k, sm_v = pool.get_kv_smooth_buffer(layer.layer_id)
            # Per-call temporaries (this path is never graph-captured): freed after the layer like the
""", """            sm_k, sm_v = pool.get_kv_smooth_buffer(layer.layer_id)
            if _PAGED_PREFIX:                             # paged chunk kernel: attends straight to the pool rows
                output = sparse_gqa_fwd_interface_paged(
                    q.contiguous(),
                    topk_indices,
                    cu_seqlens_q,
                    sequence_lens_tensor,
                    req_to_token,
                    forward_batch.req_pool_indices,
                    k_buffer,
                    v_buffer,
                    k_sf,
                    v_sf,
                    sm_k,
                    sm_v,
                    layer.scaling,
                    max(extend_lens),
                    4 if kv_bits == 4 else 8,
                    **self._kv_tier_kwargs(layer),
                )
                return self._pad_extend_output(output, num_output_rows)
            # Per-call temporaries (this path is never graph-captured): freed after the layer like the
"""),
]


def _read(p):
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def state():
    out = []
    for p, a, b in EDITS:
        t = _read(p)
        out.append((p, a in t, b in t))
    return out


def paged_applied():
    return "def _sparse_gqa_chunk_prefill_paged(" in _read(SA)


def tiers_ok():
    """kv_tiers prerequisite: its kernels in the tree and every kv_tiers edit / new file APPLIED."""
    return (kv_tiers.tiers_applied() and all(ap for _, _, ap in kv_tiers.state())
            and all(ap for _, _, ap in kv_tiers.file_state()))


# kv_tiers edit that this patch overlays: the kernel is inserted inside its sparse_attn block AND its anchor line
# (`def qwen_sparse_valid_counts_triton`) is marked, so it reads MISMATCH (neither clean nor APPLIED) while
# kv_paged_prefix is applied; every other kv_tiers edit must stay APPLIED underneath.
KV_TIERS_OVERLAID = (5,)


def tiers_intact_under_paged():
    st = kv_tiers.state()
    bad = [i for i, (_, _, ap) in enumerate(st) if i not in KV_TIERS_OVERLAID and not ap]
    if not all(ap for _, _, ap in kv_tiers.file_state()):
        bad.append("F")
    return bad


def check():
    if paged_applied():
        bad = tiers_intact_under_paged()
        print(f"  P kv_tiers overlaid by kv_paged_prefix (revert kv_paged_prefix.py before kv_tiers.py); "
              f"kv_tiers edits other than {list(KV_TIERS_OVERLAID)} + file {'intact' if not bad else 'NOT intact: ' + str(bad)}")
    else:
        print(f"  P kv_tiers prerequisite {'APPLIED' if tiers_ok() else 'MISSING/MISMATCH (kv_tiers.py --check)'}")
    for i, (p, pr, ap) in enumerate(state()):
        print(f"  {i} {'APPLIED' if ap else ('clean' if pr else 'MISMATCH'):<8} one  "
              f"{os.path.relpath(p, SG)}: {EDITS[i][1].strip().splitlines()[0][:60]}")


def apply():
    st = state()
    if not paged_applied() and not tiers_ok():
        print("  [!] prerequisite missing: apply perf/patches/kv_int4.py and kv_tiers.py first (every edit APPLIED)")
        check(); sys.exit(1)
    if not all(pr or ap for _, pr, ap in st):
        print("  [!] mismatch"); check(); sys.exit(1)
    for (p, a, b), (_, pr, ap) in zip(EDITS, st):
        if not ap:
            t = open(p, encoding="utf-8").read()
            open(p, "w", encoding="utf-8").write(t.replace(a, b, 1))
    if tiers_intact_under_paged():
        print("  [!] applied, but kv_tiers is not intact underneath:"); check(); sys.exit(1)
    print("  applied (paged prefix-chunk attention for the int8/int4/tiered QSA pools, DEFAULT ON; "
          "SGLANG_KV_PAGED_PREFIX=0 restores the materialised path)")


def revert(force=False):
    if paged_applied():
        bad = tiers_intact_under_paged()
        if bad:
            print(f"  [!] kv_tiers was reverted underneath kv_paged_prefix (out of order): kv_tiers edits {bad} not "
                  "APPLIED.  Reverting now would leave e.g. the backend import of sparse_gqa_fwd_interface_paged "
                  "dangling.  Repair: `kv_tiers.py apply` is refused while the paged kernel is in the tree, so run "
                  "`kv_paged_prefix.py revert --force`, then `kv_tiers.py --check` and hand-restore what is not APPLIED.")
            if not force:
                print("  refused (pass --force to revert kv_paged_prefix anyway)"); check(); sys.exit(1)
            print("  --force: reverting kv_paged_prefix anyway; run kv_tiers.py --check afterwards")
    for (p, a, b), (_, pr, ap) in zip(EDITS, state()):
        if ap:
            t = open(p, encoding="utf-8").read()
            open(p, "w", encoding="utf-8").write(t.replace(b, a, 1))
    left = [i for i, (_, pr, ap) in enumerate(state()) if ap or not pr]
    if left:
        print(f"  [!] reverted, but edits {left} are not clean afterwards:"); check(); sys.exit(1)
    print("  reverted")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if cmd == "revert":
        revert(force="--force" in sys.argv[2:])
    else:
        {"--check": check, "apply": apply}[cmd]()
