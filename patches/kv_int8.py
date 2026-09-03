#!/usr/bin/env python3
"""INT8-G64 KV cache for the Qwen sparse attention (QSA) layers -- stage A of perf/KV_INT8_PLAN.md.

Storage per QSA layer: int8 K/V [rows, 2, 256] (NHD as today) plus one fp16 absmax/127 scale per
(token, kv-head, 64-channel group) = fp16 [rows, 2, 4] each for K and V -> 12.4 KB/token over the
12 layers (bf16 24, fp8 12).  The scale buffers are extra KvBufferDescs (int8 [rows, 16] viewed as
fp16 [rows, 2, 4]) on the SAME lazy-VMM owner as the payload, so perf/patches/kv_lazy.py is untouched
and backs/releases scale rows in lockstep with payload rows.

  write path   : MHATokenToKVPoolInt8.set_kv_buffer -> one Triton launch (_quant_store_kv_int8) that
                 quantizes the bf16 K/V of this forward and scatters payload + scales into `loc`;
  decode/verify: _compact_kv_int8 (sibling of kv_fp8.py's _compact_kv_fp8) gathers the selected rows
                 and dequantizes (q * s, fp32 -> bf16) into the bf16 FA2/trtllm scratch;
  prefix chunk : _gather_dequant_rows_int8 replaces index_select + cat; runtime max_len is a grid
                 dimension (no per-chunk recompile) and the packed bf16 K/V (2 KB/token: 256 MB for a
                 128k prefix) are per-call torch.empty temporaries, freed after the layer exactly like
                 the cat temporaries they replace -- NOT the graph-captured FA2 scratch (its regrow
                 would break the decode graph) and not a retained buffer (lazy_ensure's headroom check
                 must see that memory as free again).

Stage A: GROUP=64, smoothing constants sm_k/sm_v = ones (kernels already take them; stage B loads
calibrated per-channel values).  Selector: --kv-cache-dtype int8_g64.

ORDERING: this patch is layered on perf/patches/kv_fp8.py (its scratch-dtype line, its prefix-chunk
block and its wrapper branch are the anchors here).  Apply kv_fp8.py first; revert kv_int8.py BEFORE
kv_fp8.py.  apply() refuses unless every kv_fp8 edit is APPLIED; while kv_int8 is applied, kv_fp8's
--check shows its overlaid edits as MISMATCH (expected) and kv_fp8 revert refuses.

  python3 kv_int8.py --check | apply | revert
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_fp8                                             # prerequisite patch (same directory)
SG = os.path.join(os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")), "python/sglang")
SA = f"{SG}/srt/layers/attention/qsa/sparse_attn.py"
BK = f"{SG}/srt/layers/attention/qwen_sparse_attn_backend.py"
MP = f"{SG}/srt/mem_cache/memory_pool.py"
KD = f"{SG}/srt/mem_cache/kv_cache_dtype.py"
KC = f"{SG}/srt/mem_cache/kv_cache_configurator.py"
PC = f"{SG}/srt/model_executor/pool_configurator.py"
SV = f"{SG}/srt/server_args.py"
POOL = f"{SG}/srt/mem_cache/int8_kv_pool.py"

# ----------------------------------------------------------------------------------------------- new file
INT8_KV_POOL_SRC = '''"""INT8-G64 KV pool for the Qwen sparse attention layers (installed by perf/patches/kv_int8.py).

Layout per local layer l (rows = size + page_size, NHD):
  k_buffer[l], v_buffer[l]             int8 [rows, H, D]
  k_scale_buffer[l], v_scale_buffer[l] fp16 [rows, H, D // GROUP]   absmax/127 per (token, head, group)
The scale buffers are extra KvBufferDescs (int8 [rows, H * D // GROUP * 2]) on the lazy VMM owner and
viewed as fp16; on the eager path they are plain zero tensors.  Stage B smoothing constants
(sm_k / sm_v and their inverses, fp16 [L, H, D]) are identity here.
"""
import os
from typing import Optional, Tuple

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.attention.qsa.sparse_attn import KV_INT8_GROUP, quant_store_kv_int8
from sglang.srt.mem_cache.memory_pool import KvBufferDesc, MHATokenToKVPool, unwrap_write_loc
from sglang.srt.utils.async_probe import maybe_detect_oob


def _is_unit_scale(s) -> bool:
    return s is None or (isinstance(s, (int, float)) and s == 1)


class MHATokenToKVPoolInt8(MHATokenToKVPool):
    """MHA KV pool storing int8 K/V with fp16 per-(token, head, GROUP-channel) absmax scales."""

    GROUP = KV_INT8_GROUP

    def __init__(self, *args, **kwargs):
        self.k_scale_buffer = None
        self.v_scale_buffer = None
        super().__init__(*args, **kwargs)
        if self.dtype != torch.int8 or self.store_dtype != torch.int8:
            raise ValueError(f"MHATokenToKVPoolInt8 needs dtype int8, got {self.dtype}/{self.store_dtype}")
        if self.use_hnd or self.kv_cache_layout != "nhd":
            raise ValueError("MHATokenToKVPoolInt8 supports the NHD layout only")
        # Stage-B static per-channel smoothing constants; identity in stage A.
        L, H = self.layer_num, self.head_num
        self.sm_k = torch.ones((L, H, self.head_dim), dtype=torch.float16, device=self.device)
        self.sm_v = torch.ones((L, H, self.v_head_dim), dtype=torch.float16, device=self.device)
        self.sm_k_inv = torch.ones_like(self.sm_k)
        self.sm_v_inv = torch.ones_like(self.sm_v)

    # -- layout ---------------------------------------------------------------------------------

    @property
    def k_groups(self) -> int:
        if self.head_dim % self.GROUP:
            raise ValueError(f"head_dim {self.head_dim} not a multiple of GROUP {self.GROUP}")
        return self.head_dim // self.GROUP

    @property
    def v_groups(self) -> int:
        if self.v_head_dim % self.GROUP:
            raise ValueError(f"v_head_dim {self.v_head_dim} not a multiple of GROUP {self.GROUP}")
        return self.v_head_dim // self.GROUP

    def _scale_shapes(self):
        rows = self.size + self.page_size
        return (rows, self.head_num, self.k_groups), (rows, self.head_num, self.v_groups)

    def _create_buffers_normal(self):
        super()._create_buffers_normal()
        ks_shape, vs_shape = self._scale_shapes()
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.k_scale_buffer = [
                torch.zeros(ks_shape, dtype=torch.float16, device=self.device)
                for _ in range(self.layer_num)
            ]
            self.v_scale_buffer = [
                torch.zeros(vs_shape, dtype=torch.float16, device=self.device)
                for _ in range(self.layer_num)
            ]

    def _build_kv_buffer_descs(self):
        descs = super()._build_kv_buffer_descs()
        if any(d.tokens_per_row != 1 for d in descs):
            raise ValueError("MHATokenToKVPoolInt8 scale descs assume one token per row (NHD)")
        ks_shape, vs_shape = self._scale_shapes()
        for prefix, shape in (("ks", ks_shape), ("vs", vs_shape)):
            row_bytes = shape[1] * shape[2] * 2                       # fp16 scales as int8 bytes
            for layer in range(self.layer_num):
                descs.append(
                    KvBufferDesc(f"{prefix}{layer}", (shape[0], row_bytes), row_bytes=row_bytes, tokens_per_row=1)
                )
        return descs

    def _assign_post_capture_tensors(self, tensors):
        L = self.layer_num
        ks_shape, vs_shape = self._scale_shapes()
        self.k_buffer = tensors[:L]
        self.v_buffer = tensors[L : 2 * L]
        self.k_scale_buffer = [t.view(torch.float16).view(ks_shape) for t in tensors[2 * L : 3 * L]]
        self.v_scale_buffer = [t.view(torch.float16).view(vs_shape) for t in tensors[3 * L : 4 * L]]
        # The owner backs exactly one page at construction; zero only those rows (never touch
        # unbacked VA).  Every other slot is written before it is read.
        for t in self.k_scale_buffer + self.v_scale_buffer:
            t[: self.page_size].zero_()

    def _pd_registerable_tensors(self):
        return self.k_buffer + self.v_buffer + self.k_scale_buffer + self.v_scale_buffer

    # -- accessors ------------------------------------------------------------------------------

    def get_kv_scale_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = layer_id - self.start_layer
        return self.k_scale_buffer[idx], self.v_scale_buffer[idx]

    def get_kv_smooth_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = layer_id - self.start_layer
        return self.sm_k[idx], self.sm_v[idx]

    # -- write path -----------------------------------------------------------------------------

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None,
        v_scale=None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        if dcp_kv_mask is not None:
            raise NotImplementedError("int8_g64 KV cache does not support DCP KV masks")
        if not (_is_unit_scale(k_scale) and _is_unit_scale(v_scale)):
            raise ValueError("int8_g64 KV cache computes its own scales; got k_scale/v_scale")
        loc, _, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MHA-INT8)")
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        idx = layer_id - self.start_layer
        if os.environ.get("SGLANG_KV_STATS") and hasattr(self, "_kv_stats"):
            self._kv_stats(layer_id, cache_k, cache_v)
        cache_k = cache_k.view(-1, self.head_num, self.head_dim)
        cache_v = cache_v.view(-1, self.head_num, self.v_head_dim)
        quant_store_kv_int8(
            cache_k,
            cache_v,
            loc,
            self.k_buffer[idx],
            self.v_buffer[idx],
            self.k_scale_buffer[idx],
            self.v_scale_buffer[idx],
            self.sm_k_inv[idx],
            self.sm_v_inv[idx],
        )

    def set_kv_buffer_prefix_valid(self, *args, **kwargs):
        raise NotImplementedError("int8_g64 KV cache: prefix-valid commit is not supported")

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("int8_g64 KV cache: CPU offload is not supported")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("int8_g64 KV cache: CPU offload is not supported")
'''

NEW_FILES = [(POOL, INT8_KV_POOL_SRC)]

# (path, old, new[, "all"])   "all" = replace every occurrence (identical text at several sites)
EDITS = [
  # ------------------------------------------------------------- selector: server_args choices
  (SV, """                "fp8_e4m3",
                "mxfp8",
                "bf16",
""", """                "fp8_e4m3",
                "mxfp8",
                "int8_g64",
                "bf16",
"""),
  # ------------------------------------------------------------- selector: torch dtype mapping
  (KD, """    torch.bfloat16: "bf16",
}
""", """    torch.bfloat16: "bf16",
    torch.int8: "int8_g64",
}
"""),
  (KD, """    elif server_args_kv_cache_dtype == "mxfp8":
        kv_cache_dtype = torch.float8_e4m3fn
""", """    elif server_args_kv_cache_dtype == "mxfp8":
        kv_cache_dtype = torch.float8_e4m3fn
    elif server_args_kv_cache_dtype == "int8_g64":
        kv_cache_dtype = torch.int8
"""),
  # ------------------------------------------------------------- selector: pool class for the hybrid full pool
  (KC, """from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
""", """from sglang.srt.mem_cache.int8_kv_pool import MHATokenToKVPoolInt8
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
"""),
  (KC, """        full_pool_class = (
            MHATokenToKVPoolMXFP8
            if self.kv_cache_dtype_str == "mxfp8" and not self.use_mla_backend
            else mha_pool_class
        )
""", """        full_pool_class = (
            MHATokenToKVPoolInt8
            if self.kv_cache_dtype_str == "int8_g64"
            else MHATokenToKVPoolMXFP8
            if self.kv_cache_dtype_str == "mxfp8" and not self.use_mla_backend
            else mha_pool_class
        )
"""),
  # ------------------------------------------------------------- selector: per-token cell size (+ fp16 scales)
  (PC, """            elif self.kv_cache_dtype_str == "mxfp8":
                scale_block_size = 32
""", """            elif self.kv_cache_dtype_str == "int8_g64":
                # fp16 absmax scale per (token, kv-head, 64-channel group) for K and V
                cell_size += (
                    n * ((model_config.head_dim + model_config.v_head_dim) // 64) * 2 * num_layers
                )
            elif self.kv_cache_dtype_str == "mxfp8":
                scale_block_size = 32
"""),
  # ------------------------------------------------------------- memory_pool: forward the smoothing constants
  (MP, """    def get_kv_scale_buffer(self, layer_id: int):
        # MXFP8 full_kv_pool exposes per-32 UE8M0 K/V scale buffers.
""", """    def get_kv_smooth_buffer(self, layer_id: int):
        # int8_g64 full_kv_pool exposes per-channel smoothing constants (identity in stage A).
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_smooth_buffer(layer_id)

    def get_kv_scale_buffer(self, layer_id: int):
        # MXFP8 full_kv_pool exposes per-32 UE8M0 K/V scale buffers.
"""),
  # ------------------------------------------------------------- sparse_attn.py: kernels
  (SA, """import triton.language as tl
""", """import triton.language as tl
from triton.language.extra import libdevice

KV_INT8_GROUP = 64          # channels per fp16 scale (group 0 = the 64 rotary dims, 1-3 = pass-through)
"""),
  (SA, """def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
""", """@triton.jit
def _quant_store_kv_int8(
    k,
    v,
    loc,
    k_buf,
    v_buf,
    k_scale,
    v_scale,
    sm_k_inv,
    sm_v_inv,
    sk_n,
    sk_h,
    sv_n,
    sv_h,
    H: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
):
    \"\"\"Program (token, head): quantize one K row and one V row (per-GROUP absmax/127, fp16 scale)
    and scatter payload + scales into slot loc[token].  Grid (N, H) is static -> capture-safe.\"\"\"
    NG: tl.constexpr = D // GROUP
    t, h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(loc + t).to(tl.int64)
    offs = tl.arange(0, D)
    goffs = tl.arange(0, NG)
    row = (slot * H + h) * D
    srow = (slot * H + h) * NG

    x = tl.load(k + t * sk_n + h * sk_h + offs).to(tl.float32)
    x = x * tl.load(sm_k_inv + h * D + offs).to(tl.float32)
    xg = tl.reshape(x, [NG, GROUP])
    a = tl.max(tl.abs(xg), axis=1)
    s = tl.minimum(tl.where(a > 0, a / 127.0, 1.0), 65504.0).to(tl.float16)   # fp16 max: no inf/NaN on extreme groups            # the stored scale (fp16) is the one used
    q = libdevice.rint(xg / s.to(tl.float32)[:, None])
    q = tl.clamp(q, -127.0, 127.0).to(tl.int8)
    tl.store(k_buf + row + offs, tl.reshape(q, [D]))
    tl.store(k_scale + srow + goffs, s)

    x = tl.load(v + t * sv_n + h * sv_h + offs).to(tl.float32)
    x = x * tl.load(sm_v_inv + h * D + offs).to(tl.float32)
    xg = tl.reshape(x, [NG, GROUP])
    a = tl.max(tl.abs(xg), axis=1)
    s = tl.minimum(tl.where(a > 0, a / 127.0, 1.0), 65504.0).to(tl.float16)   # fp16 max: no inf/NaN on extreme groups
    q = libdevice.rint(xg / s.to(tl.float32)[:, None])
    q = tl.clamp(q, -127.0, 127.0).to(tl.int8)
    tl.store(v_buf + row + offs, tl.reshape(q, [D]))
    tl.store(v_scale + srow + goffs, s)


def quant_store_kv_int8(k, v, loc, k_buf, v_buf, k_scale, v_scale, sm_k_inv, sm_v_inv):
    \"\"\"k, v: [N, H, D] (any float dtype, unit last stride); loc: [N] int32/int64 slots;
    k_buf/v_buf: int8 [rows, H, D]; k_scale/v_scale: fp16 [rows, H, D // GROUP]; sm_*_inv: fp16 [H, D].\"\"\"
    N, H, D = k.shape
    assert v.shape == (N, H, D) and k.stride(2) == 1 and v.stride(2) == 1
    assert k_buf.dtype == torch.int8 and v_buf.dtype == torch.int8
    assert k_scale.dtype == torch.float16 and v_scale.dtype == torch.float16
    assert k_buf.is_contiguous() and v_buf.is_contiguous() and k_scale.is_contiguous() and v_scale.is_contiguous()
    assert k_buf.shape[1:] == (H, D) and k_scale.shape[1:] == (H, D // KV_INT8_GROUP)
    assert sm_k_inv.shape == (H, D) and sm_v_inv.shape == (H, D)
    assert loc.numel() == N
    if N == 0:
        return
    _quant_store_kv_int8[(N, H)](
        k,
        v,
        loc,
        k_buf,
        v_buf,
        k_scale,
        v_scale,
        sm_k_inv,
        sm_v_inv,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        H=H,
        D=D,
        GROUP=KV_INT8_GROUP,
        num_warps=4,
    )


@triton.jit
def _compact_kv_int8(
    k,
    v,
    k_scale,
    v_scale,
    sm_k,
    sm_v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    \"\"\"_compact_kv for an int8 pool: gather + dequantize (q * s * sm, fp32) into the bf16 scratch.
    Same store mask as _compact_kv / _compact_kv_fp8: only valid (in-region, 0 <= pos < seq_len)
    columns are written; invalid columns are neither read nor written (on the trtllm strided tables
    cu_k spans the whole page-aligned stride, so zero-filling them would write every unused column).\"\"\"
    NG: tl.constexpr = dim // GROUP
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    ).to(tl.int64)
    dmask = dims < dim
    src = (slots[:, None] * heads + head) * dim + dims[None, :]
    ssrc = (slots[:, None] * heads + head) * NG + dims[None, :] // GROUP
    dst = ((pack_start + cols)[:, None] * heads + head) * dim + dims[None, :]
    mask = valid[:, None] & dmask[None, :]
    smr_k = tl.load(sm_k + head * dim + dims, mask=dmask, other=0).to(tl.float32)
    smr_v = tl.load(sm_v + head * dim + dims, mask=dmask, other=0).to(tl.float32)
    sc = tl.load(k_scale + ssrc, mask=mask, other=0).to(tl.float32)
    kk = tl.load(k + src, mask=mask, other=0).to(tl.float32) * sc * smr_k[None, :]
    tl.store(out_k + dst, kk.to(tl.bfloat16), mask=mask)
    sc = tl.load(v_scale + ssrc, mask=mask, other=0).to(tl.float32)
    vv = tl.load(v + src, mask=mask, other=0).to(tl.float32) * sc * smr_v[None, :]
    tl.store(out_v + dst, vv.to(tl.bfloat16), mask=mask)


@triton.jit
def _gather_dequant_rows_int8(
    k,
    v,
    k_scale,
    v_scale,
    sm_k,
    sm_v,
    req_to_token,
    req_indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    \"\"\"Program (batch, row block, head): rows t < seq_lens[batch] of request req_indices[batch]
    (slots from req_to_token) -> dequantized bf16 rows cu_k[batch] + t of out_k/out_v.  The row-block
    count is a grid dimension (runtime max_len), so chunk sizes never recompile the kernel.\"\"\"
    NG: tl.constexpr = dim // GROUP
    batch, block, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    t = block * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = t < length
    slots = tl.load(req_to_token + req * req_stride + t, mask=valid, other=0).to(tl.int64)
    dims = tl.arange(0, dim)
    src = (slots[:, None] * heads + head) * dim + dims[None, :]
    ssrc = (slots[:, None] * heads + head) * NG + dims[None, :] // GROUP
    dst = ((pack_start + t)[:, None] * heads + head) * dim + dims[None, :]
    mask = valid[:, None] & (dims[None, :] < dim)
    smr_k = tl.load(sm_k + head * dim + dims).to(tl.float32)
    smr_v = tl.load(sm_v + head * dim + dims).to(tl.float32)
    sc = tl.load(k_scale + ssrc, mask=mask, other=0).to(tl.float32)
    kk = tl.load(k + src, mask=mask, other=0).to(tl.float32) * sc * smr_k[None, :]
    tl.store(out_k + dst, kk.to(tl.bfloat16), mask=mask)
    sc = tl.load(v_scale + ssrc, mask=mask, other=0).to(tl.float32)
    vv = tl.load(v + src, mask=mask, other=0).to(tl.float32) * sc * smr_v[None, :]
    tl.store(out_v + dst, vv.to(tl.bfloat16), mask=mask)


def qwen_sparse_prefix_gather_dequant_int8(
    k, v, k_scale, v_scale, sm_k, sm_v, req_to_token, req_indices, seq_lens, cu_k, out_k, out_v, batch, max_len
):
    \"\"\"Whole-prefix gather-dequant of an int8 pool: rows [0, seq_lens[b]) of every request packed at
    cu_k[b] into the bf16 out_k/out_v.  Rows at or beyond seq_lens[b] are not written.\"\"\"
    _, heads, dim = k.shape
    assert k.dtype == torch.int8 and v.dtype == torch.int8
    assert out_k.dtype == torch.bfloat16 and out_v.dtype == torch.bfloat16, "int8 KV gather needs a bf16 scratch"
    assert dim % KV_INT8_GROUP == 0 and dim == triton.next_power_of_2(dim)
    assert k_scale.shape[1:] == (heads, dim // KV_INT8_GROUP) and k_scale.is_contiguous()
    assert v_scale.shape[1:] == (heads, dim // KV_INT8_GROUP) and v_scale.is_contiguous()
    assert sm_k.shape == (heads, dim) and sm_v.shape == (heads, dim)
    assert out_k.is_contiguous() and out_v.is_contiguous()
    block_t = 16
    if batch == 0 or max_len == 0:
        return
    _gather_dequant_rows_int8[(batch, triton.cdiv(int(max_len), block_t), heads)](
        k,
        v,
        k_scale,
        v_scale,
        sm_k,
        sm_v,
        req_to_token,
        req_indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        heads,
        dim,
        req_to_token.stride(0),
        GROUP=KV_INT8_GROUP,
        BLOCK_T=block_t,
        num_warps=8,
    )


def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
"""),
  # ------------------------------------------------------------- sparse_attn.py: wrapper branch (before fp8)
  (SA, """def qwen_sparse_kv_extraction_compact_triton(
    k, v, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk
):
    _, heads, dim = k.shape
    block_topk = 16
    if k.dtype == torch.float8_e4m3fn:""", """def qwen_sparse_kv_extraction_compact_triton(
    k, v, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk,
    k_scale=None, v_scale=None, sm_k=None, sm_v=None,
):
    _, heads, dim = k.shape
    block_topk = 16
    if k.dtype == torch.int8:                                # int8_g64 pool: dequantize into the bf16 scratch
        assert out_k.dtype == torch.bfloat16 and out_v.dtype == torch.bfloat16, "int8 KV gather needs a bf16 scratch"
        assert k_scale is not None and v_scale is not None and sm_k is not None and sm_v is not None
        assert dim % KV_INT8_GROUP == 0 and k_scale.is_contiguous() and v_scale.is_contiguous()
        _compact_kv_int8[(batch, heads, triton.cdiv(topk, block_topk))](
            k,
            v,
            k_scale,
            v_scale,
            sm_k,
            sm_v,
            req_to_token,
            req_indices,
            indices,
            seq_lens,
            cu_k,
            out_k,
            out_v,
            topk,
            heads,
            dim,
            req_to_token.stride(0),
            indices.stride(0),
            GROUP=KV_INT8_GROUP,
            BLOCK_TOPK=block_topk,
            BLOCK_D=triton.next_power_of_2(dim),
            num_warps=8,
        )
        return
    if k.dtype == torch.float8_e4m3fn:"""),
  # ------------------------------------------------------------- backend: import the prefix gather
  (BK, """    qwen_sparse_kv_extraction_compact_triton,
""", """    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_prefix_gather_dequant_int8,
"""),
  # ------------------------------------------------------------- backend: helper next to _get_fa2_scratch
  (BK, """    def _get_trtllm_sparse_tables(self, batch, pages_per_row, page, device):
""", """    def _int8_gather_kwargs(self, k_buffer: torch.Tensor, layer) -> dict:
        \"\"\"Scale + smoothing tensors for the int8_g64 gather-dequant kernels ({} for other dtypes).\"\"\"
        if k_buffer.dtype != torch.int8:
            return {}
        pool = self.token_to_kv_pool
        k_sf, v_sf = pool.get_kv_scale_buffer(layer.layer_id)
        sm_k, sm_v = pool.get_kv_smooth_buffer(layer.layer_id)
        return dict(k_scale=k_sf, v_scale=v_sf, sm_k=sm_k, sm_v=sm_v)

    def _get_trtllm_sparse_tables(self, batch, pages_per_row, page, device):
"""),
  # ------------------------------------------------------------- backend: bf16 scratch for int8 pools (decode + verify)
  (BK, """            torch.bfloat16 if k_buffer.dtype == torch.float8_e4m3fn else k_buffer.dtype,
""", """            torch.bfloat16 if k_buffer.dtype in (torch.float8_e4m3fn, torch.int8) else k_buffer.dtype,
""", "all"),
  # ------------------------------------------------------------- backend: pass scales to the compact gather (trtllm + FA2)
  (BK, """            batch,
            topk,
        )
        num_kv_heads = k_buffer.shape[1]
""", """            batch,
            topk,
            **self._int8_gather_kwargs(k_buffer, layer),
        )
        num_kv_heads = k_buffer.shape[1]
"""),
  (BK, """            batch,
            topk,
        )
        output = flash_attn_varlen_func(
""", """            batch,
            topk,
            **self._int8_gather_kwargs(k_buffer, layer),
        )
        output = flash_attn_varlen_func(
"""),
  # ------------------------------------------------------------- backend: CPU fallback rejects int8
  (BK, """            pool = self.token_to_kv_pool
            output = qsa_sparse_attention(
""", """            pool = self.token_to_kv_pool
            if pool.get_key_buffer(layer.layer_id).dtype == torch.int8:
                raise NotImplementedError("int8_g64 KV cache has no CPU attention fallback")
            output = qsa_sparse_attention(
"""),
  (BK, """            output = qsa_sparse_attention(q, k_buffer, v_buffer, slots, layer.scaling)
""", """            if k_buffer.dtype == torch.int8:
                raise NotImplementedError("int8_g64 KV cache has no CPU attention fallback")
            output = qsa_sparse_attention(q, k_buffer, v_buffer, slots, layer.scaling)
"""),
  # ------------------------------------------------------------- backend: prefix-chunk gather-dequant
  (BK, """        req_indices = forward_batch.req_pool_indices.tolist()
        _fp8 = k_buffer.dtype == torch.float8_e4m3fn
        k_src = k_buffer.view(torch.uint8) if _fp8 else k_buffer
        v_src = v_buffer.view(torch.uint8) if _fp8 else v_buffer
        k_parts = [
            k_src.index_select(
                0, req_to_token[req_indices[i], : sequence_lens[i]].long()
            )
            for i in range(len(sequence_lens))
        ]
        v_parts = [
            v_src.index_select(
                0, req_to_token[req_indices[i], : sequence_lens[i]].long()
            )
            for i in range(len(sequence_lens))
        ]
        if _fp8:                                              # fp8 pool: dequantize the gathered prefix
            k_parts = [p.view(torch.float8_e4m3fn).to(q.dtype) for p in k_parts]
            v_parts = [p.view(torch.float8_e4m3fn).to(q.dtype) for p in v_parts]
        sequence_lens_tensor = torch.tensor(
            sequence_lens, dtype=torch.int32, device=q.device
        )
        cu_seqlens_k = F.pad(sequence_lens_tensor.cumsum(0), (1, 0)).contiguous()
        output = sparse_gqa_fwd_interface_triton_ck(
            q.contiguous(),
            torch.cat(k_parts),
            torch.cat(v_parts),
""", """        sequence_lens_tensor = torch.tensor(
            sequence_lens, dtype=torch.int32, device=q.device
        )
        cu_seqlens_k = F.pad(sequence_lens_tensor.cumsum(0), (1, 0)).contiguous()
        if k_buffer.dtype == torch.int8:                      # int8_g64 pool: row gather + dequant kernel
            assert q.dtype == torch.bfloat16, "int8 KV prefix gather needs bf16 queries"
            k_sf, v_sf = pool.get_kv_scale_buffer(layer.layer_id)
            sm_k, sm_v = pool.get_kv_smooth_buffer(layer.layer_id)
            # Per-call temporaries (this path is never graph-captured): freed after the layer like the
            # cat temporaries below, so no prefill-sized buffer outlives the request (2 KB/token bf16).
            packed_shape = (sum(sequence_lens), k_buffer.shape[1], k_buffer.shape[2])
            packed_k = torch.empty(packed_shape, dtype=torch.bfloat16, device=k_buffer.device)
            packed_v = torch.empty(packed_shape, dtype=torch.bfloat16, device=k_buffer.device)
            qwen_sparse_prefix_gather_dequant_int8(
                k_buffer,
                v_buffer,
                k_sf,
                v_sf,
                sm_k,
                sm_v,
                req_to_token,
                forward_batch.req_pool_indices,
                sequence_lens_tensor,
                cu_seqlens_k,
                packed_k,
                packed_v,
                len(sequence_lens),
                max(sequence_lens),
            )
        else:
            req_indices = forward_batch.req_pool_indices.tolist()
            _fp8 = k_buffer.dtype == torch.float8_e4m3fn
            k_src = k_buffer.view(torch.uint8) if _fp8 else k_buffer
            v_src = v_buffer.view(torch.uint8) if _fp8 else v_buffer
            k_parts = [
                k_src.index_select(
                    0, req_to_token[req_indices[i], : sequence_lens[i]].long()
                )
                for i in range(len(sequence_lens))
            ]
            v_parts = [
                v_src.index_select(
                    0, req_to_token[req_indices[i], : sequence_lens[i]].long()
                )
                for i in range(len(sequence_lens))
            ]
            if _fp8:                                          # fp8 pool: dequantize the gathered prefix
                k_parts = [p.view(torch.float8_e4m3fn).to(q.dtype) for p in k_parts]
                v_parts = [p.view(torch.float8_e4m3fn).to(q.dtype) for p in v_parts]
            packed_k = torch.cat(k_parts)
            packed_v = torch.cat(v_parts)
        output = sparse_gqa_fwd_interface_triton_ck(
            q.contiguous(),
            packed_k,
            packed_v,
"""),
]


def _mode(e):
    return e[3] if len(e) > 3 else "one"


def state():
    out = []
    for e in EDITS:
        p, a, b = e[0], e[1], e[2]
        t = open(p, encoding="utf-8").read()
        out.append((p, a in t, b in t))
    return out


def file_state():
    """(path, clean, applied) per new file: clean = absent, applied = present with identical content."""
    out = []
    for p, src in NEW_FILES:
        if not os.path.exists(p):
            out.append((p, True, False))
        else:
            out.append((p, False, open(p, encoding="utf-8").read() == src))
    return out


def int8_applied():
    """True once the int8 kernels are in the tree (kv_fp8's overlaid edits then read MISMATCH)."""
    return "def _compact_kv_int8(" in open(SA, encoding="utf-8").read()


def fp8_applied():
    """kv_fp8 prerequisite: every kv_fp8 edit APPLIED (only meaningful while kv_int8 is not applied)."""
    return all(ap for _, _, ap in kv_fp8.state())


def fp8_kernels_present():
    return "def _compact_kv_fp8(" in open(SA, encoding="utf-8").read()


def check():
    if int8_applied():
        print(f"  P kv_fp8 overlaid by kv_int8 (revert kv_int8.py before kv_fp8.py); "
              f"_compact_kv_fp8 {'present' if fp8_kernels_present() else 'MISSING'}")
    else:
        print(f"  P kv_fp8 prerequisite {'APPLIED' if fp8_applied() else 'MISSING (apply kv_fp8.py first)'}")
    for i, (p, pr, ap) in enumerate(state()):
        print(f"  {i} {'APPLIED' if ap else ('clean' if pr else 'MISMATCH'):<8} {_mode(EDITS[i]):<4} "
              f"{os.path.relpath(p, SG)}: {EDITS[i][1].strip().splitlines()[0][:60]}")
    for p, pr, ap in file_state():
        print(f"  F {'APPLIED' if ap else ('clean' if pr else 'MISMATCH'):<8} new  {os.path.relpath(p, SG)}")


def apply():
    st = state()
    fs = file_state()
    if not int8_applied() and not fp8_applied():
        print("  [!] prerequisite missing: apply perf/patches/kv_fp8.py first (kv_int8 layers on its edits)")
        check(); return
    if not all(pr or ap for _, pr, ap in st) or not all(pr or ap for _, pr, ap in fs):
        print("  [!] mismatch"); check(); return
    for e, (_, pr, ap) in zip(EDITS, st):
        p, a, b = e[0], e[1], e[2]
        if not ap:
            t = open(p, encoding="utf-8").read()
            open(p, "w", encoding="utf-8").write(t.replace(a, b, -1 if _mode(e) == "all" else 1))
    for (p, src), (_, pr, ap) in zip(NEW_FILES, fs):
        if not ap:
            open(p, "w", encoding="utf-8").write(src)
    print("  applied (int8_g64 KV cache for QSA; serve with --kv-cache-dtype int8_g64)")


def revert():
    if int8_applied() and not fp8_kernels_present():
        print("  [!] warning: kv_fp8's _compact_kv_fp8 kernel is absent (kv_fp8 was reverted under kv_int8?); "
              "this revert restores kv_fp8's backend/wrapper text -- run kv_fp8.py --check afterwards")
    for e, (_, pr, ap) in zip(EDITS, state()):
        p, a, b = e[0], e[1], e[2]
        if ap:
            t = open(p, encoding="utf-8").read()
            open(p, "w", encoding="utf-8").write(t.replace(b, a, -1 if _mode(e) == "all" else 1))
    for (p, _), (_, pr, ap) in zip(NEW_FILES, file_state()):
        if ap:
            os.remove(p)
    print("  reverted")


if __name__ == "__main__":
    {"--check": check, "apply": apply, "revert": revert}[sys.argv[1] if len(sys.argv) > 1 else "--check"]()
