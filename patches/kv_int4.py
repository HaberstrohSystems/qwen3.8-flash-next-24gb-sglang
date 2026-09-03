#!/usr/bin/env python3
"""INT4-G32 KV cache for the Qwen sparse attention (QSA) layers -- stage C of perf/KV_INT4_PLAN.md.

Storage per QSA layer: uint8 K/V [rows, 2, 128] = two 4-bit channels per byte (low nibble = even
channel, high nibble = odd channel, offset-binary q + 8 with q in [-7, 7]; row_bytes 256) plus one
fp16 absmax/7 scale per (token, kv-head, 32-channel group) = fp16 [rows, 2, 8] (row_bytes 32) each
for K and V -> 6.75 KB/token over the 12 layers (bf16 24, fp8 12, int8_g64 12.4).  Groups 0-1 are the
64 rotary dims.  Like the int8 pool, the scale buffers are extra KvBufferDescs (int8 [rows, 32]
viewed as fp16 [rows, 2, 8]) on the SAME lazy-VMM owner as the payload, so perf/patches/kv_lazy.py is
untouched and backs/releases scale rows in lockstep with payload rows.

  write path   : MHATokenToKVPoolInt4.set_kv_buffer -> one Triton launch (_quant_store_kv_int4) that
                 quantizes + nibble-packs the bf16 K/V of this forward and scatters payload + scales;
  decode/verify: _compact_kv_int4 (sibling of _compact_kv_int8) gathers the selected rows, unpacks
                 and dequantizes ((nibble - 8) * s, fp32 -> bf16) into the bf16 FA2/trtllm scratch,
                 valid columns only;
  prefix chunk : _gather_dequant_rows_int4 replaces the int8 row gather (runtime max_len grid, no
                 per-chunk recompile); the packed bf16 K/V are per-call torch.empty temporaries as in
                 kv_int8.py (never the graph-captured FA2 scratch, never a retained buffer).

DISPATCH: the fp8 pool ALSO stores uint8 (viewed as float8 by get_key_buffer) and the int8 pool
stores int8, so the backend keys its dispatch on a pool attribute `kv_bits` (4 = this pool, 8 = the
int8 pool, None otherwise) resolved through HybridLinearKVPool.full_kv_pool, and passes it to the
compact wrapper (`kv_bits=` kwarg).  This patch adds `kv_bits = 8` to the int8 pool source (a
kv_int8.py NEW_FILE; kv_int8.py itself is not modified).  The int4 pool's k_buffer is [rows, H, D//2],
so the backend derives the logical head_dim through `_kv_head_dim` (scratch shape, trtllm view,
prefix pack shape).  Smoothing constants sm_k/sm_v are carried like the int8 kernels (identity).

ORDERING: this patch is layered on perf/patches/kv_int8.py (its wrapper branch, backend helper,
scratch-dtype line, CPU-fallback guards, prefix-chunk block and int8_kv_pool.py are the anchors
here), which is layered on kv_fp8.py.  Apply kv_fp8 -> kv_int8 -> kv_int4; revert kv_int4.py BEFORE
kv_int8.py.  apply() refuses unless every kv_int8 edit and its new file are APPLIED; while kv_int4 is
applied, kv_int8's --check shows its overlaid edits (9-17) and its new file as MISMATCH (expected;
kv_int8 apply then refuses, and a kv_int8 revert in that state would be out of order -- run this
revert first).  If kv_int8 WAS reverted underneath (its non-overlaid edits KV_INT8_INTACT_UNDER_INT4 no
longer APPLIED, or _compact_kv_int8 gone), revert() refuses unless --force, because it would re-materialise
kv_int8's overlaid wording with kv_int8's reverted parts (e.g. the MHATokenToKVPoolInt8 import) missing.

  python3 kv_int4.py --check | apply | revert [--force]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_int8                                            # prerequisite patch (same directory)
SG = os.path.join(os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")), "python/sglang")
SA = f"{SG}/srt/layers/attention/qsa/sparse_attn.py"
BK = f"{SG}/srt/layers/attention/qwen_sparse_attn_backend.py"
KD = f"{SG}/srt/mem_cache/kv_cache_dtype.py"
KC = f"{SG}/srt/mem_cache/kv_cache_configurator.py"
PC = f"{SG}/srt/model_executor/pool_configurator.py"
SV = f"{SG}/srt/server_args.py"
POOL8 = f"{SG}/srt/mem_cache/int8_kv_pool.py"            # kv_int8's new file (gets kv_bits = 8)
POOL = f"{SG}/srt/mem_cache/int4_kv_pool.py"

# ----------------------------------------------------------------------------------------------- new file
INT4_KV_POOL_SRC = '''"""INT4-G32 KV pool for the Qwen sparse attention layers (installed by perf/patches/kv_int4.py).

Layout per local layer l (rows = size + page_size, NHD):
  k_buffer[l], v_buffer[l]             uint8 [rows, H, D // 2]     two 4-bit channels per byte
                                       (low nibble = even channel, high = odd, stored q + 8, q in [-7, 7])
  k_scale_buffer[l], v_scale_buffer[l] fp16 [rows, H, D // GROUP]  absmax/7 per (token, head, group)
The scale buffers are extra KvBufferDescs (int8 [rows, H * D // GROUP * 2]) on the lazy VMM owner and
viewed as fp16; on the eager path they are plain zero tensors.  `kv_bits = 4` is the QSA backend's
dispatch key (the int8 pool carries 8; the fp8 pool also stores uint8, so dtype alone is ambiguous).
Smoothing constants (sm_k / sm_v and their inverses, fp16 [L, H, D]) are identity, as in the int8 pool.
"""
import os
from typing import Optional, Tuple

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.attention.qsa.sparse_attn import KV_INT4_GROUP, quant_store_kv_int4
from sglang.srt.mem_cache.memory_pool import KvBufferDesc, MHATokenToKVPool, unwrap_write_loc
from sglang.srt.utils.async_probe import maybe_detect_oob


def _is_unit_scale(s) -> bool:
    return s is None or (isinstance(s, (int, float)) and s == 1)


class MHATokenToKVPoolInt4(MHATokenToKVPool):
    """MHA KV pool storing nibble-packed int4 K/V with fp16 per-(token, head, GROUP-channel) scales."""

    GROUP = KV_INT4_GROUP
    kv_bits = 4                 # QSA backend dispatch key (the int8_g64 pool carries 8)

    def __init__(self, *args, **kwargs):
        self.k_scale_buffer = None
        self.v_scale_buffer = None
        super().__init__(*args, **kwargs)
        if self.dtype != torch.uint8 or self.store_dtype != torch.uint8:
            raise ValueError(f"MHATokenToKVPoolInt4 needs dtype uint8, got {self.dtype}/{self.store_dtype}")
        if self.use_hnd or self.kv_cache_layout != "nhd":
            raise ValueError("MHATokenToKVPoolInt4 supports the NHD layout only")
        # Static per-channel smoothing constants; identity.
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

    def _kv_buffer_shapes(self):
        """Payload rows hold D // 2 bytes per head (two channels per byte)."""
        if self.use_hnd:
            raise ValueError("MHATokenToKVPoolInt4 supports the NHD layout only")
        if self.head_dim % 2 or self.v_head_dim % 2:
            raise ValueError("MHATokenToKVPoolInt4 needs even head dims")
        rows = self.size + self.page_size
        return (
            (rows, self.head_num, self.head_dim // 2),
            (rows, self.head_num, self.v_head_dim // 2),
        )

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
            raise ValueError("MHATokenToKVPoolInt4 scale descs assume one token per row (NHD)")
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
            raise NotImplementedError("int4_g32 KV cache does not support DCP KV masks")
        if not (_is_unit_scale(k_scale) and _is_unit_scale(v_scale)):
            raise ValueError("int4_g32 KV cache computes its own scales; got k_scale/v_scale")
        loc, _, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MHA-INT4)")
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        idx = layer_id - self.start_layer
        if os.environ.get("SGLANG_KV_STATS") and hasattr(self, "_kv_stats"):
            self._kv_stats(layer_id, cache_k, cache_v)
        cache_k = cache_k.view(-1, self.head_num, self.head_dim)
        cache_v = cache_v.view(-1, self.head_num, self.v_head_dim)
        quant_store_kv_int4(
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
        raise NotImplementedError("int4_g32 KV cache: prefix-valid commit is not supported")

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("int4_g32 KV cache: CPU offload is not supported")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("int4_g32 KV cache: CPU offload is not supported")
'''

NEW_FILES = [(POOL, INT4_KV_POOL_SRC)]

# (path, old, new[, "all"])   "all" = replace every occurrence (identical text at several sites)
EDITS = [
  # ------------------------------------------------------------- selector: server_args choices
  (SV, """                "int8_g64",
                "bf16",
""", """                "int8_g64",
                "int4_g32",
                "bf16",
"""),
  # ------------------------------------------------------------- selector: torch dtype mapping
  (KD, """    torch.int8: "int8_g64",
}
""", """    torch.int8: "int8_g64",
    torch.uint8: "int4_g32",
}
"""),
  (KD, """    elif server_args_kv_cache_dtype == "int8_g64":
        kv_cache_dtype = torch.int8
""", """    elif server_args_kv_cache_dtype == "int8_g64":
        kv_cache_dtype = torch.int8
    elif server_args_kv_cache_dtype == "int4_g32":
        kv_cache_dtype = torch.uint8
"""),
  # ------------------------------------------------------------- selector: pool class for the hybrid full pool
  (KC, """from sglang.srt.mem_cache.int8_kv_pool import MHATokenToKVPoolInt8
""", """from sglang.srt.mem_cache.int4_kv_pool import MHATokenToKVPoolInt4
from sglang.srt.mem_cache.int8_kv_pool import MHATokenToKVPoolInt8
"""),
  (KC, """        full_pool_class = (
            MHATokenToKVPoolInt8
            if self.kv_cache_dtype_str == "int8_g64"
""", """        full_pool_class = (
            MHATokenToKVPoolInt4
            if self.kv_cache_dtype_str == "int4_g32"
            else MHATokenToKVPoolInt8
            if self.kv_cache_dtype_str == "int8_g64"
"""),
  # ------------------------------------------------------------- selector: per-token cell size (half payload + fp16 scales)
  (PC, """            elif self.kv_cache_dtype_str == "int8_g64":
""", """            elif self.kv_cache_dtype_str == "int4_g32":
                # two 4-bit channels per byte (the uint8 itemsize above counts one byte per channel)
                # plus one fp16 absmax scale per (token, kv-head, 32-channel group) for K and V
                cell_size = cell_size // 2 + (
                    n * ((model_config.head_dim + model_config.v_head_dim) // 32) * 2 * num_layers
                )
            elif self.kv_cache_dtype_str == "int8_g64":
"""),
  # ------------------------------------------------------------- int8 pool: dispatch key (kv_int8's new file)
  (POOL8, """    GROUP = KV_INT8_GROUP
""", """    GROUP = KV_INT8_GROUP
    kv_bits = 8                 # QSA backend dispatch key (the int4_g32 pool carries 4)
"""),
  # ------------------------------------------------------------- sparse_attn.py: kernels
  (SA, """def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
""", """KV_INT4_GROUP = 32          # channels per fp16 scale for the int4 pool (groups 0-1 = the 64 rotary dims)


@triton.jit
def _quant_store_kv_int4(
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
    \"\"\"Program (token, head): quantize one K row and one V row (per-GROUP absmax/7, fp16 scale,
    q = rint(x / s) with an IEEE-exact division, clamped to [-7, 7]; s clamped to fp16 max so it is never inf) and pack channel pairs into bytes (low nibble = even channel,
    high nibble = odd channel, offset-binary q + 8), then scatter D // 2 bytes + D // GROUP scales
    into slot loc[token].  Grid (N, H) is static -> capture-safe.\"\"\"
    DH: tl.constexpr = D // 2
    NG: tl.constexpr = D // GROUP
    GH: tl.constexpr = GROUP // 2
    t, h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(loc + t).to(tl.int64)
    pairs = tl.arange(0, DH)
    even = 2 * pairs
    odd = even + 1
    goffs = tl.arange(0, NG)
    row = (slot * H + h) * DH
    srow = (slot * H + h) * NG

    base = k + t * sk_n + h * sk_h
    xe = tl.load(base + even).to(tl.float32) * tl.load(sm_k_inv + h * D + even).to(tl.float32)
    xo = tl.load(base + odd).to(tl.float32) * tl.load(sm_k_inv + h * D + odd).to(tl.float32)
    ge = tl.reshape(xe, [NG, GH])                                  # (g, j) = channel 32 g + 2 j
    go = tl.reshape(xo, [NG, GH])                                  # (g, j) = channel 32 g + 2 j + 1
    a = tl.maximum(tl.max(tl.abs(ge), axis=1), tl.max(tl.abs(go), axis=1))
    # The stored scale (fp16) is the one used.  Clamp to fp16 max BEFORE the cast: a bf16 group absmax
    # above 7 * 65504 would otherwise give s = +inf -> nibble 8 -> (8 - 8) * inf = NaN on dequant;
    # clamped, such channels saturate to +/- 7 * 65504 (finite) instead.
    s = tl.minimum(tl.where(a > 0, a / 7.0, 1.0), 65504.0).to(tl.float16)
    # IEEE-exact division: Triton's `/` is the approximate div.full (2 ulp), which breaks exact .5 ties
    # (x/s = 6.5 -> 6.5000001 -> 7) that int4's coarse grid hits constantly; torch rounds them to even.
    sf = tl.broadcast_to(s.to(tl.float32)[:, None], [NG, GH])
    qe = tl.clamp(libdevice.rint(tl.div_rn(ge, sf)), -7.0, 7.0).to(tl.int32) + 8
    qo = tl.clamp(libdevice.rint(tl.div_rn(go, sf)), -7.0, 7.0).to(tl.int32) + 8
    packed = (qe | (qo << 4)).to(tl.uint8)
    tl.store(k_buf + row + pairs, tl.reshape(packed, [DH]))
    tl.store(k_scale + srow + goffs, s)

    base = v + t * sv_n + h * sv_h
    xe = tl.load(base + even).to(tl.float32) * tl.load(sm_v_inv + h * D + even).to(tl.float32)
    xo = tl.load(base + odd).to(tl.float32) * tl.load(sm_v_inv + h * D + odd).to(tl.float32)
    ge = tl.reshape(xe, [NG, GH])
    go = tl.reshape(xo, [NG, GH])
    a = tl.maximum(tl.max(tl.abs(ge), axis=1), tl.max(tl.abs(go), axis=1))
    s = tl.minimum(tl.where(a > 0, a / 7.0, 1.0), 65504.0).to(tl.float16)   # fp16-max clamp, see K half
    # IEEE-exact division: Triton's `/` is the approximate div.full (2 ulp), which breaks exact .5 ties
    # (x/s = 6.5 -> 6.5000001 -> 7) that int4's coarse grid hits constantly; torch rounds them to even.
    sf = tl.broadcast_to(s.to(tl.float32)[:, None], [NG, GH])
    qe = tl.clamp(libdevice.rint(tl.div_rn(ge, sf)), -7.0, 7.0).to(tl.int32) + 8
    qo = tl.clamp(libdevice.rint(tl.div_rn(go, sf)), -7.0, 7.0).to(tl.int32) + 8
    packed = (qe | (qo << 4)).to(tl.uint8)
    tl.store(v_buf + row + pairs, tl.reshape(packed, [DH]))
    tl.store(v_scale + srow + goffs, s)


def quant_store_kv_int4(k, v, loc, k_buf, v_buf, k_scale, v_scale, sm_k_inv, sm_v_inv):
    \"\"\"k, v: [N, H, D] (any float dtype, unit last stride); loc: [N] int32/int64 slots;
    k_buf/v_buf: uint8 [rows, H, D // 2]; k_scale/v_scale: fp16 [rows, H, D // GROUP]; sm_*_inv: fp16 [H, D].\"\"\"
    N, H, D = k.shape
    assert v.shape == (N, H, D) and k.stride(2) == 1 and v.stride(2) == 1
    assert D % KV_INT4_GROUP == 0 and D == triton.next_power_of_2(D)
    assert k_buf.dtype == torch.uint8 and v_buf.dtype == torch.uint8
    assert k_scale.dtype == torch.float16 and v_scale.dtype == torch.float16
    assert k_buf.is_contiguous() and v_buf.is_contiguous() and k_scale.is_contiguous() and v_scale.is_contiguous()
    assert k_buf.shape[1:] == (H, D // 2) and v_buf.shape[1:] == (H, D // 2)
    assert k_scale.shape[1:] == (H, D // KV_INT4_GROUP) and v_scale.shape[1:] == (H, D // KV_INT4_GROUP)
    assert sm_k_inv.shape == (H, D) and sm_v_inv.shape == (H, D)
    assert loc.numel() == N
    if N == 0:
        return
    _quant_store_kv_int4[(N, H)](
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
        GROUP=KV_INT4_GROUP,
        num_warps=4,
    )


@triton.jit
def _compact_kv_int4(
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
):
    \"\"\"_compact_kv for the int4 pool: gather dim // 2 packed bytes per (slot, head), unpack the two
    nibbles ((b & 15) - 8 = even channel, (b >> 4) - 8 = odd channel), dequantize (* s * sm, fp32) and
    interleave into the bf16 scratch.  `dim` is the logical head_dim.  Same store mask as
    _compact_kv / _compact_kv_int8: only valid (in-region, 0 <= pos < seq_len) columns are written;
    invalid columns are neither read nor written.\"\"\"
    DH: tl.constexpr = dim // 2
    NG: tl.constexpr = dim // GROUP
    GH: tl.constexpr = GROUP // 2
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    pairs = tl.arange(0, DH)
    dims = tl.arange(0, dim)
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
    src = (slots[:, None] * heads + head) * DH + pairs[None, :]
    ssrc = (slots[:, None] * heads + head) * NG + pairs[None, :] // GH
    dst = ((pack_start + cols)[:, None] * heads + head) * dim + dims[None, :]
    pmask = valid[:, None] & (pairs[None, :] < DH)
    mask = valid[:, None] & (dims[None, :] < dim)
    sm_e = tl.load(sm_k + head * dim + 2 * pairs).to(tl.float32)
    sm_o = tl.load(sm_k + head * dim + 2 * pairs + 1).to(tl.float32)
    b = tl.load(k + src, mask=pmask, other=0).to(tl.int32)
    sc = tl.load(k_scale + ssrc, mask=pmask, other=0).to(tl.float32)
    lo = ((b & 15) - 8).to(tl.float32) * sc * sm_e[None, :]
    hi = ((b >> 4) - 8).to(tl.float32) * sc * sm_o[None, :]
    tl.store(out_k + dst, tl.interleave(lo, hi).to(tl.bfloat16), mask=mask)
    sm_e = tl.load(sm_v + head * dim + 2 * pairs).to(tl.float32)
    sm_o = tl.load(sm_v + head * dim + 2 * pairs + 1).to(tl.float32)
    b = tl.load(v + src, mask=pmask, other=0).to(tl.int32)
    sc = tl.load(v_scale + ssrc, mask=pmask, other=0).to(tl.float32)
    lo = ((b & 15) - 8).to(tl.float32) * sc * sm_e[None, :]
    hi = ((b >> 4) - 8).to(tl.float32) * sc * sm_o[None, :]
    tl.store(out_v + dst, tl.interleave(lo, hi).to(tl.bfloat16), mask=mask)


@triton.jit
def _gather_dequant_rows_int4(
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
    (slots from req_to_token) -> unpacked + dequantized bf16 rows cu_k[batch] + t of out_k/out_v.
    The row-block count is a grid dimension (runtime max_len), so chunk sizes never recompile.\"\"\"
    DH: tl.constexpr = dim // 2
    NG: tl.constexpr = dim // GROUP
    GH: tl.constexpr = GROUP // 2
    batch, block, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    t = block * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = t < length
    slots = tl.load(req_to_token + req * req_stride + t, mask=valid, other=0).to(tl.int64)
    pairs = tl.arange(0, DH)
    dims = tl.arange(0, dim)
    src = (slots[:, None] * heads + head) * DH + pairs[None, :]
    ssrc = (slots[:, None] * heads + head) * NG + pairs[None, :] // GH
    dst = ((pack_start + t)[:, None] * heads + head) * dim + dims[None, :]
    pmask = valid[:, None] & (pairs[None, :] < DH)
    mask = valid[:, None] & (dims[None, :] < dim)
    sm_e = tl.load(sm_k + head * dim + 2 * pairs).to(tl.float32)
    sm_o = tl.load(sm_k + head * dim + 2 * pairs + 1).to(tl.float32)
    b = tl.load(k + src, mask=pmask, other=0).to(tl.int32)
    sc = tl.load(k_scale + ssrc, mask=pmask, other=0).to(tl.float32)
    lo = ((b & 15) - 8).to(tl.float32) * sc * sm_e[None, :]
    hi = ((b >> 4) - 8).to(tl.float32) * sc * sm_o[None, :]
    tl.store(out_k + dst, tl.interleave(lo, hi).to(tl.bfloat16), mask=mask)
    sm_e = tl.load(sm_v + head * dim + 2 * pairs).to(tl.float32)
    sm_o = tl.load(sm_v + head * dim + 2 * pairs + 1).to(tl.float32)
    b = tl.load(v + src, mask=pmask, other=0).to(tl.int32)
    sc = tl.load(v_scale + ssrc, mask=pmask, other=0).to(tl.float32)
    lo = ((b & 15) - 8).to(tl.float32) * sc * sm_e[None, :]
    hi = ((b >> 4) - 8).to(tl.float32) * sc * sm_o[None, :]
    tl.store(out_v + dst, tl.interleave(lo, hi).to(tl.bfloat16), mask=mask)


def qwen_sparse_prefix_gather_dequant_int4(
    k, v, k_scale, v_scale, sm_k, sm_v, req_to_token, req_indices, seq_lens, cu_k, out_k, out_v, batch, max_len
):
    \"\"\"Whole-prefix gather-dequant of the int4 pool (k/v uint8 [rows, H, D // 2]): rows [0, seq_lens[b])
    of every request packed at cu_k[b] into the bf16 out_k/out_v [.., H, D].  Rows at or beyond
    seq_lens[b] are not written.\"\"\"
    _, heads, dh = k.shape
    dim = 2 * dh
    assert k.dtype == torch.uint8 and v.dtype == torch.uint8
    assert out_k.dtype == torch.bfloat16 and out_v.dtype == torch.bfloat16, "int4 KV gather needs a bf16 scratch"
    assert out_k.shape[1:] == (heads, dim) and out_v.shape[1:] == (heads, dim)
    assert dim % KV_INT4_GROUP == 0 and dim == triton.next_power_of_2(dim)
    assert k_scale.shape[1:] == (heads, dim // KV_INT4_GROUP) and k_scale.is_contiguous()
    assert v_scale.shape[1:] == (heads, dim // KV_INT4_GROUP) and v_scale.is_contiguous()
    assert sm_k.shape == (heads, dim) and sm_v.shape == (heads, dim)
    assert out_k.is_contiguous() and out_v.is_contiguous()
    block_t = 16
    if batch == 0 or max_len == 0:
        return
    _gather_dequant_rows_int4[(batch, triton.cdiv(int(max_len), block_t), heads)](
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
        GROUP=KV_INT4_GROUP,
        BLOCK_T=block_t,
        num_warps=8,
    )


def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
"""),
  # ------------------------------------------------------------- sparse_attn.py: wrapper branch (before int8 / fp8)
  (SA, """def qwen_sparse_kv_extraction_compact_triton(
    k, v, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk,
    k_scale=None, v_scale=None, sm_k=None, sm_v=None,
):
    _, heads, dim = k.shape
    block_topk = 16
    if k.dtype == torch.int8:""", """def qwen_sparse_kv_extraction_compact_triton(
    k, v, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk,
    k_scale=None, v_scale=None, sm_k=None, sm_v=None, kv_bits=None,
):
    _, heads, dim = k.shape
    block_topk = 16
    if k.dtype == torch.uint8 and kv_bits == 4:              # int4_g32 pool (keyed on the pool: fp8 is uint8 too)
        dim = 2 * dim                                        # k is [rows, H, D // 2] packed; dim = logical D
        assert out_k.dtype == torch.bfloat16 and out_v.dtype == torch.bfloat16, "int4 KV gather needs a bf16 scratch"
        assert out_k.shape[1:] == (heads, dim) and out_v.shape[1:] == (heads, dim)
        assert k_scale is not None and v_scale is not None and sm_k is not None and sm_v is not None
        assert dim % KV_INT4_GROUP == 0 and dim == triton.next_power_of_2(dim)
        assert k_scale.is_contiguous() and v_scale.is_contiguous()
        _compact_kv_int4[(batch, heads, triton.cdiv(topk, block_topk))](
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
            GROUP=KV_INT4_GROUP,
            BLOCK_TOPK=block_topk,
            num_warps=8,
        )
        return
    if k.dtype == torch.int8:"""),
  # ------------------------------------------------------------- backend: import the prefix gather
  (BK, """    qwen_sparse_prefix_gather_dequant_int8,
""", """    qwen_sparse_prefix_gather_dequant_int4,
    qwen_sparse_prefix_gather_dequant_int8,
"""),
  # ------------------------------------------------------------- backend: pool-keyed dispatch helpers
  (BK, """    def _int8_gather_kwargs(self, k_buffer: torch.Tensor, layer) -> dict:
        \"\"\"Scale + smoothing tensors for the int8_g64 gather-dequant kernels ({} for other dtypes).\"\"\"
        if k_buffer.dtype != torch.int8:
            return {}
        pool = self.token_to_kv_pool
        k_sf, v_sf = pool.get_kv_scale_buffer(layer.layer_id)
        sm_k, sm_v = pool.get_kv_smooth_buffer(layer.layer_id)
        return dict(k_scale=k_sf, v_scale=v_sf, sm_k=sm_k, sm_v=sm_v)
""", """    def _kv_bits(self):
        \"\"\"Quantized-pool bit width from the pool (8 = int8_g64, 4 = int4_g32, None otherwise).  The
        fp8 pool stores uint8 too (viewed as float8), so dispatch is keyed on the pool, not the dtype.\"\"\"
        pool = self.token_to_kv_pool
        return getattr(getattr(pool, "full_kv_pool", pool), "kv_bits", None)

    def _kv_head_dim(self, k_buffer: torch.Tensor) -> int:
        \"\"\"Logical head_dim of the KV rows (the int4 pool packs two channels per byte).\"\"\"
        return k_buffer.shape[2] * 2 if self._kv_bits() == 4 else k_buffer.shape[2]

    def _kv_scratch_dtype(self, k_buffer: torch.Tensor) -> torch.dtype:
        \"\"\"FA2 / trtllm scratch dtype: bf16 for every quantized pool (fp8, int8_g64, int4_g32).\"\"\"
        if k_buffer.dtype in (torch.float8_e4m3fn, torch.int8) or self._kv_bits() == 4:
            return torch.bfloat16
        return k_buffer.dtype

    def _int8_gather_kwargs(self, k_buffer: torch.Tensor, layer) -> dict:
        \"\"\"Scale + smoothing tensors (+ kv_bits) for the int8_g64 / int4_g32 gather-dequant kernels
        ({} for other dtypes).\"\"\"
        kv_bits = self._kv_bits()
        if k_buffer.dtype != torch.int8 and kv_bits != 4:
            return {}
        pool = self.token_to_kv_pool
        k_sf, v_sf = pool.get_kv_scale_buffer(layer.layer_id)
        sm_k, sm_v = pool.get_kv_smooth_buffer(layer.layer_id)
        return dict(k_scale=k_sf, v_scale=v_sf, sm_k=sm_k, sm_v=sm_v, kv_bits=kv_bits)
"""),
  # ------------------------------------------------------------- backend: scratch shape/dtype from the pool (decode + verify)
  (BK, """            k_buffer.shape[2],
            torch.bfloat16 if k_buffer.dtype in (torch.float8_e4m3fn, torch.int8) else k_buffer.dtype,
""", """            self._kv_head_dim(k_buffer),
            self._kv_scratch_dtype(k_buffer),
""", "all"),
  # ------------------------------------------------------------- backend: trtllm view of the bf16 scratch
  (BK, """        num_kv_heads = k_buffer.shape[1]
        head_dim = k_buffer.shape[2]
""", """        num_kv_heads = k_buffer.shape[1]
        head_dim = self._kv_head_dim(k_buffer)
"""),
  # ------------------------------------------------------------- backend: CPU fallback rejects int4
  (BK, """            if pool.get_key_buffer(layer.layer_id).dtype == torch.int8:
                raise NotImplementedError("int8_g64 KV cache has no CPU attention fallback")
""", """            if pool.get_key_buffer(layer.layer_id).dtype == torch.int8 or self._kv_bits() == 4:
                raise NotImplementedError("int8_g64 / int4_g32 KV cache has no CPU attention fallback")
"""),
  (BK, """            if k_buffer.dtype == torch.int8:
                raise NotImplementedError("int8_g64 KV cache has no CPU attention fallback")
""", """            if k_buffer.dtype == torch.int8 or self._kv_bits() == 4:
                raise NotImplementedError("int8_g64 / int4_g32 KV cache has no CPU attention fallback")
"""),
  # ------------------------------------------------------------- backend: prefix-chunk gather-dequant (int8 or int4 kernel)
  (BK, """        if k_buffer.dtype == torch.int8:                      # int8_g64 pool: row gather + dequant kernel
            assert q.dtype == torch.bfloat16, "int8 KV prefix gather needs bf16 queries"
            k_sf, v_sf = pool.get_kv_scale_buffer(layer.layer_id)
            sm_k, sm_v = pool.get_kv_smooth_buffer(layer.layer_id)
            # Per-call temporaries (this path is never graph-captured): freed after the layer like the
            # cat temporaries below, so no prefill-sized buffer outlives the request (2 KB/token bf16).
            packed_shape = (sum(sequence_lens), k_buffer.shape[1], k_buffer.shape[2])
            packed_k = torch.empty(packed_shape, dtype=torch.bfloat16, device=k_buffer.device)
            packed_v = torch.empty(packed_shape, dtype=torch.bfloat16, device=k_buffer.device)
            qwen_sparse_prefix_gather_dequant_int8(
""", """        kv_bits = self._kv_bits()
        if k_buffer.dtype == torch.int8 or kv_bits == 4:      # int8_g64 / int4_g32 pool: row gather + dequant kernel
            assert q.dtype == torch.bfloat16, "quantized KV prefix gather needs bf16 queries"
            k_sf, v_sf = pool.get_kv_scale_buffer(layer.layer_id)
            sm_k, sm_v = pool.get_kv_smooth_buffer(layer.layer_id)
            # Per-call temporaries (this path is never graph-captured): freed after the layer like the
            # cat temporaries below, so no prefill-sized buffer outlives the request (2 KB/token bf16).
            packed_shape = (sum(sequence_lens), k_buffer.shape[1], self._kv_head_dim(k_buffer))
            packed_k = torch.empty(packed_shape, dtype=torch.bfloat16, device=k_buffer.device)
            packed_v = torch.empty(packed_shape, dtype=torch.bfloat16, device=k_buffer.device)
            gather_dequant = (
                qwen_sparse_prefix_gather_dequant_int4
                if kv_bits == 4
                else qwen_sparse_prefix_gather_dequant_int8
            )
            gather_dequant(
"""),
]


def _mode(e):
    return e[3] if len(e) > 3 else "one"


def _read(p):
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def state():
    out = []
    for e in EDITS:
        p, a, b = e[0], e[1], e[2]
        t = _read(p)
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


def int4_applied():
    """True once the int4 kernels are in the tree (kv_int8's overlaid edits then read MISMATCH)."""
    return "def _compact_kv_int4(" in _read(SA)


def int8_applied():
    """kv_int8 prerequisite: every kv_int8 edit and its new file APPLIED (meaningful while kv_int4 is not applied)."""
    return all(ap for _, _, ap in kv_int8.state()) and all(ap for _, _, ap in kv_int8.file_state())


def int8_kernels_present():
    return "def _compact_kv_int8(" in _read(SA)


def tiers_applied():
    """True while perf/patches/kv_tiers.py is layered on top of kv_int4 (revert kv_tiers.py first)."""
    return "def _compact_kv_tiered(" in _read(SA)


# kv_int8 edits that kv_int4 does NOT overlay: they read APPLIED in `kv_int8.py --check` while kv_int4 is
# applied (the others read MISMATCH, or clean where an int4 insertion splits their applied text: 8, 10, 11, 16).
# If any of these is missing while kv_int4 is applied, `kv_int8.py revert` ran underneath (out of order): it
# reverted exactly these and skipped the rest, so a kv_int4 revert would re-materialise kv_int8's overlaid
# wording (e.g. the MHATokenToKVPoolInt8 reference in the configurator) with its import already gone.
KV_INT8_INTACT_UNDER_INT4 = (2, 3, 5, 6, 7, 13, 14)


def int8_reverted_underneath():
    """kv_int8 edits (indices) that must be intact under kv_int4 but are not -- non-empty = out-of-order revert."""
    st = kv_int8.state()
    return [i for i in KV_INT8_INTACT_UNDER_INT4 if not st[i][2]]


def check():
    if int4_applied():
        bad = int8_reverted_underneath()
        print(f"  P kv_int8 overlaid by kv_int4 (revert kv_int4.py before kv_int8.py); "
              f"_compact_kv_int8 {'present' if int8_kernels_present() else 'MISSING'}; "
              f"kv_int8 edits {list(KV_INT8_INTACT_UNDER_INT4)} {'intact' if not bad else 'NOT intact: ' + str(bad)}")
    else:
        print(f"  P kv_int8 prerequisite {'APPLIED' if int8_applied() else 'MISSING (apply kv_int8.py first)'}")
    for i, (p, pr, ap) in enumerate(state()):
        print(f"  {i} {'APPLIED' if ap else ('clean' if pr else 'MISMATCH'):<8} {_mode(EDITS[i]):<4} "
              f"{os.path.relpath(p, SG)}: {EDITS[i][1].strip().splitlines()[0][:60]}")
    for p, pr, ap in file_state():
        print(f"  F {'APPLIED' if ap else ('clean' if pr else 'MISMATCH'):<8} new  {os.path.relpath(p, SG)}")


def apply():
    st = state()
    fs = file_state()
    if not int4_applied() and not int8_applied():
        print("  [!] prerequisite missing: apply perf/patches/kv_int8.py first (kv_int4 layers on its edits)")
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
    print("  applied (int4_g32 KV cache for QSA; serve with --kv-cache-dtype int4_g32)")
    # keep KV_INT8_INTACT_UNDER_INT4 honest: it must be exactly the kv_int8 edits still APPLIED now
    now = tuple(i for i, (_, _, ap) in enumerate(kv_int8.state()) if ap)
    if now != KV_INT8_INTACT_UNDER_INT4:
        print(f"  [!] KV_INT8_INTACT_UNDER_INT4 {KV_INT8_INTACT_UNDER_INT4} != kv_int8 edits APPLIED under kv_int4 {now}; "
              "update the constant (the out-of-order revert guard keys on it)")


def revert(force=False):
    if tiers_applied():
        print("  [!] kv_tiers.py is applied on top of kv_int4 (its _compact_kv_tiered kernel is in sparse_attn.py). "
              "Reverting kv_int4 now would delete int4_kv_pool.py under the tiered pool and skip kv_tiers' overlaid "
              "edits (unimportable tree). Revert kv_tiers.py first.")
        if not force:
            print("  refused (pass --force to revert kv_int4 anyway)"); check(); sys.exit(1)
        print("  --force: reverting kv_int4 anyway; run kv_tiers.py revert --force and kv_int8.py --check afterwards")
    if int4_applied():
        bad = int8_reverted_underneath()
        if bad or not int8_kernels_present():
            print(f"  [!] kv_int8 was reverted underneath kv_int4 (out of order): kv_int8 edits {bad} not APPLIED"
                  f"{'' if int8_kernels_present() else ', _compact_kv_int8 kernel absent'}. Reverting kv_int4 now would "
                  "re-materialise kv_int8's overlaid text (e.g. the MHATokenToKVPoolInt8 configurator branch) with "
                  "its reverted parts (e.g. the import) missing. Repair: `kv_int4.py revert --force`, then "
                  "`kv_int8.py --check` and hand-restore whatever is not uniformly APPLIED or clean.")
            if not force:
                print("  refused (pass --force to revert kv_int4 anyway)"); check(); sys.exit(1)
            print("  --force: reverting kv_int4 anyway; run kv_int8.py --check afterwards and repair by hand")
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
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if cmd == "revert":
        revert(force="--force" in sys.argv[2:])
    else:
        {"--check": check, "apply": apply}[cmd]()
