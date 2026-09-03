#!/usr/bin/env python3
"""Tiered ("compost") KV cache for the Qwen sparse attention (QSA) layers -- perf/KV_TIERS_PLAN.md layout (B), dual-write.

Every fresh token is written TWICE, in two Triton launches per set_kv_buffer (both static-grid, capture-safe):
  * _stamp_ring_owner: `owner[slot & (R - 1)] = slot` (int32 owner table [R]) for every token of the write;
    tokens of one write that share a ring row (slots congruent mod R: only possible with several requests in
    one batch) race here and exactly ONE of them survives as the row's owner (arbitrary but definite once the
    launch has completed);
  * _quant_store_kv_tiered: int4-g32 into the full-context row `slot` of the int4 pool (kv_int4.py layout, on
    the lazy-VMM owner) for every token, and int8-g64 into ring row `slot & (R - 1)` of a fixed ring of
    R = SGLANG_KV_TIERS_W slots (default 8192, power of two; plain torch tensors allocated once, before graph
    capture, OUTSIDE the VMM owner) only for the token that owns the row (`owner[r] == slot`, read after the
    stamp launch) -- a losing token of a same-launch collision is simply cold (its int4 row is complete), the
    winner's ring row is never mixed with a loser's bytes.
The ring row of slot s is valid until slot s + R is written (its owner entry then flips), so exactly the
last R written slots are int8 and everything older is int4; no compactor, no tier flip, no VMM change.
Every reader (decode/verify compact gather, prefix-chunk row gather) does the tier test on the device:
    r = slot & MASK; hot = valid & (owner[r] == slot); cold = valid & ~hot
and reads the int8 ring row for hot slots, the int4 row for cold slots (both always exist), selecting with
tl.where; the store mask stays `valid` (invalid columns are never written: trtllm strided tables).  The ring
writer carries the fp16-max clamp on both scales (the int8 kernel lacks it: CAMPAIGN.md).  Slot-0 dummy
writes (graph capture) stamp owner[0] = 0 and are self-healing: a later reader of slot k*R sees owner != slot
and takes its int4 row.  Bytes/token at 256k: 6,912 (int4) + 12,672 * R / 262,144 = 7,308 at R = 8192
(ring 103.8 MB + owner 32 KB fixed; pool_configurator keeps the int4 cell size).

  selector : --kv-cache-dtype int8ring_int4 (torch.uint8 like int4_g32; pool class MHATokenToKVPoolTiered)
  env      : SGLANG_KV_TIERS_W = ring slots R (default 8192; positive power of two; R <= pool size)
  new file : srt/mem_cache/tiered_kv_pool.py (MHATokenToKVPoolTiered(MHATokenToKVPoolInt4): kv_tiered = True)
  kernels  : _stamp_ring_owner + _quant_store_kv_tiered (one wrapper), _compact_kv_tiered,
             _gather_dequant_rows_tiered (+ wrappers) in sparse_attn.py
  backend  : `_kv_tier_kwargs` (ring buffers + owner + mask) folded into `_int8_gather_kwargs` -> both compact
             call sites; prefix block picks qwen_sparse_prefix_gather_dequant_tiered when the pool is tiered.

ORDERING: layered on perf/patches/kv_int4.py (its wrapper branch, gather_dequant block, _int8_gather_kwargs,
configurator/pool_configurator selector lines are the anchors), which is layered on kv_int8 -> kv_fp8.
Apply kv_fp8 -> kv_int8 -> kv_int4 -> kv_tiers; revert kv_tiers.py BEFORE kv_int4.py.  apply() refuses unless
every kv_int4 edit and its new file are APPLIED; while kv_tiers is applied, kv_int4's --check shows its
overlaid edits (4, 8, 10, 15 MISMATCH; 5, 7 clean) as not APPLIED (expected; kv_int4 apply then refuses).  If kv_int4 WAS reverted
underneath (its non-overlaid edits KV_INT4_INTACT_UNDER_TIERS no longer APPLIED, or _compact_kv_int4 gone),
revert() refuses unless --force (repair: `kv_tiers.py revert --force`, then `kv_int4.py revert`, then
`kv_int8.py --check`).  A refused revert exits 1.  kv_int4.py's revert() in turn refuses (unless --force) while
the tiered kernels are in the tree, and phase1.py reverts a step's patches in reversed(patches) order
(kv_tiers before kv_int4), so the out-of-order case cannot arise from the driver.

  python3 kv_tiers.py --check | apply | revert [--force]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_int4                                            # prerequisite patch (same directory; imports kv_int8)
SG = os.path.join(os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")), "python/sglang")
SA = f"{SG}/srt/layers/attention/qsa/sparse_attn.py"
BK = f"{SG}/srt/layers/attention/qwen_sparse_attn_backend.py"
MP = f"{SG}/srt/mem_cache/memory_pool.py"
KD = f"{SG}/srt/mem_cache/kv_cache_dtype.py"
KC = f"{SG}/srt/mem_cache/kv_cache_configurator.py"
PC = f"{SG}/srt/model_executor/pool_configurator.py"
SV = f"{SG}/srt/server_args.py"
POOL = f"{SG}/srt/mem_cache/tiered_kv_pool.py"

# ----------------------------------------------------------------------------------------------- new file
TIERED_KV_POOL_SRC = '''"""Tiered KV pool: int8-g64 ring over the int4-g32 full-context pool (installed by perf/patches/kv_tiers.py).

Layout per local layer l: the int4 pool's k/v_buffer + k/v_scale_buffer (rows = size + page_size, on the lazy
VMM owner, untouched) PLUS a fixed ring of R = SGLANG_KV_TIERS_W slots outside the owner:
  ring_k[l], ring_v[l]     int8 [R, H, D]                  ring row r = slot & (R - 1)
  ring_ks[l], ring_vs[l]   fp16 [R, H, D // 64]            absmax/127 per (row, head, 64-channel group)
  ring_owner               int32 [R]                       slot that owns ring row r (-1 = nobody)
set_kv_buffer stamps the owner (one launch) and then writes both tiers (dual-write launch; the ring row
only by the stamped owner, so same-write ring-row collisions degrade to cold instead of mixing bytes);
readers test owner[slot & mask] == slot on the device and take the int8 ring row (hot) or the int4 row (cold).
The ring is allocated once at construction (before graph capture; its pointers are baked into the decode
graph) and never reallocated; it is NOT a KvBufferDesc, so bytes_per_token()/lazy_ensure/kv_lazy.py stay
exact and PD registration (desc-aligned) is unchanged.  `kv_tiered = True` is the backend's dispatch key
(kv_bits stays 4: scratch shape, prefix pack shape and the int4 cell size are inherited).
"""
import logging
import os
from typing import Tuple

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.attention.qsa.sparse_attn import KV_INT8_GROUP, quant_store_kv_tiered
from sglang.srt.mem_cache.int4_kv_pool import MHATokenToKVPoolInt4, _is_unit_scale
from sglang.srt.mem_cache.memory_pool import unwrap_write_loc
from sglang.srt.utils.async_probe import maybe_detect_oob

logger = logging.getLogger(__name__)


def ring_slots_from_env() -> int:
    """SGLANG_KV_TIERS_W: ring slots R (default 8192); must be a positive power of two (ring index = slot & (R - 1))."""
    raw = os.environ.get("SGLANG_KV_TIERS_W", "8192")
    try:
        r = int(raw)
    except ValueError:
        raise ValueError(f"SGLANG_KV_TIERS_W={raw!r} is not an integer") from None
    if r <= 0 or (r & (r - 1)):
        raise ValueError(f"SGLANG_KV_TIERS_W must be a positive power of two, got {r}")
    return r


class MHATokenToKVPoolTiered(MHATokenToKVPoolInt4):
    """int4-g32 full-context rows (inherited) + an int8-g64 ring of the last R written slots + owner table."""

    RING_GROUP = KV_INT8_GROUP
    kv_tiered = True            # QSA backend dispatch key (kv_bits == 4 is inherited)

    def __init__(self, *args, **kwargs):
        self.ring_k = None
        self.ring_v = None
        self.ring_ks = None
        self.ring_vs = None
        self.ring_owner = None
        super().__init__(*args, **kwargs)
        R = ring_slots_from_env()
        if R > self.size:
            raise ValueError(f"SGLANG_KV_TIERS_W={R} exceeds the KV pool size {self.size}")
        if self.head_dim % self.RING_GROUP or self.v_head_dim % self.RING_GROUP:
            raise ValueError(f"head dims {self.head_dim}/{self.v_head_dim} not multiples of RING_GROUP {self.RING_GROUP}")
        self.ring_slots = R
        L, H = self.layer_num, self.head_num
        ks_shape = (R, H, self.head_dim // self.RING_GROUP)
        vs_shape = (R, H, self.v_head_dim // self.RING_GROUP)
        # Plain torch tensors (torch allocator, no 2 MiB granule rounding), allocated once before capture.
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.ring_k = [torch.zeros((R, H, self.head_dim), dtype=torch.int8, device=self.device) for _ in range(L)]
            self.ring_v = [torch.zeros((R, H, self.v_head_dim), dtype=torch.int8, device=self.device) for _ in range(L)]
            self.ring_ks = [torch.zeros(ks_shape, dtype=torch.float16, device=self.device) for _ in range(L)]
            self.ring_vs = [torch.zeros(vs_shape, dtype=torch.float16, device=self.device) for _ in range(L)]
            self.ring_owner = torch.full((R,), -1, dtype=torch.int32, device=self.device)
        ring_bytes = sum(t.numel() * t.element_size() for t in self.ring_k + self.ring_v + self.ring_ks + self.ring_vs)
        logger.info(
            "KV tiers: ring R=%d slots (int8_g%d over int4_g%d), %.0f MB, owner %d KB",
            R, self.RING_GROUP, self.GROUP, ring_bytes / 2**20, self.ring_owner.numel() * 4 // 1024,
        )

    # -- layout ---------------------------------------------------------------------------------

    @property
    def ring_mask(self) -> int:
        return self.ring_slots - 1

    def _ring_bytes(self) -> Tuple[int, int]:
        kb = sum(t.numel() * t.element_size() for t in self.ring_k + self.ring_ks)
        vb = sum(t.numel() * t.element_size() for t in self.ring_v + self.ring_vs)
        return kb + self.ring_owner.numel() * self.ring_owner.element_size(), vb

    def get_kv_size_bytes(self):
        k_size_bytes, v_size_bytes = super().get_kv_size_bytes()
        if self.ring_k is not None:
            rk, rv = self._ring_bytes()
            k_size_bytes += rk
            v_size_bytes += rv
        return k_size_bytes, v_size_bytes

    # _pd_registerable_tensors is inherited on purpose: get_contiguous_buf_infos zips it with
    # _kv_buffer_descs, and the ring has no desc (it is not on the owner).  PD transfer of a tiered pool
    # would move the int4 tier only; PD is unsupported here anyway (CPU offload raises in the int4 pool).

    def _clear_buffers(self):
        super()._clear_buffers()
        self.ring_k = self.ring_v = self.ring_ks = self.ring_vs = self.ring_owner = None

    # -- accessors ------------------------------------------------------------------------------

    def get_kv_ring_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = layer_id - self.start_layer
        return self.ring_k[idx], self.ring_v[idx], self.ring_ks[idx], self.ring_vs[idx]

    def get_kv_ring_owner(self) -> torch.Tensor:
        return self.ring_owner

    # -- write path -----------------------------------------------------------------------------

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None,
        v_scale=None,
        layer_id_override=None,
        dcp_kv_mask=None,
    ):
        if dcp_kv_mask is not None:
            raise NotImplementedError("int8ring_int4 KV cache does not support DCP KV masks")
        if not (_is_unit_scale(k_scale) and _is_unit_scale(v_scale)):
            raise ValueError("int8ring_int4 KV cache computes its own scales; got k_scale/v_scale")
        loc, _, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MHA-TIERED)")
        if loc.numel() > self.ring_slots:
            # more tokens than ring rows would make every write collide (chunked prefill 1024 <= R); a
            # collision is safe (the stamp launch picks one owner per ring row, the others are cold) but
            # this can only be a misconfiguration
            raise ValueError(f"int8ring_int4: {loc.numel()} tokens in one write exceed the ring of {self.ring_slots} slots")
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        idx = layer_id - self.start_layer
        if os.environ.get("SGLANG_KV_STATS") and hasattr(self, "_kv_stats"):
            self._kv_stats(layer_id, cache_k, cache_v)
        cache_k = cache_k.view(-1, self.head_num, self.head_dim)
        cache_v = cache_v.view(-1, self.head_num, self.v_head_dim)
        quant_store_kv_tiered(
            cache_k,
            cache_v,
            loc,
            self.k_buffer[idx],
            self.v_buffer[idx],
            self.k_scale_buffer[idx],
            self.v_scale_buffer[idx],
            self.ring_k[idx],
            self.ring_v[idx],
            self.ring_ks[idx],
            self.ring_vs[idx],
            self.ring_owner,
            self.sm_k_inv[idx],
            self.sm_v_inv[idx],
            self.ring_mask,
        )

    # -- lazy VMM hooks -------------------------------------------------------------------------

    def lazy_release(self) -> None:
        super().lazy_release()
        # Hygiene only: every slot is re-stamped by its own write before any read, so correctness never
        # depends on this; it just keeps a stale owner from pointing a reader at an old ring row.
        if self.ring_owner is not None:
            self.ring_owner.fill_(-1)
'''

NEW_FILES = [(POOL, TIERED_KV_POOL_SRC)]

# (path, old, new[, "all"])   "all" = replace every occurrence (identical text at several sites)
EDITS = [
  # ------------------------------------------------------------- selector: server_args choices (after bfloat16: keeps kv_int4's edit 0 intact)
  (SV, """                "bf16",
                "bfloat16",
""", """                "bf16",
                "bfloat16",
                "int8ring_int4",
"""),
  # ------------------------------------------------------------- selector: torch dtype mapping (uint8 like int4_g32)
  (KD, """    elif server_args_kv_cache_dtype == "int4_g32":
        kv_cache_dtype = torch.uint8
""", """    elif server_args_kv_cache_dtype == "int4_g32":
        kv_cache_dtype = torch.uint8
    elif server_args_kv_cache_dtype == "int8ring_int4":       # int4_g32 rows + int8_g64 ring (tiered pool)
        kv_cache_dtype = torch.uint8
"""),
  # ------------------------------------------------------------- selector: pool class for the hybrid full pool
  # (import placed ABOVE the int4 import: kv_int4's edit 3 (int4 import + int8 import) stays contiguous / APPLIED)
  (KC, """from sglang.srt.mem_cache.int4_kv_pool import MHATokenToKVPoolInt4
""", """from sglang.srt.mem_cache.tiered_kv_pool import MHATokenToKVPoolTiered
from sglang.srt.mem_cache.int4_kv_pool import MHATokenToKVPoolInt4
"""),
  (KC, """        full_pool_class = (
            MHATokenToKVPoolInt4
            if self.kv_cache_dtype_str == "int4_g32"
""", """        full_pool_class = (
            MHATokenToKVPoolTiered
            if self.kv_cache_dtype_str == "int8ring_int4"
            else MHATokenToKVPoolInt4
            if self.kv_cache_dtype_str == "int4_g32"
"""),
  # ------------------------------------------------------------- selector: per-token cell size = the int4 cell size (ring is fixed)
  (PC, """            elif self.kv_cache_dtype_str == "int4_g32":
""", """            elif self.kv_cache_dtype_str in ("int4_g32", "int8ring_int4"):   # tiered: int4 rows + a fixed-size int8 ring
"""),
  # ------------------------------------------------------------- sparse_attn.py: kernels (after the int4 kernels, before valid_counts)
  (SA, """def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
""", """@triton.jit
def _stamp_ring_owner(loc, owner, N, RING_MASK: tl.constexpr, BLOCK: tl.constexpr):
    \"\"\"owner[slot & RING_MASK] = slot for every slot of loc[0:N].  Tokens of one write whose slots share a
    ring row race on the same int32 word; one of them wins (definite once the launch is complete) and only
    the winner writes the ring row in the dual-write launch that follows.  Static grid -> capture-safe.\"\"\"
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    slot = tl.load(loc + offs, mask=m, other=0).to(tl.int64)
    tl.store(owner + (slot & RING_MASK), slot.to(tl.int32), mask=m)


@triton.jit
def _quant_store_kv_tiered(
    k,
    v,
    loc,
    k_buf,
    v_buf,
    k_scale,
    v_scale,
    rk,
    rv,
    rks,
    rvs,
    owner,
    sm_k_inv,
    sm_v_inv,
    sk_n,
    sk_h,
    sv_n,
    sv_h,
    H: tl.constexpr,
    D: tl.constexpr,
    GROUP4: tl.constexpr,
    GROUP8: tl.constexpr,
    RING_MASK: tl.constexpr,
):
    \"\"\"Program (token, head): dual-write of one K row and one V row.  (1) int8-g64 (absmax/127, fp16 scale
    clamped to fp16 max, q = rint(x / s) clamped to [-127, 127]) into ring row r = slot & RING_MASK of
    rk/rv/rks/rvs -- the body of _quant_store_kv_int8 plus the clamp -- stored ONLY if owner[r] == slot
    (stamped by _stamp_ring_owner in the preceding launch: a token that lost a same-write ring-row collision
    stores nothing into the ring and is cold); (2) int4-g32 nibble-packed into the full-context row `slot`
    of k_buf/v_buf/k_scale/v_scale for every token -- the body of _quant_store_kv_int4, verbatim.
    Grid (N, H) is static -> capture-safe.\"\"\"
    DH: tl.constexpr = D // 2
    NG4: tl.constexpr = D // GROUP4
    GH: tl.constexpr = GROUP4 // 2
    NG8: tl.constexpr = D // GROUP8
    t, h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(loc + t).to(tl.int64)
    r = slot & RING_MASK
    hot = tl.load(owner + r).to(tl.int64) == slot          # this token owns its ring row (stamp launch done)

    # ---- (1) int8 ring row (owner only) ------------------------------------------------------
    offs = tl.arange(0, D)
    hmask = (offs < D) & hot
    hsmask = (tl.arange(0, NG8) < NG8) & hot
    goffs8 = tl.arange(0, NG8)
    row8 = (r * H + h) * D
    srow8 = (r * H + h) * NG8
    x = tl.load(k + t * sk_n + h * sk_h + offs).to(tl.float32)
    x = x * tl.load(sm_k_inv + h * D + offs).to(tl.float32)
    xg = tl.reshape(x, [NG8, GROUP8])
    a = tl.max(tl.abs(xg), axis=1)
    # fp16-max clamp (the int8 kernel lacks it): a bf16 group absmax above 127 * 65504 would give s = inf
    s = tl.minimum(tl.where(a > 0, a / 127.0, 1.0), 65504.0).to(tl.float16)
    q = libdevice.rint(xg / s.to(tl.float32)[:, None])
    q = tl.clamp(q, -127.0, 127.0).to(tl.int8)
    tl.store(rk + row8 + offs, tl.reshape(q, [D]), mask=hmask)
    tl.store(rks + srow8 + goffs8, s, mask=hsmask)

    x = tl.load(v + t * sv_n + h * sv_h + offs).to(tl.float32)
    x = x * tl.load(sm_v_inv + h * D + offs).to(tl.float32)
    xg = tl.reshape(x, [NG8, GROUP8])
    a = tl.max(tl.abs(xg), axis=1)
    s = tl.minimum(tl.where(a > 0, a / 127.0, 1.0), 65504.0).to(tl.float16)   # fp16-max clamp, see K half
    q = libdevice.rint(xg / s.to(tl.float32)[:, None])
    q = tl.clamp(q, -127.0, 127.0).to(tl.int8)
    tl.store(rv + row8 + offs, tl.reshape(q, [D]), mask=hmask)
    tl.store(rvs + srow8 + goffs8, s, mask=hsmask)

    # ---- (2) int4 full-context row (= _quant_store_kv_int4) ----------------------------------
    pairs = tl.arange(0, DH)
    even = 2 * pairs
    odd = even + 1
    goffs = tl.arange(0, NG4)
    row = (slot * H + h) * DH
    srow = (slot * H + h) * NG4

    base = k + t * sk_n + h * sk_h
    xe = tl.load(base + even).to(tl.float32) * tl.load(sm_k_inv + h * D + even).to(tl.float32)
    xo = tl.load(base + odd).to(tl.float32) * tl.load(sm_k_inv + h * D + odd).to(tl.float32)
    ge = tl.reshape(xe, [NG4, GH])                                 # (g, j) = channel 32 g + 2 j
    go = tl.reshape(xo, [NG4, GH])                                 # (g, j) = channel 32 g + 2 j + 1
    a = tl.maximum(tl.max(tl.abs(ge), axis=1), tl.max(tl.abs(go), axis=1))
    s = tl.minimum(tl.where(a > 0, a / 7.0, 1.0), 65504.0).to(tl.float16)
    sf = tl.broadcast_to(s.to(tl.float32)[:, None], [NG4, GH])
    qe = tl.clamp(libdevice.rint(tl.div_rn(ge, sf)), -7.0, 7.0).to(tl.int32) + 8
    qo = tl.clamp(libdevice.rint(tl.div_rn(go, sf)), -7.0, 7.0).to(tl.int32) + 8
    packed = (qe | (qo << 4)).to(tl.uint8)
    tl.store(k_buf + row + pairs, tl.reshape(packed, [DH]))
    tl.store(k_scale + srow + goffs, s)

    base = v + t * sv_n + h * sv_h
    xe = tl.load(base + even).to(tl.float32) * tl.load(sm_v_inv + h * D + even).to(tl.float32)
    xo = tl.load(base + odd).to(tl.float32) * tl.load(sm_v_inv + h * D + odd).to(tl.float32)
    ge = tl.reshape(xe, [NG4, GH])
    go = tl.reshape(xo, [NG4, GH])
    a = tl.maximum(tl.max(tl.abs(ge), axis=1), tl.max(tl.abs(go), axis=1))
    s = tl.minimum(tl.where(a > 0, a / 7.0, 1.0), 65504.0).to(tl.float16)
    sf = tl.broadcast_to(s.to(tl.float32)[:, None], [NG4, GH])
    qe = tl.clamp(libdevice.rint(tl.div_rn(ge, sf)), -7.0, 7.0).to(tl.int32) + 8
    qo = tl.clamp(libdevice.rint(tl.div_rn(go, sf)), -7.0, 7.0).to(tl.int32) + 8
    packed = (qe | (qo << 4)).to(tl.uint8)
    tl.store(v_buf + row + pairs, tl.reshape(packed, [DH]))
    tl.store(v_scale + srow + goffs, s)


def quant_store_kv_tiered(k, v, loc, k_buf, v_buf, k_scale, v_scale, rk, rv, rks, rvs, owner, sm_k_inv, sm_v_inv, ring_mask):
    \"\"\"k, v: [N, H, D] (any float dtype, unit last stride); loc: [N] int32/int64 slots (N <= R; slots that
    share a ring row within one write are safe: the stamp launch picks one owner per row, only it writes the
    ring row, the others are cold); k_buf/v_buf: uint8 [rows, H, D // 2]; k_scale/v_scale:
    fp16 [rows, H, D // 32]; rk/rv: int8 [R, H, D]; rks/rvs: fp16 [R, H, D // 64]; owner: int32 [R];
    sm_*_inv: fp16 [H, D]; ring_mask = R - 1 (R a power of two).\"\"\"
    N, H, D = k.shape
    R = int(ring_mask) + 1
    assert R > 0 and R & (R - 1) == 0, f"ring_mask + 1 = {R} is not a power of two"
    assert v.shape == (N, H, D) and k.stride(2) == 1 and v.stride(2) == 1
    assert D % KV_INT4_GROUP == 0 and D % KV_INT8_GROUP == 0 and D == triton.next_power_of_2(D)
    assert k_buf.dtype == torch.uint8 and v_buf.dtype == torch.uint8
    assert k_scale.dtype == torch.float16 and v_scale.dtype == torch.float16
    assert k_buf.is_contiguous() and v_buf.is_contiguous() and k_scale.is_contiguous() and v_scale.is_contiguous()
    assert k_buf.shape[1:] == (H, D // 2) and v_buf.shape[1:] == (H, D // 2)
    assert k_scale.shape[1:] == (H, D // KV_INT4_GROUP) and v_scale.shape[1:] == (H, D // KV_INT4_GROUP)
    assert rk.dtype == torch.int8 and rv.dtype == torch.int8 and rk.shape == (R, H, D) and rv.shape == (R, H, D)
    assert rks.dtype == torch.float16 and rvs.dtype == torch.float16
    assert rks.shape == (R, H, D // KV_INT8_GROUP) and rvs.shape == (R, H, D // KV_INT8_GROUP)
    assert rk.is_contiguous() and rv.is_contiguous() and rks.is_contiguous() and rvs.is_contiguous()
    assert owner.dtype == torch.int32 and owner.shape == (R,) and owner.is_contiguous()
    assert sm_k_inv.shape == (H, D) and sm_v_inv.shape == (H, D)
    assert loc.numel() == N
    assert N <= R, f"{N} tokens in one tiered write exceed the ring of {R} slots"
    if N == 0:
        return
    STAMP_BLOCK = 128
    _stamp_ring_owner[(triton.cdiv(N, STAMP_BLOCK),)](loc, owner, N, RING_MASK=R - 1, BLOCK=STAMP_BLOCK, num_warps=4)
    _quant_store_kv_tiered[(N, H)](
        k,
        v,
        loc,
        k_buf,
        v_buf,
        k_scale,
        v_scale,
        rk,
        rv,
        rks,
        rvs,
        owner,
        sm_k_inv,
        sm_v_inv,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        H=H,
        D=D,
        GROUP4=KV_INT4_GROUP,
        GROUP8=KV_INT8_GROUP,
        RING_MASK=R - 1,
        num_warps=4,
    )


@triton.jit
def _compact_kv_tiered(
    k,
    v,
    k_scale,
    v_scale,
    rk,
    rv,
    rks,
    rvs,
    owner,
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
    GROUP4: tl.constexpr,
    GROUP8: tl.constexpr,
    RING_MASK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    \"\"\"_compact_kv for the tiered pool.  Per selected slot the tier is decided on the device:
    hot = valid & (owner[slot & RING_MASK] == slot) -> int8 ring row (dequant q * s * sm as _compact_kv_int8),
    cold = valid & ~hot -> int4 full-context row (unpack + dequant as _compact_kv_int4); masked-off lanes of
    the other tier issue no loads; tl.where selects.  Store mask unchanged: only valid columns are written
    (trtllm strided tables rely on unused columns never being touched).\"\"\"
    DH: tl.constexpr = dim // 2
    NG4: tl.constexpr = dim // GROUP4
    GH: tl.constexpr = GROUP4 // 2
    NG8: tl.constexpr = dim // GROUP8
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
    # device-side tier test
    r = slots & RING_MASK
    o = tl.load(owner + r, mask=valid, other=-1).to(tl.int64)
    hot = valid & (o == slots)
    cold = valid & (o != slots)
    dst = ((pack_start + cols)[:, None] * heads + head) * dim + dims[None, :]
    mask = valid[:, None] & (dims[None, :] < dim)
    # cold: int4 rows
    src4 = (slots[:, None] * heads + head) * DH + pairs[None, :]
    ssrc4 = (slots[:, None] * heads + head) * NG4 + pairs[None, :] // GH
    pmask = cold[:, None] & (pairs[None, :] < DH)
    # hot: int8 ring rows
    src8 = (r[:, None] * heads + head) * dim + dims[None, :]
    ssrc8 = (r[:, None] * heads + head) * NG8 + dims[None, :] // GROUP8
    mask8 = hot[:, None] & (dims[None, :] < dim)

    sm_e = tl.load(sm_k + head * dim + 2 * pairs).to(tl.float32)
    sm_o = tl.load(sm_k + head * dim + 2 * pairs + 1).to(tl.float32)
    smr = tl.load(sm_k + head * dim + dims).to(tl.float32)
    b = tl.load(k + src4, mask=pmask, other=0).to(tl.int32)
    sc = tl.load(k_scale + ssrc4, mask=pmask, other=0).to(tl.float32)
    lo = ((b & 15) - 8).to(tl.float32) * sc * sm_e[None, :]
    hi = ((b >> 4) - 8).to(tl.float32) * sc * sm_o[None, :]
    k4 = tl.interleave(lo, hi)
    sc8 = tl.load(rks + ssrc8, mask=mask8, other=0).to(tl.float32)
    k8 = tl.load(rk + src8, mask=mask8, other=0).to(tl.float32) * sc8 * smr[None, :]
    tl.store(out_k + dst, tl.where(hot[:, None], k8, k4).to(tl.bfloat16), mask=mask)

    sm_e = tl.load(sm_v + head * dim + 2 * pairs).to(tl.float32)
    sm_o = tl.load(sm_v + head * dim + 2 * pairs + 1).to(tl.float32)
    smr = tl.load(sm_v + head * dim + dims).to(tl.float32)
    b = tl.load(v + src4, mask=pmask, other=0).to(tl.int32)
    sc = tl.load(v_scale + ssrc4, mask=pmask, other=0).to(tl.float32)
    lo = ((b & 15) - 8).to(tl.float32) * sc * sm_e[None, :]
    hi = ((b >> 4) - 8).to(tl.float32) * sc * sm_o[None, :]
    v4 = tl.interleave(lo, hi)
    sc8 = tl.load(rvs + ssrc8, mask=mask8, other=0).to(tl.float32)
    v8 = tl.load(rv + src8, mask=mask8, other=0).to(tl.float32) * sc8 * smr[None, :]
    tl.store(out_v + dst, tl.where(hot[:, None], v8, v4).to(tl.bfloat16), mask=mask)


@triton.jit
def _gather_dequant_rows_tiered(
    k,
    v,
    k_scale,
    v_scale,
    rk,
    rv,
    rks,
    rvs,
    owner,
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
    GROUP4: tl.constexpr,
    GROUP8: tl.constexpr,
    RING_MASK: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    \"\"\"Program (batch, row block, head): rows t < seq_lens[batch] of request req_indices[batch] (slots from
    req_to_token) -> bf16 rows cu_k[batch] + t of out_k/out_v, each row from its tier (owner test as in
    _compact_kv_tiered).  The row-block count is a grid dimension (runtime max_len): no per-chunk recompile.\"\"\"
    DH: tl.constexpr = dim // 2
    NG4: tl.constexpr = dim // GROUP4
    GH: tl.constexpr = GROUP4 // 2
    NG8: tl.constexpr = dim // GROUP8
    batch, block, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    t = block * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = t < length
    slots = tl.load(req_to_token + req * req_stride + t, mask=valid, other=0).to(tl.int64)
    pairs = tl.arange(0, DH)
    dims = tl.arange(0, dim)
    r = slots & RING_MASK
    o = tl.load(owner + r, mask=valid, other=-1).to(tl.int64)
    hot = valid & (o == slots)
    cold = valid & (o != slots)
    dst = ((pack_start + t)[:, None] * heads + head) * dim + dims[None, :]
    mask = valid[:, None] & (dims[None, :] < dim)
    src4 = (slots[:, None] * heads + head) * DH + pairs[None, :]
    ssrc4 = (slots[:, None] * heads + head) * NG4 + pairs[None, :] // GH
    pmask = cold[:, None] & (pairs[None, :] < DH)
    src8 = (r[:, None] * heads + head) * dim + dims[None, :]
    ssrc8 = (r[:, None] * heads + head) * NG8 + dims[None, :] // GROUP8
    mask8 = hot[:, None] & (dims[None, :] < dim)

    sm_e = tl.load(sm_k + head * dim + 2 * pairs).to(tl.float32)
    sm_o = tl.load(sm_k + head * dim + 2 * pairs + 1).to(tl.float32)
    smr = tl.load(sm_k + head * dim + dims).to(tl.float32)
    b = tl.load(k + src4, mask=pmask, other=0).to(tl.int32)
    sc = tl.load(k_scale + ssrc4, mask=pmask, other=0).to(tl.float32)
    lo = ((b & 15) - 8).to(tl.float32) * sc * sm_e[None, :]
    hi = ((b >> 4) - 8).to(tl.float32) * sc * sm_o[None, :]
    k4 = tl.interleave(lo, hi)
    sc8 = tl.load(rks + ssrc8, mask=mask8, other=0).to(tl.float32)
    k8 = tl.load(rk + src8, mask=mask8, other=0).to(tl.float32) * sc8 * smr[None, :]
    tl.store(out_k + dst, tl.where(hot[:, None], k8, k4).to(tl.bfloat16), mask=mask)

    sm_e = tl.load(sm_v + head * dim + 2 * pairs).to(tl.float32)
    sm_o = tl.load(sm_v + head * dim + 2 * pairs + 1).to(tl.float32)
    smr = tl.load(sm_v + head * dim + dims).to(tl.float32)
    b = tl.load(v + src4, mask=pmask, other=0).to(tl.int32)
    sc = tl.load(v_scale + ssrc4, mask=pmask, other=0).to(tl.float32)
    lo = ((b & 15) - 8).to(tl.float32) * sc * sm_e[None, :]
    hi = ((b >> 4) - 8).to(tl.float32) * sc * sm_o[None, :]
    v4 = tl.interleave(lo, hi)
    sc8 = tl.load(rvs + ssrc8, mask=mask8, other=0).to(tl.float32)
    v8 = tl.load(rv + src8, mask=mask8, other=0).to(tl.float32) * sc8 * smr[None, :]
    tl.store(out_v + dst, tl.where(hot[:, None], v8, v4).to(tl.bfloat16), mask=mask)


def _check_tier_args(heads, dim, k_scale, v_scale, ring_k, ring_v, ring_ks, ring_vs, owner, ring_mask):
    assert ring_k is not None and ring_v is not None and ring_ks is not None and ring_vs is not None
    assert owner is not None and ring_mask is not None, "tiered gather needs the owner table and ring_mask"
    R = int(ring_mask) + 1
    assert R > 0 and R & (R - 1) == 0, f"ring_mask + 1 = {R} is not a power of two"
    assert dim % KV_INT4_GROUP == 0 and dim % KV_INT8_GROUP == 0 and dim == triton.next_power_of_2(dim)
    assert k_scale.shape[1:] == (heads, dim // KV_INT4_GROUP) and k_scale.is_contiguous()
    assert v_scale.shape[1:] == (heads, dim // KV_INT4_GROUP) and v_scale.is_contiguous()
    assert ring_k.dtype == torch.int8 and ring_v.dtype == torch.int8
    assert ring_k.shape == (R, heads, dim) and ring_v.shape == (R, heads, dim)
    assert ring_ks.dtype == torch.float16 and ring_vs.dtype == torch.float16
    assert ring_ks.shape == (R, heads, dim // KV_INT8_GROUP) and ring_vs.shape == (R, heads, dim // KV_INT8_GROUP)
    assert ring_k.is_contiguous() and ring_v.is_contiguous() and ring_ks.is_contiguous() and ring_vs.is_contiguous()
    assert owner.dtype == torch.int32 and owner.shape == (R,) and owner.is_contiguous()
    return R


def qwen_sparse_prefix_gather_dequant_tiered(
    k, v, k_scale, v_scale, sm_k, sm_v, req_to_token, req_indices, seq_lens, cu_k, out_k, out_v, batch, max_len,
    ring_k=None, ring_v=None, ring_ks=None, ring_vs=None, owner=None, ring_mask=None,
):
    \"\"\"Whole-prefix gather-dequant of the tiered pool (int4 rows k/v uint8 [rows, H, D // 2] + int8 ring):
    rows [0, seq_lens[b]) of every request packed at cu_k[b] into the bf16 out_k/out_v [.., H, D], each row
    from its tier.  Rows at or beyond seq_lens[b] are not written.\"\"\"
    _, heads, dh = k.shape
    dim = 2 * dh
    assert k.dtype == torch.uint8 and v.dtype == torch.uint8
    assert out_k.dtype == torch.bfloat16 and out_v.dtype == torch.bfloat16, "tiered KV gather needs a bf16 scratch"
    assert out_k.shape[1:] == (heads, dim) and out_v.shape[1:] == (heads, dim)
    assert sm_k.shape == (heads, dim) and sm_v.shape == (heads, dim)
    assert out_k.is_contiguous() and out_v.is_contiguous()
    R = _check_tier_args(heads, dim, k_scale, v_scale, ring_k, ring_v, ring_ks, ring_vs, owner, ring_mask)
    block_t = 16
    if batch == 0 or max_len == 0:
        return
    _gather_dequant_rows_tiered[(batch, triton.cdiv(int(max_len), block_t), heads)](
        k,
        v,
        k_scale,
        v_scale,
        ring_k,
        ring_v,
        ring_ks,
        ring_vs,
        owner,
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
        GROUP4=KV_INT4_GROUP,
        GROUP8=KV_INT8_GROUP,
        RING_MASK=R - 1,
        BLOCK_T=block_t,
        num_warps=8,
    )


def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
"""),
  # ------------------------------------------------------------- sparse_attn.py: wrapper branch (before int4 / int8 / fp8)
  (SA, """    k_scale=None, v_scale=None, sm_k=None, sm_v=None, kv_bits=None,
):
    _, heads, dim = k.shape
    block_topk = 16
    if k.dtype == torch.uint8 and kv_bits == 4:              # int4_g32 pool (keyed on the pool: fp8 is uint8 too)
""", """    k_scale=None, v_scale=None, sm_k=None, sm_v=None, kv_bits=None,
    ring_k=None, ring_v=None, ring_ks=None, ring_vs=None, owner=None, ring_mask=None,
):
    _, heads, dim = k.shape
    block_topk = 16
    if k.dtype == torch.uint8 and kv_bits == 4 and owner is not None:   # tiered pool: int8 ring over int4 rows
        dim = 2 * dim                                        # k is [rows, H, D // 2] packed; dim = logical D
        assert out_k.dtype == torch.bfloat16 and out_v.dtype == torch.bfloat16, "tiered KV gather needs a bf16 scratch"
        assert out_k.shape[1:] == (heads, dim) and out_v.shape[1:] == (heads, dim)
        assert k_scale is not None and v_scale is not None and sm_k is not None and sm_v is not None
        R = _check_tier_args(heads, dim, k_scale, v_scale, ring_k, ring_v, ring_ks, ring_vs, owner, ring_mask)
        _compact_kv_tiered[(batch, heads, triton.cdiv(topk, block_topk))](
            k,
            v,
            k_scale,
            v_scale,
            ring_k,
            ring_v,
            ring_ks,
            ring_vs,
            owner,
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
            GROUP4=KV_INT4_GROUP,
            GROUP8=KV_INT8_GROUP,
            RING_MASK=R - 1,
            BLOCK_TOPK=block_topk,
            num_warps=8,
        )
        return
    if k.dtype == torch.uint8 and kv_bits == 4:              # int4_g32 pool (keyed on the pool: fp8 is uint8 too)
"""),
  # ------------------------------------------------------------- backend: import the prefix gather
  (BK, """    qwen_sparse_prefix_gather_dequant_int4,
    qwen_sparse_prefix_gather_dequant_int8,
""", """    qwen_sparse_prefix_gather_dequant_int4,
    qwen_sparse_prefix_gather_dequant_int8,
    qwen_sparse_prefix_gather_dequant_tiered,
"""),
  # ------------------------------------------------------------- backend: ring/owner kwargs (both compact call sites go through _int8_gather_kwargs)
  (BK, """        return dict(k_scale=k_sf, v_scale=v_sf, sm_k=sm_k, sm_v=sm_v, kv_bits=kv_bits)
""", """        return dict(k_scale=k_sf, v_scale=v_sf, sm_k=sm_k, sm_v=sm_v, kv_bits=kv_bits, **self._kv_tier_kwargs(layer))

    def _kv_tier_kwargs(self, layer) -> dict:
        \"\"\"Ring buffers + owner table + mask of the tiered pool (int8 ring over int4 rows; {} otherwise).
        The tensors are fixed for the pool's lifetime (allocated before capture): capture-safe.\"\"\"
        pool = self.token_to_kv_pool
        full_pool = getattr(pool, "full_kv_pool", pool)
        if not getattr(full_pool, "kv_tiered", False):
            return {}
        ring_k, ring_v, ring_ks, ring_vs = pool.get_kv_ring_buffer(layer.layer_id)
        return dict(
            ring_k=ring_k, ring_v=ring_v, ring_ks=ring_ks, ring_vs=ring_vs,
            owner=pool.get_kv_ring_owner(), ring_mask=full_pool.ring_mask,
        )
"""),
  # ------------------------------------------------------------- backend: prefix-chunk gather-dequant picks the tiered kernel
  (BK, """            gather_dequant = (
                qwen_sparse_prefix_gather_dequant_int4
                if kv_bits == 4
                else qwen_sparse_prefix_gather_dequant_int8
            )
""", """            tier_kwargs = self._kv_tier_kwargs(layer)
            gather_dequant = (
                qwen_sparse_prefix_gather_dequant_tiered
                if tier_kwargs
                else qwen_sparse_prefix_gather_dequant_int4
                if kv_bits == 4
                else qwen_sparse_prefix_gather_dequant_int8
            )
"""),
  (BK, """                len(sequence_lens),
                max(sequence_lens),
            )
        else:
""", """                len(sequence_lens),
                max(sequence_lens),
                **tier_kwargs,
            )
        else:
"""),
  # ------------------------------------------------------------- memory_pool: HybridLinearKVPool forwarders
  (MP, """    def get_kv_scale_buffer(self, layer_id: int):
        # MXFP8 full_kv_pool exposes per-32 UE8M0 K/V scale buffers.
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_scale_buffer(layer_id)
""", """    def get_kv_scale_buffer(self, layer_id: int):
        # MXFP8 full_kv_pool exposes per-32 UE8M0 K/V scale buffers.
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_scale_buffer(layer_id)

    def get_kv_ring_buffer(self, layer_id: int):
        # int8ring_int4 full_kv_pool exposes the per-layer int8 ring (K, V, K scales, V scales).
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_ring_buffer(layer_id)

    def get_kv_ring_owner(self):
        # int8ring_int4 full_kv_pool: ring-row owner table (shared by all layers).
        return self.full_kv_pool.get_kv_ring_owner()
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


def tiers_applied():
    """True once the tiered kernels are in the tree (kv_int4's overlaid edits then read MISMATCH)."""
    return "def _compact_kv_tiered(" in _read(SA)


def int4_applied():
    """kv_int4 prerequisite: every kv_int4 edit and its new file APPLIED (meaningful while kv_tiers is not applied)."""
    return all(ap for _, _, ap in kv_int4.state()) and all(ap for _, _, ap in kv_int4.file_state())


def int4_kernels_present():
    return "def _compact_kv_int4(" in _read(SA)


def paged_applied():
    """True while perf/patches/kv_paged_prefix.py is applied on top of kv_tiers (its kernel sits inside kv_tiers'
    sparse_attn block and marks edit 5's anchor): kv_tiers must be neither applied nor reverted underneath it."""
    return "def _sparse_gqa_chunk_prefill_paged(" in _read(SA)


def _refuse_under_paged(what):
    if paged_applied():
        print(f"  [!] kv_paged_prefix is applied on top of kv_tiers (_sparse_gqa_chunk_prefill_paged in sparse_attn.py): "
              f"refusing to {what} kv_tiers underneath it; run `kv_paged_prefix.py revert` first (order: "
              "kv_int4 < kv_tiers < kv_paged_prefix)")
        check(); sys.exit(1)


# kv_int4 edits that kv_tiers does NOT overlay: they read APPLIED in `kv_int4.py --check` while kv_tiers is
# applied (the others read MISMATCH or clean: 4 = configurator full_pool_class, 5 = pool_configurator elif,
# 7 = sparse_attn kernels (the tiered kernels split its applied text), 8 = wrapper branch, 10 = _int8_gather_kwargs,
# 15 = prefix block).  If any of these is missing while kv_tiers is applied, `kv_int4.py revert` ran underneath
# (out of order): it reverted exactly these and skipped the rest, so a kv_tiers revert would re-materialise
# kv_int4's overlaid wording (e.g. the MHATokenToKVPoolInt4 configurator branch) with its reverted parts
# (e.g. the import, int4_kv_pool.py) missing.
KV_INT4_INTACT_UNDER_TIERS = (0, 1, 2, 3, 6, 9, 11, 12, 13, 14)


def int4_reverted_underneath():
    """kv_int4 edits (indices) that must be intact under kv_tiers but are not -- non-empty = out-of-order revert."""
    st = kv_int4.state()
    bad = [i for i in KV_INT4_INTACT_UNDER_TIERS if not st[i][2]]
    if not all(ap for _, _, ap in kv_int4.file_state()):
        bad.append("F")
    return bad


def check():
    if paged_applied():
        print("  P kv_paged_prefix applied on top (revert kv_paged_prefix.py before kv_tiers.py; edit 5 reads MISMATCH meanwhile)")
    if tiers_applied():
        bad = int4_reverted_underneath()
        print(f"  P kv_int4 overlaid by kv_tiers (revert kv_tiers.py before kv_int4.py); "
              f"_compact_kv_int4 {'present' if int4_kernels_present() else 'MISSING'}; "
              f"kv_int4 edits {list(KV_INT4_INTACT_UNDER_TIERS)} + file {'intact' if not bad else 'NOT intact: ' + str(bad)}")
    else:
        print(f"  P kv_int4 prerequisite {'APPLIED' if int4_applied() else 'MISSING (apply kv_int4.py first)'}")
    for i, (p, pr, ap) in enumerate(state()):
        print(f"  {i} {'APPLIED' if ap else ('clean' if pr else 'MISMATCH'):<8} {_mode(EDITS[i]):<4} "
              f"{os.path.relpath(p, SG)}: {EDITS[i][1].strip().splitlines()[0][:60]}")
    for p, pr, ap in file_state():
        print(f"  F {'APPLIED' if ap else ('clean' if pr else 'MISMATCH'):<8} new  {os.path.relpath(p, SG)}")


def apply():
    _refuse_under_paged("apply")
    st = state()
    fs = file_state()
    if not tiers_applied() and not int4_applied():
        print("  [!] prerequisite missing: apply perf/patches/kv_int4.py first (kv_tiers layers on its edits)")
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
    print("  applied (tiered int8-ring-over-int4 KV cache for QSA; serve with --kv-cache-dtype int8ring_int4, "
          "SGLANG_KV_TIERS_W=<ring slots, default 8192>)")
    # keep KV_INT4_INTACT_UNDER_TIERS honest: it must be exactly the kv_int4 edits still APPLIED now
    now = tuple(i for i, (_, _, ap) in enumerate(kv_int4.state()) if ap)
    if now != KV_INT4_INTACT_UNDER_TIERS:
        print(f"  [!] KV_INT4_INTACT_UNDER_TIERS {KV_INT4_INTACT_UNDER_TIERS} != kv_int4 edits APPLIED under kv_tiers {now}; "
              "update the constant (the out-of-order revert guard keys on it)")


def revert(force=False):
    _refuse_under_paged("revert")
    if tiers_applied():
        bad = int4_reverted_underneath()
        if bad or not int4_kernels_present():
            print(f"  [!] kv_int4 was reverted underneath kv_tiers (out of order): kv_int4 edits {bad} not APPLIED"
                  f"{'' if int4_kernels_present() else ', _compact_kv_int4 kernel absent'}. Reverting kv_tiers now would "
                  "re-materialise kv_int4's overlaid text (e.g. the MHATokenToKVPoolInt4 configurator branch) with "
                  "its reverted parts (e.g. the import, int4_kv_pool.py) missing. Repair: `kv_tiers.py revert --force`, "
                  "then `kv_int4.py revert`, then `kv_int8.py --check` and hand-restore whatever is not uniformly APPLIED.")
            if not force:
                print("  refused (pass --force to revert kv_tiers anyway)"); check(); sys.exit(1)
            print("  --force: reverting kv_tiers anyway; run kv_int4.py revert and kv_int8.py --check afterwards")
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
