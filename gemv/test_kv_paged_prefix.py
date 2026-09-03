"""Unit test for the paged prefix-chunk kernel (run after patches/kv_int4.py + kv_tiers.py + kv_paged_prefix.py apply;
< 200 MB VRAM).

Pools (int8_g64, int4_g32, tiered = int8 ring R over the int4 pool) are filled by the REAL writers
(quant_store_kv_int8 / _int4 / _tiered) from the same random bf16 K/V into random ascending slot maps
(req_to_token[b, :seq_len[b]] = sorted random slots; the rest of the row is junk), with RANDOM per-channel smoothing
factors sm_k != sm_v in [0.5, 2) (writers get sm_inv = (1 / sm).half(), readers get sm: the sm / sm_inv contract).
Batch of 3 requests: (prefix 0, chunk 1024), (prefix 6000, chunk 300), (prefix 2500, chunk 512); topk width 2051 like
the indexer, per query sorted positions < visible, valid-first then -1 padding, some rows short, some all -1, some with
exactly ONE valid position, one row only from the current chunk.  cu_q is built exactly as the backend builds
cu_seqlens_q (F.pad(int32.cumsum(0), (1, 0)) -> int64).  Tiered: R = 1024 with ~10% of the owner entries cleared
(-1: post-lazy_release) -> a hot/cold mix.
Reference = the materialised path exactly as the backend runs it (qwen_sparse_prefix_gather_dequant_{mode} into fresh
bf16 packed buffers + _sparse_gqa_chunk_prefill).
ACCEPTANCE (DEVIATION from "bit-exact or max |diff| <= 2 bf16 ulps per element", stated here and in the ok line):
the two kernels are NOT bit-exact on Triton 3.7.1 (a computed K operand changes the fp32 MMA accumulation order of
the QK dot).  Both kernels cast the probabilities to bf16 for the PV dot, so a tiny fp32 score difference that flips
one bf16 rounding of a dominant p_i moves the output by ~2^-8 * |p_i v_i|: proportional to the ROW scale, not to the
element -- a per-element ulp bound is not attainable (measured: up to ~15 element ulps on elements >= rowmax/64; the
production kernel against itself at another tile config, section 1c, shows the same behaviour).  Bound applied:
|diff| <= 2 ulps of the (query, head) row's absmax (measured max 1.0).  The DISCRIMINATING checks are bit-exact by
construction:  1a. the K/V operand tiles the paged kernel feeds its dots (dumped through the same `_paged_tile`
device function, K in the transposed [D, N] orientation) == the gather's bf16 rows, torch.equal over every selected
row of sampled queries, all three modes, random smoothing (catches any dequant / scale-group / nibble / tier / sm
defect regardless of softmax weight);  1b. rows with exactly one valid position (p = exp2(0) = 1, normalizer 1)
return the gather's bf16 V row exactly (torch.equal), per head.
  1. matched tile config (16,1,2) / (16,4,2) / (32,4,2), all three modes: the report above;
  1a. operand-tile dump == gather rows, bit-exact;
  1b. single-position rows: paged output == the packed bf16 V row of that position (and == the materialised output);
  1c. control: the packed kernel at (16,1,2) vs itself at (32,4,2) (accumulation-order change of the production kernel);
  2. production pair: paged at _PAGED_CONFIG vs sparse_gqa_fwd_interface_triton_ck (its table config), same report;
  3. dirty indices (entries >= kv_len inside the read range): paged(dirty) == paged(clean) bit-exact and == the
     materialised path on the clean indices (the materialised kernel would read foreign packed rows on the dirty ones);
  4. torch fp32 reference (softmax over the gathered bf16 rows of the selected valid positions) for 24 sampled
     queries: allclose at bf16 tolerance (validates visibility / masking / softmax independently of Triton);
  5. timing at prefix 60,000 (one request, 1024-query chunk = the production chunk, ONE kv head, tiered R = 8192, last
     8192 slots hot): gather alone, packed kernel at the interface's table config for 1024 queries and explicitly at
     (16,1,2), materialised total (alloc + gather + kernel), paged kernel at three configs -- each L2-WARM (back to back)
     and L2-COLD (a 64 MB memset before every timed launch evicts the 48 MB L2: the 258k regime where the 2 x 528 MB
     scratch cannot be resident); plus the bit-exact report at that size at matched config.  The GPU is shared with a
     running server: timings are noisy.  The 258k slope-fit of the plan (section 6.1) stays the decision criterion.
"""
import gc
import time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sglang.srt.layers.attention.qsa.sparse_attn import (
    KV_INT4_GROUP as G4,
    KV_INT8_GROUP as G8,
    _PAGED_CONFIG,
    _get_best_config,
    _paged_tile,
    _sparse_gqa_chunk_prefill,
    quant_store_kv_int4,
    quant_store_kv_int8,
    quant_store_kv_tiered,
    qwen_sparse_prefix_gather_dequant_int4,
    qwen_sparse_prefix_gather_dequant_int8,
    qwen_sparse_prefix_gather_dequant_tiered,
    sparse_gqa_fwd_interface_paged,
    sparse_gqa_fwd_interface_triton_ck,
)

torch.manual_seed(0)
dev = "cuda"
H, D, QH = 2, 256, 24
GS = QH // H
TOPK = 2051
SCALE = D ** -0.5
NG4, DH, NG8 = D // G4, D // 2, D // G8
MB = 1024 * 1024


def make_smooth(heads):
    """Random per-channel smoothing factors in [0.5, 2), separate for K and V; sm_inv for the writers."""
    g = torch.Generator().manual_seed(11 + heads)
    sm_k = (0.5 + 1.5 * torch.rand(heads, D, generator=g)).to(torch.float16).to(dev)
    sm_v = (0.5 + 1.5 * torch.rand(heads, D, generator=g)).to(torch.float16).to(dev)
    return sm_k, sm_v, (1.0 / sm_k.float()).half(), (1.0 / sm_v.float()).half()


SM_K, SM_V, SMI_K, SMI_V = make_smooth(H)


def ulps(a, b):
    """bf16 ulp distance per element (sign-magnitude -> monotone integer)."""
    ia = a.contiguous().view(torch.int16).to(torch.int32)
    ib = b.contiguous().view(torch.int16).to(torch.int32)
    ma = torch.where(ia < 0, -(ia & 0x7FFF), ia)
    mb = torch.where(ib < 0, -(ib & 0x7FFF), ib)
    return (ma - mb).abs()


def bit_equal(a, b):
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def cpu(x):
    return x.detach().to("cpu")


torch.cuda.memory._record_memory_history(max_entries=200000)      # allocation stacks for the lingering dump below


def peak(tag, lingering=False):
    """Per-section peak of live device allocations (the test budget is < 200 MB); resets the counter.
    lingering=True also lists every still-allocated block >= 1 MB with the Python frames that allocated it
    (what holds VRAM after a `del`: e.g. a cuBLAS workspace, which is why the fp32 reference runs on the CPU)."""
    torch.cuda.synchronize()
    gc.collect()
    print(f"      [VRAM {tag}: peak {torch.cuda.max_memory_allocated() / MB:.1f} MB, live now {torch.cuda.memory_allocated() / MB:.1f} MB]")
    if lingering:
        for seg in torch.cuda.memory._snapshot()["segments"]:
            for bl in seg["blocks"]:
                if bl["state"] == "active_allocated" and bl["size"] >= MB:
                    fr = [f"{f['filename'].split('/')[-1]}:{f['line']} {f['name']}" for f in bl.get("frames", [])
                          if f["filename"].endswith(".py")][:4]
                    print(f"        lingering block {bl['size'] / MB:.1f} MB: {fr}")
    torch.cuda.reset_peak_memory_stats()


def row_ulps(a, b):
    """|a - b| in units of the bf16 ulp of the (query, head) output vector's absmax: the meaningful distance when
    fp32 accumulation order differs (near-zero cancellation outputs make per-element ulps arbitrary).
    Returns (row-scaled ulps, rowmax broadcast)."""
    af, bf = a.float(), b.float()
    rowmax = torch.maximum(af.abs().amax(-1, keepdim=True), bf.abs().amax(-1, keepdim=True)).clamp(min=1e-30)
    ulp = torch.exp2(torch.floor(torch.log2(rowmax)) - 7)              # bf16: 8 significand bits incl. hidden
    d = (af - bf).abs() / ulp
    return torch.where(torch.isnan(af) & torch.isnan(bf), torch.zeros_like(d), d), rowmax


ELEM_FRAC = 1.0 / 64                                                   # element-ulp bound applies at |x| >= rowmax / 64


def report_diff(name, a, b, limit=2.0):
    a, b = cpu(a), cpu(b)                               # all diff bookkeeping on the host: no GPU temporaries
    if bit_equal(a, b):
        print(f"    {name}: bit-exact")
        return True
    nan_mismatch = int((torch.isnan(a) != torch.isnan(b)).sum())
    assert nan_mismatch == 0, f"{name}: {nan_mismatch} elements NaN in one path only"
    u = ulps(a, b)
    u[torch.isnan(a) & torch.isnan(b)] = 0
    ru, rowmax = row_ulps(a, b)
    big = torch.maximum(a.float().abs(), b.float().abs()) >= rowmax * ELEM_FRAC
    u_big = torch.where(big, u, torch.zeros_like(u))
    nz = int((u > 0).sum())
    print(f"    {name}: NOT bit-exact: {nz} / {u.numel()} elements differ ({100.0 * nz / u.numel():.3f}%), "
          f"max |diff| {float((a.float() - b.float()).abs().nan_to_num(0).max()):.3e}, "
          f"element ulps: ==1 {int((u == 1).sum())}, ==2 {int((u == 2).sum())}, >2 {int((u > 2).sum())} "
          f"(max {int(u_big.max())} on the {int(big.sum())} elements >= rowmax/64); "
          f"max diff in ulps of the row absmax {float(ru.max()):.3f} (bound {limit})")
    assert float(ru.max()) <= limit, f"{name}: {float(ru.max()):.3f} row-scaled ulps > {limit}"
    return False


# ---------------------------------------------------------------- operand-tile dump (the discriminating check)
@triton.jit
def _dump_tiles(qids, qbatch, indices, kv_lens, req_to_token, req_indices, k4, v4, ks4, vs4, rk, rv, rks, rvs, owner,
                sm_k, sm_v, out_k, out_v, topk, si_m, si_n,
                NUM_KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, REQ_STRIDE: tl.constexpr,
                GROUP4: tl.constexpr, GROUP8: tl.constexpr, RING_MASK: tl.constexpr, MODE: tl.constexpr):
    """The paged kernel's per-tile prologue (index -> slot -> tier) + _paged_tile, storing the bf16 K ([D, N], as fed to
    the QK dot) and V ([N, D]) operands of every valid lane of sampled queries: out_[k|v][(i * H + group) * topk + n, d]."""
    i = tl.program_id(0)
    group = tl.program_id(1)
    query = tl.load(qids + i).to(tl.int64)
    batch = tl.load(qbatch + i)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    req = tl.load(req_indices + batch).to(tl.int64)
    slot_row = req_to_token + req * REQ_STRIDE
    idx_row = indices + query * si_m
    head = group
    dims = tl.arange(0, HEAD_DIM)
    smr_k = tl.load(sm_k + head * HEAD_DIM + dims).to(tl.float32)
    smr_v = tl.load(sm_v + head * HEAD_DIM + dims).to(tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    base = (i * NUM_KV_HEADS + group) * topk
    for start in range(0, topk, BLOCK_N):
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
        keys = _paged_tile(k4, ks4, rk, rks, slots, r, hot, cold, head, dims, smr_k,
                           NUM_KV_HEADS, HEAD_DIM, GROUP4, GROUP8, MODE, True).to(tl.bfloat16)
        values = _paged_tile(v4, vs4, rv, rvs, slots, r, hot, cold, head, dims, smr_v,
                             NUM_KV_HEADS, HEAD_DIM, GROUP4, GROUP8, MODE, False).to(tl.bfloat16)
        tl.store(out_k + (base + current[None, :]) * HEAD_DIM + dims[:, None], keys, mask=valid[None, :])
        tl.store(out_v + (base + current[:, None]) * HEAD_DIM + dims[None, :], values, mask=valid[:, None])


def dump_tiles(case, mode, qids, block_n=16):
    """Returns (out_k, out_v) bf16 [len(qids), H, TOPK, D] on the device (rows of invalid lanes = 0)."""
    p, t = case["pools"][mode], case["pools"][mode]["tier"]
    idx = case["idx"]
    cu = case["cu_q"].cpu()
    qbatch = torch.tensor([int((cu <= q).sum()) - 1 for q in qids], dtype=torch.int32, device=dev)
    qids_t = torch.tensor(qids, dtype=torch.int32, device=dev)
    out_k = torch.zeros((len(qids), H, TOPK, D), dtype=torch.bfloat16, device=dev)
    out_v = torch.zeros_like(out_k)
    if mode == "tiered":
        mode_id, mask = 2, t["ring_mask"]
        k4, v4, ks4, vs4 = p["k"], p["v"], p["k_scale"], p["v_scale"]
        rk, rv, rks, rvs, own = t["ring_k"], t["ring_v"], t["ring_ks"], t["ring_vs"], t["owner"]
    elif mode == "int4":
        mode_id, mask = 1, 0
        k4, v4, ks4, vs4 = p["k"], p["v"], p["k_scale"], p["v_scale"]
        rk, rv, rks, rvs, own = k4, v4, ks4, vs4, case["req_indices"]
    else:
        mode_id, mask = 0, 0
        rk, rv, rks, rvs = p["k"], p["v"], p["k_scale"], p["v_scale"]
        k4, v4, ks4, vs4, own = rk, rv, rks, rvs, case["req_indices"]
    _dump_tiles[(len(qids), H)](
        qids_t, qbatch, idx, case["seq_lens_t"], case["req_to_token"], case["req_indices"],
        k4, v4, ks4, vs4, rk, rv, rks, rvs, own, SM_K, SM_V, out_k, out_v, TOPK, idx.stride(0), idx.stride(1),
        NUM_KV_HEADS=H, HEAD_DIM=D, BLOCK_N=block_n, REQ_STRIDE=case["req_to_token"].stride(0),
        GROUP4=G4, GROUP8=G8, RING_MASK=mask, MODE=mode_id, num_warps=4)
    return out_k, out_v


# ---------------------------------------------------------------- pool construction
def new_pools(slots, R, modes):
    """Fresh (poisoned) int8 / int4 / tiered pools of `slots` rows."""
    pools = {}
    if "int8" in modes:
        k8 = torch.full((slots, H, D), 0x33, dtype=torch.int8, device=dev); v8 = torch.full_like(k8, 0x44)
        ks8 = torch.full((slots, H, NG8), 7.0, dtype=torch.float16, device=dev); vs8 = torch.full_like(ks8, 9.0)
        pools["int8"] = dict(k=k8, v=v8, k_scale=ks8, v_scale=vs8, kv_bits=8, tier={})
    for mode in ("int4", "tiered"):
        if mode not in modes:
            continue
        k4 = torch.full((slots, H, DH), 0x77, dtype=torch.uint8, device=dev); v4 = torch.full_like(k4, 0x99)
        ks4 = torch.full((slots, H, NG4), 3.0, dtype=torch.float16, device=dev); vs4 = torch.full_like(ks4, 5.0)
        tier = {}
        if mode == "tiered":
            rk = torch.full((R, H, D), 0x33, dtype=torch.int8, device=dev); rv = torch.full_like(rk, 0x44)
            rks = torch.full((R, H, NG8), 7.0, dtype=torch.float16, device=dev); rvs = torch.full_like(rks, 9.0)
            owner = torch.full((R,), -1, dtype=torch.int32, device=dev)
            tier = dict(ring_k=rk, ring_v=rv, ring_ks=rks, ring_vs=rvs, owner=owner, ring_mask=R - 1)
        pools[mode] = dict(k=k4, v=v4, k_scale=ks4, v_scale=vs4, kv_bits=4, tier=tier)
    return pools


def write_chunk(pools, xk, xv, loc, R):
    """Write the tokens (xk, xv) at ascending slots `loc` into every pool with the REAL writers."""
    if "int8" in pools:
        p = pools["int8"]
        quant_store_kv_int8(xk, xv, loc, p["k"], p["v"], p["k_scale"], p["v_scale"], SMI_K, SMI_V)
    if "int4" in pools:
        p = pools["int4"]
        quant_store_kv_int4(xk, xv, loc, p["k"], p["v"], p["k_scale"], p["v_scale"], SMI_K, SMI_V)
    if "tiered" in pools:
        p, t = pools["tiered"], pools["tiered"]["tier"]
        lc = loc.tolist()                                   # alias-free ascending pieces (span < R, N <= R)
        start = 0
        for i in range(1, len(lc) + 1):
            if i == len(lc) or lc[i] - lc[start] >= R or i - start >= R:
                quant_store_kv_tiered(xk[start:i], xv[start:i], loc[start:i], p["k"], p["v"], p["k_scale"], p["v_scale"],
                                      t["ring_k"], t["ring_v"], t["ring_ks"], t["ring_vs"], t["owner"], SMI_K, SMI_V, R - 1)
                start = i


def make_case(prefix_lens, chunk_lens, slots, R, modes, clear_owner_frac=0.0, chunk=4096):
    batch = len(prefix_lens)
    seq_lens = [p + c for p, c in zip(prefix_lens, chunk_lens)]
    total = sum(seq_lens)
    assert total <= slots
    perm = torch.randperm(slots, device=dev)[:total]
    req_to_token = torch.randint(0, slots, (batch, slots), dtype=torch.int32, device=dev)   # junk beyond seq_len
    a = 0
    for b in range(batch):
        req_to_token[b, : seq_lens[b]] = perm[a:a + seq_lens[b]].sort().values.to(torch.int32)
        a += seq_lens[b]
    loc = perm.sort().values                                        # global ascending write order
    pools = new_pools(slots, R, modes)
    for a in range(0, total, chunk):                                # chunked: no prefix-sized bf16 source
        n = min(chunk, total - a)
        xk = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16) * 3
        xv = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16) * 3
        write_chunk(pools, xk, xv, loc[a:a + n], R)
    del xk, xv
    torch.cuda.synchronize()
    if "tiered" in pools and clear_owner_frac > 0:
        owner = pools["tiered"]["tier"]["owner"]
        clear = torch.rand(R, device=dev) < clear_owner_frac
        owner[clear] = -1
    total_q = sum(chunk_lens)
    q = torch.randn(total_q, QH, D, device=dev, dtype=torch.bfloat16)
    # exactly as the backend builds cu_seqlens_q (qwen_sparse_attn_backend.py): int32 cumsum -> int64
    cu_q = F.pad(torch.tensor(chunk_lens, device=dev, dtype=torch.int32).cumsum(0), (1, 0)).contiguous()
    assert cu_q.dtype == torch.int64, cu_q.dtype
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    req_indices = torch.arange(batch, dtype=torch.int64, device=dev)   # req_pool_indices are int64 in the backend
    # indices like the indexer's output: per query sorted positions < visible, valid first, -1 padding
    idx = torch.full((total_q, TOPK), -1, dtype=torch.int32)
    g = torch.Generator().manual_seed(1)
    row = 0
    special = {"short": 0, "empty": 0, "chunk_only": 0, "single": 0}
    single_rows = []                                                # (row, batch, position)
    for b in range(batch):
        for j in range(chunk_lens[b]):
            visible = prefix_lens[b] + j + 1
            n = min(TOPK, visible)
            u = torch.rand(1, generator=g).item()
            if u < 0.03 and special["empty"] < 8:
                special["empty"] += 1; row += 1; continue                     # all -1
            if u < 0.06 and prefix_lens[b] > 0 and special["chunk_only"] < 8:  # only current-chunk positions
                special["chunk_only"] += 1
                pos = torch.arange(prefix_lens[b], visible)[:n]
            elif u < 0.10 and special["single"] < 8:                            # exactly one valid position
                special["single"] += 1
                pos = torch.randint(0, visible, (1,), generator=g)
                single_rows.append((row, b, int(pos)))
            else:
                if u < 0.25:
                    n = int(torch.randint(0, n + 1, (1,), generator=g)); special["short"] += 1
                pos = torch.randperm(visible, generator=g)[:n].sort().values
            idx[row, : pos.numel()] = pos.to(torch.int32)
            row += 1
    idx = idx.to(dev)
    return dict(batch=batch, prefix_lens=prefix_lens, chunk_lens=chunk_lens, seq_lens=seq_lens, seq_lens_t=seq_lens_t,
                req_to_token=req_to_token, req_indices=req_indices, q=q, cu_q=cu_q, idx=idx, pools=pools,
                max_q=max(chunk_lens), special=special, single_rows=single_rows)


# ---------------------------------------------------------------- the two paths
def gather_packed(case, mode):
    """The backend's materialised gather: fresh bf16 packed buffers + gather-dequant kernel; returns (cu_k, pk, pv)."""
    p = case["pools"][mode]
    cu_k = F.pad(case["seq_lens_t"].cumsum(0), (1, 0)).contiguous()
    pk = torch.empty((sum(case["seq_lens"]), H, D), dtype=torch.bfloat16, device=dev)
    pv = torch.empty_like(pk)
    fn = {"int8": qwen_sparse_prefix_gather_dequant_int8, "int4": qwen_sparse_prefix_gather_dequant_int4,
          "tiered": qwen_sparse_prefix_gather_dequant_tiered}[mode]
    fn(p["k"], p["v"], p["k_scale"], p["v_scale"], SM_K, SM_V, case["req_to_token"], case["req_indices"],
       case["seq_lens_t"], cu_k, pk, pv, case["batch"], max(case["seq_lens"]), **p["tier"])
    return cu_k, pk, pv


def packed_kernel(case, idx, cu_k, pk, pv, config):
    """_sparse_gqa_chunk_prefill launched directly at a chosen (BLOCK_N, num_warps, num_stages)."""
    q = case["q"]
    block_n, warps, stages = config
    out = torch.empty_like(q)
    _sparse_gqa_chunk_prefill[(case["max_q"], case["batch"] * H)](
        q, pk, pv, out, idx, case["cu_q"], cu_k, case["seq_lens_t"], SCALE, idx.shape[-1],
        q.stride(0), q.stride(1), q.stride(2), pk.stride(0), pk.stride(1), pk.stride(2),
        pv.stride(0), pv.stride(1), pv.stride(2), out.stride(0), out.stride(1), out.stride(2),
        idx.stride(0), 0, idx.stride(1),
        NUM_KV_HEADS=H, GROUP_SIZE=GS, BLOCK_M=16, BLOCK_N=block_n, HEAD_DIM=D, num_warps=warps, num_stages=stages)
    return out


def materialized(case, mode, idx, config=None):
    cu_k, pk, pv = gather_packed(case, mode)
    if config is None:
        return sparse_gqa_fwd_interface_triton_ck(case["q"], pk, pv, idx, case["cu_q"], cu_k, case["seq_lens_t"], SCALE)
    return packed_kernel(case, idx, cu_k, pk, pv, config)


def paged(case, mode, idx, config=None):
    p = case["pools"][mode]
    return sparse_gqa_fwd_interface_paged(
        case["q"], idx, case["cu_q"], case["seq_lens_t"], case["req_to_token"], case["req_indices"],
        p["k"], p["v"], p["k_scale"], p["v_scale"], SM_K, SM_V, SCALE, case["max_q"], p["kv_bits"],
        config=config, **p["tier"])


# ================================================================ small case: 3 requests, all three pools
torch.cuda.reset_peak_memory_stats()
R_small = 1024
case = make_case([0, 6000, 2500], [1024, 300, 512], slots=12288, R=R_small, modes=("int8", "int4", "tiered"),
                 clear_owner_frac=0.1)
own = case["pools"]["tiered"]["tier"]["owner"]
hot_slots = 0
for b, l in enumerate(case["seq_lens"]):
    rt_b = case["req_to_token"][b, :l].long()
    hot_slots += int((own.long()[rt_b & (R_small - 1)] == rt_b).sum())
print(f"small case: seq_lens {case['seq_lens']} chunks {case['chunk_lens']} total_q {case['q'].shape[0]} topk {TOPK}; "
      f"cu_q dtype {case['cu_q'].dtype}; sm_k/sm_v random in [{float(SM_K.min()):.2f}, {float(SM_K.max()):.2f}]; "
      f"tiered ring R={R_small}: {hot_slots} hot of {sum(case['seq_lens'])} context slots; "
      f"index rows: {case['special']}")

# ---------------------------------------------------------------- 1. matched configs, bit-exact
CONFIGS = [(16, 1, 2), (16, 4, 2), (32, 4, 2)]
all_exact = True
for mode in ("int8", "int4", "tiered"):
    for cfg in CONFIGS:
        om = materialized(case, mode, case["idx"], cfg)
        op = paged(case, mode, case["idx"], cfg)
        torch.cuda.synchronize()
        all_exact &= report_diff(f"1. {mode:<6} matched config {cfg}: paged vs packed", op, om)
        del om, op
print(f"  1. matched-config comparison: {'ALL BIT-EXACT' if all_exact else 'differences within the stated bounds (see above)'}")
peak("section 1")

# ---------------------------------------------------------------- 1a. operand tiles == gather rows, bit-exact
samp = torch.randperm(case["q"].shape[0], generator=torch.Generator().manual_seed(5))[:6].tolist()
for mode in ("int8", "int4", "tiered"):
    cu_k, pk, pv = gather_packed(case, mode)
    dk, dv = dump_tiles(case, mode, samp)
    torch.cuda.synchronize()
    ek = torch.zeros((len(samp), H, TOPK, D), dtype=torch.bfloat16, device="cpu")
    ev = torch.zeros_like(ek)
    n_rows, n_hot = 0, 0
    for i, r in enumerate(samp):
        b = int((case["cu_q"].cpu() <= r).sum()) - 1
        pos = case["idx"][r].long()
        ok = (pos >= 0) & (pos < case["seq_lens"][b])
        rows = int(cu_k[b]) + pos[ok]
        ek[i, :, ok.cpu()] = pk[rows].transpose(0, 1).cpu()             # [H, n, D]
        ev[i, :, ok.cpu()] = pv[rows].transpose(0, 1).cpu()
        n_rows += int(ok.sum())
        if mode == "tiered":
            s = case["req_to_token"][b, pos[ok]].long()
            n_hot += int((own.long()[s & (R_small - 1)] == s).sum())
    assert bit_equal(cpu(dk), ek), f"1a. {mode}: paged K operand tiles != gather K rows"
    assert bit_equal(cpu(dv), ev), f"1a. {mode}: paged V operand tiles != gather V rows"
    print(f"  1a. {mode:<6} operand tiles of {len(samp)} queries x {H} heads ({n_rows} selected rows"
          f"{f', {n_hot} hot' if mode == 'tiered' else ''}): K [D, N] and V [N, D] == gather rows, bit-exact")
    del cu_k, pk, pv, dk, dv, ek, ev
peak("section 1a")

# ---------------------------------------------------------------- 1b. single-position rows: bit-exact regardless of MMA order
single = case["single_rows"]
assert len(single) == 8, single
for mode in ("int8", "int4", "tiered"):
    cu_k, pk, pv = gather_packed(case, mode)
    n_hot = 0
    for label, fn in (("paged", paged), ("packed", materialized)):
        o = fn(case, mode, case["idx"])
        torch.cuda.synchronize()
        for r, b, pos in single:
            slot = int(case["req_to_token"][b, pos])
            if mode == "tiered" and label == "paged":
                n_hot += int(own[slot & (R_small - 1)]) == slot
            vrow = pv[int(cu_k[b]) + pos]                                # [H, D] bf16: the gather's V row
            for h in range(H):
                want = vrow[h][None, :].expand(GS, D)
                assert bit_equal(o[r, h * GS:(h + 1) * GS], want), f"1b. {mode} row {r} head {h}: {label} != V row"
        del o, vrow, want
    print(f"  1b. {mode:<6} {len(single)} single-position rows{f' ({n_hot} hot)' if mode == 'tiered' else ''}: "
          f"paged == packed == the gather's bf16 V row, bit-exact")
    del cu_k, pk, pv
peak("section 1b")

# ---------------------------------------------------------------- 1c. control: the packed kernel against itself
cu_k, pk, pv = gather_packed(case, "int8")
oa = packed_kernel(case, case["idx"], cu_k, pk, pv, (16, 1, 2))
ob = packed_kernel(case, case["idx"], cu_k, pk, pv, (32, 4, 2))
torch.cuda.synchronize()
report_diff("1c. control: packed (16,1,2) vs packed (32,4,2), same scratch", oa, ob)
del cu_k, pk, pv, oa, ob
peak("section 1c")

# ---------------------------------------------------------------- 2. production pair
for mode in ("int8", "int4", "tiered"):
    om = materialized(case, mode, case["idx"])                    # sparse_gqa_fwd_interface_triton_ck: table config
    op = paged(case, mode, case["idx"])                            # _PAGED_CONFIG
    torch.cuda.synchronize()
    report_diff(f"2. {mode:<6} production: paged {_PAGED_CONFIG} vs interface_triton_ck {_get_best_config(case['q'].shape[0])}", op, om)
    del om, op
peak("section 2")

# ---------------------------------------------------------------- 3. dirty indices (>= kv_len inside the read range)
dirty = case["idx"].clone()
clean = case["idx"].clone()
g = torch.Generator().manual_seed(2)
rows = torch.randperm(dirty.shape[0], generator=g)[:64]
n_dirty = 0
for r in rows.tolist():
    b = int((case["cu_q"].cpu() <= r).sum()) - 1
    kv_len = case["seq_lens"][b]
    j = r - int(case["cu_q"][b])
    visible = case["prefix_lens"][b] + j + 1
    n = min(TOPK, visible)
    cols = torch.randperm(n, generator=g)[: max(1, n // 50)]
    dirty[r, cols] = (kv_len + torch.randint(0, 100, (cols.numel(),), generator=g)).to(torch.int32).to(dev)
    clean[r, cols] = -1
    n_dirty += cols.numel()
for mode in ("int8", "int4", "tiered"):
    op_d = paged(case, mode, dirty, (16, 4, 2))
    op_c = paged(case, mode, clean, (16, 4, 2))
    torch.cuda.synchronize()
    assert bit_equal(op_d, op_c), f"3. {mode}: paged(dirty) != paged(clean): out-of-range indices not ignored"
    del op_c
    om_c = materialized(case, mode, clean, (16, 4, 2))
    torch.cuda.synchronize()
    report_diff(f"3. {mode:<6} dirty indices ({n_dirty} entries >= kv_len in {rows.numel()} rows): paged(dirty) vs packed(clean)", op_d, om_c)
    del op_d, om_c
print("  3. paged(dirty) == paged(clean) bit-exact for all modes")
peak("section 3")

# ---------------------------------------------------------------- 4. torch fp32 reference on sampled queries
sample = torch.randperm(case["q"].shape[0], generator=torch.Generator().manual_seed(3))[:24].tolist()
for mode in ("int8", "int4", "tiered"):
    cu_k, pk, pv = gather_packed(case, mode)
    op = paged(case, mode, case["idx"])
    torch.cuda.synchronize()
    worst = 0.0
    for r in sample:
        b = int((case["cu_q"].cpu() <= r).sum()) - 1
        j = r - int(case["cu_q"][b])
        visible = case["prefix_lens"][b] + j + 1
        n = min(TOPK, visible)
        pos = case["idx"][r, :n].long()
        pos = pos[(pos >= 0) & (pos < visible)]
        qf = case["q"][r].float().cpu() * SCALE                             # [QH, D]; reference on the CPU: a CUDA
        for h in range(H):                                                  # matmul would pin a 32 MB cuBLAS workspace
            qh = qf[h * GS:(h + 1) * GS]
            got = op[r, h * GS:(h + 1) * GS].float().cpu()
            if pos.numel() == 0:
                assert not torch.isfinite(got).any() or torch.isnan(got).all(), "empty selection must be nan (0/0) as in the packed kernel"
                continue
            kk = pk[int(cu_k[b]) + pos, h].float().cpu(); vv = pv[int(cu_k[b]) + pos, h].float().cpu()
            ref = torch.softmax(qh @ kk.T, dim=-1) @ vv
            worst = max(worst, float((got - ref).abs().max()))
            assert torch.allclose(got, ref, atol=3e-2, rtol=3e-2), f"4. {mode} query {r} head {h}: max |diff| {float((got - ref).abs().max())}"
    print(f"  4. {mode:<6} torch fp32 reference on {len(sample)} queries: allclose (max |diff| {worst:.3e})")
    del cu_k, pk, pv, op, kk, vv, got, ref, qf, qh, pos
peak("section 4")

# ================================================================ 5. timing at prefix 60k (tiered R = 8192)
del case, own, dirty, clean, rt_b
torch.cuda.empty_cache()
peak("small case freed", lingering=True)
PREFIX, CHUNK, R_big = 60000, 1024, 8192
H, QH = 1, 12                        # one kv head for the 60k case: the materialised path's 2 KB/token bf16 scratch
GS = QH // H                         # alone is 124 MB at 2 heads; both paths scale identically with the head count
SM_K, SM_V, SMI_K, SMI_V = make_smooth(H)
big = make_case([PREFIX], [CHUNK], slots=PREFIX + CHUNK, R=R_big, modes=("tiered",), chunk=4096)
own = big["pools"]["tiered"]["tier"]["owner"]
rt = big["req_to_token"][0, : PREFIX + CHUNK].long()
n_hot = int((own.long()[rt & (R_big - 1)] == rt).sum())
sel = big["idx"][big["idx"] >= 0].long()
n_sel = sel.numel()
sel_slots = rt[sel]
sel_hot = float((own.long()[sel_slots & (R_big - 1)] == sel_slots).float().mean())
del sel, sel_slots, rt
print(f"timing case: 1 request, prefix {PREFIX}, chunk {CHUNK}, {H} kv head x {GS} q heads, tiered R={R_big}: {n_hot} hot slots, "
      f"selected rows {n_sel} ({100 * sel_hot:.1f}% hot); packed scratch 2 x {(PREFIX + CHUNK) * H * D * 2 / MB:.1f} MB, "
      f"L2 {torch.cuda.get_device_properties(0).L2_cache_size / MB:.0f} MB")
om = materialized(big, "tiered", big["idx"], (16, 4, 2))
op = paged(big, "tiered", big["idx"], (16, 4, 2))
torch.cuda.synchronize()
report_diff("5. 60k tiered matched config (16,4,2): paged vs packed", op, om)
del om, op
peak("section 5 setup")

FLUSH = torch.empty(64 * MB, dtype=torch.uint8, device=dev)     # > L2 (48 MB): a memset evicts the L2 before a cold launch


def bench(fn, iters=10, warm=3, cold=False):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if cold:
            FLUSH.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3                     # median ms


def bench2(fn, **kw):
    return bench(fn, **kw), bench(fn, cold=True, **kw)


cu_k, pk, pv = gather_packed(big, "tiered")
torch.cuda.synchronize()
p = big["pools"]["tiered"]
cfg_if = _get_best_config(CHUNK)
t_gather = bench2(lambda: qwen_sparse_prefix_gather_dequant_tiered(
    p["k"], p["v"], p["k_scale"], p["v_scale"], SM_K, SM_V, big["req_to_token"], big["req_indices"],
    big["seq_lens_t"], cu_k, pk, pv, 1, PREFIX + CHUNK, **p["tier"]))
t_ck = bench2(lambda: sparse_gqa_fwd_interface_triton_ck(big["q"], pk, pv, big["idx"], big["cu_q"], cu_k, big["seq_lens_t"], SCALE))
t_pk = bench2(lambda: packed_kernel(big, big["idx"], cu_k, pk, pv, (16, 1, 2)))
del pk, pv
t_mat = bench2(lambda: materialized(big, "tiered", big["idx"]))
rows_sel = n_sel * H                                        # (query, kv-head) row selections
bytes_ck = rows_sel * 2 * D * 2                              # bf16 K + V per selected row (pre-L2)
bytes_paged = rows_sel * (2 * (DH + NG4 * 2) * (1 - sel_hot) + 2 * (D + NG8 * 2) * sel_hot) + n_sel * 12
bytes_gather = (PREFIX + CHUNK) * H * (2 * (DH + NG4 * 2) + 2 * D * 2)
print(f"  5. prefix {PREFIX} chunk {CHUNK} (median of 10, shared GPU; 'warm' = back to back, 'cold' = 64 MB memset before each launch):")
print(f"     materialised: gather warm {t_gather[0]:.3f} / cold {t_gather[1]:.3f} ms ({bytes_gather / t_gather[1] / 1e6:.0f} GB/s cold)")
print(f"     materialised: packed kernel via interface {cfg_if}: warm {t_ck[0]:.3f} / cold {t_ck[1]:.3f} ms "
      f"({bytes_ck / t_ck[1] / 1e6:.0f} GB/s of bf16 row requests cold)")
print(f"     materialised: packed kernel (16, 1, 2) explicit: warm {t_pk[0]:.3f} / cold {t_pk[1]:.3f} ms")
print(f"     materialised: alloc + gather + kernel {cfg_if}: warm {t_mat[0]:.3f} / cold {t_mat[1]:.3f} ms "
      f"(gather + kernel = warm {t_gather[0] + t_ck[0]:.3f} / cold {t_gather[1] + t_ck[1]:.3f} ms)")
best = None
for cfg in [(16, 4, 2), (16, 8, 2), (32, 4, 2)]:
    tw, tc = bench2(lambda: paged(big, "tiered", big["idx"], cfg))
    best = tc if best is None else min(best, tc)
    print(f"     paged {cfg}: warm {tw:.3f} / cold {tc:.3f} ms ({bytes_paged / tc / 1e6:.0f} GB/s of pool row requests cold) -> "
          f"cold: {t_mat[1] / tc:.2f}x vs materialised total, {t_ck[1] / tc:.2f}x vs packed {cfg_if}, {t_pk[1] / tc:.2f}x vs packed (16,1,2)")
print(f"TIMING prefix={PREFIX} chunk={CHUNK} tiered 1 kv head: materialised warm {t_mat[0]:.3f} / cold {t_mat[1]:.3f} ms "
      f"(gather {t_gather[1]:.3f} + kernel {cfg_if} {t_ck[1]:.3f} cold; packed (16,1,2) {t_pk[1]:.3f} cold) "
      f"vs paged best cold {best:.3f} ms = {t_mat[1] / best:.2f}x")
print(f"  timing case peak VRAM: {torch.cuda.max_memory_allocated() / MB:.1f} MB allocated (incl. the 64 MB flush buffer), "
      f"{torch.cuda.max_memory_reserved() / MB:.1f} MB reserved")
# the paged kernel alone on the pure pools at the same size (the packed kernel's time above does not depend on the pool)
del big, own, cu_k, p
torch.cuda.empty_cache()
peak("timing case freed", lingering=True)
for mode in ("int8", "int4"):
    torch.cuda.reset_peak_memory_stats()
    one = make_case([PREFIX], [CHUNK], slots=PREFIX + CHUNK, R=R_big, modes=(mode,), chunk=4096)
    tw, tc = bench2(lambda: paged(one, mode, one["idx"], (16, 4, 2)))
    print(f"     paged {mode} (16, 4, 2): warm {tw:.3f} / cold {tc:.3f} ms (cold: {t_ck[1] / tc:.2f}x vs packed {cfg_if}, "
          f"{t_pk[1] / tc:.2f}x vs packed (16,1,2); peak VRAM {torch.cuda.max_memory_allocated() / MB:.1f} MB)")
    del one
    torch.cuda.empty_cache()
print("test_kv_paged_prefix: ok (criterion, a DEVIATION from per-element 2 ulps: operand tiles and single-position rows "
      "bit-exact, attention output |diff| <= 2 ulps of the row absmax -- see the module docstring)")
