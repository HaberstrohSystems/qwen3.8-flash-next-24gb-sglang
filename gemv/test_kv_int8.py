"""Unit test for the int8_g64 QSA KV path (run after patches/kv_int8.py apply; ~50 MB VRAM).

Checks the three Triton kernels against a torch reference (per-token-per-head-per-64-group absmax/127,
fp16 scale, q = rint(x / s) clamped to [-127, 127], dequant q * s in fp32 -> bf16):
  1. _quant_store_kv_int8: quantize + scatter into random slots (int32 and int64 loc); payload and
     scales bit-exact vs the reference; untouched slots keep their sentinel; zero rows get s = 1;
  2. scale index arithmetic for GROUP=64 with 2 heads: flat index (slot*2 + h)*4 + g;
  3. _compact_kv_int8: batch of 3 requests (300 / 1200 / 4000 tokens, permuted req_to_token), bit-exact
     dequant vs torch, rows beyond the packed range untouched, an invalid position inside the packed
     region is neither read nor written (same store mask as _compact_kv / _compact_kv_fp8: on the
     trtllm strided tables cu_k spans the page stride, so zero-filling would write every unused column);
  4. _gather_dequant_rows_int8: 3 requests with different lengths (partial last row block, non-identity
     req_indices) packed with a 32-row gap after EACH request, bit-exact, every gap row and the tail
     untouched (an over-write at an inner boundary cannot be hidden by the neighbour's own writes);
  5. end-to-end relative RMS error of gather-dequant(quant(x)) for x ~ N(0, 3) < 1.2 %;
  6. MHATokenToKVPoolInt8 itself: eager path and the lazy-VMM path (SGLANG_KV_LAZY=1), set_kv_buffer
     through the pool, scale descs / bytes-per-token, first-page scale rows zeroed.
"""
import os
import torch
import torch.nn.functional as F
from sglang.srt.layers.attention.qsa.sparse_attn import (
    KV_INT8_GROUP as G,
    quant_store_kv_int8,
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_prefix_gather_dequant_int8,
    qwen_sparse_fa2_cu_seqlens_triton,
)

torch.manual_seed(0)
dev = "cuda"
slots, heads, dim, batch, topk = 4096, 2, 256, 3, 64
NG = dim // G
ones = torch.ones(heads, dim, dtype=torch.float16, device=dev)


def ref_quant(x):
    """x [N, H, D] -> (q int8 [N, H, D], s fp16 [N, H, NG])."""
    xf = x.float().reshape(*x.shape[:-1], NG, G)
    a = xf.abs().amax(-1)
    s = torch.where(a > 0, a / 127.0, torch.ones_like(a)).half()
    q = torch.clamp(torch.round(xf / s.float()[..., None]), -127, 127).to(torch.int8).reshape(x.shape)
    return q, s


def ref_dequant(q, s):
    return (q.float() * s.float().repeat_interleave(G, dim=-1)).to(torch.bfloat16)


def relerr(a, b):
    return float(((a.float() - b.float()).pow(2).mean() / b.float().pow(2).mean()).sqrt())


# ---------------------------------------------------------------- 1. quantize + scatter into random slots
N = 1000
xk = torch.randn(N, heads, dim, device=dev, dtype=torch.bfloat16) * 3
xv = torch.randn(N, heads, dim, device=dev, dtype=torch.bfloat16) * 3
xk[5] = 0                                   # zero row -> s = 1, q = 0
xv[7, 1, 64:128] = 0                        # one zero group in one head
qk_ref, sk_ref = ref_quant(xk)
qv_ref, sv_ref = ref_quant(xv)
for loc_dtype in (torch.int32, torch.int64):
    loc = torch.randperm(slots, device=dev)[:N].to(loc_dtype)
    k_buf = torch.full((slots, heads, dim), 7, dtype=torch.int8, device=dev)
    v_buf = torch.full((slots, heads, dim), -7, dtype=torch.int8, device=dev)
    ks = torch.full((slots, heads, NG), 3.0, dtype=torch.float16, device=dev)
    vs = torch.full((slots, heads, NG), 5.0, dtype=torch.float16, device=dev)
    quant_store_kv_int8(xk, xv, loc, k_buf, v_buf, ks, vs, ones, ones)
    torch.cuda.synchronize()
    ll = loc.long()
    assert torch.equal(k_buf[ll], qk_ref) and torch.equal(v_buf[ll], qv_ref), f"payload mismatch ({loc_dtype})"
    assert torch.equal(ks[ll], sk_ref) and torch.equal(vs[ll], sv_ref), f"scale mismatch ({loc_dtype})"
    untouched = torch.ones(slots, dtype=torch.bool, device=dev); untouched[ll] = False
    assert (k_buf[untouched] == 7).all() and (v_buf[untouched] == -7).all(), "payload of other slots changed"
    assert (ks[untouched] == 3.0).all() and (vs[untouched] == 5.0).all(), "scales of other slots changed"
    assert (ks[ll[5]] == 1.0).all() and (k_buf[ll[5]] == 0).all(), "zero row must give s = 1, q = 0"
    assert float(vs[ll[7], 1, 1]) == 1.0, "zero group must give s = 1"
    assert int(qk_ref.abs().max()) == 127 and int(k_buf[ll].abs().max()) == 127
    print(f"  quant+scatter ({loc_dtype}): bit-exact payload + scales for {N} tokens x {heads} heads, others untouched: ok")

# ---------------------------------------------------------------- 2. scale index arithmetic (GROUP=64, 2 heads)
flat = ks.view(-1)
for _ in range(64):
    t = int(torch.randint(N, (1,))); h = int(torch.randint(heads, (1,))); g = int(torch.randint(NG, (1,)))
    slot = int(loc[t])
    assert float(flat[(slot * heads + h) * NG + g]) == float(sk_ref[t, h, g]), "scale index (slot*H+h)*NG+g mismatch"
assert not torch.equal(sk_ref[:, 0], sk_ref[:, 1]), "heads must carry distinct scales for the test to be meaningful"
print("  scale index (slot*2 + h)*4 + g: ok (64 random probes)")

# ---------------------------------------------------------------- full pool for the gather tests
k16 = torch.randn(slots, heads, dim, device=dev, dtype=torch.bfloat16) * 3
v16 = torch.randn(slots, heads, dim, device=dev, dtype=torch.bfloat16) * 3
k8 = torch.empty(slots, heads, dim, dtype=torch.int8, device=dev); v8 = torch.empty_like(k8)
ks8 = torch.empty(slots, heads, NG, dtype=torch.float16, device=dev); vs8 = torch.empty_like(ks8)
quant_store_kv_int8(k16, v16, torch.arange(slots, device=dev, dtype=torch.int32), k8, v8, ks8, vs8, ones, ones)
torch.cuda.synchronize()
qk, sk = ref_quant(k16); qv, sv = ref_quant(v16)
assert torch.equal(k8, qk) and torch.equal(ks8, sk) and torch.equal(v8, qv) and torch.equal(vs8, sv)
dk_all = ref_dequant(qk, sk); dv_all = ref_dequant(qv, sv)          # reference dequant of every slot

# ---------------------------------------------------------------- 3. compact gather (decode / verify path)
seq_lens = torch.tensor([300, 1200, 4000], dtype=torch.int32, device=dev)
req_to_token = torch.stack([torch.randperm(slots, device=dev) for _ in range(batch)]).to(torch.int32)
req_indices = torch.arange(batch, dtype=torch.int32, device=dev)
indices = torch.stack([torch.randperm(int(l), device=dev)[:topk].sort().values for l in seq_lens]).to(torch.int32)


def run_compact(idx):
    cu_k = torch.empty(batch + 1, dtype=torch.int32, device=dev)
    counts = torch.empty(batch, dtype=torch.int32, device=dev)
    qwen_sparse_fa2_cu_seqlens_triton(seq_lens, idx, counts, cu_k, batch, topk)
    n = int(cu_k[-1])
    out_k = torch.full((batch * topk + 16, heads, dim), float("nan"), device=dev, dtype=torch.bfloat16)
    out_v = torch.full_like(out_k, float("nan"))
    qwen_sparse_kv_extraction_compact_triton(
        k8, v8, req_to_token, req_indices, idx, seq_lens, cu_k, out_k, out_v, batch, topk,
        k_scale=ks8, v_scale=vs8, sm_k=ones, sm_v=ones)
    torch.cuda.synchronize()
    return cu_k, n, out_k, out_v


cu_k, n, out_k, out_v = run_compact(indices)
assert n == batch * topk
for b in range(batch):
    rows = req_to_token[b, indices[b].long()].long()
    a, e = int(cu_k[b]), int(cu_k[b + 1])
    assert torch.equal(out_k[a:e], dk_all[rows]) and torch.equal(out_v[a:e], dv_all[rows]), f"compact gather req {b} != torch dequant"
assert torch.isnan(out_k[n:]).all() and torch.isnan(out_v[n:]).all(), "rows beyond the packed range were written"
print(f"  compact gather-dequant bit-exact vs torch (q*s -> bf16), {n} rows, {batch} requests: ok")

bad = indices.clone(); bad[1, 0] = int(seq_lens[1]) + 5             # invalid position inside request 1's region
cu_b, n_b, ok_b, ov_b = run_compact(bad)
assert n_b == batch * topk - 1
a = int(cu_b[1])
assert torch.isnan(ok_b[a]).all() and torch.isnan(ov_b[a]).all(), "invalid position inside the packed region must not be written"
rows = req_to_token[1, bad[1, 1:topk - 1].long()].long()
assert torch.equal(ok_b[a + 1:a + topk - 1], dk_all[rows]) and torch.equal(ov_b[a + 1:a + topk - 1], dv_all[rows])
assert torch.equal(ok_b[:a], out_k[:a]) and torch.equal(ok_b[int(cu_b[2]):n_b], out_k[int(cu_k[2]):n])
assert torch.isnan(ok_b[n_b:]).all()
print("  compact gather: invalid position -> row untouched, neighbours intact, tail untouched: ok")

# ---------------------------------------------------------------- 4. prefix row gather (chunk-prefill path)
lens = torch.tensor([300, 1203, 4001], dtype=torch.int32, device=dev)    # 4001 > slots? no: slots=4096
req_idx = torch.tensor([2, 0, 1], dtype=torch.int64, device=dev)          # non-identity request mapping
GAP = 32                                                                  # > BLOCK_T (16): a stray row block lands in the gap
cu = F.pad((lens + GAP).cumsum(0), (1, 0)).to(torch.int32).contiguous()   # request b packed at cu[b], then GAP unused rows
total = int(cu[-1])
pk = torch.full((total + 100, heads, dim), float("nan"), device=dev, dtype=torch.bfloat16)
pv = torch.full_like(pk, float("nan"))
qwen_sparse_prefix_gather_dequant_int8(k8, v8, ks8, vs8, ones, ones, req_to_token, req_idx, lens, cu, pk, pv,
                                       batch, int(lens.max()))
torch.cuda.synchronize()
written = torch.zeros(pk.shape[0], dtype=torch.bool, device=dev)
for b in range(batch):
    rows = req_to_token[int(req_idx[b]), : int(lens[b])].long()
    a, e = int(cu[b]), int(cu[b]) + int(lens[b])
    assert torch.equal(pk[a:e], dk_all[rows]) and torch.equal(pv[a:e], dv_all[rows]), f"row gather req {b} != torch dequant"
    written[a:e] = True
    assert torch.isnan(pk[e:e + GAP]).all() and torch.isnan(pv[e:e + GAP]).all(), f"rows beyond seq_len of req {b} were written"
assert torch.isnan(pk[~written]).all() and torch.isnan(pv[~written]).all(), "rows outside every request were written"
assert int(written.sum()) == int(lens.sum())
print(f"  prefix row gather-dequant bit-exact, lens {lens.tolist()} (req_idx {req_idx.tolist()}), "
      f"{GAP}-row gaps after each request + tail untouched: ok")

# ---------------------------------------------------------------- 5. end-to-end error
gathered_k = torch.cat([k16[req_to_token[int(req_idx[b]), : int(lens[b])].long()] for b in range(batch)])
gathered_v = torch.cat([v16[req_to_token[int(req_idx[b]), : int(lens[b])].long()] for b in range(batch)])
ek, ev = relerr(pk[written], gathered_k), relerr(pv[written], gathered_v)
print(f"  int8_g64 relative RMS error on N(0,3): K {ek*100:.3f} %  V {ev*100:.3f} %  (e4m3 reference ~2.7 %)")
assert ek < 0.012 and ev < 0.012, "relative RMS error above 1.2 %"

# ---------------------------------------------------------------- 6. the pool class (eager + lazy VMM)
from sglang.srt.mem_cache.int8_kv_pool import MHATokenToKVPoolInt8

for lazy in (False, True):
    if lazy:
        os.environ["SGLANG_KV_LAZY"] = "1"; os.environ["SGLANG_KV_LAZY_FLOOR"] = "512"
    else:
        os.environ.pop("SGLANG_KV_LAZY", None)
    size, page = 1024, 64
    pool = MHATokenToKVPoolInt8(size=size, page_size=page, dtype=torch.int8, head_num=heads, head_dim=dim,
                                layer_num=2, device=dev, enable_memory_saver=False, start_layer=0)
    rows = size + page
    assert pool.dtype == pool.store_dtype == torch.int8
    assert len(pool.k_buffer) == len(pool.v_buffer) == len(pool.k_scale_buffer) == len(pool.v_scale_buffer) == 2
    assert pool.k_buffer[0].shape == (rows, heads, dim) and pool.k_buffer[0].dtype == torch.int8
    assert pool.k_scale_buffer[0].shape == (rows, heads, NG) and pool.k_scale_buffer[0].dtype == torch.float16
    assert pool.get_key_buffer(1).dtype == torch.int8 and pool.get_key_buffer(1).data_ptr() == pool.k_buffer[1].data_ptr()
    descs = pool._kv_buffer_descs
    assert [d.name for d in descs] == ["k0", "k1", "v0", "v1", "ks0", "ks1", "vs0", "vs1"]
    assert all(d.row_bytes == 16 and d.shape == (rows, 16) for d in descs[4:])
    kb, vb = pool.get_kv_size_bytes()
    assert kb == vb == 2 * rows * (heads * dim + heads * NG * 2)
    if lazy:
        o = pool._post_capture_owner
        assert o is not None and len(o.tensors) == 8 and o.bytes_per_token() == 2 * (512 + 512 + 16 + 16)
        assert pool.k_scale_buffer[0].data_ptr() == o.tensors[4].data_ptr()
        assert (pool.k_scale_buffer[1][:page] == 0).all() and (pool.v_scale_buffer[0][:page] == 0).all()
    else:
        assert (pool.k_scale_buffer[0] == 0).all()
    n = 200
    loc = torch.randperm(512, device=dev)[:n].to(torch.int64)          # inside the backed floor
    xk = torch.randn(n, heads, dim, device=dev, dtype=torch.bfloat16) * 3
    xv = torch.randn(n, heads, dim, device=dev, dtype=torch.bfloat16) * 3
    pool.set_kv_buffer(None, loc, xk, xv, 1.0, 1.0, layer_id_override=1)   # HybridLinearKVPool style
    pool.set_kv_buffer(None, loc, xk.reshape(n, -1), xv.reshape(n, -1), layer_id_override=0)   # [N, H*D] form
    torch.cuda.synchronize()
    qk_r, sk_r = ref_quant(xk); qv_r, sv_r = ref_quant(xv)
    for l in (0, 1):
        ksb, vsb = pool.get_kv_scale_buffer(l)
        assert torch.equal(pool.k_buffer[l][loc], qk_r) and torch.equal(ksb[loc], sk_r)
        assert torch.equal(pool.v_buffer[l][loc], qv_r) and torch.equal(vsb[loc], sv_r)
    smk, smv = pool.get_kv_smooth_buffer(1)
    assert smk.shape == (heads, dim) and bool((smk == 1).all()) and bool((smv == 1).all())
    for bad_kw in (dict(k_scale=0.5), dict(v_scale=torch.ones(1, device=dev)), dict(dcp_kv_mask=torch.ones(n, device=dev))):
        try:
            pool.set_kv_buffer(None, loc, xk, xv, layer_id_override=0, **bad_kw); raise AssertionError("expected a raise")
        except (ValueError, NotImplementedError):
            pass
    for fn in (lambda: pool.get_cpu_copy(loc), lambda: pool.load_cpu_copy(None, loc), lambda: pool.set_kv_buffer_prefix_valid()):
        try:
            fn(); raise AssertionError("expected NotImplementedError")
        except NotImplementedError:
            pass
    pool._clear_buffers()
    print(f"  MHATokenToKVPoolInt8 ({'lazy VMM' if lazy else 'eager'}): descs, sizes, set_kv_buffer bit-exact, guards: ok")
os.environ.pop("SGLANG_KV_LAZY", None)
print("  OK")
