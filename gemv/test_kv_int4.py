"""Unit test for the int4_g32 QSA KV path (run after patches/kv_int4.py apply; ~30 MB VRAM).

Checks the three Triton kernels against a torch reference (per-token-per-head-per-32-group absmax/7,
fp16 scale clamped to fp16 max, q = rint(x / s) clamped to [-7, 7], nibble packing low = even channel / high = odd channel
with offset +8, dequant (nibble - 8) * s in fp32 -> bf16):
  0. pack/unpack round trip: every q in [-7, 7] survives pack -> unpack bit-exact (torch reference);
  1. _quant_store_kv_int4: quantize + scatter into random slots (int32 and int64 loc); payload and
     scales bit-exact vs the reference; a row that hits every value -7..7 (s = 1); untouched slots keep
     their sentinel; zero rows and a zero group get s = 1 and nibble 8;
  2. scale index arithmetic for GROUP=32 with 2 heads: flat index (slot*2 + h)*8 + g; nibble order;
  3. _compact_kv_int4: batch of 3 requests (300 / 1200 / 4000 tokens, permuted req_to_token), bit-exact
     dequant vs torch, rows beyond the packed range untouched, an invalid position inside the packed
     region is neither read nor written; plus the trtllm strided layout (cu_k = arange * stride with
     stride = 64 > topk = 40, topk not a multiple of BLOCK_TOPK, -1 padded indices, a position >= seq_len):
     valid rows bit-exact, every padding / invalid row of the strided scratch untouched;
  4. _gather_dequant_rows_int4: 3 requests with different lengths (partial last row block, non-identity
     req_indices) packed with a 32-row gap after EACH request, bit-exact, every gap row and the tail
     untouched;
  5. end-to-end relative RMS error of gather-dequant(quant(x)) for x ~ N(0, 3): ~9-10 % (g32);
  5b. fp16-max scale clamp: groups with absmax > 7 * 65504 (1e6, bf16 max) store s = 65504 (never inf),
     bit-exact vs the reference, and both gather kernels dequantize them to finite +/- 7 * 65504;
  6. MHATokenToKVPoolInt4 itself: eager path and the lazy-VMM path (SGLANG_KV_LAZY=1), set_kv_buffer
     through the pool, payload/scale descs (row_bytes 256 / 32), bytes-per-token 1152 for 2 layers,
     first-page scale rows zeroed, kv_bits = 4 (and 8 on the int8 pool).
"""
import os
import torch
import torch.nn.functional as F
from sglang.srt.layers.attention.qsa.sparse_attn import (
    KV_INT4_GROUP as G,
    quant_store_kv_int4,
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_prefix_gather_dequant_int4,
    qwen_sparse_fa2_cu_seqlens_triton,
)

torch.manual_seed(0)
dev = "cuda"
slots, heads, dim, batch, topk = 4096, 2, 256, 3, 64
NG, DH = dim // G, dim // 2
ones = torch.ones(heads, dim, dtype=torch.float16, device=dev)


def ref_pack(q):
    """q int [.., D] in [-7, 7] -> uint8 [.., D // 2]: low nibble = even channel, high = odd, offset +8."""
    return ((q[..., 0::2] + 8) | ((q[..., 1::2] + 8) << 4)).to(torch.uint8)


def ref_unpack(p):
    """uint8 [.., D // 2] -> int32 [.., D]."""
    b = p.to(torch.int32)
    return torch.stack([(b & 15) - 8, (b >> 4) - 8], dim=-1).reshape(*p.shape[:-1], -1)


def ref_quant(x):
    """x [N, H, D] -> (packed uint8 [N, H, D // 2], s fp16 [N, H, NG], q int32 [N, H, D])."""
    xf = x.float().reshape(*x.shape[:-1], NG, G)
    a = xf.abs().amax(-1)
    s = torch.where(a > 0, a / 7.0, torch.ones_like(a)).clamp(max=65504.0).half()   # never inf (kernel clamps too)
    q = torch.clamp(torch.round(xf / s.float()[..., None]), -7, 7).to(torch.int32).reshape(x.shape)
    return ref_pack(q), s, q


def ref_dequant(p, s):
    return (ref_unpack(p).float() * s.float().repeat_interleave(G, dim=-1)).to(torch.bfloat16)


def relerr(a, b):
    return float(((a.float() - b.float()).pow(2).mean() / b.float().pow(2).mean()).sqrt())


# ---------------------------------------------------------------- 0. pack/unpack round trip over the full range
qa = torch.arange(-7, 8, device=dev, dtype=torch.int32)                       # 15 values
qq = torch.cartesian_prod(qa, qa).reshape(-1)                                  # every (even, odd) pair
qq = torch.cat([qq, torch.zeros(dim - qq.numel() % dim, device=dev, dtype=torch.int32)]).reshape(-1, dim)
pp = ref_pack(qq)
assert pp.dtype == torch.uint8 and int(pp.min()) >= 0x11 and int(pp.max()) <= 0xFF
assert torch.equal(ref_unpack(pp), qq), "pack -> unpack is not the identity on [-7, 7]"
assert int(ref_pack(torch.tensor([[-7, 7]], device=dev, dtype=torch.int32))[0, 0]) == 0x01 | (0x0F << 4)
assert int(ref_pack(torch.tensor([[0, 0]], device=dev, dtype=torch.int32))[0, 0]) == 0x88
print(f"  pack/unpack round trip: {qq.numel()} values covering every (-7..7, -7..7) pair, low nibble = even: ok")

# ---------------------------------------------------------------- 1. quantize + scatter into random slots
N = 1000
xk = torch.randn(N, heads, dim, device=dev, dtype=torch.bfloat16) * 3
xv = torch.randn(N, heads, dim, device=dev, dtype=torch.bfloat16) * 3
xk[5] = 0                                   # zero row -> s = 1, nibble 8
xv[7, 1, 64:96] = 0                         # one zero group (g = 2) in one head
full = torch.tensor([-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 7] * 16, device=dev, dtype=torch.bfloat16)
xk[3, 0] = full                             # absmax 7 -> s = 1 -> q hits every value in [-7, 7]
xv[3, 1] = -full
pk_ref, sk_ref, qk_ref = ref_quant(xk)
pv_ref, sv_ref, qv_ref = ref_quant(xv)
assert torch.equal(qk_ref[3, 0].unique(), qa) and torch.equal(qv_ref[3, 1].unique(), qa), "test row must cover -7..7"
for loc_dtype in (torch.int32, torch.int64):
    loc = torch.randperm(slots, device=dev)[:N].to(loc_dtype)
    k_buf = torch.full((slots, heads, DH), 0x77, dtype=torch.uint8, device=dev)
    v_buf = torch.full((slots, heads, DH), 0x99, dtype=torch.uint8, device=dev)
    ks = torch.full((slots, heads, NG), 3.0, dtype=torch.float16, device=dev)
    vs = torch.full((slots, heads, NG), 5.0, dtype=torch.float16, device=dev)
    quant_store_kv_int4(xk, xv, loc, k_buf, v_buf, ks, vs, ones, ones)
    torch.cuda.synchronize()
    ll = loc.long()
    assert torch.equal(k_buf[ll], pk_ref) and torch.equal(v_buf[ll], pv_ref), f"payload mismatch ({loc_dtype})"
    assert torch.equal(ks[ll], sk_ref) and torch.equal(vs[ll], sv_ref), f"scale mismatch ({loc_dtype})"
    untouched = torch.ones(slots, dtype=torch.bool, device=dev); untouched[ll] = False
    assert (k_buf[untouched] == 0x77).all() and (v_buf[untouched] == 0x99).all(), "payload of other slots changed"
    assert (ks[untouched] == 3.0).all() and (vs[untouched] == 5.0).all(), "scales of other slots changed"
    assert (ks[ll[5]] == 1.0).all() and (k_buf[ll[5]] == 0x88).all(), "zero row must give s = 1, nibbles 8"
    assert float(vs[ll[7], 1, 2]) == 1.0 and (v_buf[ll[7], 1, 32:48] == 0x88).all(), "zero group must give s = 1, nibbles 8"
    assert (ks[ll[3], 0] == 1.0).all() and torch.equal(ref_unpack(k_buf[ll[3], 0]).unique(), qa), "full-range row"
    assert (vs[ll[3], 1] == 1.0).all() and torch.equal(ref_unpack(v_buf[ll[3], 1]).unique(), qa), "full-range row"
    assert int(ref_unpack(k_buf[ll]).abs().max()) == 7
    print(f"  quant+scatter ({loc_dtype}): bit-exact packed payload + scales for {N} tokens x {heads} heads, "
          f"-7..7 row, zero row/group, others untouched: ok")

# ---------------------------------------------------------------- 2. scale index arithmetic (GROUP=32, 2 heads) + nibble order
flat = ks.view(-1)
for _ in range(64):
    t = int(torch.randint(N, (1,))); h = int(torch.randint(heads, (1,))); g = int(torch.randint(NG, (1,)))
    slot = int(loc[t])
    assert float(flat[(slot * heads + h) * NG + g]) == float(sk_ref[t, h, g]), "scale index (slot*H+h)*NG+g mismatch"
    c = int(torch.randint(DH, (1,)))
    b = int(k_buf[slot, h, c])
    assert (b & 15) - 8 == int(qk_ref[t, h, 2 * c]) and (b >> 4) - 8 == int(qk_ref[t, h, 2 * c + 1]), "nibble order"
assert not torch.equal(sk_ref[:, 0], sk_ref[:, 1]), "heads must carry distinct scales for the test to be meaningful"
print("  scale index (slot*2 + h)*8 + g and nibble order (low = even channel): ok (64 random probes)")

# ---------------------------------------------------------------- full pool for the gather tests
k16 = torch.randn(slots, heads, dim, device=dev, dtype=torch.bfloat16) * 3
v16 = torch.randn(slots, heads, dim, device=dev, dtype=torch.bfloat16) * 3
k4 = torch.empty(slots, heads, DH, dtype=torch.uint8, device=dev); v4 = torch.empty_like(k4)
ks4 = torch.empty(slots, heads, NG, dtype=torch.float16, device=dev); vs4 = torch.empty_like(ks4)
quant_store_kv_int4(k16, v16, torch.arange(slots, device=dev, dtype=torch.int32), k4, v4, ks4, vs4, ones, ones)
torch.cuda.synchronize()
pk, sk, _ = ref_quant(k16); pv, sv, _ = ref_quant(v16)
assert torch.equal(k4, pk) and torch.equal(ks4, sk) and torch.equal(v4, pv) and torch.equal(vs4, sv)
dk_all = ref_dequant(pk, sk); dv_all = ref_dequant(pv, sv)          # reference dequant of every slot

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
        k4, v4, req_to_token, req_indices, idx, seq_lens, cu_k, out_k, out_v, batch, topk,
        k_scale=ks4, v_scale=vs4, sm_k=ones, sm_v=ones, kv_bits=4)
    torch.cuda.synchronize()
    return cu_k, n, out_k, out_v


cu_k, n, out_k, out_v = run_compact(indices)
assert n == batch * topk
for b in range(batch):
    rows = req_to_token[b, indices[b].long()].long()
    a, e = int(cu_k[b]), int(cu_k[b + 1])
    assert torch.equal(out_k[a:e], dk_all[rows]) and torch.equal(out_v[a:e], dv_all[rows]), f"compact gather req {b} != torch dequant"
assert torch.isnan(out_k[n:]).all() and torch.isnan(out_v[n:]).all(), "rows beyond the packed range were written"
print(f"  compact gather-unpack-dequant bit-exact vs torch ((nibble-8)*s -> bf16), {n} rows, {batch} requests: ok")

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

# trtllm decode layout (_forward_trtllm_sparse): cu_k = arange(batch + 1) * stride with stride = ceil(topk / page) * page
# > topk, so valid_count (= stride) never masks anything and the `cols < topk` load mask (other = -1) is what keeps the
# page-padding columns [topk, stride) untouched; topk = 40 is not a multiple of BLOCK_TOPK (16); rows carry -1 padding
# (fewer selections than topk) and one position >= seq_len inside the selected region.
page_s, topk_s = 64, 40
stride_s = -(-topk_s // page_s) * page_s
assert stride_s > topk_s and topk_s % 16 != 0
cu_s = torch.arange(batch + 1, dtype=torch.int32, device=dev) * stride_s
nsel = [topk_s, 25, 33]                                                    # valid selections per request; rest -1
idx_s = torch.full((batch, topk_s), -1, dtype=torch.int32, device=dev)
for b in range(batch):
    idx_s[b, : nsel[b]] = torch.randperm(int(seq_lens[b]), device=dev)[: nsel[b]].sort().values.to(torch.int32)
idx_s[2, 5] = int(seq_lens[2]) + 3                                         # >= seq_len inside request 2's selection
ok_s = torch.full((batch * stride_s, heads, dim), float("nan"), device=dev, dtype=torch.bfloat16)
ov_s = torch.full_like(ok_s, float("nan"))
qwen_sparse_kv_extraction_compact_triton(
    k4, v4, req_to_token, req_indices, idx_s, seq_lens, cu_s, ok_s, ov_s, batch, topk_s,
    k_scale=ks4, v_scale=vs4, sm_k=ones, sm_v=ones, kv_bits=4)
torch.cuda.synchronize()
written_s = torch.zeros(batch * stride_s, dtype=torch.bool, device=dev)
for b in range(batch):
    for c in range(topk_s):
        pos = int(idx_s[b, c])
        if 0 <= pos < int(seq_lens[b]):
            r = b * stride_s + c
            slot = int(req_to_token[b, pos])
            assert torch.equal(ok_s[r], dk_all[slot]) and torch.equal(ov_s[r], dv_all[slot]), f"strided row ({b}, {c})"
            written_s[r] = True
assert int(written_s.sum()) == sum(nsel) - 1
assert torch.isnan(ok_s[~written_s]).all() and torch.isnan(ov_s[~written_s]).all(), \
    "strided scratch: a padding column [topk, stride), a -1 index or an out-of-range position was written"
assert not written_s.view(batch, stride_s)[:, topk_s:].any()
print(f"  compact gather, trtllm strided layout (stride {stride_s} > topk {topk_s}, -1 padding, pos >= seq_len): "
      f"{int(written_s.sum())} valid rows bit-exact, {int((~written_s).sum())} other rows untouched: ok")

# a bf16 scratch of the packed width must be rejected (the backend derives the logical head_dim from the pool)
try:
    qwen_sparse_kv_extraction_compact_triton(
        k4, v4, req_to_token, req_indices, indices, seq_lens, cu_k,
        torch.empty(n, heads, DH, device=dev, dtype=torch.bfloat16), torch.empty(n, heads, DH, device=dev, dtype=torch.bfloat16),
        batch, topk, k_scale=ks4, v_scale=vs4, sm_k=ones, sm_v=ones, kv_bits=4)
    raise RuntimeError("expected an AssertionError for a packed-width scratch")
except AssertionError:
    pass

# ---------------------------------------------------------------- 4. prefix row gather (chunk-prefill path)
lens = torch.tensor([300, 1203, 4001], dtype=torch.int32, device=dev)
req_idx = torch.tensor([2, 0, 1], dtype=torch.int64, device=dev)          # non-identity request mapping
GAP = 32                                                                  # > BLOCK_T (16): a stray row block lands in the gap
cu = F.pad((lens + GAP).cumsum(0), (1, 0)).to(torch.int32).contiguous()   # request b packed at cu[b], then GAP unused rows
total = int(cu[-1])
pkk = torch.full((total + 100, heads, dim), float("nan"), device=dev, dtype=torch.bfloat16)
pvv = torch.full_like(pkk, float("nan"))
qwen_sparse_prefix_gather_dequant_int4(k4, v4, ks4, vs4, ones, ones, req_to_token, req_idx, lens, cu, pkk, pvv,
                                       batch, int(lens.max()))
torch.cuda.synchronize()
written = torch.zeros(pkk.shape[0], dtype=torch.bool, device=dev)
for b in range(batch):
    rows = req_to_token[int(req_idx[b]), : int(lens[b])].long()
    a, e = int(cu[b]), int(cu[b]) + int(lens[b])
    assert torch.equal(pkk[a:e], dk_all[rows]) and torch.equal(pvv[a:e], dv_all[rows]), f"row gather req {b} != torch dequant"
    written[a:e] = True
    assert torch.isnan(pkk[e:e + GAP]).all() and torch.isnan(pvv[e:e + GAP]).all(), f"rows beyond seq_len of req {b} were written"
assert torch.isnan(pkk[~written]).all() and torch.isnan(pvv[~written]).all(), "rows outside every request were written"
assert int(written.sum()) == int(lens.sum())
print(f"  prefix row gather-unpack-dequant bit-exact, lens {lens.tolist()} (req_idx {req_idx.tolist()}), "
      f"{GAP}-row gaps after each request + tail untouched: ok")

# ---------------------------------------------------------------- 5. end-to-end error
gathered_k = torch.cat([k16[req_to_token[int(req_idx[b]), : int(lens[b])].long()] for b in range(batch)])
gathered_v = torch.cat([v16[req_to_token[int(req_idx[b]), : int(lens[b])].long()] for b in range(batch)])
ek, ev = relerr(pkk[written], gathered_k), relerr(pvv[written], gathered_v)
print(f"  int4_g32 relative RMS error on N(0,3): K {ek*100:.3f} %  V {ev*100:.3f} %  (expected ~9-10 %; int8_g64 ~0.9 %, e4m3 ~2.7 %)")
assert 0.08 < ek < 0.115 and 0.08 < ev < 0.115, "relative RMS error outside 8-11.5 %"

# ---------------------------------------------------------------- 5b. fp16-max scale clamp (absmax > 7 * 65504 -> s = 65504, never inf)
big = torch.zeros(1, heads, dim, device=dev, dtype=torch.bfloat16)
big[0, 0, :32] = torch.tensor([1e6, -1e6, 5e5, -5e5, 4.6e5, -4.6e5, 1e5, -1e5] * 4, device=dev).to(torch.bfloat16)
big[0, 1, 64:96] = torch.finfo(torch.bfloat16).max                         # 3.39e38: every channel of group 2 at bf16 max
bigv = -big
pos7 = 17
slot7 = int(req_to_token[0, pos7])
quant_store_kv_int4(big, bigv, torch.tensor([slot7], dtype=torch.int32, device=dev), k4, v4, ks4, vs4, ones, ones)
torch.cuda.synchronize()
pk_b, sk_b, q_b = ref_quant(big); pv_b, sv_b, _ = ref_quant(bigv)
assert torch.isfinite(sk_b).all() and float(sk_b[0, 0, 0]) == 65504.0 and float(sk_b[0, 1, 2]) == 65504.0
assert torch.isfinite(ks4[slot7]).all() and torch.isfinite(vs4[slot7]).all(), "kernel stored an inf scale"
assert torch.equal(k4[slot7], pk_b[0]) and torch.equal(ks4[slot7], sk_b[0]), "clamped-scale payload/scale != reference"
assert torch.equal(v4[slot7], pv_b[0]) and torch.equal(vs4[slot7], sv_b[0])
sat = torch.tensor(7 * 65504.0, device=dev).to(torch.bfloat16)
d_b, dv_b = ref_dequant(pk_b, sk_b)[0], ref_dequant(pv_b, sv_b)[0]
assert torch.isfinite(d_b).all() and float(d_b[0, 0]) == float(sat) and float(d_b[0, 1]) == -float(sat)
assert (d_b[1, 64:96] == sat).all() and (dv_b[1, 64:96] == -sat).all()
assert int(q_b[0, 0, :32].abs().max()) == 7 and int(q_b[0, 0, 6].abs()) < 7                     # 1e5 / 65504 -> 2
idx7 = indices.clone(); idx7[0, 0] = pos7
cu7, n7, ok7, ov7 = run_compact(idx7)
r7 = int(cu7[0])
assert torch.isfinite(ok7[:n7]).all() and torch.isfinite(ov7[:n7]).all(), "compact gather produced inf/NaN"
assert torch.equal(ok7[r7], d_b) and torch.equal(ov7[r7], dv_b), "compact gather of the clamped-scale row != reference"
pk7 = torch.full((pos7 + 8, heads, dim), float("nan"), device=dev, dtype=torch.bfloat16); pv7 = torch.full_like(pk7, float("nan"))
qwen_sparse_prefix_gather_dequant_int4(k4, v4, ks4, vs4, ones, ones, req_to_token,
                                       torch.tensor([0], dtype=torch.int64, device=dev),
                                       torch.tensor([pos7 + 8], dtype=torch.int32, device=dev),
                                       torch.tensor([0, pos7 + 8], dtype=torch.int32, device=dev), pk7, pv7, 1, pos7 + 8)
torch.cuda.synchronize()
assert torch.isfinite(pk7).all() and torch.isfinite(pv7).all(), "prefix gather produced inf/NaN"
assert torch.equal(pk7[pos7], d_b) and torch.equal(pv7[pos7], dv_b), "prefix gather of the clamped-scale row != reference"
print(f"  fp16-max scale clamp: absmax 1e6 / bf16-max groups -> s = 65504 (no inf), quant bit-exact, "
      f"compact + prefix gather finite (+/- {float(sat):.0f} saturation): ok")

# ---------------------------------------------------------------- 6. the pool class (eager + lazy VMM)
from sglang.srt.mem_cache.int4_kv_pool import MHATokenToKVPoolInt4
from sglang.srt.mem_cache.int8_kv_pool import MHATokenToKVPoolInt8

assert MHATokenToKVPoolInt4.kv_bits == 4 and MHATokenToKVPoolInt8.kv_bits == 8, "kv_bits dispatch keys"
for lazy in (False, True):
    if lazy:
        os.environ["SGLANG_KV_LAZY"] = "1"; os.environ["SGLANG_KV_LAZY_FLOOR"] = "512"
    else:
        os.environ.pop("SGLANG_KV_LAZY", None)
    size, page = 1024, 64
    pool = MHATokenToKVPoolInt4(size=size, page_size=page, dtype=torch.uint8, head_num=heads, head_dim=dim,
                                layer_num=2, device=dev, enable_memory_saver=False, start_layer=0)
    rows = size + page
    assert pool.kv_bits == 4 and pool.dtype == pool.store_dtype == torch.uint8 and pool.head_dim == dim
    assert len(pool.k_buffer) == len(pool.v_buffer) == len(pool.k_scale_buffer) == len(pool.v_scale_buffer) == 2
    assert pool.k_buffer[0].shape == (rows, heads, DH) and pool.k_buffer[0].dtype == torch.uint8
    assert pool.v_buffer[1].shape == (rows, heads, DH) and pool.v_buffer[1].dtype == torch.uint8
    assert pool.k_scale_buffer[0].shape == (rows, heads, NG) and pool.k_scale_buffer[0].dtype == torch.float16
    assert pool.get_key_buffer(1).dtype == torch.uint8 and pool.get_key_buffer(1).data_ptr() == pool.k_buffer[1].data_ptr()
    descs = pool._kv_buffer_descs
    assert [d.name for d in descs] == ["k0", "k1", "v0", "v1", "ks0", "ks1", "vs0", "vs1"]
    assert all(d.row_bytes == heads * DH and d.shape == (rows, heads, DH) and d.tokens_per_row == 1 for d in descs[:4])
    assert all(d.row_bytes == 32 and d.shape == (rows, 32) for d in descs[4:])
    kb, vb = pool.get_kv_size_bytes()
    assert kb == vb == 2 * rows * (heads * DH + heads * NG * 2)
    if lazy:
        o = pool._post_capture_owner
        assert o is not None and len(o.tensors) == 8 and o.bytes_per_token() == 2 * (256 + 256 + 32 + 32)
        assert pool.k_scale_buffer[0].data_ptr() == o.tensors[4].data_ptr()
        assert (pool.k_scale_buffer[1][:page] == 0).all() and (pool.v_scale_buffer[0][:page] == 0).all()
    else:
        assert (pool.k_scale_buffer[0] == 0).all() and (pool.k_buffer[0] == 0).all()
    n = 200
    loc = torch.randperm(512, device=dev)[:n].to(torch.int64)          # inside the backed floor
    xk = torch.randn(n, heads, dim, device=dev, dtype=torch.bfloat16) * 3
    xv = torch.randn(n, heads, dim, device=dev, dtype=torch.bfloat16) * 3
    pool.set_kv_buffer(None, loc, xk, xv, 1.0, 1.0, layer_id_override=1)   # HybridLinearKVPool style
    pool.set_kv_buffer(None, loc, xk.reshape(n, -1), xv.reshape(n, -1), layer_id_override=0)   # [N, H*D] form
    torch.cuda.synchronize()
    pk_r, sk_r, _ = ref_quant(xk); pv_r, sv_r, _ = ref_quant(xv)
    for l in (0, 1):
        ksb, vsb = pool.get_kv_scale_buffer(l)
        assert torch.equal(pool.k_buffer[l][loc], pk_r) and torch.equal(ksb[loc], sk_r)
        assert torch.equal(pool.v_buffer[l][loc], pv_r) and torch.equal(vsb[loc], sv_r)
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
    print(f"  MHATokenToKVPoolInt4 ({'lazy VMM' if lazy else 'eager'}): descs (256 B payload / 32 B scale rows), sizes, "
          f"set_kv_buffer bit-exact, guards: ok")
os.environ.pop("SGLANG_KV_LAZY", None)
print("  OK")
