"""Unit test for the tiered (int8 ring over int4) QSA KV path (run after patches/kv_int4.py + kv_tiers.py apply; ~40 MB VRAM).

Small ring R = 64 over a 4096-slot pool so that ring rows are overwritten by later slots (s + R).  Torch references:
int4-g32 (test_kv_int4.py's ref_quant / ref_dequant) for cold rows, int8-g64 dequant (q8 * s8, fp32 -> bf16) of the
ring row for hot rows; ring rows are compared bit-exact with the int8 kernel (quant_store_kv_int8) itself.
  1. _quant_store_kv_tiered: N = 1000 tokens into random ascending slots (incl. slot 0) written in alias-free
     chunks (span < R, as chunked prefill with chunk <= R); int32 and int64 loc; int4 rows/scales of EVERY slot
     bit-exact vs ref (never overwritten); owner[r] == the last slot of class r; ring rows of every hot slot
     bit-exact vs quant_store_kv_int8; cold slots' ring rows hold the (int8 of) the owning slot;
  2. hot/cold boundary: fresh buffers; write s then s + R -> owner flips to s + R (s cold, s + R hot, int4 row of s
     intact); write s + R then s -> owner == s (s hot, s + R cold); N > R in one launch raises;
  2b. same-launch ring-row collisions (slots congruent mod R in ONE write, as with several requests in a batch):
     exactly one owner per contested ring row, its ring row bit-exact, the losers cold, every int4 row exact,
     uncontested rows hot; repeated 20x with random collision patterns (the winner is arbitrary but definite);
  3. _compact_kv_tiered: 3 requests (300 / 1200 / 4000 tokens, permuted req_to_token), random owner pattern
     (per ring row: -1 or a random member of its slot class, with the ring row re-quantized from that slot),
     poisoned ring rows for the unowned classes -> rows bit-exact vs the per-tier torch reference; rows beyond
     the packed range untouched; an invalid position inside the packed region untouched; trtllm strided layout
     (cu_k = arange * 64, topk 40, -1 padding, pos >= seq_len); stale owner (owner = slot + R) -> int4 path;
  4. _gather_dequant_rows_tiered: 3 lengths, 32-row gaps after each request, mixed tiers, bit-exact, gaps untouched;
  5. fp16-max scale clamp on the ring: absmax 1e6 / bf16-max groups -> s8 = 65504 (the int8 kernel clamps too),
     ring row bit-exact vs the clamped torch reference, int4 half bit-exact, both gathers finite + bit-exact;
  6. MHATokenToKVPoolTiered (SGLANG_KV_TIERS_W=64): eager and lazy-VMM paths; ring shapes/dtypes, owner -1,
     kv_tiered / kv_bits / ring_mask, descs identical to the int4 pool (ring not on the owner, bytes_per_token
     1152 for 2 layers), get_kv_size_bytes includes the ring, set_kv_buffer bit-exact (both tiers + owner),
     N > R raises, lazy_release resets the owner, guards; bad SGLANG_KV_TIERS_W rejected;
  7. microbench (informational): tiered compact gather vs _compact_kv_int4 (all cold: owner = -1) and vs
     _compact_kv_int8 (all hot: a 4096-row ring with owner = arange, every selected slot verified hot and the
     output bit-exact vs the int8 kernel); plan target <= 1.10x each.
"""
import os
import time
import torch
import torch.nn.functional as F
from sglang.srt.layers.attention.qsa.sparse_attn import (
    KV_INT4_GROUP as G4,
    KV_INT8_GROUP as G8,
    quant_store_kv_int4,
    quant_store_kv_int8,
    quant_store_kv_tiered,
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_prefix_gather_dequant_tiered,
    qwen_sparse_fa2_cu_seqlens_triton,
)

torch.manual_seed(0)
dev = "cuda"
slots, heads, dim, batch, topk = 4096, 2, 256, 3, 64
R = 64
MASK = R - 1
NG4, DH, NG8 = dim // G4, dim // 2, dim // G8
ones = torch.ones(heads, dim, dtype=torch.float16, device=dev)


# ---------------------------------------------------------------- torch references
def ref_pack(q):
    return ((q[..., 0::2] + 8) | ((q[..., 1::2] + 8) << 4)).to(torch.uint8)


def ref_unpack(p):
    b = p.to(torch.int32)
    return torch.stack([(b & 15) - 8, (b >> 4) - 8], dim=-1).reshape(*p.shape[:-1], -1)


def ref_quant4(x):
    xf = x.float().reshape(*x.shape[:-1], NG4, G4)
    a = xf.abs().amax(-1)
    s = torch.where(a > 0, a / 7.0, torch.ones_like(a)).clamp(max=65504.0).half()
    q = torch.clamp(torch.round(xf / s.float()[..., None]), -7, 7).to(torch.int32).reshape(x.shape)
    return ref_pack(q), s


def ref_dequant4(p, s):
    return (ref_unpack(p).float() * s.float().repeat_interleave(G4, dim=-1)).to(torch.bfloat16)


def ref_quant8(x):
    """int8-g64 with the fp16-max clamp (what the tiered writer stores in the ring)."""
    xf = x.float().reshape(*x.shape[:-1], NG8, G8)
    a = xf.abs().amax(-1)
    s = torch.where(a > 0, a / 127.0, torch.ones_like(a)).clamp(max=65504.0).half()
    q = torch.clamp(torch.round(xf / s.float()[..., None]), -127, 127).to(torch.int8).reshape(x.shape)
    return q, s


def ref_dequant8(q, s):
    return (q.float() * s.float().repeat_interleave(G8, dim=-1)).to(torch.bfloat16)


def chunks_alias_free(sl):
    """Split an ascending slot list into consecutive chunks whose span is < R (no two slots share a ring row)."""
    out, start = [], 0
    for i in range(1, len(sl) + 1):
        if i == len(sl) or int(sl[i]) - int(sl[start]) >= R:
            out.append((start, i)); start = i
    return out


def new_bufs():
    k4 = torch.full((slots, heads, DH), 0x77, dtype=torch.uint8, device=dev)
    v4 = torch.full((slots, heads, DH), 0x99, dtype=torch.uint8, device=dev)
    ks4 = torch.full((slots, heads, NG4), 3.0, dtype=torch.float16, device=dev)
    vs4 = torch.full((slots, heads, NG4), 5.0, dtype=torch.float16, device=dev)
    rk = torch.full((R, heads, dim), 0x33, dtype=torch.int8, device=dev)
    rv = torch.full((R, heads, dim), 0x44, dtype=torch.int8, device=dev)
    rks = torch.full((R, heads, NG8), 7.0, dtype=torch.float16, device=dev)
    rvs = torch.full((R, heads, NG8), 9.0, dtype=torch.float16, device=dev)
    owner = torch.full((R,), -1, dtype=torch.int32, device=dev)
    return k4, v4, ks4, vs4, rk, rv, rks, rvs, owner


def tiered_write(xk, xv, loc, bufs):
    k4, v4, ks4, vs4, rk, rv, rks, rvs, owner = bufs
    quant_store_kv_tiered(xk, xv, loc, k4, v4, ks4, vs4, rk, rv, rks, rvs, owner, ones, ones, MASK)


# ---------------------------------------------------------------- 1. dual write into ascending slots with ring wrap
N = 1000
xk = torch.randn(N, heads, dim, device=dev, dtype=torch.bfloat16) * 3
xv = torch.randn(N, heads, dim, device=dev, dtype=torch.bfloat16) * 3
xk[5] = 0                                   # zero row -> s = 1 in both tiers
sl = torch.randperm(slots, device=dev)[:N].sort().values
sl[0] = 0                                   # slot 0 (the dummy-write slot) is part of the set
assert int(sl.unique().numel()) == N
pk_ref, sk_ref = ref_quant4(xk); pv_ref, sv_ref = ref_quant4(xv)
# int8 reference straight from the int8 kernel (identical arithmetic): full-pool sized buffers at loc = slots
k8_ref = torch.empty(slots, heads, dim, dtype=torch.int8, device=dev); v8_ref = torch.empty_like(k8_ref)
ks8_ref = torch.empty(slots, heads, NG8, dtype=torch.float16, device=dev); vs8_ref = torch.empty_like(ks8_ref)
quant_store_kv_int8(xk, xv, sl.to(torch.int32), k8_ref, v8_ref, ks8_ref, vs8_ref, ones, ones)
torch.cuda.synchronize()
q8k, s8k = ref_quant8(xk); q8v, s8v = ref_quant8(xv)
assert torch.equal(k8_ref[sl], q8k) and torch.equal(ks8_ref[sl], s8k), "int8 kernel != torch int8 reference (test premise)"
ch = chunks_alias_free(sl)
assert len(ch) > 20 and max(e - a for a, e in ch) <= R
exp_owner = torch.full((R,), -1, dtype=torch.int64, device=dev)
for s in sl.tolist():
    exp_owner[s & MASK] = s                                             # ascending -> the last writer of each class
n_wrapped = int(((sl & MASK).bincount(minlength=R) > 1).sum())
assert n_wrapped > 40, "the slot set must make most ring rows wrap"
for loc_dtype in (torch.int32, torch.int64):
    bufs = new_bufs(); k4, v4, ks4, vs4, rk, rv, rks, rvs, owner = bufs
    for a, e in ch:
        tiered_write(xk[a:e], xv[a:e], sl[a:e].to(loc_dtype), bufs)
    torch.cuda.synchronize()
    ll = sl.long()
    assert torch.equal(k4[ll], pk_ref) and torch.equal(ks4[ll], sk_ref), f"int4 K rows/scales ({loc_dtype})"
    assert torch.equal(v4[ll], pv_ref) and torch.equal(vs4[ll], sv_ref), f"int4 V rows/scales ({loc_dtype})"
    untouched = torch.ones(slots, dtype=torch.bool, device=dev); untouched[ll] = False
    assert (k4[untouched] == 0x77).all() and (ks4[untouched] == 3.0).all(), "int4 rows of other slots changed"
    assert torch.equal(owner.long(), exp_owner), f"owner table ({loc_dtype})"
    assert int(owner[0]) == int(exp_owner[0]) and int(exp_owner[0]) >= 0
    hot = ll[owner.long()[ll & MASK] == ll]                            # slots whose ring row they still own
    cold = ll[owner.long()[ll & MASK] != ll]
    assert hot.numel() == int((exp_owner >= 0).sum()) and cold.numel() == N - hot.numel() and cold.numel() > 900
    assert torch.equal(rk[hot & MASK], k8_ref[hot]) and torch.equal(rks[hot & MASK], ks8_ref[hot]), "ring K rows of hot slots"
    assert torch.equal(rv[hot & MASK], v8_ref[hot]) and torch.equal(rvs[hot & MASK], vs8_ref[hot]), "ring V rows of hot slots"
    own_of_cold = owner.long()[cold & MASK]
    assert torch.equal(rk[cold & MASK], k8_ref[own_of_cold]), "ring row of a cold slot must hold its owner's int8 data"
    assert (ks4[ll[5]] == 1.0).all() and (k4[ll[5]] == 0x88).all()             # zero row: s = 1, nibbles 8
    if int(owner[ll[5] & MASK]) == int(ll[5]):
        assert (rks[ll[5] & MASK] == 1.0).all() and (rk[ll[5] & MASK] == 0).all()   # ... and s8 = 1, q8 = 0 while hot
    print(f"  tiered write ({loc_dtype}): {N} tokens in {len(ch)} alias-free chunks, {n_wrapped}/{R} ring rows wrapped; "
          f"int4 rows of all {N} slots bit-exact, owner == last writer, {hot.numel()} hot ring rows bit-exact vs "
          f"quant_store_kv_int8, {cold.numel()} cold slots' ring rows hold their owner's data: ok")

# ---------------------------------------------------------------- 2. hot/cold boundary exactly at s and s +/- R
s0 = 100
bufs = new_bufs(); k4, v4, ks4, vs4, rk, rv, rks, rvs, owner = bufs
one = lambda i: torch.tensor([i], dtype=torch.int32, device=dev)
tiered_write(xk[:1], xv[:1], one(s0), bufs)
torch.cuda.synchronize()
assert int(owner[s0 & MASK]) == s0 and torch.equal(rk[s0 & MASK], q8k[0]) and torch.equal(k4[s0], pk_ref[0])
tiered_write(xk[1:2], xv[1:2], one(s0 + R), bufs)                        # s0 + R evicts s0 from the ring
torch.cuda.synchronize()
assert int(owner[s0 & MASK]) == s0 + R, "owner must flip to s + R"
assert torch.equal(rk[s0 & MASK], q8k[1]) and torch.equal(rks[s0 & MASK], s8k[1]), "ring row now holds s + R"
assert torch.equal(k4[s0], pk_ref[0]) and torch.equal(ks4[s0], sk_ref[0]), "int4 row of s intact after eviction"
assert torch.equal(k4[s0 + R], pk_ref[1]) and torch.equal(v4[s0 + R], pv_ref[1])
assert (owner == -1).sum() == R - 1
bufs = new_bufs(); k4, v4, ks4, vs4, rk, rv, rks, rvs, owner = bufs
tiered_write(xk[1:2], xv[1:2], one(s0 + R), bufs)
tiered_write(xk[:1], xv[:1], one(s0), bufs)                              # s written after s + R: s owns the row
torch.cuda.synchronize()
assert int(owner[s0 & MASK]) == s0 and torch.equal(rk[s0 & MASK], q8k[0]) and torch.equal(k4[s0 + R], pk_ref[1])
try:
    tiered_write(xk[: R + 1], xv[: R + 1], torch.arange(R + 1, device=dev, dtype=torch.int32), bufs)
    raise RuntimeError("expected an AssertionError for N > R")
except AssertionError:
    pass
print(f"  hot/cold boundary: write {s0} then {s0 + R} -> owner {s0 + R} (slot {s0} cold, int4 row intact); "
      f"write {s0 + R} then {s0} -> owner {s0}; N = R + 1 in one launch rejected: ok")

# ---------------------------------------------------------------- 2b. ring-row collisions inside ONE launch
# slots congruent mod R in the same write (several requests in one batch): the stamp launch picks one owner per
# ring row, only the owner's programs write the ring row, the losers are cold; no row may mix two tokens' bytes.
n_contested_total = 0
for trial in range(20):
    bufs = new_bufs(); k4, v4, ks4, vs4, rk, rv, rks, rvs, owner = bufs
    n_c = R                                                           # N = R tokens, heavy aliasing
    classes = torch.randint(0, 8, (n_c,), device=dev)                 # ring rows 0..7 only -> ~8 slots per row
    members = torch.randperm(slots // R, device=dev)[:n_c]            # distinct class members -> distinct slots
    loc_c = (members * R + classes).to(torch.int32)
    assert int(loc_c.unique().numel()) == n_c
    perm = torch.randperm(N, device=dev)[:n_c]
    tiered_write(xk[perm], xv[perm], loc_c, bufs)
    torch.cuda.synchronize()
    ll = loc_c.long()
    assert torch.equal(k4[ll], pk_ref[perm]) and torch.equal(ks4[ll], sk_ref[perm]), "int4 rows of all colliding tokens"
    assert torch.equal(v4[ll], pv_ref[perm]) and torch.equal(vs4[ll], sv_ref[perm])
    for r_ in range(R):
        cand = ll[(ll & MASK) == r_]
        if cand.numel() == 0:
            assert int(owner[r_]) == -1 and (rk[r_] == 0x33).all() and (rks[r_] == 7.0).all(), "untouched ring row"
            continue
        w = int(owner[r_])
        assert w in set(cand.tolist()), f"owner of contested row {r_} = {w} not among its writers"
        i_w = int(perm[int(torch.nonzero(loc_c == w).item())])           # the token (row of xk) written to slot w
        assert torch.equal(rk[r_], q8k[i_w]) and torch.equal(rks[r_], s8k[i_w]), f"ring row {r_} mixed / not the owner's"
        assert torch.equal(rv[r_], q8v[i_w]) and torch.equal(rvs[r_], s8v[i_w])
        n_contested_total += int(cand.numel() > 1)
    hot_c = ll[owner.long()[ll & MASK] == ll]
    assert hot_c.numel() == int((owner >= 0).sum()) <= 8
assert n_contested_total > 100
# an uncontested token in the same launch as a contested row stays hot (mixed launch)
bufs = new_bufs(); k4, v4, ks4, vs4, rk, rv, rks, rvs, owner = bufs
loc_m = torch.tensor([5, 5 + R, 5 + 2 * R, 7], dtype=torch.int32, device=dev)
tiered_write(xk[:4], xv[:4], loc_m, bufs)
torch.cuda.synchronize()
w5 = int(owner[5]); assert w5 in (5, 5 + R, 5 + 2 * R) and int(owner[7]) == 7
i5 = {5: 0, 5 + R: 1, 5 + 2 * R: 2}[w5]
assert torch.equal(rk[5], q8k[i5]) and torch.equal(rks[5], s8k[i5]) and torch.equal(rv[5], q8v[i5]) and torch.equal(rvs[5], s8v[i5])
assert torch.equal(rk[7], q8k[3]) and torch.equal(rv[7], q8v[3]) and torch.equal(k4[loc_m.long()], pk_ref[:4])
assert int((owner >= 0).sum()) == 2
print(f"  same-launch ring-row collisions: 20 x {n_c} tokens on 8 ring rows ({n_contested_total} contested rows): one owner per row, "
      f"its ring row bit-exact, losers cold, int4 rows of every token exact; mixed launch [5, 69, 133, 7] -> owner[5] = {w5}, slot 7 hot: ok")

# ---------------------------------------------------------------- full pool with a random owner pattern for the gather tests
k16 = torch.randn(slots, heads, dim, device=dev, dtype=torch.bfloat16) * 3
v16 = torch.randn(slots, heads, dim, device=dev, dtype=torch.bfloat16) * 3
bufs = new_bufs(); k4, v4, ks4, vs4, rk, rv, rks, rvs, owner = bufs
for a in range(0, slots, R):                                              # ascending chunks of exactly R slots
    tiered_write(k16[a:a + R], v16[a:a + R], torch.arange(a, a + R, device=dev, dtype=torch.int32), bufs)
torch.cuda.synchronize()
pk, sk = ref_quant4(k16); pv, sv = ref_quant4(v16)
assert torch.equal(k4, pk) and torch.equal(ks4, sk) and torch.equal(v4, pv) and torch.equal(vs4, sv)
assert torch.equal(owner.long(), torch.arange(slots - R, slots, device=dev))
# random owner pattern: per ring row r either -1 (nobody) or a random member s' of the class {r, r + R, ...};
# the ring row is then re-quantized from s' (int8 kernel at loc = r into the ring), so "hot" rows carry s'.
n_cls = slots // R
pick = torch.randint(0, n_cls, (R,), device=dev)                          # class index ...
pick = torch.where(torch.rand(R, device=dev) < 0.35, torch.full_like(pick, -1), pick)   # ... or -1 (~1/3 of the rows)
own = torch.where(pick < 0, torch.full_like(pick, -1), pick * R + torch.arange(R, device=dev))
own[0] = 0                                                                # ring row 0 owned by slot 0 (dummy-write slot)
own[1] = -1
own[2] = 2 + R * (n_cls - 1)                                              # the last slot of its class
owner.copy_(own.to(torch.int32))
hot_rows = torch.nonzero(own >= 0).flatten()
quant_store_kv_int8(k16[own[hot_rows]], v16[own[hot_rows]], hot_rows.to(torch.int32), rk, rv, rks, rvs, ones, ones)
# poison every unowned ring row: garbage payload and NaN scales must never reach the output
dead = torch.nonzero(own < 0).flatten()
rk[dead] = -128; rv[dead] = 127; rks[dead] = float("nan"); rvs[dead] = float("inf")
torch.cuda.synchronize()
assert 10 < hot_rows.numel() < R - 5
k8_all = torch.empty(slots, heads, dim, dtype=torch.int8, device=dev); v8_all = torch.empty_like(k8_all)
ks8_all = torch.empty(slots, heads, NG8, dtype=torch.float16, device=dev); vs8_all = torch.empty_like(ks8_all)
quant_store_kv_int8(k16, v16, torch.arange(slots, device=dev, dtype=torch.int32), k8_all, v8_all, ks8_all, vs8_all, ones, ones)
torch.cuda.synchronize()
is_hot = owner.long()[torch.arange(slots, device=dev) & MASK] == torch.arange(slots, device=dev)      # [slots] bool
assert int(is_hot.sum()) == hot_rows.numel() and bool(is_hot[0]) and not bool(is_hot[1]) and bool(is_hot[2 + R * (n_cls - 1)])
d4k, d4v = ref_dequant4(pk, sk), ref_dequant4(pv, sv)
d8k, d8v = ref_dequant8(k8_all, ks8_all), ref_dequant8(v8_all, vs8_all)
dk_all = torch.where(is_hot[:, None, None], d8k, d4k)                     # per-tier reference for every slot
dv_all = torch.where(is_hot[:, None, None], d8v, d4v)
assert not torch.equal(d8k[is_hot], d4k[is_hot]), "the tiers must differ for the test to discriminate"


def tier_kwargs():
    return dict(k_scale=ks4, v_scale=vs4, sm_k=ones, sm_v=ones, kv_bits=4,
                ring_k=rk, ring_v=rv, ring_ks=rks, ring_vs=rvs, owner=owner, ring_mask=MASK)


# ---------------------------------------------------------------- 3. compact gather (decode / verify path)
seq_lens = torch.tensor([300, 1200, 4000], dtype=torch.int32, device=dev)
req_to_token = torch.stack([torch.randperm(slots, device=dev) for _ in range(batch)]).to(torch.int32)
req_indices = torch.arange(batch, dtype=torch.int32, device=dev)
indices = torch.stack([torch.randperm(int(l), device=dev)[:topk].sort().values for l in seq_lens]).to(torch.int32)
# the random owner pattern leaves only ~2/3 R hot slots in 4096: make sure every request selects up to 8 of them
for b in range(batch):
    hot_pos = torch.nonzero(is_hot[req_to_token[b, : int(seq_lens[b])].long()]).flatten()
    if hot_pos.numel():
        take = hot_pos[torch.randperm(hot_pos.numel(), device=dev)[:8]]
        rest = torch.tensor([p for p in indices[b].tolist() if p not in set(take.tolist())], device=dev, dtype=torch.int64)
        indices[b] = torch.cat([rest[: topk - take.numel()], take]).sort().values.to(torch.int32)
assert indices.shape == (batch, topk) and all(int(indices[b].unique().numel()) == topk for b in range(batch))


def run_compact(idx, kw=None):
    cu_k = torch.empty(batch + 1, dtype=torch.int32, device=dev)
    counts = torch.empty(batch, dtype=torch.int32, device=dev)
    qwen_sparse_fa2_cu_seqlens_triton(seq_lens, idx, counts, cu_k, batch, topk)
    n = int(cu_k[-1])
    out_k = torch.full((batch * topk + 16, heads, dim), float("nan"), device=dev, dtype=torch.bfloat16)
    out_v = torch.full_like(out_k, float("nan"))
    qwen_sparse_kv_extraction_compact_triton(
        k4, v4, req_to_token, req_indices, idx, seq_lens, cu_k, out_k, out_v, batch, topk, **(kw or tier_kwargs()))
    torch.cuda.synchronize()
    return cu_k, n, out_k, out_v


cu_k, n, out_k, out_v = run_compact(indices)
assert n == batch * topk
n_hot_sel = 0
for b in range(batch):
    rows = req_to_token[b, indices[b].long()].long()
    a, e = int(cu_k[b]), int(cu_k[b + 1])
    assert torch.equal(out_k[a:e], dk_all[rows]) and torch.equal(out_v[a:e], dv_all[rows]), f"compact gather req {b} != per-tier reference"
    n_hot_sel += int(is_hot[rows].sum())
assert 10 < n_hot_sel < n, f"compact test needs a real hot/cold mix, got {n_hot_sel} hot of {n}"
assert torch.isnan(out_k[n:]).all() and torch.isnan(out_v[n:]).all(), "rows beyond the packed range were written"
assert torch.isfinite(out_k[:n]).all(), "a poisoned (unowned) ring row leaked into the output"
# the same call with the int4 kwargs only (no owner) must take the plain int4 path: every row cold
kw4 = dict(k_scale=ks4, v_scale=vs4, sm_k=ones, sm_v=ones, kv_bits=4)
_, _, ok4, ov4 = run_compact(indices, kw4)
for b in range(batch):
    rows = req_to_token[b, indices[b].long()].long()
    a, e = int(cu_k[b]), int(cu_k[b + 1])
    assert torch.equal(ok4[a:e], d4k[rows]) and torch.equal(ov4[a:e], d4v[rows])
print(f"  compact gather bit-exact vs per-tier torch reference ({n} rows, {n_hot_sel} hot / {n - n_hot_sel} cold, "
      f"{dead.numel()} poisoned ring rows never read), no-owner call = plain int4: ok")

bad = indices.clone(); bad[1, 0] = int(seq_lens[1]) + 5
cu_b, n_b, ok_b, ov_b = run_compact(bad)
assert n_b == batch * topk - 1
a = int(cu_b[1])
assert torch.isnan(ok_b[a]).all() and torch.isnan(ov_b[a]).all(), "invalid position inside the packed region must not be written"
rows = req_to_token[1, bad[1, 1:topk - 1].long()].long()
assert torch.equal(ok_b[a + 1:a + topk - 1], dk_all[rows]) and torch.equal(ov_b[a + 1:a + topk - 1], dv_all[rows])
assert torch.equal(ok_b[:a], out_k[:a]) and torch.equal(ok_b[int(cu_b[2]):n_b], out_k[int(cu_k[2]):n])
assert torch.isnan(ok_b[n_b:]).all()
print("  compact gather: invalid position -> row untouched, neighbours intact, tail untouched: ok")

# trtllm strided layout: cu_k = arange * stride (stride 64 > topk 40), -1 padding, a position >= seq_len
page_s, topk_s = 64, 40
stride_s = -(-topk_s // page_s) * page_s
cu_s = torch.arange(batch + 1, dtype=torch.int32, device=dev) * stride_s
nsel = [topk_s, 25, 33]
idx_s = torch.full((batch, topk_s), -1, dtype=torch.int32, device=dev)
for b in range(batch):
    idx_s[b, : nsel[b]] = torch.randperm(int(seq_lens[b]), device=dev)[: nsel[b]].sort().values.to(torch.int32)
idx_s[2, 5] = int(seq_lens[2]) + 3
ok_s = torch.full((batch * stride_s, heads, dim), float("nan"), device=dev, dtype=torch.bfloat16)
ov_s = torch.full_like(ok_s, float("nan"))
qwen_sparse_kv_extraction_compact_triton(
    k4, v4, req_to_token, req_indices, idx_s, seq_lens, cu_s, ok_s, ov_s, batch, topk_s, **tier_kwargs())
torch.cuda.synchronize()
written_s = torch.zeros(batch * stride_s, dtype=torch.bool, device=dev)
for b in range(batch):
    for c in range(topk_s):
        pos = int(idx_s[b, c])
        if 0 <= pos < int(seq_lens[b]):
            r_ = b * stride_s + c
            slot = int(req_to_token[b, pos])
            assert torch.equal(ok_s[r_], dk_all[slot]) and torch.equal(ov_s[r_], dv_all[slot]), f"strided row ({b}, {c})"
            written_s[r_] = True
assert int(written_s.sum()) == sum(nsel) - 1
assert torch.isnan(ok_s[~written_s]).all() and torch.isnan(ov_s[~written_s]).all(), "strided scratch: an unused row was written"
print(f"  compact gather, trtllm strided layout (stride {stride_s} > topk {topk_s}, -1 padding, pos >= seq_len): "
      f"{int(written_s.sum())} valid rows bit-exact, {int((~written_s).sum())} other rows untouched: ok")

# stale owner: a hot slot's ring row is re-owned by slot + R (as after an eviction) -> the slot must read int4
sel_slot = int(req_to_token[0, indices[0, 3].long()])
saved = int(owner[sel_slot & MASK])
owner[sel_slot & MASK] = (sel_slot + R) % slots if sel_slot + R < slots else sel_slot - R
cu_t, n_t, ok_t, ov_t = run_compact(indices)
r0 = int(cu_t[0]) + 3
assert torch.equal(ok_t[r0], d4k[sel_slot]) and torch.equal(ov_t[r0], d4v[sel_slot]), "stale owner must fall back to int4"
owner[sel_slot & MASK] = saved
print(f"  stale owner (owner = slot +/- R) for slot {sel_slot} -> int4 path: ok")

# ---------------------------------------------------------------- 4. prefix row gather (chunk-prefill path)
lens = torch.tensor([300, 1203, 4001], dtype=torch.int32, device=dev)
req_idx = torch.tensor([2, 0, 1], dtype=torch.int64, device=dev)
GAP = 32
cu = F.pad((lens + GAP).cumsum(0), (1, 0)).to(torch.int32).contiguous()
total = int(cu[-1])
pkk = torch.full((total + 100, heads, dim), float("nan"), device=dev, dtype=torch.bfloat16)
pvv = torch.full_like(pkk, float("nan"))
qwen_sparse_prefix_gather_dequant_tiered(k4, v4, ks4, vs4, ones, ones, req_to_token, req_idx, lens, cu, pkk, pvv,
                                         batch, int(lens.max()), ring_k=rk, ring_v=rv, ring_ks=rks, ring_vs=rvs,
                                         owner=owner, ring_mask=MASK)
torch.cuda.synchronize()
written = torch.zeros(pkk.shape[0], dtype=torch.bool, device=dev)
n_hot_pre = 0
for b in range(batch):
    rows = req_to_token[int(req_idx[b]), : int(lens[b])].long()
    a, e = int(cu[b]), int(cu[b]) + int(lens[b])
    assert torch.equal(pkk[a:e], dk_all[rows]) and torch.equal(pvv[a:e], dv_all[rows]), f"row gather req {b} != per-tier reference"
    written[a:e] = True
    n_hot_pre += int(is_hot[rows].sum())
    assert torch.isnan(pkk[e:e + GAP]).all() and torch.isnan(pvv[e:e + GAP]).all(), f"rows beyond seq_len of req {b} were written"
assert torch.isnan(pkk[~written]).all() and torch.isnan(pvv[~written]).all(), "rows outside every request were written"
assert int(written.sum()) == int(lens.sum()) and 0 < n_hot_pre < int(lens.sum())
try:
    qwen_sparse_prefix_gather_dequant_tiered(k4, v4, ks4, vs4, ones, ones, req_to_token, req_idx, lens, cu, pkk, pvv,
                                             batch, int(lens.max()))
    raise RuntimeError("expected an AssertionError without ring/owner")
except AssertionError:
    pass
print(f"  prefix row gather bit-exact, lens {lens.tolist()} (req_idx {req_idx.tolist()}), {n_hot_pre} hot rows, "
      f"{GAP}-row gaps after each request + tail untouched, missing ring args rejected: ok")

# ---------------------------------------------------------------- 5. fp16-max scale clamp on both tiers
big = torch.zeros(1, heads, dim, device=dev, dtype=torch.bfloat16)
big[0, 0, :32] = torch.tensor([1e6, -1e6, 5e5, -5e5, 4.6e5, -4.6e5, 1e5, -1e5] * 4, device=dev).to(torch.bfloat16)
big[0, 1, 64:96] = torch.finfo(torch.bfloat16).max
bigv = -big
pos7 = 17
slot7 = int(req_to_token[0, pos7])
# the int8 kernel clamps the fp16 scale to 65504 like the tiered writer (the clamp was added to every
# write kernel after this test was written; the premise used to be an inf scale for a bf16-max group)
tk = torch.empty(1, heads, dim, dtype=torch.int8, device=dev); tks = torch.empty(1, heads, NG8, dtype=torch.float16, device=dev)
quant_store_kv_int8(big, bigv, one(0), tk, tk.clone(), tks, tks.clone(), ones, ones)
torch.cuda.synchronize()
assert torch.isfinite(tks).all() and float(tks[0, 1, 1]) == 65504.0, "premise: the int8 kernel clamps the bf16-max group's scale to 65504"
tiered_write(big, bigv, one(slot7), bufs)
torch.cuda.synchronize()
q8b, s8b = ref_quant8(big); q8bv, s8bv = ref_quant8(bigv)
p4b, s4b = ref_quant4(big); p4bv, s4bv = ref_quant4(bigv)
assert float(s8b[0, 1, 1]) == 65504.0 and float(s4b[0, 1, 2]) == 65504.0 and torch.isfinite(s8b).all()
assert int(owner[slot7 & MASK]) == slot7
assert torch.isfinite(rks[slot7 & MASK]).all() and torch.isfinite(rvs[slot7 & MASK]).all(), "ring scale inf"
assert torch.equal(rk[slot7 & MASK], q8b[0]) and torch.equal(rks[slot7 & MASK], s8b[0]), "clamped ring row != reference"
assert torch.equal(rv[slot7 & MASK], q8bv[0]) and torch.equal(rvs[slot7 & MASK], s8bv[0])
assert torch.equal(k4[slot7], p4b[0]) and torch.equal(ks4[slot7], s4b[0]) and torch.equal(v4[slot7], p4bv[0]) and torch.equal(vs4[slot7], s4bv[0])
d8b, d8bv = ref_dequant8(q8b, s8b)[0], ref_dequant8(q8bv, s8bv)[0]
assert torch.isfinite(d8b).all() and float(d8b[1, 64]) == float(torch.tensor(127 * 65504.0).to(torch.bfloat16))
idx7 = indices.clone(); idx7[0, 0] = pos7
cu7, n7, ok7, ov7 = run_compact(idx7)
r7 = int(cu7[0])
assert torch.isfinite(ok7[:n7]).all() and torch.isfinite(ov7[:n7]).all(), "compact gather produced inf/NaN"
assert torch.equal(ok7[r7], d8b) and torch.equal(ov7[r7], d8bv), "compact gather of the clamped hot row != int8 reference"
pk7 = torch.full((pos7 + 8, heads, dim), float("nan"), device=dev, dtype=torch.bfloat16); pv7 = torch.full_like(pk7, float("nan"))
qwen_sparse_prefix_gather_dequant_tiered(k4, v4, ks4, vs4, ones, ones, req_to_token,
                                         torch.tensor([0], dtype=torch.int64, device=dev),
                                         torch.tensor([pos7 + 8], dtype=torch.int32, device=dev),
                                         torch.tensor([0, pos7 + 8], dtype=torch.int32, device=dev), pk7, pv7, 1, pos7 + 8,
                                         ring_k=rk, ring_v=rv, ring_ks=rks, ring_vs=rvs, owner=owner, ring_mask=MASK)
torch.cuda.synchronize()
assert torch.isfinite(pk7).all() and torch.isfinite(pv7).all()
assert torch.equal(pk7[pos7], d8b) and torch.equal(pv7[pos7], d8bv), "prefix gather of the clamped hot row != int8 reference"
# evict it (write slot7 +/- R) and read again: the int4 row (clamped too) is served
owner[slot7 & MASK] = slot7 + R if slot7 + R < slots else slot7 - R
cu7c, n7c, ok7c, ov7c = run_compact(idx7)
assert torch.equal(ok7c[int(cu7c[0])], ref_dequant4(p4b, s4b)[0]) and torch.isfinite(ok7c[:n7c]).all()
print(f"  fp16-max scale clamp: bf16-max group -> ring s8 = 65504 (int8 kernel: inf), ring + int4 rows bit-exact, "
      f"compact + prefix gathers finite and bit-exact (hot, then evicted -> int4): ok")

# ---------------------------------------------------------------- 6. the pool class (eager + lazy VMM)
from sglang.srt.mem_cache.int4_kv_pool import MHATokenToKVPoolInt4
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.mem_cache.tiered_kv_pool import MHATokenToKVPoolTiered, ring_slots_from_env

assert MHATokenToKVPoolTiered.kv_bits == 4 and MHATokenToKVPoolTiered.kv_tiered is True
assert not getattr(MHATokenToKVPoolInt4, "kv_tiered", False)
assert callable(getattr(HybridLinearKVPool, "get_kv_ring_buffer", None)) and callable(getattr(HybridLinearKVPool, "get_kv_ring_owner", None))
for bad_w in ("0", "-8", "100", "abc"):
    os.environ["SGLANG_KV_TIERS_W"] = bad_w
    try:
        ring_slots_from_env(); raise RuntimeError(f"expected ValueError for SGLANG_KV_TIERS_W={bad_w}")
    except ValueError:
        pass
os.environ.pop("SGLANG_KV_TIERS_W", None)
assert ring_slots_from_env() == 8192
os.environ["SGLANG_KV_TIERS_W"] = str(R)
size, page = 1024, 64
mk = dict(size=size, page_size=page, dtype=torch.uint8, head_num=heads, head_dim=dim, layer_num=2, device=dev,
          enable_memory_saver=False, start_layer=0)
for lazy in (False, True):
    if lazy:
        os.environ["SGLANG_KV_LAZY"] = "1"; os.environ["SGLANG_KV_LAZY_FLOOR"] = "512"
    else:
        os.environ.pop("SGLANG_KV_LAZY", None)
    ref_pool = MHATokenToKVPoolInt4(**mk)
    ref_descs = [(d.name, d.shape, d.row_bytes, d.tokens_per_row) for d in ref_pool._kv_buffer_descs]
    ref_kb, ref_vb = ref_pool.get_kv_size_bytes()
    ref_pool._clear_buffers()
    pool = MHATokenToKVPoolTiered(**mk)
    rows = size + page
    assert pool.kv_tiered and pool.kv_bits == 4 and pool.ring_slots == R and pool.ring_mask == MASK
    assert pool.dtype == pool.store_dtype == torch.uint8 and pool.head_dim == dim
    assert len(pool.ring_k) == len(pool.ring_v) == len(pool.ring_ks) == len(pool.ring_vs) == 2
    assert pool.ring_k[0].shape == (R, heads, dim) and pool.ring_k[0].dtype == torch.int8
    assert pool.ring_v[1].shape == (R, heads, dim) and pool.ring_v[1].dtype == torch.int8
    assert pool.ring_ks[0].shape == (R, heads, NG8) and pool.ring_ks[0].dtype == torch.float16
    assert pool.ring_vs[1].shape == (R, heads, NG8) and pool.ring_vs[1].dtype == torch.float16
    assert pool.ring_owner.shape == (R,) and pool.ring_owner.dtype == torch.int32 and (pool.ring_owner == -1).all()
    assert pool.get_kv_ring_owner().data_ptr() == pool.ring_owner.data_ptr()
    rb = pool.get_kv_ring_buffer(1)
    assert [t.data_ptr() for t in rb] == [pool.ring_k[1].data_ptr(), pool.ring_v[1].data_ptr(), pool.ring_ks[1].data_ptr(), pool.ring_vs[1].data_ptr()]
    assert pool.k_buffer[0].shape == (rows, heads, DH) and pool.k_scale_buffer[0].shape == (rows, heads, NG4)
    descs = [(d.name, d.shape, d.row_bytes, d.tokens_per_row) for d in pool._kv_buffer_descs]
    assert descs == ref_descs, "descs must be identical to the int4 pool (ring is not a desc)"
    kb, vb = pool.get_kv_size_bytes()
    ring_k_bytes = 2 * R * heads * (dim + NG8 * 2)
    assert kb == ref_kb + ring_k_bytes + R * 4 and vb == ref_vb + ring_k_bytes, "get_kv_size_bytes must add the ring (+ owner in K)"
    if lazy:
        o = pool._post_capture_owner
        assert o is not None and len(o.tensors) == 8 and o.bytes_per_token() == 2 * (256 + 256 + 32 + 32), "owner unchanged vs int4"
        lo = min(t.data_ptr() for t in o.tensors); hi = max(t.data_ptr() + t.numel() * t.element_size() for t in o.tensors)
        for t in pool.ring_k + pool.ring_v + pool.ring_ks + pool.ring_vs + [pool.ring_owner]:
            assert not (lo <= t.data_ptr() < hi), "ring tensors must live outside the VMM owner"
        assert (pool.k_scale_buffer[1][:page] == 0).all()
    else:
        assert (pool.k_scale_buffer[0] == 0).all() and (pool.k_buffer[0] == 0).all()
    n = 50
    loc = (torch.randperm(400, device=dev)[:n] + 64).sort().values.to(torch.int64)   # inside the backed floor
    # make the loc alias-free mod R (the pool only enforces N <= R)
    seen, keep = set(), []
    for s in loc.tolist():
        if (s & MASK) not in seen:
            seen.add(s & MASK); keep.append(s)
    loc = torch.tensor(keep, dtype=torch.int64, device=dev); n = loc.numel()
    assert n >= 30
    xk_p = torch.randn(n, heads, dim, device=dev, dtype=torch.bfloat16) * 3
    xv_p = torch.randn(n, heads, dim, device=dev, dtype=torch.bfloat16) * 3
    pool.set_kv_buffer(None, loc, xk_p, xv_p, 1.0, 1.0, layer_id_override=1)
    pool.set_kv_buffer(None, loc, xk_p.reshape(n, -1), xv_p.reshape(n, -1), layer_id_override=0)
    torch.cuda.synchronize()
    pk_p, sk_p = ref_quant4(xk_p); pv_p, sv_p = ref_quant4(xv_p)
    q8_p, s8_p = ref_quant8(xk_p); q8v_p, s8v_p = ref_quant8(xv_p)
    for l in (0, 1):
        ksb, vsb = pool.get_kv_scale_buffer(l)
        assert torch.equal(pool.k_buffer[l][loc], pk_p) and torch.equal(ksb[loc], sk_p)
        assert torch.equal(pool.v_buffer[l][loc], pv_p) and torch.equal(vsb[loc], sv_p)
        rk_p, rv_p, rks_p, rvs_p = pool.get_kv_ring_buffer(l)
        assert torch.equal(rk_p[loc & MASK], q8_p) and torch.equal(rks_p[loc & MASK], s8_p)
        assert torch.equal(rv_p[loc & MASK], q8v_p) and torch.equal(rvs_p[loc & MASK], s8v_p)
    assert torch.equal(pool.ring_owner.long()[loc & MASK], loc) and int((pool.ring_owner == -1).sum()) == R - n
    try:
        pool.set_kv_buffer(None, torch.arange(R + 1, device=dev, dtype=torch.int64), torch.zeros(R + 1, heads, dim, device=dev, dtype=torch.bfloat16),
                           torch.zeros(R + 1, heads, dim, device=dev, dtype=torch.bfloat16), layer_id_override=0)
        raise AssertionError("expected a ValueError for N > R")
    except ValueError:
        pass
    for bad_kw in (dict(k_scale=0.5), dict(v_scale=torch.ones(1, device=dev)), dict(dcp_kv_mask=torch.ones(n, device=dev))):
        try:
            pool.set_kv_buffer(None, loc, xk_p, xv_p, layer_id_override=0, **bad_kw); raise AssertionError("expected a raise")
        except (ValueError, NotImplementedError):
            pass
    pool.lazy_release()
    assert (pool.ring_owner == -1).all(), "lazy_release must reset the owner table"
    smk, smv = pool.get_kv_smooth_buffer(1)
    assert smk.shape == (heads, dim) and bool((smk == 1).all())
    pool._clear_buffers()
    assert pool.ring_k is None and pool.ring_owner is None
    print(f"  MHATokenToKVPoolTiered ({'lazy VMM' if lazy else 'eager'}, R={R}): ring shapes/dtypes, owner -1, descs == int4 pool, "
          f"ring outside the owner, sizes incl. ring, set_kv_buffer bit-exact (int4 + ring + owner, {n} tokens), "
          f"N > R rejected, lazy_release resets owner, guards: ok")
os.environ.pop("SGLANG_KV_LAZY", None); os.environ.pop("SGLANG_KV_TIERS_W", None)

# ---------------------------------------------------------------- 7. microbench (informational; the GPU may be shared)
tk_ = 2048
seq_b = torch.tensor([4000], dtype=torch.int32, device=dev)
idx_b = torch.randperm(4000, device=dev)[:tk_].sort().values.to(torch.int32)[None]
cu_b_ = torch.tensor([0, tk_], dtype=torch.int32, device=dev)
ob_k = torch.empty(tk_, heads, dim, device=dev, dtype=torch.bfloat16); ob_v = torch.empty_like(ob_k)
kw_t = tier_kwargs(); kw_i4 = kw4
k8_full, v8_full = k8_all, v8_all
kw_i8 = dict(k_scale=ks8_all, v_scale=vs8_all, sm_k=ones, sm_v=ones, kv_bits=8)


def bench(fn, iters=200):
    for _ in range(20): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / iters * 1e6


owner_all_cold = torch.full((R,), -1, dtype=torch.int32, device=dev)
# all-hot: a ring as large as the pool (R_b = slots, so slot & MASK_b == slot) whose rows ARE the int8 pool and
# whose owner table is arange -> every selected slot is hot (verified below, output bit-exact vs the int8 kernel)
R_b = slots
owner_all_hot = torch.arange(R_b, device=dev, dtype=torch.int32)
kw_cold = dict(kw_t, owner=owner_all_cold)
kw_hot = dict(k_scale=ks4, v_scale=vs4, sm_k=ones, sm_v=ones, kv_bits=4, ring_k=k8_full, ring_v=v8_full,
              ring_ks=ks8_all, ring_vs=vs8_all, owner=owner_all_hot, ring_mask=R_b - 1)
f_cold = lambda: qwen_sparse_kv_extraction_compact_triton(k4, v4, req_to_token, req_indices, idx_b, seq_b, cu_b_, ob_k, ob_v, 1, tk_, **kw_cold)
f_hot = lambda: qwen_sparse_kv_extraction_compact_triton(k4, v4, req_to_token, req_indices, idx_b, seq_b, cu_b_, ob_k, ob_v, 1, tk_, **kw_hot)
f_i4 = lambda: qwen_sparse_kv_extraction_compact_triton(k4, v4, req_to_token, req_indices, idx_b, seq_b, cu_b_, ob_k, ob_v, 1, tk_, **kw_i4)
f_i8 = lambda: qwen_sparse_kv_extraction_compact_triton(k8_full, v8_full, req_to_token, req_indices, idx_b, seq_b, cu_b_, ob_k, ob_v, 1, tk_, **kw_i8)
rows_b = req_to_token[0, idx_b[0].long()].long()
n_hot_b = int((owner_all_hot.long()[rows_b & (R_b - 1)] == rows_b).sum())
assert n_hot_b == tk_, f"all-hot bench must select only hot slots, got {n_hot_b}/{tk_}"
f_hot(); torch.cuda.synchronize(); hot_k, hot_v = ob_k.clone(), ob_v.clone()
f_i8(); torch.cuda.synchronize()
assert torch.equal(hot_k[:tk_], ob_k[:tk_]) and torch.equal(hot_v[:tk_], ob_v[:tk_]), "all-hot tiered gather != _compact_kv_int8"
assert torch.equal(hot_k[:tk_], d8k[rows_b]) and torch.equal(hot_v[:tk_], d8v[rows_b])
f_cold(); torch.cuda.synchronize(); cold_k = ob_k.clone()
f_i4(); torch.cuda.synchronize()
# (d4k is stale for slot7 since section 5 rewrote its int4 row: recompute the reference from the current buffers)
assert torch.equal(cold_k[:tk_], ob_k[:tk_]) and torch.equal(cold_k[:tk_], ref_dequant4(k4, ks4)[rows_b]), "all-cold tiered gather != _compact_kv_int4"
t_cold, t_hot, t_i4, t_i8 = bench(f_cold), bench(f_hot), bench(f_i4), bench(f_i8)
print(f"  microbench topk {tk_} bs 1 (us/launch, shared GPU; plan target <= 1.10x each): tiered all-cold {t_cold:.1f} vs int4 {t_i4:.1f} "
      f"({t_cold / t_i4:.2f}x); tiered all-hot ({n_hot_b}/{tk_} hot, bit-exact vs int8) {t_hot:.1f} vs int8 {t_i8:.1f} ({t_hot / t_i8:.2f}x)")
print("  OK")
