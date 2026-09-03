#!/usr/bin/env python3
"""fp8_e4m3 KV cache for the Qwen sparse attention (QSA) read path.

SGLang already stores fp8 KV (uint8 store dtype, unit scale) and the lazy VMM backing handles it
unchanged (12 KB/token instead of 24). What is missing is the READ side of the QSA backend:

  * decode / verify: the selected blocks are gathered into a scratch and handed to FlashAttention-2,
    which only takes bf16/fp16 -> gather-dequant kernel (_compact_kv_fp8: uint8 -> fp8 bitcast -> bf16)
    and a bf16 scratch;
  * prefix-chunk prefill: the whole prefix is gathered with index_select + cat and fed to the Triton
    chunk kernel, which does tl.dot(q_bf16, k) -> gather on the uint8 view, bitcast, convert to q.dtype;
  * write path: skip the no-op div_ by a unit scale and saturate to +-448 before the fp8 cast (the
    fp32 -> e4m3fn cast returns NaN beyond the range instead of saturating).

Serve with --kv-cache-dtype fp8_e4m3 ('auto' stays bf16; 'fp8' is rejected by the parser).

ORDERING: perf/patches/kv_int8.py is layered on edits 0-3 of this patch (it wraps the scratch-dtype
line, the prefix-chunk block and the wrapper branch).  Apply kv_fp8 before kv_int8 and revert kv_int8
BEFORE kv_fp8; while kv_int8 is applied this --check shows edits 1-3 as MISMATCH (expected) and revert
refuses.

  python3 kv_fp8.py --check | apply | revert
"""
import os, sys
SG = os.path.join(os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")), "python/sglang")
SA = f"{SG}/srt/layers/attention/qsa/sparse_attn.py"
BK = f"{SG}/srt/layers/attention/qwen_sparse_attn_backend.py"
MP = f"{SG}/srt/mem_cache/memory_pool.py"

# (path, old, new[, "all"])   "all" = replace every occurrence (identical text at several sites)
EDITS = [
  # ------------------------------------------------------------- sparse_attn.py: gather-dequant kernel
  (SA, """def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
""", """@triton.jit
def _compact_kv_fp8(
    k,
    v,
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
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    \"\"\"_compact_kv for an fp8_e4m3 pool viewed as uint8: dequantize into the bf16 scratch.\"\"\"
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
    )
    src = slots[:, None] * heads * dim + head * dim + dims[None, :]
    dst = (pack_start + cols)[:, None] * heads * dim + head * dim + dims[None, :]
    mask = valid[:, None] & (dims[None, :] < dim)
    kk = tl.load(k + src, mask=mask, other=0).to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
    vv = tl.load(v + src, mask=mask, other=0).to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
    tl.store(out_k + dst, kk, mask=mask)
    tl.store(out_v + dst, vv, mask=mask)


def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
"""),
  (SA, """    _, heads, dim = k.shape
    block_topk = 16
    _compact_kv[(batch, heads, triton.cdiv(topk, block_topk))](
        k,
        v,
""", """    _, heads, dim = k.shape
    block_topk = 16
    if k.dtype == torch.float8_e4m3fn:                       # fp8 pool: dequantize into the bf16 scratch
        assert out_k.dtype == torch.bfloat16, "fp8 KV gather needs a bf16 scratch"
        _compact_kv_fp8[(batch, heads, triton.cdiv(topk, block_topk))](
            k.view(torch.uint8),
            v.view(torch.uint8),
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
            BLOCK_TOPK=block_topk,
            BLOCK_D=triton.next_power_of_2(dim),
            num_warps=8,
        )
        return
    _compact_kv[(batch, heads, triton.cdiv(topk, block_topk))](
        k,
        v,
"""),
  # ------------------------------------------------------------- backend: bf16 scratch for fp8 pools (decode + verify)
  (BK, """            k_buffer.shape[1],
            k_buffer.shape[2],
            k_buffer.dtype,
            k_buffer.device,
        )
""", """            k_buffer.shape[1],
            k_buffer.shape[2],
            torch.bfloat16 if k_buffer.dtype == torch.float8_e4m3fn else k_buffer.dtype,
            k_buffer.device,
        )
""", "all"),
  # ------------------------------------------------------------- backend: prefix-chunk gather with dequant
  (BK, """        k_parts = [
            k_buffer.index_select(
                0, req_to_token[req_indices[i], : sequence_lens[i]].long()
            )
            for i in range(len(sequence_lens))
        ]
        v_parts = [
            v_buffer.index_select(
                0, req_to_token[req_indices[i], : sequence_lens[i]].long()
            )
            for i in range(len(sequence_lens))
        ]
""", """        _fp8 = k_buffer.dtype == torch.float8_e4m3fn
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
"""),
  # ------------------------------------------------------------- memory_pool: unit-scale skip + saturation
  (MP, """        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
""", """        if cache_k.dtype != self.dtype:
            if k_scale is not None and not (isinstance(k_scale, (int, float)) and k_scale == 1):
                cache_k.div_(k_scale)
            if v_scale is not None and not (isinstance(v_scale, (int, float)) and v_scale == 1):
                cache_v.div_(v_scale)
            if self.dtype == torch.float8_e4m3fn:            # fp32/bf16 -> e4m3fn returns NaN beyond +-448
                cache_k = cache_k.clamp(-448.0, 448.0)
                cache_v = cache_v.clamp(-448.0, 448.0)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
""", "all"),
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


def check():
    for i, (p, pr, ap) in enumerate(state()):
        print(f"  {i} {'APPLIED' if ap else ('clean' if pr else 'MISMATCH'):<8} {_mode(EDITS[i]):<4} "
              f"{os.path.relpath(p, SG)}: {EDITS[i][1].strip().splitlines()[0][:60]}")


def apply():
    st = state()
    if not all(pr or ap for _, pr, ap in st):
        print("  [!] mismatch"); check(); return
    for e, (_, pr, ap) in zip(EDITS, st):
        p, a, b = e[0], e[1], e[2]
        if not ap:
            t = open(p, encoding="utf-8").read()
            open(p, "w", encoding="utf-8").write(t.replace(a, b, -1 if _mode(e) == "all" else 1))
    print("  applied (fp8_e4m3 KV read path for QSA; serve with --kv-cache-dtype fp8_e4m3)")


def revert():
    if "def _compact_kv_int8(" in open(SA, encoding="utf-8").read():
        print("  [!] refusing: perf/patches/kv_int8.py is applied on top of this patch -- run kv_int8.py revert first")
        return
    for e, (_, pr, ap) in zip(EDITS, state()):
        p, a, b = e[0], e[1], e[2]
        if ap:
            t = open(p, encoding="utf-8").read()
            open(p, "w", encoding="utf-8").write(t.replace(b, a, -1 if _mode(e) == "all" else 1))
    print("  reverted")


if __name__ == "__main__":
    {"--check": check, "apply": apply, "revert": revert}[sys.argv[1] if len(sys.argv) > 1 else "--check"]()
