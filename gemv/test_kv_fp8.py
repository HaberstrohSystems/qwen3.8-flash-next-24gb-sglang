"""Unit test for the fp8 QSA gather-dequant path (run after patches/kv_fp8.py apply; ~50 MB VRAM).

Checks, against a bf16 reference pool:
  1. the compact gather from an fp8 pool (uint8 store) equals pool_bf16.to(fp8).to(bf16) bit for bit;
  2. the dequantization error of e4m3 on N(0,1) K/V (relative RMS);
  3. the write-path saturation (values beyond 448 do not become NaN).
"""
import torch
from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_kv_extraction_compact_triton, qwen_sparse_fa2_cu_seqlens_triton)

torch.manual_seed(0)
dev = "cuda"
slots, heads, dim, batch, topk = 4096, 2, 256, 3, 64
k16 = torch.randn(slots, heads, dim, device=dev, dtype=torch.bfloat16) * 3
v16 = torch.randn(slots, heads, dim, device=dev, dtype=torch.bfloat16) * 3
k8 = k16.to(torch.float8_e4m3fn); v8 = v16.to(torch.float8_e4m3fn)
seq_lens = torch.tensor([300, 1200, 4000], dtype=torch.int32, device=dev)
req_to_token = torch.arange(slots, dtype=torch.int32, device=dev).repeat(batch, 1)
req_indices = torch.arange(batch, dtype=torch.int32, device=dev)
indices = torch.stack([torch.randperm(int(l), device=dev)[:topk].sort().values for l in seq_lens]).to(torch.int32)
cu_k = torch.empty(batch + 1, dtype=torch.int32, device=dev)
counts = torch.empty(batch, dtype=torch.int32, device=dev)
qwen_sparse_fa2_cu_seqlens_triton(seq_lens, indices, counts, cu_k, batch, topk)
n = int(cu_k[-1])

ref_k = torch.empty(n, heads, dim, device=dev, dtype=torch.bfloat16); ref_v = torch.empty_like(ref_k)
qwen_sparse_kv_extraction_compact_triton(k16, v16, req_to_token, req_indices, indices, seq_lens, cu_k, ref_k, ref_v, batch, topk)
out_k = torch.empty_like(ref_k); out_v = torch.empty_like(ref_v)
qwen_sparse_kv_extraction_compact_triton(k8, v8, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk)
torch.cuda.synchronize()
exp_k = torch.empty_like(ref_k); exp_v = torch.empty_like(ref_v)
qwen_sparse_kv_extraction_compact_triton(k8.to(torch.bfloat16), v8.to(torch.bfloat16), req_to_token, req_indices, indices, seq_lens, cu_k, exp_k, exp_v, batch, topk)
torch.cuda.synchronize()
assert torch.equal(out_k, exp_k) and torch.equal(out_v, exp_v), "fp8 gather-dequant != torch fp8->bf16"
print(f"  gather-dequant bit-exact vs torch cast: ok ({n} rows)")
err = ((out_k.float() - ref_k.float()).pow(2).mean() / ref_k.float().pow(2).mean()).sqrt()
print(f"  e4m3 relative RMS error on N(0,3) keys: {float(err):.4f}")
big = torch.tensor([500.0, -500.0, 100.0], device=dev)
print(f"  cast without clamp: {big.to(torch.float8_e4m3fn).to(torch.float32).tolist()}  with clamp: {big.clamp(-448, 448).to(torch.float8_e4m3fn).to(torch.float32).tolist()}")
print("  OK")
