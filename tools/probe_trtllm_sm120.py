# probe_trtllm_sm120.py: standalone check of FlashInfer's trtllm_batch_decode_with_kv_cache,
# the sparse-decode route that SGLang's qwen4-main-squashed takes on exact SM120 since #36806
# (sglang/UPSTREAM.md, "How the series was verified"; sglang/upstream/PR-4.md, "Decode routing").
# Shapes are the QSA backend's for Qwen3.8-Flash-Next: 24 query heads, 2 KV heads, head_dim 256,
# page 64; topk 2051 = indexer_budget 2048 + indexer_compress_ratio 4 - 1, the index-row width
# `token_topk + compress_ratio - 1` of qsa_indexer.py (2051 rows -> 33 pages -> stride 2112).
# Reference: a torch softmax attention in fp32 over the same rows. Needs flashinfer (0.6.17 was
# used), torch with CUDA, and an nvcc >= 12.9 at CUDA_HOME for the XQA JIT (flashinfer/
# compilation_context.py; with a CUDA 12.0 nvcc it aborts with "No supported CUDA architectures
# found"). No server, ~200 MB of VRAM. Output recorded on the reference machine (RTX PRO 4000
# Blackwell, nvcc 13.3 from the virtualenv): docs/logs/probe_trtllm_sm120.log.
#
# Probe: does FlashInfer's trtllm_batch_decode_with_kv_cache (the route #36806 enables on exact SM120)
# run and agree with a torch reference on this card, called exactly as _forward_trtllm_sparse calls it.
import torch, math, sys
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
torch.manual_seed(0)
dev = "cuda"
NQ, NKV, D, PAGE = 24, 2, 256, 64
scale = 1.0 / math.sqrt(D)
ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
def run(batch, topk):
    pages_per_row = (topk + PAGE - 1) // PAGE
    stride = pages_per_row * PAGE
    packed_k = torch.randn(batch * stride, NKV, D, device=dev, dtype=torch.bfloat16)
    packed_v = torch.randn(batch * stride, NKV, D, device=dev, dtype=torch.bfloat16)
    q = torch.randn(batch, NQ, D, device=dev, dtype=torch.bfloat16)
    valid = torch.randint(1, topk + 1, (batch,), device=dev, dtype=torch.int32)
    valid[0] = topk
    block_tables = (torch.arange(batch, dtype=torch.int32, device=dev)[:, None] * pages_per_row
                    + torch.arange(pages_per_row, dtype=torch.int32, device=dev)[None, :]).contiguous()
    kc = packed_k.view(-1, PAGE, NKV, D).permute(0, 2, 1, 3)
    vc = packed_v.view(-1, PAGE, NKV, D).permute(0, 2, 1, 3)
    out = trtllm_batch_decode_with_kv_cache(query=q.contiguous(), kv_cache=(kc, vc), workspace_buffer=ws,
        block_tables=block_tables, seq_lens=valid, max_seq_len=stride, bmm1_scale=scale, bmm2_scale=1.0)
    out = out.reshape(batch, NQ, D).float()
    # reference
    ref = torch.empty_like(out)
    for b in range(batch):
        n = int(valid[b]); k = packed_k[b*stride:b*stride+n].float(); v = packed_v[b*stride:b*stride+n].float()
        qb = q[b].float()  # [NQ, D]
        for h in range(NQ):
            kvh = h // (NQ // NKV)
            s = (qb[h] @ k[:, kvh].T) * scale
            ref[b, h] = torch.softmax(s, 0) @ v[:, kvh]
    err = (out - ref).abs().max().item(); rel = err / (ref.abs().max().item() + 1e-6)
    print(f"batch={batch} topk={topk} stride={stride} max_abs_err={err:.4e} rel={rel:.3e} nan={torch.isnan(out).any().item()}")
    return err
errs = [run(1, 2051), run(4, 2051), run(3, 130)]
print("PROBE_OK" if max(errs) < 2e-2 else "PROBE_MISMATCH")
