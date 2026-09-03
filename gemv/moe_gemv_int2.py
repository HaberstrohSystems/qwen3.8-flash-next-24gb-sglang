"""Batch-1 int2 MoE GEMV that reads each expert in place through a pointer table.

Idea 1 of perf/IDEAS-own.md. At batch 1 the MoE is T independent 2-bit GEMVs per
layer. A GEMV streams its weight matrix once and is memory-bound, so it does not
care whether the bytes live in HBM or in pinned host memory - only the achieved
bandwidth differs. Given a table of per-expert base addresses, the kernel indexes
with the ORIGINAL expert ids. No gather, no staging buffer, no renumbering.

Weight layout per expert (what MoeWNA16 already holds after load):
    qweight  [N, K/4]   uint8   value k of row n: byte k//4, bits (k%4)*2, & 3
    scales   [N, K/128] fp16    symmetric, zero point 2  ->  w = (q - 2) * s

Tables are int64 tensors of data_ptr() values, one per expert, on the device.
Under unified addressing a pinned host tensor's data_ptr() is directly loadable
from a kernel - that is how the existing gather kernel reaches host memory today.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_int2_tab(
    x_ptr, stride_xt,            # activations [T or 1, K] bf16; stride_xt=0 shares one x
    wtab_ptr, stab_ptr,          # int64 [E]: base address of qweight / scales per expert
    eids_ptr,                    # int32/int64 [T]: original expert ids
    out_ptr, stride_ot,          # fp32 [T, N]
    N, K,
    BLOCK_N: tl.constexpr,
    GPC: tl.constexpr,           # groups (of 128 values) per K chunk: 4 for K=2560, 5 for K=640
):
    t = tl.program_id(0)
    pn = tl.program_id(1)

    e = tl.load(eids_ptr + t).to(tl.int64)
    wbase = tl.load(wtab_ptr + e).to(tl.pointer_type(tl.uint8))
    sbase = tl.load(stab_ptr + e).to(tl.pointer_type(tl.float16))

    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N

    KBYTES: tl.constexpr = GPC * 32          # bytes per row per chunk
    offs_b = tl.arange(0, KBYTES)
    row_w = offs_n[:, None] * (K // 4)       # row stride in packed bytes
    row_s = offs_n[:, None] * (K // 128)     # row stride in scale entries
    x_row = x_ptr + t * stride_xt

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for c in range(0, K // (GPC * 128)):
        k0 = c * GPC * 128
        b = tl.load(wbase + row_w + k0 // 4 + offs_b[None, :],
                    mask=nmask[:, None], other=0)
        s = tl.load(sbase + row_s + k0 // 128 + offs_b[None, :] // 32,
                    mask=nmask[:, None], other=0.0).to(tl.float32)
        for j in tl.static_range(4):
            q = ((b >> (2 * j)) & 0x3).to(tl.float32) - 2.0
            xj = tl.load(x_row + k0 + offs_b * 4 + j).to(tl.float32)
            acc += tl.sum(q * s * xj[None, :], axis=1)

    tl.store(out_ptr + t * stride_ot + offs_n, acc, mask=nmask)


def gemv_int2(x: torch.Tensor, wtab: torch.Tensor, stab: torch.Tensor,
              eids: torch.Tensor, N: int, K: int, block_n: int = 32,
              num_warps: int = 4) -> torch.Tensor:
    """x: [K] (shared) or [T, K]. Returns fp32 [T, N]."""
    T = eids.numel()
    if x.dim() == 1:
        x2, sx = x, 0
    else:
        x2, sx = x, x.stride(0)
    out = torch.empty((T, N), dtype=torch.float32, device=eids.device)
    # tl.arange needs a power of two: 4 groups = 128 B per row-chunk for K=2560,
    # 1 group = 32 B for K=640 (5 groups, not a power of two).
    gpc = 4 if K % 512 == 0 else 1
    assert K % (gpc * 128) == 0, (K, gpc)
    grid = (T, triton.cdiv(N, block_n))
    _gemv_int2_tab[grid](
        x2, sx, wtab, stab, eids, out, out.stride(0), N, K,
        BLOCK_N=block_n, GPC=gpc, num_warps=num_warps,
    )
    return out


def make_tables(qweight: torch.Tensor, scales: torch.Tensor):
    """qweight [E, N, K/4] uint8 and scales [E, N, K/128] fp16 (device or pinned host)
    -> int64 tables of per-expert base addresses, on the CUDA device."""
    assert qweight.dtype == torch.uint8 and scales.dtype == torch.float16
    assert qweight.is_contiguous() and scales.is_contiguous()
    E = qweight.shape[0]
    wt = torch.tensor([qweight.data_ptr() + i * qweight.stride(0) for i in range(E)],
                      dtype=torch.int64, device="cuda")
    st = torch.tensor([scales.data_ptr() + i * scales.stride(0) * 2 for i in range(E)],
                      dtype=torch.int64, device="cuda")
    return wt, st
