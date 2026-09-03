import torch, triton, triton.language as tl

@triton.jit
def deq_w2(b_ptr, s_ptr, out_ptr, K, N, group_size,
           stride_bk, stride_bn, stride_sk, stride_sn,
           BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_k = tl.program_id(0); pid_n = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_ptrs = b_ptr + (offs_k[:, None] // 4) * stride_bk + offs_n[None, :] * stride_bn
    b_shifter = (offs_k[:, None] % 4) * 2
    m = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    b = tl.load(b_ptrs, mask=m, other=0)
    b = (b >> b_shifter) & 0x3
    s_ptrs = s_ptr + (offs_k[:, None] // group_size) * stride_sk + offs_n[None, :] * stride_sn
    s = tl.load(s_ptrs, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs_k[:, None] * N + offs_n[None, :],
             (b.to(tl.float32) - 2) * s, mask=m)

torch.manual_seed(7)
for (K, N, gs) in ((2560, 640, 128), (640, 2560, 128), (256, 64, 64)):
    vals = torch.randint(0, 4, (K, N), dtype=torch.int32)
    qw = torch.zeros((K // 16, N), dtype=torch.int32)
    for j in range(16):
        qw |= (vals[j::16] & 0x3) << (2 * j)
    sc = (torch.rand(K // gs, N) * 0.02 + 0.001).float()
    ref = (vals.float() - 2) * sc.repeat_interleave(gs, dim=0)

    std = qw.T.contiguous().view(torch.uint8).cuda()
    scg = sc.cuda()
    out = torch.empty((K, N), dtype=torch.float32, device="cuda")
    BK = BN = 64
    deq_w2[(triton.cdiv(K, BK), triton.cdiv(N, BN))](
        std, scg, out, K, N, gs, std.stride(1), std.stride(0),
        scg.stride(0), scg.stride(1), BLOCK_K=BK, BLOCK_N=BN)
    ok = torch.equal(out.cpu(), ref)
    print(f"  K={K:5d} N={N:5d} gs={gs:4d} -> {K//gs:3d} Gruppen   "
          f"identisch={ok}   maxAbw={(out.cpu()-ref).abs().max():.1e}")
