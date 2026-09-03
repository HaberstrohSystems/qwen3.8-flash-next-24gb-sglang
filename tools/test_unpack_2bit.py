"""Check the patched kernel's 2-bit unpacking against a reference.

These are exactly the expressions from fused_moe_triton_kernels.py:
    b_ptrs    = ... + (offs_k[:, None] // 4) * stride_bk + offs_bn[None, :] * stride_bn
    b_shifter = (offs_k[:, None] % 4) * 2
    b         = (b >> b_shifter) & 0x3
    b_zp_num  = 2
"""
import torch, triton, triton.language as tl, glob
import os
from safetensors import safe_open

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
    val = (b.to(tl.float32) - 2) * s          # b_zp_num = 2 (sym)
    tl.store(out_ptr + offs_k[:, None] * N + offs_n[None, :], val, mask=m)

d = glob.glob(os.path.expanduser("~/quant/out/mini-sub4/*/model.safetensors"))[0]
f = safe_open(d, "pt")
key = "model.language_model.layers.0.mlp.experts.0.gate_proj"
qw = f.get_tensor(key + ".qweight")            # int32 [K/16, N]
sc = f.get_tensor(key + ".scales").float()     # [K/gs, N]
K, N = qw.shape[0] * 16, qw.shape[1]
gs = K // sc.shape[0]
print(f"  Tensor {key.split('.')[-1]}: K={K} N={N} group_size={gs}")

# --- Referenz in PyTorch ---
ref_q = torch.empty((K, N), dtype=torch.int32)
for j in range(16):
    ref_q[j::16] = (qw >> (2 * j)) & 0x3
ref = (ref_q.float() - 2) * sc.repeat_interleave(gs, dim=0)

# --- so, wie MoeWNA16 laedt: [N, K/4] uint8 ---
std = qw.T.contiguous().view(torch.uint8).cuda()      # [N, K/4]
scg = sc.cuda()
out = torch.empty((K, N), dtype=torch.float32, device="cuda")
BK, BN = 64, 64
deq_w2[(triton.cdiv(K, BK), triton.cdiv(N, BN))](
    std, scg, out, K, N, gs,
    std.stride(1), std.stride(0),       # stride_bk laeuft entlang K, stride_bn entlang N
    scg.stride(0), scg.stride(1),
    BLOCK_K=BK, BLOCK_N=BN)

got = out.cpu()
same = torch.equal(got, ref)
maxdiff = (got - ref).abs().max().item()
print(f"  bit-identisch : {same}")
print(f"  max. Abweichung: {maxdiff:.3e}")
print(f"  Wertebereich   : {ref.min():.4f} .. {ref.max():.4f}")
print(f"  {'BESTANDEN' if same else 'FEHLGESCHLAGEN'}")
