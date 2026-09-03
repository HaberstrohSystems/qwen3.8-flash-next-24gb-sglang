"""GPU test of the agent's N-contiguous int2 GEMV on real expert tensors.

Weights stay in the checkpoint's native orientation: qweight int32 [K/16, N] (N contiguous),
scales fp16 [K/128, N]. Compares against a torch fp32 reference and reports GB/s from
device memory and from pinned host memory.

  python3 test_gemv_ncontig.py [--experts 64] [--topk 10] [--bn 64] [--bk 128] [--warps 4]
"""
import argparse, json, os, sys, time
import torch, triton
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "int2_gemv"))
from moe_gemv_int2_ncontig_v4 import moe_gemv_int2_ncontig

Q = os.path.expanduser(os.environ.get("Q", "~/quant/model"))  # the 2-bit checkpoint directory (Hub download)


def load(layer, n_exp):
    idx = json.load(open(f"{Q}/model.safetensors.index.json"))["weight_map"]
    pre = f"model.language_model.layers.{layer}.mlp.experts."
    hs = {}
    def get(k):
        f = idx[k]
        if f not in hs:
            hs[f] = safe_open(f"{Q}/{f}", "pt")
        return hs[f].get_tensor(k)
    w13, s13, w2, s2 = [], [], [], []
    for e in range(n_exp):
        w13.append(torch.cat([get(f"{pre}{e}.gate_proj.qweight"), get(f"{pre}{e}.up_proj.qweight")], 1))
        s13.append(torch.cat([get(f"{pre}{e}.gate_proj.scales"), get(f"{pre}{e}.up_proj.scales")], 1))
        w2.append(get(f"{pre}{e}.down_proj.qweight")); s2.append(get(f"{pre}{e}.down_proj.scales"))
    st = lambda L: torch.stack(L).contiguous()
    return st(w13), st(s13), st(w2), st(s2)        # [E,K/16,N] i32, [E,K/128,N] f16


def reference(w32, s, x):
    """w32 [T,K/16,N] int32, s [T,K/128,N] f16, x [K] or [T,K] -> [T,N] fp32"""
    T, KW, N = w32.shape; K = KW * 16
    q = torch.empty((T, K, N), dtype=torch.int32)
    for j in range(16):
        q[:, j::16, :] = (w32 >> (2 * j)) & 0x3
    W = (q.float() - 2.0) * s.float().repeat_interleave(128, dim=1)
    if x.dim() == 1:
        return torch.einsum("tkn,k->tn", W, x.float())
    return torch.einsum("tkn,tk->tn", W, x.float())


def launch(x, w32, s, ids, N, K, BN, BK, warps, topk):
    R = ids.numel()
    c = torch.empty(R, N, dtype=torch.bfloat16, device="cuda")
    tw = torch.ones(R, device="cuda")
    a = x if x.dim() == 2 else x.unsqueeze(0)          # [M,K]; TOP_K maps r -> r//TOP_K
    moe_gemv_int2_ncontig[(triton.cdiv(N, BN), R)](
        a, w32, c, s, ids, tw, N, K,
        a.stride(0), w32.stride(0), w32.stride(1), c.stride(0), s.stride(0), s.stride(1),
        TOP_K=topk, GROUP=128, BLOCK_N=BN, BLOCK_K=BK, MUL_ROUTED_WEIGHT=False,
        num_warps=warps)
    return c


def bench(fn, iters=50):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5); ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--bn", type=int, default=64); ap.add_argument("--bk", type=int, default=128)
    ap.add_argument("--warps", type=int, default=4)
    a = ap.parse_args(); torch.manual_seed(0)
    w13, s13, w2, s2 = load(a.layer, a.experts)
    E, KW13, N13 = w13.shape; K13 = KW13 * 16
    _, KW2, N2 = w2.shape;    K2 = KW2 * 16
    b13 = w13[0].numel() * 4 + s13[0].numel() * 2; b2 = w2[0].numel() * 4 + s2[0].numel() * 2
    print(f"  w13 [{E},{KW13},{N13}] i32 K={K13}  {b13/1e6:.2f} MB/expert;  w2 [{E},{KW2},{N2}] K={K2}  {b2/1e6:.2f} MB/expert")
    ids = torch.randperm(E)[: a.topk].to(torch.int32)
    x13 = torch.randn(K13, dtype=torch.bfloat16); x2 = torch.randn(a.topk, K2, dtype=torch.bfloat16)
    ref13 = reference(w13[ids.long()], s13[ids.long()], x13)
    ref2 = reference(w2[ids.long()], s2[ids.long()], x2)
    for name, mv in (("DEVICE", lambda t: t.cuda()), ("PINNED HOST", lambda t: t.pin_memory())):
        W13, S13, W2, S2 = (mv(t) for t in (w13, s13, w2, s2))
        i_d = ids.cuda(); x13_d = x13.cuda(); x2_d = x2.cuda()
        # w13: one shared x -> TOP_K=topk with a [1,K] activation; w2: per-expert x -> TOP_K=1
        o13 = launch(x13_d, W13, S13, i_d, N13, K13, a.bn, a.bk, a.warps, a.topk).float().cpu()
        o2 = launch(x2_d, W2, S2, i_d, N2, K2, a.bn, a.bk, a.warps, 1).float().cpu()
        e13 = ((o13 - ref13).norm() / ref13.norm()).item(); e2 = ((o2 - ref2).norm() / ref2.norm()).item()
        print(f"\n=== source: {name} ===")
        print(f"  w13 rel err {e13:.2e}   w2 rel err {e2:.2e}   (bf16 output; ~4e-3 is bf16 rounding)")
        t13 = bench(lambda: launch(x13_d, W13, S13, i_d, N13, K13, a.bn, a.bk, a.warps, a.topk))
        t2 = bench(lambda: launch(x2_d, W2, S2, i_d, N2, K2, a.bn, a.bk, a.warps, 1))
        mb13 = a.topk * b13 / 1e6; mb2 = a.topk * b2 / 1e6
        print(f"  w13: {t13*1e3:6.3f} ms {mb13:6.1f} MB = {mb13/t13/1e3:6.1f} GB/s")
        print(f"  w2 : {t2*1e3:6.3f} ms {mb2:6.1f} MB = {mb2/t2/1e3:6.1f} GB/s")
        print(f"  per layer {(t13+t2)*1e3:.3f} ms -> x48 = {(t13+t2)*48*1e3:.1f} ms/token  (today 13.7 kernel + 9.5 movement)")


if __name__ == "__main__":
    main()
