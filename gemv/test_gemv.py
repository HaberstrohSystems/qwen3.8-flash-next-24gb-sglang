"""Correctness and bandwidth test for moe_gemv_int2 against real expert tensors.

Needs the GPU. Run in a window where the server is down (or has >1.5 GiB free).

  python3 test_gemv.py [--layer 5] [--experts 64] [--topk 10]

Reports:
  1. bit-level agreement with a torch fp32 reference (dequant then matmul)
  2. achieved GB/s reading experts from DEVICE memory
  3. achieved GB/s reading experts from PINNED HOST memory through the same kernel
  4. the same for the down projection (K=640)
"""
import argparse, json, os, sys, time
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_gemv_int2 import gemv_int2, make_tables

Q = os.path.expanduser(os.environ.get("Q", "~/quant/model"))  # the 2-bit checkpoint directory (Hub download)


def load_experts(layer, n_exp):
    """Return w13 [E,N,K/4] u8, s13 [E,N,K/128] f16, w2, s2 - packed as MoeWNA16 does."""
    idx = json.load(open(f"{Q}/model.safetensors.index.json"))["weight_map"]
    pre = f"model.language_model.layers.{layer}.mlp.experts."
    handles = {}
    def get(k):
        f = idx[k]
        if f not in handles:
            handles[f] = safe_open(f"{Q}/{f}", "pt")
        return handles[f].get_tensor(k)
    w13, s13, w2, s2 = [], [], [], []
    for e in range(n_exp):
        g = get(f"{pre}{e}.gate_proj.qweight"); u = get(f"{pre}{e}.up_proj.qweight")
        gs = get(f"{pre}{e}.gate_proj.scales"); us = get(f"{pre}{e}.up_proj.scales")
        d = get(f"{pre}{e}.down_proj.qweight"); ds = get(f"{pre}{e}.down_proj.scales")
        # checkpoint: qweight [K/16, N] int32, scales [K/128, N] f16
        qw = torch.cat([g, u], dim=1)                     # [K/16, 2N]
        sc = torch.cat([gs, us], dim=1)                   # [K/128, 2N]
        w13.append(qw.T.contiguous().view(torch.uint8))   # [2N, K/4]
        s13.append(sc.T.contiguous())                     # [2N, K/128]
        w2.append(d.T.contiguous().view(torch.uint8))     # [N, K2/4]
        s2.append(ds.T.contiguous())                      # [N, K2/128]
    return (torch.stack(w13), torch.stack(s13), torch.stack(w2), torch.stack(s2))


def reference(w_u8, s_f16, x):
    """w_u8 [T,N,K/4], s [T,N,K/128], x [K] or [T,K] -> fp32 [T,N]"""
    T, N, KB = w_u8.shape; K = KB * 4
    q = torch.empty((T, N, K), dtype=torch.int32)
    for j in range(4):
        q[:, :, j::4] = ((w_u8 >> (2 * j)) & 0x3).to(torch.int32)
    s = s_f16.float().repeat_interleave(128, dim=2)      # [T,N,K]
    W = (q.float() - 2.0) * s
    if x.dim() == 1:
        return torch.einsum("tnk,k->tn", W, x.float())
    return torch.einsum("tnk,tk->tn", W, x.float())


def bench(fn, iters=50):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--block-n", type=int, default=32)
    ap.add_argument("--warps", type=int, default=4)
    a = ap.parse_args()
    torch.manual_seed(0)

    print(f"\n=== loading {a.experts} experts of layer {a.layer} ===")
    w13, s13, w2, s2 = load_experts(a.layer, a.experts)
    E, N13, KB13 = w13.shape; K13 = KB13 * 4
    _, N2, KB2 = w2.shape;    K2 = KB2 * 4
    bytes13 = N13 * KB13 + N13 * (K13 // 128) * 2
    bytes2 = N2 * KB2 + N2 * (K2 // 128) * 2
    print(f"  w13 [{E},{N13},{KB13}] u8  K={K13}   {bytes13/1e6:.2f} MB/expert")
    print(f"  w2  [{E},{N2},{KB2}] u8   K={K2}    {bytes2/1e6:.2f} MB/expert")

    eids = torch.randperm(E)[: a.topk].to(torch.int32)
    x13 = torch.randn(K13, dtype=torch.bfloat16)
    x2 = torch.randn(a.topk, K2, dtype=torch.bfloat16)

    print("\n=== reference (CPU fp32) ===")
    ref13 = reference(w13[eids.long()], s13[eids.long()], x13)
    ref2 = reference(w2[eids.long()], s2[eids.long()], x2)

    for name, src in (("DEVICE", "cuda"), ("PINNED HOST", "pinned")):
        print(f"\n=== source: {name} ===")
        if src == "cuda":
            W13, S13, W2, S2 = (t.cuda() for t in (w13, s13, w2, s2))
        else:
            W13, S13, W2, S2 = (t.pin_memory() for t in (w13, s13, w2, s2))
        wt13, st13 = make_tables(W13, S13)
        wt2, st2 = make_tables(W2, S2)
        e_d = eids.cuda(); x13_d = x13.cuda(); x2_d = x2.cuda()

        out13 = gemv_int2(x13_d, wt13, st13, e_d, N13, K13, a.block_n, a.warps).cpu()
        out2 = gemv_int2(x2_d, wt2, st2, e_d, N2, K2, a.block_n, a.warps).cpu()
        d13 = (out13 - ref13).abs().max().item() / (ref13.abs().max().item() + 1e-9)
        d2 = (out2 - ref2).abs().max().item() / (ref2.abs().max().item() + 1e-9)
        ok13 = torch.allclose(out13, ref13, rtol=1e-3, atol=1e-2)
        ok2 = torch.allclose(out2, ref2, rtol=1e-3, atol=1e-2)
        print(f"  w13 correct: {ok13}  (max rel err {d13:.2e})")
        print(f"  w2  correct: {ok2}   (max rel err {d2:.2e})")

        t13 = bench(lambda: gemv_int2(x13_d, wt13, st13, e_d, N13, K13, a.block_n, a.warps))
        t2 = bench(lambda: gemv_int2(x2_d, wt2, st2, e_d, N2, K2, a.block_n, a.warps))
        mb13 = a.topk * bytes13 / 1e6; mb2 = a.topk * bytes2 / 1e6
        print(f"  w13: {t13*1e3:6.3f} ms for {mb13:6.1f} MB  = {mb13/t13/1e3:6.1f} GB/s")
        print(f"  w2 : {t2*1e3:6.3f} ms for {mb2:6.1f} MB  = {mb2/t2/1e3:6.1f} GB/s")
        per_layer = t13 + t2
        print(f"  per layer {per_layer*1e3:.3f} ms  -> x48 layers = {per_layer*48*1e3:.1f} ms/token"
              f"  (today: kernel 13.7 + movement 9.5 = 23.2 ms)")


if __name__ == "__main__":
    main()
