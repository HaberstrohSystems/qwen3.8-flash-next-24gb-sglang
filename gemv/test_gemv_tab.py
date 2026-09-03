"""GPU test of moe_gemv_int2_tab (int32-word N-contiguous layout + pointer table).

  python3 test_gemv_tab.py [--experts 64] [--topk 10]

1. to_word_ncontig(): loader layout [E, N, K/4] uint8 -> [E, K/16, N] int32 must equal the
   checkpoint's native int32 [K/16, N] stacked (bit-exact re-arrangement).
2. Full decode MoE for one layer against a torch fp32 reference:
       w13 GEMV -> silu(gate) * up -> w2 GEMV with routed weights -> sum over top-k
3. GB/s for the expert reads from device and from pinned host memory.
"""
import argparse, json, os, sys, time
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_gemv_int2_tab import moe_gemv_int2_tab, make_tables, to_word_ncontig

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
    return st(w13), st(s13), st(w2), st(s2)      # int32 [E,K/16,N], f16 [E,K/128,N]


def dequant(w32, s):
    T, KW, N = w32.shape; K = KW * 16
    q = torch.empty((T, K, N), dtype=torch.int32)
    for j in range(16):
        q[:, j::16, :] = (w32 >> (2 * j)) & 0x3
    return (q.float() - 2.0) * s.float().repeat_interleave(128, dim=1)


def bench(fn, iters=50):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5); ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--topk", type=int, default=10); ap.add_argument("--bn", type=int, default=64)
    a = ap.parse_args(); torch.manual_seed(0)
    w13, s13, w2, s2 = load(a.layer, a.experts)
    E, KW13, N13 = w13.shape; K13 = KW13 * 16
    _, KW2, N2 = w2.shape;    K2 = KW2 * 16
    inter = N13 // 2
    print(f"  w13 i32 [{E},{KW13},{N13}] K={K13}  w2 i32 [{E},{KW2},{N2}] K={K2}  inter={inter}")

    # 1. layout conversion check: loader layout [E, N, K/4] u8 -> words
    loader_u8 = torch.stack([w.T.contiguous().view(torch.uint8) for w in w13])   # [E, N, K/4]
    back = to_word_ncontig(loader_u8)
    print(f"  to_word_ncontig bit-exact vs checkpoint words: {torch.equal(back, w13)}")

    ids = torch.randperm(E)[: a.topk].to(torch.int32)
    tw = torch.rand(a.topk); tw = tw / tw.sum()
    x = torch.randn(1, K13, dtype=torch.bfloat16)

    W13 = dequant(w13[ids.long()], s13[ids.long()]); W2 = dequant(w2[ids.long()], s2[ids.long()])
    h13 = torch.einsum("k,tkn->tn", x[0].float(), W13)
    h2 = torch.nn.functional.silu(h13[:, :inter]) * h13[:, inter:]
    ref = (torch.einsum("tk,tkn->tn", h2, W2) * tw[:, None]).sum(0)

    for name, mv in (("DEVICE", lambda t: t.cuda()), ("PINNED HOST", lambda t: t.pin_memory())):
        W13d, S13d, W2d, S2d = (mv(t) for t in (w13, s13, w2, s2))
        wt13, st13, sw13, ss13 = make_tables(W13d, S13d)
        wt2, st2, sw2, ss2 = make_tables(W2d, S2d)
        ids_d, tw_d, x_d = ids.cuda(), tw.cuda(), x.cuda()

        def run():
            c13 = moe_gemv_int2_tab(x_d, wt13, st13, ids_d, tw_d, N13, K13, sw13, ss13,
                                    top_k=a.topk, mul_routed_weight=False, block_n=a.bn, scale_bf16=False)
            hh = (torch.nn.functional.silu(c13[:, :inter].float()) * c13[:, inter:].float()).to(torch.bfloat16)
            c2 = moe_gemv_int2_tab(hh, wt2, st2, ids_d, tw_d, N2, K2, sw2, ss2,
                                   top_k=1, mul_routed_weight=True, block_n=a.bn, scale_bf16=False)
            return c2.float().sum(0)
        out = run().cpu()
        err = ((out - ref).norm() / ref.norm()).item()
        t = bench(run)
        mb = a.topk * (w13[0].numel() * 4 + s13[0].numel() * 2 + w2[0].numel() * 4 + s2[0].numel() * 2) / 1e6
        print(f"\n=== source: {name} ===")
        print(f"  end-to-end rel err {err:.2e}  (bf16 intermediates; ~5e-3 expected)")
        print(f"  layer MoE: {t*1e3:.3f} ms for {mb:.1f} MB = {mb/t/1e3:.0f} GB/s  -> x48 = {t*48*1e3:.1f} ms/token")


if __name__ == "__main__":
    main()
