"""A/B of the tiled MoE kernel path through the REAL invoke_fused_moe_kernel:

  python3 test_tiled_word.py ref    # pristine tree: original kernel on the byte layout -> saves ref
  python3 test_tiled_word.py cmp    # patched tree (ncontig_gemv.py applied): word layout -> compares

Same real experts (layer 5), same seeded random tokens and routing, prefill-sized M. Exercises
exactly the code the server runs for M > 16: config resolution is bypassed (fixed config) but the
grid, even_Ks, strides and the kernel are the patched invoke's own.
"""
import argparse, json, os, sys
import torch, triton
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_gemv_int2_tab import to_word_ncontig

Q = os.path.expanduser(os.environ.get("Q", "~/quant/model"))  # the 2-bit checkpoint directory (Hub download)
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiled_ref.pt")
CFG = (64, 64, 128)


def load_words(layer, n_exp):
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
    return st(w13), st(s13), st(w2), st(s2)


def run_pair(mode, words, s_kn, m, topk, tag):
    """One GEMM (w13-like or w2-like) through invoke. mode: 'byte' or 'word'."""
    import sglang.kernels.ops.moe.fused_moe_triton_kernels as KM
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import moe_align_block_size
    E, KW, N = words.shape; K = KW * 16
    dev = "cuda"
    g = torch.Generator().manual_seed(1234 + len(tag))
    x = (torch.randn(m, K, generator=g) / 8).to(torch.bfloat16).to(dev)
    topk_ids = torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(m)]).to(torch.int32).to(dev)
    topk_w = torch.rand(m, topk, generator=g); topk_w = (topk_w / topk_w.sum(1, keepdim=True)).to(dev)
    config = {"BLOCK_SIZE_M": CFG[0], "BLOCK_SIZE_N": CFG[1], "BLOCK_SIZE_K": CFG[2], "GROUP_SIZE_M": 8,
              "num_warps": 4, "num_stages": 3}
    sorted_ids, expert_ids, n_post = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E)
    if mode == "byte":
        B = torch.stack([w.T.contiguous().view(torch.uint8) for w in words]).contiguous().to(dev)
        Bs = s_kn.transpose(1, 2).contiguous().to(dev).to(torch.bfloat16)
    else:
        B = words.to(dev); Bs = s_kn.to(dev).to(torch.bfloat16)
    C = torch.empty((m * topk, N), dtype=torch.bfloat16, device=dev)
    KM.invoke_fused_moe_kernel(
        x, B, None, C, None, Bs, None, topk_w, topk_ids, sorted_ids, expert_ids, n_post,
        True, topk, config, compute_type=triton.language.bfloat16,
        use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False,
        use_int2_w2a16=True, per_channel_quant=False, block_shape=[0, 128])
    torch.cuda.synchronize()
    return C.float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ref", "cmp"])
    ap.add_argument("--m", type=int, default=96); ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--topk", type=int, default=10); ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--cfg", default="64,64,128", help="BLOCK_SIZE_M,N,K")
    a = ap.parse_args()
    global CFG, REF
    CFG = tuple(int(v) for v in a.cfg.split(","))
    REF = REF.replace(".pt", f"_{a.cfg.replace(',', 'x')}.pt")
    w13, s13, w2, s2 = load_words(a.layer, a.experts)
    assert torch.equal(to_word_ncontig(torch.stack([w.T.contiguous().view(torch.uint8) for w in w13])), w13)
    if a.cmd == "ref":
        out = {"w13": run_pair("byte", w13, s13, a.m, a.topk, "w13"),
               "w2": run_pair("byte", w2, s2, a.m, a.topk, "w2")}
        torch.save(out, REF); print(f"  ref saved (byte layout, original kernel): {REF}")
        return
    ref = torch.load(REF)
    for tag, w, s in (("w13", w13, s13), ("w2", w2, s2)):
        got = run_pair("word", w, s, a.m, a.topk, tag)
        r = ref[tag]
        err = ((got - r).norm() / r.norm()).item(); mx = (got - r).abs().max().item()
        print(f"  {tag}: word layout vs byte ref  rel err {err:.2e}  max abs {mx:.3e}  "
              f"-> {'IDENTICAL' if err < 1e-2 else 'WRONG'}")


if __name__ == "__main__":
    main()
