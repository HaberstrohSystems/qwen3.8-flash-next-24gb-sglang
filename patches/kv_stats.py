#!/usr/bin/env python3
"""KV statistics hook for the QSA layers (design data for our own 8-bit KV scheme).

With SGLANG_KV_STATS=/path/out.pt every set_kv_buffer call on the main K/V pool accumulates, per
layer and KV head: per-channel absmax and sum of squares (for K and V), per-token absmax
histograms, and the fraction of tokens whose absmax lands in the top channel. The file is written
every 64 calls and at exit. Diagnostics only; values are unchanged.

  python3 kv_stats.py --check | apply | revert
"""
import os, sys
SG = os.path.join(os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")), "python/sglang")
MP = f"{SG}/srt/mem_cache/memory_pool.py"

EDITS = [
  (MP, """    def lazy_ensure(self, num_tokens: int) -> None:""", """    _KVSTATS = None

    def _kv_stats(self, layer_id, cache_k, cache_v):
        \"\"\"Accumulate K/V statistics (SGLANG_KV_STATS). cache_k/v: [N, H, D] bf16.\"\"\"
        path = os.environ.get("SGLANG_KV_STATS")
        if not path or cache_k.dim() != 3:
            return
        st = MHATokenToKVPool._KVSTATS
        if st is None:
            st = MHATokenToKVPool._KVSTATS = {"calls": 0, "layers": {}}
        H, D = cache_k.shape[1], cache_k.shape[2]
        L = st["layers"].setdefault(int(layer_id), {
            "n": 0,
            "k_absmax_ch": torch.zeros(H, D, device=cache_k.device), "v_absmax_ch": torch.zeros(H, D, device=cache_k.device),
            "k_sumsq_ch": torch.zeros(H, D, device=cache_k.device), "v_sumsq_ch": torch.zeros(H, D, device=cache_k.device),
            "k_tok_absmax_hist": torch.zeros(64, device=cache_k.device), "v_tok_absmax_hist": torch.zeros(64, device=cache_k.device),
            "k_tok_ratio_hist": torch.zeros(32, device=cache_k.device), "v_tok_ratio_hist": torch.zeros(32, device=cache_k.device),
        })
        k = cache_k.float(); v = cache_v.float()
        L["n"] += int(k.shape[0])
        L["k_absmax_ch"] = torch.maximum(L["k_absmax_ch"], k.abs().amax(0)); L["v_absmax_ch"] = torch.maximum(L["v_absmax_ch"], v.abs().amax(0))
        L["k_sumsq_ch"] += (k * k).sum(0); L["v_sumsq_ch"] += (v * v).sum(0)
        for name, t in (("k", k), ("v", v)):
            am = t.abs().amax(2)                                              # [N, H] per-token absmax
            L[f"{name}_tok_absmax_hist"] += torch.histc(am.log2().clamp(-16, 15), bins=64, min=-16, max=16)
            ratio = am / (t.abs().mean(2) + 1e-6)                             # absmax / mean|x| per token (outlier-ness)
            L[f"{name}_tok_ratio_hist"] += torch.histc(ratio.clamp(0, 64), bins=32, min=0, max=64)
        st["calls"] += 1
        if st["calls"] % 64 == 0:
            torch.save({lid: {kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in Ld.items()} for lid, Ld in st["layers"].items()}, path)

    def lazy_ensure(self, num_tokens: int) -> None:"""),
  (MP, """        if cache_k.dtype != self.dtype:
            if k_scale is not None and not (isinstance(k_scale, (int, float)) and k_scale == 1):
                cache_k.div_(k_scale)""", """        if os.environ.get("SGLANG_KV_STATS"):
            self._kv_stats(layer_id, cache_k, cache_v)
        if cache_k.dtype != self.dtype:
            if k_scale is not None and not (isinstance(k_scale, (int, float)) and k_scale == 1):
                cache_k.div_(k_scale)"""),
]


def state():
    return [(p, a in open(p, encoding="utf-8").read(), b in open(p, encoding="utf-8").read()) for p, a, b in EDITS]


def check():
    for i, (p, pr, ap) in enumerate(state()):
        print(f"  {i} {'APPLIED' if ap else ('clean' if pr else 'MISMATCH'):<8} {os.path.relpath(p, SG)}: {EDITS[i][1].strip().splitlines()[0][:60]}")


def apply():
    st = state()
    if not all(pr or ap for _, pr, ap in st):
        print("  [!] mismatch (needs kv_lazy.py and kv_fp8.py applied)"); check(); return
    for (p, a, b), (_, pr, ap) in zip(EDITS, st):
        if not ap:
            t = open(p, encoding="utf-8").read(); open(p, "w", encoding="utf-8").write(t.replace(a, b, 1))
    print("  applied (KV stats hook; set SGLANG_KV_STATS=/path/out.pt)")


def revert():
    for (p, a, b), (_, pr, ap) in zip(EDITS, state()):
        if ap:
            t = open(p, encoding="utf-8").read(); open(p, "w", encoding="utf-8").write(t.replace(b, a, 1))
    print("  reverted")


if __name__ == "__main__":
    {"--check": check, "apply": apply, "revert": revert}[sys.argv[1] if len(sys.argv) > 1 else "--check"]()
