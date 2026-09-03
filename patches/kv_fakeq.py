#!/usr/bin/env python3
"""Stage A0 of the own KV scheme: fake quantization on the write path (accuracy go/no-go).

SGLANG_KV_FAKEQ selects a scheme; K and V are quantized and immediately dequantized in
set_kv_buffer, then stored in the pool's normal dtype. Pool layout, lazy backing and both QSA
read paths stay untouched, so nll_long.py measures exactly the quantization error the real
storage scheme would have.

  int8_tok   int8, symmetric absmax scale per (token, kv-head) over 256 channels
  int8_g64   int8, scale per (token, kv-head, 64-channel group)   <- design winner
  int8_g32   int8, scale per (token, kv-head, 32-channel group)
  e4m3       fp8_e4m3 unit scale (what --kv-cache-dtype fp8_e4m3 stores), for the same-protocol comparison

  python3 kv_fakeq.py --check | apply | revert
"""
import os, sys
SG = os.path.join(os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")), "python/sglang")
MP = f"{SG}/srt/mem_cache/memory_pool.py"

EDITS = [
  (MP, """        if os.environ.get("SGLANG_KV_STATS"):
            self._kv_stats(layer_id, cache_k, cache_v)
""", """        if os.environ.get("SGLANG_KV_STATS"):
            self._kv_stats(layer_id, cache_k, cache_v)
        _fq = os.environ.get("SGLANG_KV_FAKEQ")
        if _fq and cache_k.dim() == 3:
            cache_k = _fake_quant_kv(cache_k, _fq, layer_id - self.start_layer, "k")
            cache_v = _fake_quant_kv(cache_v, _fq, layer_id - self.start_layer, "v")
"""),
  (MP, """class KVCache(abc.ABC):
""", """_FQ_SMOOTH = {}


def _fq_smooth(layer_id: int, which: str, H: int, D: int, device):
    \"\"\"Static per-channel RMS from the SGLANG_KV_STATS_FILE dump (normalised to median 1); ones if absent.\"\"\"
    key = (layer_id, which)
    if key not in _FQ_SMOOTH:
        st = _FQ_SMOOTH.setdefault("_stats", None)
        if st is None:
            path = os.environ.get("SGLANG_KV_STATS_FILE", "")
            st = torch.load(path) if path and os.path.exists(path) else {}
            _FQ_SMOOTH["_stats"] = st
        L = st.get(layer_id)
        if L is None:
            _FQ_SMOOTH[key] = torch.ones(H, D)
        else:
            rms = (L[f"{which}_sumsq_ch"] / max(L["n"], 1)).sqrt().float()
            rms = rms.clamp(min=1e-3 * rms.median())
            _FQ_SMOOTH[key] = rms / rms.median()
    return _FQ_SMOOTH[key].to(device)


def _fake_quant_kv(x: torch.Tensor, scheme: str, layer_id: int = 0, which: str = "k") -> torch.Tensor:
    \"\"\"Quantize -> dequantize [N, H, D] K or V per the SGLANG_KV_FAKEQ scheme; returns x.dtype.\"\"\"
    if not _FQ_SMOOTH.get("_logged"):
        _FQ_SMOOTH["_logged"] = True
        logger.info("KV fake quantization active: scheme %s (layer %d, %s)", scheme, layer_id, which)
    if scheme == "e4m3":
        return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(x.dtype)
    if scheme == "zero":                                   # destructive control: is the metric blind to the cache?
        return torch.zeros_like(x)
    if scheme == "noise":                                  # destructive control: random values of the same scale
        return torch.randn_like(x) * x.float().pow(2).mean().sqrt().to(x.dtype)
    bits = {"int8": 8, "int4": 4, "int3": 3, "int2": 2}[scheme[:4]]
    base = scheme.replace("_sm", "")
    group = {"int8_tok": x.shape[-1], "int8_g64": 64, "int8_g32": 32, "int4_g32": 32, "int4_g64": 64, "int4_g16": 16,
             "int3_g16": 16, "int3_g32": 32, "int2_g16": 16, "int2_g8": 8}[base]
    qmax = float(2 ** (bits - 1) - 1)                       # symmetric: 127 / 7 / 3 / 1
    N, H, D = x.shape
    xf = x.float()
    sm = None
    if scheme.endswith("_sm") and not torch.cuda.is_current_stream_capturing():   # capture writes dummy tokens; no H2D copies allowed there
        sm = _fq_smooth(layer_id, which, H, D, x.device)
    if sm is not None:
        xf = xf / sm                                     # flatten channel scale before grouping
    xf = xf.reshape(N, H, D // group, group)
    amax = xf.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
    q = torch.clamp(torch.round(xf / scale), -qmax, qmax)
    out = (q * scale).reshape(N, H, D)
    if sm is not None:
        out = out * sm
    return out.to(x.dtype)


class KVCache(abc.ABC):
"""),
]


def state():
    return [(p, a in open(p, encoding="utf-8").read(), b in open(p, encoding="utf-8").read()) for p, a, b in EDITS]


def check():
    for i, (p, pr, ap) in enumerate(state()):
        print(f"  {i} {'APPLIED' if ap else ('clean' if pr else 'MISMATCH'):<8} {os.path.relpath(p, SG)}: {EDITS[i][1].strip().splitlines()[0][:60]}")


def apply():
    st = state()
    if not all(pr or ap for _, pr, ap in st):
        print("  [!] mismatch (needs kv_stats.py applied)"); check(); return
    for (p, a, b), (_, pr, ap) in zip(EDITS, st):
        if not ap:
            t = open(p, encoding="utf-8").read(); open(p, "w", encoding="utf-8").write(t.replace(a, b, 1))
    print("  applied (fake KV quantization; set SGLANG_KV_FAKEQ=int8_g64|int8_tok|int8_g32|e4m3)")


def revert():
    for (p, a, b), (_, pr, ap) in zip(EDITS, state()):
        if ap:
            t = open(p, encoding="utf-8").read(); open(p, "w", encoding="utf-8").write(t.replace(b, a, 1))
    print("  reverted")


if __name__ == "__main__":
    {"--check": check, "apply": apply, "revert": revert}[sys.argv[1] if len(sys.argv) > 1 else "--check"]()
