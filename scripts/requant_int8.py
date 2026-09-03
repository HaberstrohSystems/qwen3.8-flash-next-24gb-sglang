#!/usr/bin/env python3
"""Requantize the dense bf16 groups to INT8 g128 (RTN, symmetric) into a NEW model directory.

Groups (per the second-pass error survey, all below the error the sealed int8 in_proj already
carries): linear_attn.out_proj (36), self_attn.{q,k,v,o}_proj (12 each), lm_head. Hyper-connection
tensors stay bf16 here (they need a kernel change, not just a config entry). Routers, norms,
indexer, conv, A_log, dt_bias, in_proj_a/b stay bf16.

Saves ~3.6 GB of VRAM reads per decode token (~3.1 ms) and ~1.8 GB of VRAM.

  python3 requant_int8.py --selftest                 # unpack->repack roundtrip on a sealed int8 tensor
  python3 requant_int8.py --out DIR [--dry-run]      # build the new directory (~46 GB, ~1 h, CPU only)

The sealed directory is read only; src-bf16 is read only. Output shards are written streaming,
one input shard at a time, so host memory stays bounded (< 8 GB).
"""
import argparse, json, os, re, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SEALED = os.path.expanduser(os.environ.get("SEALED", "~/quant/sealed-w2g128"))  # the sealed AutoRound output
SRC = os.path.expanduser(os.environ.get("SRC", "~/quant/src-bf16"))  # the bf16 source checkpoint
GS = 128
DENSE_RX = re.compile(
    r"^(model\.language_model\.layers\.\d+\.(linear_attn\.out_proj|self_attn\.(q|k|v|o)_proj)\.weight"
    r"|lm_head\.weight)$"
)


# ----------------------------------------------------------------- packing
def pack_int8_gptq(w: torch.Tensor):
    """bf16 [N_out, K_in] -> (qweight int32 [K/4, N], qzeros int32 [K/gs, N/4], scales f16 [K/gs, N]).

    auto_gptq layout, symmetric: zero point 128 stored as 127; 4 int8 per int32 along K for
    qweight and along N for qzeros (little-endian byte order = shift 8*(i%4)).
    """
    N, K = w.shape
    assert K % GS == 0 and N % 4 == 0, (N, K)
    wf = w.float().T.contiguous()                        # [K, N]
    g = wf.view(K // GS, GS, N)
    scale = g.abs().amax(dim=1) / 127.0                  # [K/gs, N]
    scale = torch.where(scale == 0, torch.full_like(scale, 1e-8), scale)
    q = torch.round(g / scale[:, None, :]) + 128.0
    q = q.clamp(0, 255).to(torch.int32).view(K, N)       # [K, N] in 0..255
    qw = torch.zeros((K // 4, N), dtype=torch.int32)
    for i in range(4):
        qw |= q[i::4, :] << (8 * i)
    qz = torch.full((K // GS, N // 4), 0x7F7F7F7F, dtype=torch.int32)   # 127 in every byte
    return qw.contiguous(), qz.contiguous(), scale.to(torch.float16).contiguous()


def unpack_int8_gptq(qw, qz, sc):
    """inverse of pack (for the self-test); returns fp32 [N, K] dequantized."""
    KW, N = qw.shape; K = KW * 4
    q = torch.empty((K, N), dtype=torch.int32)
    for i in range(4):
        q[i::4, :] = (qw >> (8 * i)) & 0xFF
    z = torch.empty((qz.shape[0], N), dtype=torch.int32)
    for i in range(4):
        z[:, i::4] = ((qz >> (8 * i)) & 0xFF) + 1           # stored zero-1
    w = (q.float() - z.float().repeat_interleave(GS, dim=0)) * sc.float().repeat_interleave(GS, dim=0)
    return w.T.contiguous()


def selftest():
    idx = json.load(open(f"{SEALED}/model.safetensors.index.json"))["weight_map"]
    k = "model.language_model.layers.2.linear_attn.in_proj_qkv"
    with safe_open(f"{SEALED}/{idx[k + '.qweight']}", "pt") as h:
        qw, qz, sc = h.get_tensor(k + ".qweight"), h.get_tensor(k + ".qzeros"), h.get_tensor(k + ".scales")
    print(f"  sealed {k.split('.')[-2]}: qweight {list(qw.shape)} qzeros {list(qz.shape)} scales {list(sc.shape)}")
    assert (qz == 0x7F7F7F7F).all(), "sealed int8 is not symmetric zero=128"
    w = unpack_int8_gptq(qw, qz, sc)                     # dequantized [N, K]
    qw2, qz2, sc2 = pack_int8_gptq(w.to(torch.bfloat16).float())
    # repacking a dequantized tensor must reproduce q exactly when scales agree
    same_q = torch.equal(qw2, qw)
    print(f"  roundtrip qweight identical: {same_q}   qzeros identical: {torch.equal(qz2, qz)}")
    if not same_q:
        # scales may differ (RTN max-abs vs AutoRound-tuned); compare dequantized values instead
        w2 = unpack_int8_gptq(qw2, qz2, sc2)
        print(f"  dequant rel err after repack: {((w2 - w).norm() / w.norm()).item():.2e}  (RTN scale != tuned scale is expected)")
    # RTN quality on a real bf16 dense tensor from the source
    sidx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    key = "model.language_model.layers.2.linear_attn.out_proj.weight"
    with safe_open(f"{SRC}/{sidx[key]}", "pt") as h:
        wb = h.get_tensor(key)
    qw3, qz3, sc3 = pack_int8_gptq(wb)
    w3 = unpack_int8_gptq(qw3, qz3, sc3)
    x = torch.randn(64, wb.shape[1])
    rel_w = ((w3 - wb.float()).norm() / wb.float().norm()).item()
    rel_y = ((x @ w3.T - x @ wb.float().T).norm() / (x @ wb.float().T).norm()).item()
    print(f"  RTN int8 g128 on {key.split('.')[-2]} {list(wb.shape)}: weight rel err {rel_w:.2e}, output rel err {rel_y:.2e}")
    print("  SELFTEST OK" if rel_y < 0.02 else "  SELFTEST: error too large")


# ----------------------------------------------------------------- build
DENSE_MOD_RX = re.compile(
    r"^model\.language_model\.layers\.\d+\.(linear_attn\.out_proj|self_attn\.(q|k|v|o)_proj)$"
)
INT8_SPEC = {"bits": 8, "group_size": GS, "sym": True}


def write_config(out):
    """config.json for the requantized directory.

    SGLang's AutoRound config resolution (auto_round.py get_layer_config): an EXACT module
    name in extra_config wins over any regex, and for lm_head (outside block_name_to_quantize)
    the regex path resolves group_size to -1 unless it is given explicitly. So: rewrite the
    exact entries of every requantized module, and spell out group_size/sym for lm_head.
    """
    cfg = json.load(open(f"{SEALED}/config.json"))
    ex = cfg["quantization_config"]["extra_config"]
    new_ex = {r"^lm_head.*": dict(INT8_SPEC)}
    n_exact = 0
    for pat, v in ex.items():
        if pat == r"^lm_head.*":
            continue
        if DENSE_MOD_RX.match(pat):
            new_ex[pat] = dict(INT8_SPEC); n_exact += 1
        else:
            new_ex[pat] = v
    cfg["quantization_config"]["extra_config"] = new_ex
    json.dump(cfg, open(f"{out}/config.json", "w"), indent=2)
    print(f"  config.json: {n_exact} exact modules -> int8 g{GS}, lm_head regex -> int8 g{GS}")


def build(out, dry):
    os.makedirs(out, exist_ok=True)
    idx = json.load(open(f"{SEALED}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    sidx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    dense = sorted(k for k in wm if DENSE_RX.match(k))
    print(f"  dense tensors to requantize: {len(dense)}")
    files = sorted(set(wm.values()))
    new_map = {}
    total = 0
    # 1. copy every non-dense tensor shard by shard (streaming)
    for f in files:
        keep = [k for k, v in wm.items() if v == f and not DENSE_RX.match(k)]
        print(f"  {f}: keeping {keep and len(keep)} tensors")
        if dry:
            continue
        tensors = {}
        with safe_open(f"{SEALED}/{f}", "pt") as h:
            meta = h.metadata()
            for k in keep:
                tensors[k] = h.get_tensor(k)
        save_file(tensors, f"{out}/{f}", metadata=meta or {"format": "pt"})
        for k in keep:
            new_map[k] = f
        total += sum(t.numel() * t.element_size() for t in tensors.values())
        del tensors
    # 2. requantize the dense tensors from the bf16 source into new shards
    shard, shard_bytes, n_out = {}, 0, 0
    def flush():
        nonlocal shard, shard_bytes, n_out
        if not shard:
            return
        name = f"model-int8dense-{n_out:03d}.safetensors"
        save_file(shard, f"{out}/{name}", metadata={"format": "pt"})
        for k in shard:
            new_map[k] = name
        n_out += 1; shard, shard_bytes = {}, 0
    t0 = time.time()
    for i, k in enumerate(dense, 1):
        base = k[: -len(".weight")]
        if dry:
            continue
        with safe_open(f"{SRC}/{sidx[k]}", "pt") as h:
            wb = h.get_tensor(k)
        qw, qz, sc = pack_int8_gptq(wb)
        shard[base + ".qweight"], shard[base + ".qzeros"], shard[base + ".scales"] = qw, qz, sc
        shard_bytes += qw.numel() * 4 + qz.numel() * 4 + sc.numel() * 2
        total += qw.numel() * 4 + qz.numel() * 4 + sc.numel() * 2
        if shard_bytes > 4e9:
            flush()
        if i % 20 == 0:
            print(f"    {i}/{len(dense)}  {time.time()-t0:.0f} s", flush=True)
    flush()
    if dry:
        return
    # 3. index + config
    json.dump({"metadata": {"total_size": total}, "weight_map": new_map},
              open(f"{out}/model.safetensors.index.json", "w"), indent=2)
    for f in os.listdir(SEALED):
        if f.endswith((".json", ".txt", ".jinja", ".py")) and f not in ("model.safetensors.index.json", "config.json", "MANIFEST.json"):
            shutil.copy2(f"{SEALED}/{f}", f"{out}/{f}")
    write_config(out)
    print(f"\n  wrote {out}: {total/1e9:.1f} GB, {len(new_map)} tensors, {n_out} int8 shards")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config-only", action="store_true", help="rewrite config.json of --out only")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.out and a.config_only:
        write_config(a.out)
    elif a.out:
        build(a.out, a.dry_run)
    else:
        print(__doc__)
