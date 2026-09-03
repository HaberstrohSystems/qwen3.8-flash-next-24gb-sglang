#!/usr/bin/env python3
"""
Rewrite the FP8 checkpoint into stacked bf16, in a single pass.

TWO PROBLEMS, ONE PASS
----------------------
1. Qwen's FP8 repo stores the experts INDIVIDUALLY:
       layers.N.mlp.experts.<i>.gate_proj.weight            (640, 2560)  fp8
       layers.N.mlp.experts.<i>.gate_proj.weight_scale_inv  (5, 20)      f32
   transformers expects them STACKED. Assembling 512 separate file regions
   into one tensor cannot be zero-copy, so the loader materializes the entire
   expert stack as anonymous memory (~120 GB) and the machine dies.

   Measured on the real checkpoint: 112,007 materializations for 288
   checkpoint tensors, with anonymous memory growing monotonically until the
   OOM. The bf16 release of the same model already stores stacked, which is
   why that one loads mmap-backed.

2. AutoRound cannot tune fp8:
       NotImplementedError: "min_cuda" not implemented for 'Float8_e4m3fn'
   Unstacking also moves the weights into per-expert Linears while the block
   scales (gate_up_proj_scale_inv) stay behind on the parent module, so
   dequantizing inside the run means rebuilding that mapping by hand.

One rewrite solves both:
    gate_up_proj  [512, 1280, 2560]  bf16   (gate|up concatenated, scales folded)
    down_proj     [512, 2560,  640]  bf16

Block scales are 128x128 and divide evenly:
    w_bf16 = (w_fp8.float().reshape(nb_o,128,nb_i,128) * s[:,None,:,None]).reshape(o,i)

DISK
----
Source shards are deleted as soon as they have been read to completion, so
free space falls monotonically instead of peaking. --keep-src leaves
everything in place (needs ~136 GB more).

  python3 02_prepare_source.py ~/quant/src-flashnext -o ~/quant/src-bf16
  python3 02_prepare_source.py ... --dry-run
  python3 02_prepare_source.py ... --verify        # spot-check against the source
"""
import argparse, gc, json, os, re, shutil, sys
from collections import defaultdict

EXP_RX = re.compile(r"^(?P<prefix>.*\.experts)\.(?P<idx>\d+)\."
                    r"(?P<proj>gate_proj|up_proj|down_proj)\."
                    r"(?P<kind>weight|weight_scale_inv)$")
BLOCK = 128


def plan(src):
    wm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    groups = defaultdict(dict)      # (prefix, proj, kind) -> {i: key}
    passthrough, skipped_ple, missing = [], 0, []
    for k in wm:
        m = EXP_RX.match(k)
        if m:
            groups[(m["prefix"], m["proj"], m["kind"])][int(m["idx"])] = k
            continue
        if "ngram_embedding" in k:
            skipped_ple += 1          # lives in its own mmap file
            continue
        if not os.path.exists(os.path.join(src, wm[k])):
            missing.append(k)
            continue
        passthrough.append(k)
    return wm, groups, passthrough, skipped_ple, missing


def dequant(w, s):
    """fp8 + 128x128-Blockskalen -> bf16."""
    import torch
    o, i = w.shape
    nb_o, nb_i = s.shape
    assert o == nb_o * BLOCK and i == nb_i * BLOCK, f"{w.shape} does not match {s.shape}"
    x = w.float().reshape(nb_o, BLOCK, nb_i, BLOCK) * s.float().reshape(nb_o, 1, nb_i, 1)
    return x.reshape(o, i).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--shard-gb", type=float, default=5.0)
    ap.add_argument("--keep-src", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    wm, groups, passthrough, n_ple, missing = plan(a.src)
    prefixes = sorted({g[0] for g in groups})
    n_exp = len(groups[(prefixes[0], "gate_proj", "weight")]) if prefixes else 0
    print(f"\n=== Preparing the source ===")
    print(f"  {a.src}\n  -> {a.out}")
    print(f"  expert blocks : {len(prefixes)} x {n_exp} experts")
    print(f"  passed through: {len(passthrough)}")
    print(f"  PLE skipped   : {n_ple}  (lives in its own mmap file)")
    if missing:
        print(f"  [!] no source file: {len(missing)}, e.g. {missing[:2]}")
    if a.verify:
        return verify(a, wm, groups)
    if a.dry_run:
        print("\n  --dry-run, stopping here.\n"); return

    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.src):
        p = os.path.join(a.src, f)
        if os.path.isfile(p) and not f.endswith(".safetensors") \
           and f != "model.safetensors.index.json":
            shutil.copy2(p, os.path.join(a.out, f))
    # The weights are bf16 afterwards - an fp8 quantization_config would lie
    cp = os.path.join(a.out, "config.json")
    cfg = json.load(open(cp))
    if cfg.pop("quantization_config", None) is not None:
        print("  removed quantization_config (weights are bf16 now)")
    if "text_config" in cfg:
        cfg["text_config"].pop("quantization_config", None)
        cfg["text_config"]["dtype"] = "bfloat16"
    cfg["dtype"] = "bfloat16"
    json.dump(cfg, open(cp, "w"), indent=4)

    remaining = defaultdict(int)
    for k in wm:
        remaining[wm[k]] += 1
    # PLE tensors do not count; their files are absent anyway
    for k in wm:
        if "ngram_embedding" in k or not os.path.exists(os.path.join(a.src, wm[k])):
            remaining[wm[k]] -= 1

    handles = {}
    def get(key):
        f = wm[key]
        if f not in handles:
            handles[f] = safe_open(os.path.join(a.src, f), framework="pt")
        return handles[f].get_tensor(key)

    def done(key):
        f = wm[key]
        remaining[f] -= 1
        if remaining[f] <= 0 and not a.keep_src:
            handles.pop(f, None); gc.collect()
            p = os.path.join(a.src, f)
            if os.path.exists(p):
                os.remove(p)

    LIMIT = int(a.shard_gb * 1e9)
    shard, size, no, weight_map, total = {}, 0, 0, {}, 0

    def flush():
        nonlocal shard, size, no, total
        if not shard: return
        no += 1
        name = f"model-{no:05d}.safetensors"
        save_file(shard, os.path.join(a.out, name), metadata={"format": "pt"})
        for k, v in shard.items():
            weight_map[k] = name
            total += v.numel() * v.element_size()
        st = os.statvfs(a.out)
        print(f"  -> {name}  {size/1e9:5.2f} GB   free {st.f_bavail*st.f_frsize/1e9:.0f} GB",
              flush=True)
        shard, size = {}, 0

    def add(k, t):
        nonlocal size
        shard[k] = t; size += t.numel() * t.element_size()
        if size >= LIMIT: flush()

    print(f"\n  Stacking and dequantizing ...")
    for n, prefix in enumerate(prefixes, 1):
        for tgt, srcs, cat in (("gate_up_proj", ("gate_proj", "up_proj"), True),
                               ("down_proj", ("down_proj",), False)):
            parts = []
            for i in range(n_exp):
                pieces = []
                for s in srcs:
                    wk = groups[(prefix, s, "weight")][i]
                    sk = groups[(prefix, s, "weight_scale_inv")][i]
                    pieces.append(dequant(get(wk), get(sk)))
                    done(wk); done(sk)
                parts.append(torch.cat(pieces, dim=0) if cat else pieces[0])
            add(f"{prefix}.{tgt}", torch.stack(parts, dim=0).contiguous())
            del parts
            gc.collect()
        print(f"  block {n:>3}/{len(prefixes)}", flush=True)

    print(f"\n  Passing the rest through ...")
    for j, k in enumerate(passthrough, 1):
        add(k, get(k)); done(k)
        if j % 300 == 0:
            print(f"    {j}/{len(passthrough)}", flush=True)
    flush(); handles.clear()

    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    print(f"\n  {no} shards, {total/1e9:.1f} GB\n  -> {a.out}\n")


def verify(a, wm, groups, n=3):
    """Spot-check: stacked bf16 tensor against an independently dequantized source."""
    import torch, random
    from safetensors import safe_open
    owm = json.load(open(os.path.join(a.out, "model.safetensors.index.json")))["weight_map"]
    prefixes = sorted({g[0] for g in groups})
    rnd = random.Random(0); bad = 0
    print("\n=== Stichproben ===")
    for prefix in rnd.sample(prefixes, min(3, len(prefixes))):
        for tgt, srcs, cat in (("gate_up_proj", ("gate_proj", "up_proj"), True),
                               ("down_proj", ("down_proj",), False)):
            key = f"{prefix}.{tgt}"
            if key not in owm:
                print(f"  {key}: FEHLT im Ziel"); bad += 1; continue
            with safe_open(os.path.join(a.out, owm[key]), framework="pt") as h:
                st = h.get_tensor(key)
            for i in rnd.sample(range(st.shape[0]), n):
                ref = []
                for s in srcs:
                    wk = groups[(prefix, s, "weight")][i]
                    sk = groups[(prefix, s, "weight_scale_inv")][i]
                    with safe_open(os.path.join(a.src, wm[wk]), framework="pt") as h:
                        w = h.get_tensor(wk)
                    with safe_open(os.path.join(a.src, wm[sk]), framework="pt") as h:
                        sc = h.get_tensor(sk)
                    ref.append(dequant(w, sc))
                r = torch.cat(ref, dim=0) if cat else ref[0]
                ok = torch.equal(r, st[i])
                bad += 0 if ok else 1
                print(f"  {tgt:<13} Experte {i:>3}: {'BITGENAU' if ok else 'ABWEICHUNG'}")
    print(f"\n  {'ALLE STICHPROBEN BITGENAU' if not bad else f'{bad} ABWEICHUNGEN'}\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
