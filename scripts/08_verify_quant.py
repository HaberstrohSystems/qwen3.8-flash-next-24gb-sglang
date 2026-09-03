#!/usr/bin/env python3
"""
Size audit after quantization.

Checks whether the result ACTUALLY fits the budget. AutoScheme averages
avg_bits over quantized layers only; tensors pinned to bf16 are added on top,
so a run that reports the target bpw can still be too large on disk.

  python3 08_verify_quant.py ./out/model
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B
from importlib.machinery import SourceFileLoader

_insp = SourceFileLoader("insp", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "01_inspect_model.py")).load_module()
GIB = 1024 ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-d", "--discovered", default="discovered.json")
    # Compute the budget here rather than reading it from discovered.json:
    # that file may come from an earlier configuration and would then report
    # "over budget" incorrectly.
    ap.add_argument("--ram-reserve", type=float, default=10.0)
    ap.add_argument("--ple-ring", type=float, default=0.0625)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--max-model-len", type=int, default=32768)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.path, "*.safetensors")))
    if not files:
        print(f"  [!] Keine .safetensors in {a.path}")
        sys.exit(1)

    on_disk_gib = sum(os.path.getsize(f) for f in files) / GIB

    from safetensors import safe_open
    per_group_bytes, per_group_params = defaultdict(int), defaultdict(int)
    DT = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I32": 4, "U8": 1, "I8": 1, "I64": 8}

    for f in files:
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                sl = h.get_slice(k)
                shape, dt = list(sl.get_shape()), sl.get_dtype()
                n = 1
                for d in shape:
                    n *= d
                g = _insp.classify(k)
                per_group_bytes[g] += n * DT.get(dt, 2)
                # packed int32 containers: parameter count is estimated differently
                per_group_params[g] += n

    print(f"\n=== Audit: {a.path} ===\n")
    print(f"  {'group':<12} {'on disk':>12} {'%':>7}")
    print("  " + "-" * 34)
    tot = sum(per_group_bytes.values())
    for g, b in sorted(per_group_bytes.items(), key=lambda x: -x[1]):
        print(f"  {g:<12} {b/GIB:>10.2f} GiB {100*b/tot:>6.1f}%")
    print(f"  {'TOTAL':<12} {on_disk_gib:>10.2f} GiB")

    # Budget from the current operating mode, not from a stale file
    cfgp = os.path.join(a.path, "config.json")
    if not os.path.exists(cfgp):
        print("\n  (no config.json - no budget comparison)\n"); return
    cfg = json.load(open(cfgp))
    if a.headless:
        B.VRAM_DESKTOP_GIB, B.MEM_FRACTION_STATIC = 0.10, 0.95
    B.RAM_RESERVED_GIB = a.ram_reserve + a.ple_ring
    resident_b = (sum(per_group_params.values()) - per_group_params["ngram"]
                  - per_group_params["token_embd"]) / 1e9
    _b = B.compute(cfg, resident_b, max_model_len=a.max_model_len)
    bud = {"total_weights_gib": _b.total_weights_gib}
    disc = {"params_b_resident": resident_b}
    print(f"\n  mode: headless={a.headless}, OS reserve {a.ram_reserve} GiB, "
          f"PLE staging {a.ple_ring} GiB, context {a.max_model_len}")
    mmap_gib = (per_group_bytes["ngram"] + per_group_bytes["token_embd"]) / GIB
    resident_gib = on_disk_gib - mmap_gib
    limit = bud["total_weights_gib"]

    print(f"\n  of which mmap-able (ngram+embd): {mmap_gib:>8.2f} GiB  (does not count)")
    print(f"  resident requirement          : {resident_gib:>8.2f} GiB")
    print(f"  budget                        : {limit:>8.2f} GiB")
    print(f"  effective bpw (resident)      : "
          f"{B.bpw_for(disc['params_b_resident'], resident_gib):>8.3f}")

    head = limit - resident_gib
    if head >= 0.5:
        print(f"\n  [OK] {head:.2f} GiB headroom. Optional: raise avg_bits by "
              f"{head*GIB*8/(disc['params_b_resident']*1e9):.2f} and requantize.")
    elif head >= 0:
        print(f"\n  [OK] {head:.2f} GiB headroom. It fits.")
    else:
        print(f"\n  [!!] {-head:.2f} GiB OVER BUDGET.")
        print(f"       Work through these in order:")
        print(f"       1. check whether a bf16 tensor is unexpectedly large (table above)")
        print(f"       2. --group-size 256 for exp_upgate")
        print(f"       3. lower the context -> smaller KV budget -> more room for weights")
        print(f"       4. only as a last resort: lower avg_bits")
        sys.exit(2)

    # Sanity check: is the router really bf16?
    r = per_group_bytes["router"] / max(per_group_params["router"], 1)
    if per_group_params["router"] and r < 1.9:
        print(f"\n  [!] The router appears quantized ({r:.2f} bytes/param, expected 2.0).")
        print(f"      A layer_config regex is not matching. The router must stay bf16.")
    print()


if __name__ == "__main__":
    main()
