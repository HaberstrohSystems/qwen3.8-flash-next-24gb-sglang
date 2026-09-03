#!/usr/bin/env python3
"""
Seal a finished quantization: checksums, provenance, write protection.

Days of compute are easy to lose to a single careless command. This makes that
structurally harder:

  * The output directory carries a timestamp and is never overwritten.
  * After sealing, the directory is write-protected (chmod a-w), so `rm -rf`
    prompts instead of deleting silently.
  * MANIFEST.json records a SHA-256 per shard plus provenance - which source
    checkpoint, which AutoRound version, which patches were active - so the
    checkpoint can later be proven unchanged.

  python3 09_seal_output.py <dir> --seal          # seal
  python3 09_seal_output.py <dir> --check         # re-verify checksums
  python3 09_seal_output.py <dir> --unseal        # lift write protection
"""
import argparse, hashlib, json, os, stat, subprocess, sys, time


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect(d):
    return sorted(f for f in os.listdir(d)
                  if os.path.isfile(os.path.join(d, f)) and f != "MANIFEST.json")


def provenance(d):
    p = {"versiegelt_am": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
         "verzeichnis": os.path.abspath(d)}
    cfgp = os.path.join(d, "config.json")
    if os.path.exists(cfgp):
        c = json.load(open(cfgp))
        q = c.get("quantization_config") or {}
        p["architektur"] = (c.get("architectures") or ["?"])[0]
        p["quant"] = {k: q.get(k) for k in
                      ("quant_method", "bits", "group_size", "sym", "packing_format")}
        p["extra_config_eintraege"] = len(q.get("extra_config") or {})
    try:
        import auto_round, transformers, torch
        p["versionen"] = {"auto_round": auto_round.__version__,
                          "transformers": transformers.__version__,
                          "torch": torch.__version__}
    except Exception:
        pass
    # Which patches were active during the run?
    try:
        import importlib.util
        base = os.path.dirname(importlib.util.find_spec("auto_round").origin)
        marks = {
            "group_size_collision": ("compressors/utils.py",
                                     "[qwen4-exp] group_size collision"),
            "no_3d_contiguous": ("modeling/fused_moe/moe_experts_interface.py",
                                 "[qwen4-exp] no 3D contiguous"),
            "p_dtype_fp16": ("wrapper.py", "[qwen4-exp] p_dtype fp16"),
        }
        p["auto_round_patches"] = {
            k: (m in open(os.path.join(base, f)).read())
            for k, (f, m) in marks.items() if os.path.exists(os.path.join(base, f))}
    except Exception:
        pass
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seal", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--unseal", action="store_true")
    a = ap.parse_args()
    d = a.dir
    mp = os.path.join(d, "MANIFEST.json")

    if a.unseal:
        n = 0
        for root, dirs, files in os.walk(d):
            for x in dirs + files:
                p = os.path.join(root, x)
                os.chmod(p, os.stat(p).st_mode | stat.S_IWUSR); n += 1
        os.chmod(d, os.stat(d).st_mode | stat.S_IWUSR)
        print(f"  write protection lifted ({n} entries)\n"); return

    if a.check:
        if not os.path.exists(mp):
            sys.exit("  [!] no MANIFEST.json - never sealed?")
        man = json.load(open(mp))
        # Manifests written by an earlier revision of this script used German
        # key names. Read both so an already-sealed directory still verifies.
        entries = man.get("files", man.get("dateien"))
        if entries is None:
            sys.exit("  [!] MANIFEST.json has neither 'files' nor 'dateien'")
        bad = miss = 0
        print(f"\n=== Checking {len(entries)} files ===")
        for f, rec in sorted(entries.items()):
            p = os.path.join(d, f)
            if not os.path.exists(p):
                print(f"  MISSING    {f}"); miss += 1; continue
            if os.path.getsize(p) != rec["bytes"]:
                print(f"  SIZE       {f}"); bad += 1; continue
            if sha256(p) != rec["sha256"]:
                print(f"  CHECKSUM   {f}"); bad += 1
        print(f"\n  {len(entries)-bad-miss} unchanged, {bad} changed, {miss} missing")
        print(f"  {'INTACT' if not (bad or miss) else 'DAMAGED'}\n")
        sys.exit(1 if (bad or miss) else 0)

    # --- seal ---
    files = collect(d)
    if not files:
        sys.exit(f"  [!] {d} is empty")
    print(f"\n=== Sealing: {d} ===")
    print(f"  {len(files)} files, computing checksums ...")
    rec, total = {}, 0
    for i, f in enumerate(files, 1):
        p = os.path.join(d, f)
        sz = os.path.getsize(p)
        rec[f] = {"bytes": sz, "sha256": sha256(p)}
        total += sz
        if i % 10 == 0 or i == len(files):
            print(f"    {i}/{len(files)}  {total/1e9:.1f} GB", flush=True)
    man = {"provenance": provenance(d), "bytes_total": total, "files": rec}
    json.dump(man, open(mp, "w"), indent=2, ensure_ascii=False)

    for f in files + ["MANIFEST.json"]:
        p = os.path.join(d, f)
        os.chmod(p, os.stat(p).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    os.chmod(d, os.stat(d).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    print(f"\n  {total/1e9:.1f} GB sealed, MANIFEST.json written")
    print(f"  The directory is now write-protected.")
    print(f"  verify : python3 09_seal_output.py {d} --check")
    print(f"  unseal : python3 09_seal_output.py {d} --unseal\n")


if __name__ == "__main__":
    main()
