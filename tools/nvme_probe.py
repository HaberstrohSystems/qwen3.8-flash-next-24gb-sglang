#!/usr/bin/env python3
"""
Can the NVMe sustain the PLE access pattern?

The pattern is known: 16 hash-scattered rows of 160 bytes per token, no
spatial locality, and about a 42% cache hit rate as the upper bound. The open
question is whether the remainder can come off the disk in time.

This measures the WORST CASE: O_DIRECT, bypassing the page cache. In practice
roughly 40% is served from cache, so the numbers here are pessimistic.

Run this before assuming the mmap PLE path will work on your hardware.

  python3 nvme_probe.py --file <large file> --tokens 512 --qd 16
"""
import argparse, ctypes, mmap, os, random, statistics, sys, threading, time

ALIGN = 4096


def aligned_buf(n):
    return mmap.mmap(-1, ((n + ALIGN - 1) // ALIGN) * ALIGN)


def probe(path, n_reads, qd, direct=True, row=160):
    size = os.path.getsize(path)
    flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
    fd = os.open(path, flags)
    try:
        max_off = (size - ALIGN) // ALIGN
        offs = [random.randrange(max_off) * ALIGN for _ in range(n_reads)]
        lat, lock = [], threading.Lock()
        idx = [0]

        def worker():
            buf = aligned_buf(ALIGN)
            while True:
                with lock:
                    i = idx[0]
                    if i >= len(offs): return
                    idx[0] += 1
                t0 = time.perf_counter()
                os.preadv(fd, [buf], offs[i])
                dt = time.perf_counter() - t0
                with lock: lat.append(dt)

        t0 = time.perf_counter()
        ts = [threading.Thread(target=worker) for _ in range(qd)]
        for t in ts: t.start()
        for t in ts: t.join()
        wall = time.perf_counter() - t0
    finally:
        os.close(fd)
    lat.sort()
    return {"iops": len(offs)/wall, "wall": wall,
            "p50": lat[len(lat)//2]*1e6, "p99": lat[int(len(lat)*0.99)]*1e6,
            "mean": statistics.mean(lat)*1e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--tokens", type=int, default=512, help="Tokens pro Decode-Schritt (Batch)")
    ap.add_argument("--rows-per-token", type=int, default=16)
    ap.add_argument("--reads", type=int, default=4000)
    args = ap.parse_args()

    f = args.file
    if not f:
        import glob
        c = sorted(glob.glob(os.path.expanduser("~/quant/src-flashnext/*.safetensors")),
                   key=os.path.getsize, reverse=True)
        if not c: sys.exit("  no large file found; pass --file")
        f = c[0]
    print(f"\n=== NVMe-Sondierung ===")
    print(f"  file: {os.path.basename(f)}  ({os.path.getsize(f)/1e9:.1f} GB)")
    print(f"  WORST CASE: O_DIRECT, am Seiten-Cache vorbei\n")
    print(f"  {'QD':>4}  {'IOPS':>10}  {'p50 us':>8}  {'p99 us':>8}  {'MB/s (4K)':>10}")
    print("  " + "-"*50)
    res = {}
    for qd in (1, 4, 16, 32, 64):
        r = probe(f, args.reads, qd)
        res[qd] = r
        print(f"  {qd:>4}  {r['iops']:>10,.0f}  {r['p50']:>8.0f}  {r['p99']:>8.0f}  {r['iops']*4096/1e6:>10.1f}")

    print(f"\n=== What does this mean for the PLE? ===")
    rpt = args.rows_per_token
    for label, toks, tps in [("batch 1,  30 tok/s", 1, 30),
                             ("batch 8,  120 tok/s total", 8, 120),
                             ("batch 32, 400 tok/s total", 32, 400),
                             ("prefill 8k (chunked)", 8192, 0)]:
        need_reads = toks * rpt
        if tps:
            rate = tps * rpt
            best = max(r["iops"] for r in res.values())
            print(f"  {label:<28} {rate:>9,.0f} reads/s  = {100*rate/best:5.2f} % of the NVMe")
        else:
            best_qd = max(res, key=lambda k: res[k]["iops"])
            t = need_reads / res[best_qd]["iops"]
            print(f"  {label:<28} {need_reads:>9,} reads      = {t*1000:6.1f} ms once")
    print()


if __name__ == "__main__":
    main()
