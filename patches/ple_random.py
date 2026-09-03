#!/usr/bin/env python3
"""PLE n-gram table: no readahead on the 51 GB mmap / pread path.

Every PLE row is 160 bytes at a random offset. mmap faults default to read-around (128 KB+),
so a 1024-token prefill chunk pulls ~1 GB of page cache for a few MB of rows; on this box
(2 GB of free RAM next to 24 GB of pinned weights) that churn drives memory pressure past the
systemd-oomd limit at ~55k-token prompts. MADV_RANDOM / POSIX_FADV_RANDOM make faults 4 KB.

  python3 ple_random.py --check | apply | revert
"""
import os, sys
SG = os.path.join(os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")), "python/sglang")
Q = f"{SG}/srt/models/qwen4_exp.py"
EDITS = [
  # prefill gather: drop the just-read PLE pages from the page cache. Random text touches a new 4 KB
  # page per row (~1M pages for a 250k-token prompt); keeping them evicts the host's hot pages and
  # trips systemd-oomd. rows is a copy (numpy fancy indexing), so the pages are not needed afterwards.
  (Q, """            rows = torch.from_numpy(self._mm[local.numpy()])      # uint8, (N, dim*b)
""", """            rows = torch.from_numpy(self._mm[local.numpy()])      # uint8, (N, dim*b)
            if local.numel() >= 512:
                try:
                    os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except Exception:                              # pragma: no cover
                    pass
"""),
  (Q, """        self._mm = np.memmap(path, dtype=np.uint8, mode="r",
                             shape=(self._rows, self._dim * itemsize))
""", """        self._mm = np.memmap(path, dtype=np.uint8, mode="r",
                             shape=(self._rows, self._dim * itemsize))
        try:                                   # rows are 160 B at random offsets: no read-around
            import mmap as _mmap
            self._mm._mmap.madvise(_mmap.MADV_RANDOM)
        except Exception as _ex:               # pragma: no cover
            logger.warning("Qwen4 PLE: madvise(MADV_RANDOM) failed: %s", _ex)
"""),
  (Q, """        self._fd = os.open(path, os.O_RDONLY)
""", """        self._fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_RANDOM)
        except Exception as _ex:               # pragma: no cover
            logger.warning("Qwen4 PLE: posix_fadvise(RANDOM) failed: %s", _ex)
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
        print("  [!] mismatch"); check(); return
    for (p, a, b), (_, pr, ap) in zip(EDITS, st):
        if not ap:
            t = open(p, encoding="utf-8").read(); open(p, "w", encoding="utf-8").write(t.replace(a, b, 1))
    print("  applied (PLE mmap/pread without readahead)")


def revert():
    for (p, a, b), (_, pr, ap) in zip(EDITS, state()):
        if ap:
            t = open(p, encoding="utf-8").read(); open(p, "w", encoding="utf-8").write(t.replace(b, a, 1))
    print("  reverted")


if __name__ == "__main__":
    {"--check": check, "apply": apply, "revert": revert}[sys.argv[1] if len(sys.argv) > 1 else "--check"]()
