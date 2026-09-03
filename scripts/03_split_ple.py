#!/usr/bin/env python3
"""
Serve the n-gram/PLE table from NVMe instead of from RAM.

WHY
---
The table holds 51.2B parameters. In the FP8 checkpoint that is 51.2 GB; after
dequantization to bf16, 95 GB. Neither fits in 30 GB of host RAM, for
quantization or for serving.

It does not need to be in memory at all. The structure is ideal for this:

    128 shards of [2,500,012, 160] in F8_E4M3, with no block scales
    = 16 n-gram heads ((ngram_size-1) x heads_per_ngram = 2 x 8)
      of roughly 20M entries each, 160 fp8 values per entry
    -> 16 rows of 160 bytes are read per token = 2560 bytes
    -> at 30 tok/s that is 75 KB/s of random reads

In the model it hangs off a single, simple module:

    self.ngram_embedding = nn.Embedding(320_001_536, 160)

That is what gets replaced here with an mmap-backed equivalent. The
activations stay BIT-IDENTICAL and the memory cost drops to zero. transformers
anticipates exactly this:
    _no_placement_params = ["ple.ple_embedding.ngram_embedding.weight"]

USAGE
-----
  # 1. Extract the table from the checkpoint (once, ~51 GB)
  python3 03_split_ple.py split ~/quant/src-flashnext -o ~/quant/ple

  # 2. In the quantization run: install before loading
  from importlib.machinery import SourceFileLoader
  ple = SourceFileLoader("ple", "03_split_ple.py").load_module()
  ple.install("~/quant/ple")               # patches the module class
  model = AutoModel.from_pretrained(...)   # never allocates the 95 GB

  # 3. Self-test
  python3 03_split_ple.py verify ~/quant/ple --against ~/quant/src-flashnext

Serving additionally needs the mmap path inside SGLang; the file format
written here is what that path reads. See WRITEUP.md, finding 1.

NOTE: `weight_scale` is written into ple.json. It does not survive
quantization into the checkpoint, so the serving side has to read it from
here - a table served without its scale produces plausible-looking garbage.
"""
import argparse, glob, json, math, os, re, sys

SHARD_RX = re.compile(r"\.ngram_embedding\.shard_(\d+)\.weight$")
DTYPE_BYTES = {"F8_E4M3": 1, "F8_E5M2": 1, "BF16": 2, "F16": 2, "F32": 4}


# ══════════════════════════════════════════════════════════ the module

class MmapNGramEmbedding:
    """Drop-in for nn.Embedding that reads rows from disk via mmap.

    Deliberately NOT an nn.Module. That way it does not appear in
    state_dict()/named_parameters(), the loader does not try to write into it,
    and AutoRound does not mistake it for a quantization target.
    """

    def __init__(self, manifest_dir, device=None):
        import numpy as np, torch
        self.dir = manifest_dir
        man = json.load(open(os.path.join(manifest_dir, "ple.json")))
        self.man = man
        self.rows, self.dim = man["rows"], man["dim"]
        self.dtype_str = man["dtype"]
        self.itemsize = DTYPE_BYTES[self.dtype_str]
        self.scale = man.get("weight_scale")
        binp = os.path.join(manifest_dir, man["file"])
        expect = self.rows * self.dim * self.itemsize
        actual = os.path.getsize(binp)
        if actual != expect:
            raise ValueError(f"{binp}: {actual} bytes, expected {expect}")
        # uint8 view because numpy has no fp8; torch reinterprets it later
        self._mm = np.memmap(binp, dtype=np.uint8, mode="r",
                             shape=(self.rows, self.dim * self.itemsize))
        self._torch_dtype = {"F8_E4M3": torch.float8_e4m3fn,
                             "F8_E5M2": torch.float8_e5m2,
                             "BF16": torch.bfloat16,
                             "F16": torch.float16,
                             "F32": torch.float32}[self.dtype_str]
        self.device = device or torch.device("cpu")
        self.hits = 0

        class _WeightProxy:
            # Der Aufrufer macht ngram_ids.to(self.ngram_embedding.weight.device)
            def __init__(s, dev, dt, shape): s.device, s.dtype, s.shape = dev, dt, shape
        self.weight = _WeightProxy(self.device, self._torch_dtype, (self.rows, self.dim))

    def to(self, *a, **k):
        import torch
        for x in list(a) + list(k.values()):
            if isinstance(x, (str, torch.device)):
                self.device = torch.device(x)
                self.weight.device = self.device
        return self

    def __call__(self, idx):
        import numpy as np, torch
        shape = tuple(idx.shape)
        flat = idx.reshape(-1).to("cpu", torch.int64).numpy()
        raw = self._mm[flat]                                  # [N, dim*itemsize] uint8
        self.hits += flat.size
        t = torch.from_numpy(np.ascontiguousarray(raw))
        if self.itemsize == 1:
            t = t.view(self._torch_dtype)
        else:
            t = t.view(self._torch_dtype).reshape(-1, self.dim)
        out = t.to(torch.bfloat16)
        if self.scale is not None:
            out = out * float(self.scale)
        return out.reshape(*shape, self.dim).to(self.device)

    # so nn.Module.__setattr__ treats this as an attribute, not a submodule
    def parameters(self, *a, **k):  return iter(())
    def buffers(self, *a, **k):     return iter(())
    def children(self):             return iter(())
    def state_dict(self, *a, **k):  return {}
    def eval(self):                 return self
    def train(self, mode=True):     return self
    def __repr__(self):
        return (f"MmapNGramEmbedding({self.rows:,}, {self.dim}, {self.dtype_str}, "
                f"mmap={os.path.basename(self.man['file'])})")


def install(manifest_dir, device=None):
    """Patch Qwen4ExpTextNGramEmbedding so the table is never allocated."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M
    cls = M.Qwen4ExpTextNGramEmbedding
    if getattr(cls, "_mmap_ple_installed", False):
        print("  [i] mmap PLE already installed")
        return
    orig_init = cls.__init__
    shared = {}

    def patched_init(self, config, embedding_dim, layer_idx, ple_layer_index=0):
        orig_init(self, config, embedding_dim, layer_idx, ple_layer_index)
        want = (self.ngram_embedding.num_embeddings, self.ngram_embedding.embedding_dim)
        if "emb" not in shared:
            shared["emb"] = MmapNGramEmbedding(manifest_dir, device=device)
        emb = shared["emb"]
        if (emb.rows, emb.dim) != want:
            raise ValueError(f"PLE table mismatch: file has {(emb.rows, emb.dim)}, "
                             f"model expects {want}")
        # Remove the nn.Embedding from the module tree, or its parameters stay
        del self._modules["ngram_embedding"]
        object.__setattr__(self, "ngram_embedding", emb)
        print(f"  [mmap-PLE] {layer_idx}: {emb}")

    cls.__init__ = patched_init
    cls._mmap_ple_installed = True
    print(f"  mmap-PLE installiert aus {manifest_dir}")


# ══════════════════════════════════════════════════════════ Extraktion

def _iter_ple(src):
    """(shard_index, tensor) aus einem Checkpoint, ohne alles zu laden."""
    from safetensors import safe_open
    idxp = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(idxp):
        wm = json.load(open(idxp))["weight_map"]
        files = {}
        for k, f in wm.items():
            m = SHARD_RX.search(k)
            if m:
                files.setdefault(f, []).append((int(m.group(1)), k))
        scale_key = next((k for k in wm if k.endswith("ngram_embedding.weight_scale")), None)
        scale_file = wm.get(scale_key)
    else:
        files, scale_key, scale_file = {}, None, None
        for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
            with safe_open(f, framework="pt") as h:
                for k in h.keys():
                    m = SHARD_RX.search(k)
                    if m:
                        files.setdefault(os.path.basename(f), []).append((int(m.group(1)), k))
                    if k.endswith("ngram_embedding.weight_scale"):
                        scale_key, scale_file = k, os.path.basename(f)
    return files, scale_key, scale_file


def split_ple(src, out):
    import numpy as np, torch
    from safetensors import safe_open
    os.makedirs(out, exist_ok=True)

    files, scale_key, scale_file = _iter_ple(src)
    total = sum(len(v) for v in files.values())
    if not total:
        sys.exit(f"  [!] no ngram_embedding.shard_* in {src}")
    print(f"\n=== PLE herausloesen ===")
    print(f"  source : {src}")
    print(f"  shards : {total} across {len(files)} files")

    # Determine shard order and geometry
    order, dtype_str, dim, rows_per = {}, None, None, {}
    for f, entries in files.items():
        with safe_open(os.path.join(src, f), framework="pt") as h:
            for si, key in entries:
                sl = h.get_slice(key)
                sh, dt = list(sl.get_shape()), sl.get_dtype()
                order[si] = (f, key)
                rows_per[si] = sh[0]
                dtype_str = dtype_str or dt
                dim = dim or sh[1]
                if dt != dtype_str or sh[1] != dim:
                    sys.exit(f"  [!] inconsistent shards: {key} {sh} {dt}")
    assert sorted(order) == list(range(total)), "shard indices are not contiguous"
    rows = sum(rows_per.values())
    ib = DTYPE_BYTES[dtype_str]
    print(f"  geometry: {rows:,} x {dim}  {dtype_str}  = {rows*dim*ib/1e9:.1f} GB")

    scale = None
    if scale_key:
        with safe_open(os.path.join(src, scale_file), framework="pt") as h:
            scale = float(h.get_tensor(scale_key).reshape(-1)[0])
        print(f"  weight_scale: {scale}")

    binname = f"ple.{dtype_str.lower()}.bin"
    binp = os.path.join(out, binname)
    free = os.statvfs(out).f_bavail * os.statvfs(out).f_frsize
    need = rows * dim * ib
    if free < need * 1.02:
        sys.exit(f"  [!] zu wenig Platz: {free/1e9:.0f} GB frei, {need/1e9:.0f} GB noetig")

    print(f"  -> {binp}")
    written, offsets = 0, {}
    with open(binp, "wb", buffering=1024 * 1024) as fo:
        for si in range(total):
            f, key = order[si]
            with safe_open(os.path.join(src, f), framework="pt") as h:
                t = h.get_tensor(key)
            offsets[si] = written // (dim * ib)
            # roh als Bytes schreiben, unabhaengig vom dtype
            arr = t.contiguous().view(torch.uint8).numpy()
            fo.write(arr.tobytes())
            written += arr.size
            if si % 16 == 0 or si == total - 1:
                print(f"     Shard {si+1:>3}/{total}  {written/1e9:6.1f} GB", flush=True)

    man = {"file": binname, "rows": rows, "dim": dim, "dtype": dtype_str,
           "n_shards": total, "rows_per_shard": [rows_per[i] for i in range(total)],
           "shard_row_offsets": [offsets[i] for i in range(total)],
           "weight_scale": scale, "source": os.path.abspath(src),
           "bytes": written}
    json.dump(man, open(os.path.join(out, "ple.json"), "w"), indent=2)
    print(f"\n  Manifest: {os.path.join(out,'ple.json')}")
    print(f"  Fertig: {written/1e9:.1f} GB\n")


def verify(out, against=None, n=64):
    """Stichproben gegen den Original-Checkpoint."""
    import numpy as np, torch
    from safetensors import safe_open
    emb = MmapNGramEmbedding(out)
    print(f"\n=== Selbsttest ===\n  {emb}")
    if not against:
        idx = torch.randint(0, emb.rows, (4, 3, 16))
        v = emb(idx)
        print(f"  Gather {tuple(idx.shape)} -> {tuple(v.shape)} {v.dtype}")
        print(f"  Wertebereich: [{v.min().item():.4f}, {v.max().item():.4f}]\n")
        return

    man = json.load(open(os.path.join(out, "ple.json")))
    files, _, _ = _iter_ple(against)
    order = {}
    for f, entries in files.items():
        for si, key in entries:
            order[si] = (f, key)
    rng = np.random.default_rng(0)
    bad = 0
    for si in rng.choice(man["n_shards"], size=min(6, man["n_shards"]), replace=False):
        f, key = order[int(si)]
        with safe_open(os.path.join(against, f), framework="pt") as h:
            t = h.get_tensor(key)
        base = man["shard_row_offsets"][int(si)]
        local = rng.choice(t.shape[0], size=n, replace=False)
        ref = t[torch.from_numpy(local).long()].to(torch.bfloat16)
        if man["weight_scale"] is not None:
            ref = ref * float(man["weight_scale"])
        got = emb(torch.from_numpy(local).long() + base)
        same = torch.equal(ref, got)
        bad += 0 if same else 1
        print(f"  Shard {int(si):>3}: {n} Zeilen  {'BITGENAU' if same else 'ABWEICHUNG'}")
    print(f"\n  {'ALLE STICHPROBEN BITGENAU' if not bad else f'{bad} ABWEICHUNGEN'}\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    import torch  # noqa
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split");  s.add_argument("src"); s.add_argument("-o", "--out", required=True)
    v = sub.add_parser("verify"); v.add_argument("out");  v.add_argument("--against", default=None)
    a = ap.parse_args()
    if a.cmd == "split":
        split_ple(a.src, a.out)
    else:
        verify(a.out, a.against)
