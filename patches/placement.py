#!/usr/bin/env python3
"""Frequency-based expert placement, memory-neutral (v3, deferred + interleaved).

The hottest S experts of EVERY layer live on the GPU, the rest in pinned host memory,
without allocating new pinned memory and without transient VRAM growth.

Mechanism: residency is an address. Every (layer, tensor kind) carries an int64 table
addr[e] = row address of expert e, on the GPU or inside some host layer's pinned tensor.
The decode GEMV already indexes through such tables; the prefill streamer gets a
table-driven row gather. Values never change - only where they live.

Memory neutrality: a host layer copies its hot rows to a new GPU tensor and keeps its cold
rows in place; the hot rows' slots in its pinned tensor become free. A GPU layer writes its
cold rows into such free slots and repacks its hot rows into a small GPU tensor; its big
original is freed. At S=184 donated slots (30.75 layers x 184) equal the cold rows to place
(17.25 x 328). Layers are collected during loading and placed in one interleaved pass at the
last layer (host, host, GPU, host, host, GPU, ...), so slots exist before they are needed and
the transient VRAM is < 1 GB. The offloader's host/GPU split is discovered, not assumed.

Env: SGLANG_MOE_PLACEMENT=/path/expert_freq.pt   SGLANG_MOE_PLACEMENT_S=184
  python3 placement.py --check | apply | revert      (apply on top of ncontig_gemv.py)
"""
import os, sys
SG = os.path.join(os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")), "python/sglang")
M = f"{SG}/srt/layers/quantization/moe_wna16.py"
S = f"{SG}/srt/layers/moe/expert_stream.py"

EDITS = [
  # ---------------------------------------------------------------- expert_stream.py
  (S, """_STAGING: Dict[Tuple, torch.Tensor] = {}
""", """_STAGING: Dict[Tuple, torch.Tensor] = {}


@triton.jit
def _gather_rows_tab_kernel(tab_ptr, idx_ptr, out_ptr, row_bytes, BLOCK: tl.constexpr):
    \"\"\"Row gather through an address table: expert id -> int64 row address (GPU or pinned host).\"\"\"
    e = tl.load(idx_ptr + tl.program_id(0)).to(tl.int64)
    base = tl.load(tab_ptr + e)
    src = base.to(tl.pointer_type(tl.uint8))
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < row_bytes
    vals = tl.load(src + offs, mask=mask, other=0)
    tl.store(out_ptr + tl.program_id(0).to(tl.int64) * row_bytes + offs, vals, mask=mask)
"""),
  (S, """        out: Dict[str, torch.Tensor] = {}
        for name in self.names:
            src = getattr(self.layer, name)
            src = src.data if isinstance(src, torch.nn.Parameter) else src
            buf = _staging(name, k, src.shape[0], tuple(src.shape[1:]),
                           src.dtype, self.device)
            if src.is_cuda:
""", """        out: Dict[str, torch.Tensor] = {}
        placed = getattr(self.layer, "_placed", None)
        for name in self.names:
            if placed is not None:
                proto = placed["proto"][name]          # a hot GPU tensor: shape[1:], dtype
                buf = _staging(name, k, placed["E"], tuple(proto.shape[1:]), proto.dtype, self.device)
                row_bytes = proto[0].numel() * proto.element_size()
                BLOCK = 1024
                _gather_rows_tab_kernel[(k, triton.cdiv(row_bytes, BLOCK))](
                    placed["addr"][name], uniq, buf.view(torch.uint8), row_bytes, BLOCK=BLOCK)
                out[name] = buf
                continue
            src = getattr(self.layer, name)
            src = src.data if isinstance(src, torch.nn.Parameter) else src
            buf = _staging(name, k, src.shape[0], tuple(src.shape[1:]),
                           src.dtype, self.device)
            if src.is_cuda:
"""),
  # ---------------------------------------------------------------- moe_wna16.py
  (M, """            if was_pinned:
                t2 = t2.pin_memory()
            p.data = t2
            del t
        layer._b_n_contig = True
""", """            if was_pinned:
                t2 = t2.pin_memory()
            p.data = t2
            del t
        layer._b_n_contig = True
        self._collect_for_placement(layer)

    _PLACE_LAYERS = []          # layers seen so far (loader order)
    _PLACE_FREQ = None

    def _collect_for_placement(self, layer):
        path = os.environ.get("SGLANG_MOE_PLACEMENT")
        if not path:
            return
        cls = type(self)
        if cls._PLACE_FREQ is None:
            cls._PLACE_FREQ = torch.load(path)["mass"]
        n_layers = int(cls._PLACE_FREQ.shape[0])
        lid = int(getattr(layer, "layer_id", -1))
        if 0 <= lid < n_layers:
            cls._PLACE_LAYERS.append(layer)
        if len(cls._PLACE_LAYERS) == n_layers:
            self._run_placement()
            cls._PLACE_LAYERS = []

    def _run_placement(self):
        \"\"\"Interleaved, memory-neutral hot/cold split over all layers.\"\"\"
        cls = type(self)
        freq = cls._PLACE_FREQ
        S_ = int(os.environ.get("SGLANG_MOE_PLACEMENT_S", "184"))
        names = ("w13_qweight", "w2_qweight", "w13_scales", "w2_scales")
        layers = cls._PLACE_LAYERS
        host = [l for l in layers if not l.w13_qweight.data.is_cuda]
        gpu = [l for l in layers if l.w13_qweight.data.is_cuda]
        logger.info("expert placement: %d host layers, %d GPU layers, S=%d", len(host), len(gpu), S_)
        pools = {n: [] for n in names}        # free host slots: (tensor, row)
        extra_keep = []

        def split_ids(l):
            order = torch.argsort(freq[int(l.layer_id)], descending=True)
            return order[:S_].sort().values, order[S_:].sort().values

        def place_host(l):
            hot_ids, cold_ids = split_ids(l)
            E = int(l.w13_qweight.data.shape[0]); addr, proto, keep = {}, {}, []
            for name in names:
                p = getattr(l, name, None)
                if p is None: continue
                t = p.data; rb = t[0].numel() * t.element_size()
                tab = torch.empty(E, dtype=torch.int64)
                if t.is_cuda:                                 # split layer: this kind is on the GPU
                    tab, h, kp = place_gpu_kind(t, hot_ids, cold_ids, name, rb, E)
                    keep += kp
                else:
                    h = t.index_select(0, hot_ids).to("cuda").contiguous()
                    tab[hot_ids] = h.data_ptr() + torch.arange(S_) * rb
                    tab[cold_ids] = t.data_ptr() + cold_ids * rb
                    pools[name].extend((t, int(r)) for r in hot_ids.tolist())
                    keep.append(t)
                p.data = h; addr[name] = tab.cuda(); proto[name] = h
                del t
            l._placed = {"addr": addr, "proto": proto, "E": E, "S": S_, "keep": keep}
            l._gemv_tabs = None

        def place_gpu_kind(t, hot_ids, cold_ids, name, rb, E):
            tab = torch.empty(E, dtype=torch.int64); keep = []
            need = int(cold_ids.numel()); pool = pools[name]
            if len(pool) < need:
                extra = torch.empty((need - len(pool),) + tuple(t.shape[1:]), dtype=t.dtype).pin_memory()
                pool.extend((extra, i) for i in range(need - len(pool)))
                extra_keep.append(extra); keep.append(extra)
            slots = [pool.pop() for _ in range(need)]
            cold_cpu = t.index_select(0, cold_ids.to(t.device)).to("cpu")
            for i, (dst, r) in enumerate(slots):
                dst[r].copy_(cold_cpu[i])
                tab[int(cold_ids[i])] = dst.data_ptr() + r * rb
                keep.append(dst)
            del cold_cpu
            h = t.index_select(0, hot_ids.to(t.device)).contiguous()
            tab[hot_ids] = h.data_ptr() + torch.arange(S_) * rb
            return tab, h, keep

        def place_gpu(l):
            hot_ids, cold_ids = split_ids(l)
            E = int(l.w13_qweight.data.shape[0]); addr, proto, keep = {}, {}, []
            for name in names:
                p = getattr(l, name, None)
                if p is None: continue
                t = p.data; rb = t[0].numel() * t.element_size()
                tab, h, kp = place_gpu_kind(t, hot_ids, cold_ids, name, rb, E)
                keep += kp; p.data = h; addr[name] = tab.cuda(); proto[name] = h
                del t
            torch.cuda.empty_cache()
            l._placed = {"addr": addr, "proto": proto, "E": E, "S": S_, "keep": keep}
            l._gemv_tabs = None

        # interleave so donated slots exist before they are consumed: ~1.8 host layers per GPU layer
        hi = gi = 0
        ratio = max(1.0, (E_cold := (512 - S_)) / max(S_, 1))
        credit = 0.0
        while hi < len(host) or gi < len(gpu):
            if hi < len(host) and (credit < ratio or gi >= len(gpu)):
                place_host(host[hi]); hi += 1; credit += 1.0
            else:
                place_gpu(gpu[gi]); gi += 1; credit -= ratio
        free_left = {n: len(v) for n, v in pools.items()}
        logger.info("expert placement done: free host slots left %s, pinned fallbacks %d",
                    free_left, len(extra_keep))
        cls._PLACE_EXTRA = extra_keep
"""),
  (M, """        _names = ("w13_qweight", "w2_qweight", "w13_scales", "w2_scales")
        _ts = [getattr(layer, n).data for n in _names if hasattr(layer, n)]
        if _ts and all(t.is_cuda for t in _ts):
            layer._expert_streamer = None
            return None
""", """        _names = ("w13_qweight", "w2_qweight", "w13_scales", "w2_scales")
        _ts = [getattr(layer, n).data for n in _names if hasattr(layer, n)]
        if _ts and all(t.is_cuda for t in _ts) and getattr(layer, "_placed", None) is None:
            layer._expert_streamer = None
            return None
"""),
  (M, """    def _gemv_tables(self, layer):
        tabs = getattr(layer, "_gemv_tabs", None)
        if tabs is None:
            from sglang.srt.layers.moe.expert_gemv import make_tables
            w13, s13 = layer.w13_qweight.data, layer.w13_scales.data
            w2, s2 = layer.w2_qweight.data, layer.w2_scales.data
            tabs = (make_tables(w13, s13), make_tables(w2, s2),
                    w13.shape[2], w13.shape[1] * 16, w2.shape[2], w2.shape[1] * 16,
                    s13.dtype == torch.bfloat16)
            layer._gemv_tabs = tabs
        return tabs
""", """    def _gemv_tables(self, layer):
        tabs = getattr(layer, "_gemv_tabs", None)
        if tabs is None:
            from sglang.srt.layers.moe.expert_gemv import make_tables
            placed = getattr(layer, "_placed", None)
            if placed is None:
                w13, s13 = layer.w13_qweight.data, layer.w13_scales.data
                w2, s2 = layer.w2_qweight.data, layer.w2_scales.data
                tabs = (make_tables(w13, s13), make_tables(w2, s2),
                        w13.shape[2], w13.shape[1] * 16, w2.shape[2], w2.shape[1] * 16,
                        s13.dtype == torch.bfloat16)
            else:
                a, pr = placed["addr"], placed["proto"]
                h13, h2 = pr["w13_qweight"], pr["w2_qweight"]
                tabs = ((a["w13_qweight"], a["w13_scales"], h13.stride(1), pr["w13_scales"].stride(1)),
                        (a["w2_qweight"], a["w2_scales"], h2.stride(1), pr["w2_scales"].stride(1)),
                        h13.shape[2], h13.shape[1] * 16, h2.shape[2], h2.shape[1] * 16,
                        pr["w13_scales"].dtype == torch.bfloat16)
            layer._gemv_tabs = tabs
        return tabs
"""),
]


def state():
    return [(p, a in open(p, encoding="utf-8").read(), b in open(p, encoding="utf-8").read())
            for p, a, b in EDITS]


def check():
    for i, (p, pr, ap) in enumerate(state()):
        print(f"  {i} {'APPLIED' if ap else ('clean' if pr else 'MISMATCH'):<8} "
              f"{os.path.relpath(p, SG)}: {EDITS[i][1].strip().splitlines()[0][:60]}")


def apply():
    st = state()
    if all(ap for _, _, ap in st):
        print("  already applied"); return
    if not all(pr or ap for _, pr, ap in st):
        print("  [!] mismatch (is ncontig_gemv.py applied?)"); check(); return
    for (p, a, b), (_, pr, ap) in zip(EDITS, st):
        if not ap:
            t = open(p, encoding="utf-8").read()
            open(p, "w", encoding="utf-8").write(t.replace(a, b, 1))
    print("  applied (placement v3, deferred interleaved, memory-neutral)")


def revert():
    for (p, a, b), (_, pr, ap) in zip(EDITS, state()):
        if ap:
            t = open(p, encoding="utf-8").read()
            open(p, "w", encoding="utf-8").write(t.replace(b, a, 1))
    print("  reverted")


if __name__ == "__main__":
    {"--check": check, "apply": apply, "revert": revert}[sys.argv[1] if len(sys.argv) > 1 else "--check"]()
