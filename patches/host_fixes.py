#!/usr/bin/env python3
"""Host-side fixes from the first-pass plan (items 1, 3, 5a, 5b) plus routing instrumentation.

Each edit is an exact string replacement, individually revertible.

  python3 host_fixes.py --check      show which are applied
  python3 host_fixes.py apply ITEM   apply one   (ITEM in: hook, skipgather, memo, rope, dump, all)
  python3 host_fixes.py revert ITEM  revert one
"""
import sys, os
SG = os.path.join(os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")), "python/sglang/srt")

EDITS = {
  # ---- item 1: do not install the offloader forward hook when every offloaded
  #      parameter of the module is a streamed expert param (the hook then does a
  #      state_dict() + no-op .to() + functional_call for nothing, 31x per token)
  "hook": (f"{SG}/utils/offloader.py", [
    ("""        for p in _params:
            if self._cpu_offload_bytes >= self._cpu_offload_max_bytes:
                # we use per-parameter offloading
                # one module might have some parameters offloaded and some not
                break
""",
     """        _n_offloaded = 0
        _n_streamed = 0
        for p in _params:
            if self._cpu_offload_bytes >= self._cpu_offload_max_bytes:
                # we use per-parameter offloading
                # one module might have some parameters offloaded and some not
                break
            _n_offloaded += 1
            _n_streamed += int(any(p is q for n, q in module.named_parameters()
                                   if _is_streamed_expert_param(n)))
"""),
    ("""        if offloaded_parameters:
            original_forward = module.forward
""",
     """        # When every offloaded parameter is a streamed expert param, the hook
        # below would build device_state from a state_dict() that EXCLUDES all
        # of them and reparametrize the module with tensors that are already
        # on the device. That is pure overhead on every forward - skip it.
        if offloaded_parameters and _n_offloaded == _n_streamed and _n_streamed > 0:
            offloaded_parameters = False
        if offloaded_parameters:
            original_forward = module.forward
"""),
  ]),

  # ---- item 3: for a layer whose expert tensors ALL live on the GPU, do not use
  #      the streamer at all. The plain path passes the full tensor with the
  #      original ids; the gather+renumber was a device-to-device copy for nothing.
  "skipgather": (f"{SG}/layers/quantization/moe_wna16.py", [
    ("""        if not enabled() or not hasattr(layer, "w13_qweight"):
            layer._expert_streamer = None
            return None
        layer._expert_streamer = ExpertStreamer(layer)
        return layer._expert_streamer
""",
     """        if not enabled() or not hasattr(layer, "w13_qweight"):
            layer._expert_streamer = None
            return None
        # All four expert tensors resident on the device: the plain path is
        # byte-for-byte what the streamer would produce, minus the copy. The
        # gate is per LAYER, never per tensor - one moe_align result numbers
        # both GEMMs, so a split layer must keep the streamer.
        _names = ("w13_qweight", "w2_qweight", "w13_scales", "w2_scales")
        _ts = [getattr(layer, n).data for n in _names if hasattr(layer, n)]
        if _ts and all(t.is_cuda for t in _ts):
            layer._expert_streamer = None
            return None
        layer._expert_streamer = ExpertStreamer(layer)
        return layer._expert_streamer
"""),
  ]),

  # ---- item 5a: the decode path allocated torch.arange(10) and an int64->int32
  #      cast per layer per token: 96 launches for two loop-invariant constants.
  "memo": (f"{SG}/layers/moe/expert_stream.py", [
    ("""_NO_DEDUP_LIMIT = 64
""",
     """_NO_DEDUP_LIMIT = 64
_ARANGE_CACHE: Dict[Tuple, torch.Tensor] = {}


def _cached_arange(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (n, str(device), dtype)
    t = _ARANGE_CACHE.get(key)
    if t is None:
        t = torch.arange(n, device=device, dtype=dtype)
        _ARANGE_CACHE[key] = t
    return t
"""),
    ("""            uniq = flat
            k = n
            inverse = torch.arange(n, device=flat.device)
""",
     """            uniq = flat
            k = n
            inverse = _cached_arange(n, flat.device, topk_ids.dtype)
"""),
  ]),

  # ---- item 5b: positions.max().item() is a full device sync, 12x per token
  #      (once per QSA layer) plus once more in the rope helper. Pre-size the
  #      cos/sin cache to the server context length once; afterwards the Python
  #      length check is sufficient and no sync is needed.
  "rope": (f"{SG}/layers/attention/qsa/qsa_indexer.py", [
    ("""            if not get_is_capture_mode() and hasattr(
                self.rotary_emb, "_ensure_cos_sin_cache_length"
            ):
                self.rotary_emb._ensure_cos_sin_cache_length(
                    int(positions.max().item())
                )
""",
     """            if not get_is_capture_mode() and hasattr(
                self.rotary_emb, "_ensure_cos_sin_cache_length"
            ):
                _qsa_ensure_rope(self.rotary_emb, positions)
"""),
    ("""        if not get_is_capture_mode() and hasattr(self.rotary_emb, "_ensure_cos_sin_cache_length"):
            self.rotary_emb._ensure_cos_sin_cache_length(int(positions.max().item()))
""",
     """        if not get_is_capture_mode() and hasattr(self.rotary_emb, "_ensure_cos_sin_cache_length"):
            _qsa_ensure_rope(self.rotary_emb, positions)
"""),
    # helper appended after the imports: find the first blank line after the
    # last top-level import and insert there
    ("""from sglang.srt.model_executor.runner import get_is_capture_mode
""",
     """from sglang.srt.model_executor.runner import get_is_capture_mode


def _qsa_ensure_rope(rotary_emb, positions):
    \"\"\"Grow the cos/sin cache without a device sync on the hot path.

    positions.max().item() synchronizes the device once per QSA layer. The
    cache only ever needs to reach the server context length, so size it to
    that once (a Python-side comparison afterwards) and fall back to the
    synchronizing path only if a position somehow exceeds it.
    \"\"\"
    from sglang.srt.server_args import get_global_server_args
    ctx = int(getattr(get_global_server_args(), "context_length", 0) or 0)
    if ctx > 0:
        if int(rotary_emb.cos_sin_cache.shape[0]) <= ctx:
            rotary_emb._ensure_cos_sin_cache_length(ctx)
        return
    rotary_emb._ensure_cos_sin_cache_length(int(positions.max().item()))
"""),
  ]),

  # ---- instrumentation: dump per-layer topk ids + weights when SGLANG_ROUTE_DUMP
  #      names a file. Used for adjacent-token reuse, gate-mass, and cache design.
  "dump": (f"{SG}/layers/quantization/moe_wna16.py", [
    ("""        streamer = self._maybe_expert_streamer(layer)
        if streamer is None:
            quant_info = self.get_triton_quant_info(layer)
            return self.runner.run(dispatch_output, quant_info)
""",
     """        _dump = os.environ.get("SGLANG_ROUTE_DUMP")
        if _dump:
            _route_dump(_dump, layer, dispatch_output.topk_output)
        streamer = self._maybe_expert_streamer(layer)
        if streamer is None:
            quant_info = self.get_triton_quant_info(layer)
            return self.runner.run(dispatch_output, quant_info)
"""),
    ("""class MoeWNA16Method""",
     """_ROUTE = {"ids": [], "w": [], "steps": [], "n": 0, "flushed": 0}


def _route_dump(path, layer, topk):
    \"\"\"Sync-free routing probe: clone ids/weights on the GPU per layer, stack per
    forward, D2H only every 64 decode steps. Records decode (M=1) steps only.\"\"\"
    ids = topk.topk_ids
    if ids.shape[0] != 1:
        _ROUTE["ids"].clear(); _ROUTE["w"].clear()
        return
    st = _ROUTE
    st["ids"].append(ids[0].to(torch.int16).clone())
    st["w"].append(topk.topk_weights[0].float().clone())
    if int(getattr(layer, "layer_id", -1)) != 47:      # last MoE layer of this forward
        return
    st["steps"].append((torch.stack(st["ids"]), torch.stack(st["w"])))   # [48,10] each
    st["ids"].clear(); st["w"].clear(); st["n"] += 1
    if st["n"] % 64 == 0:
        _route_flush(path)


def _route_flush(path):
    st = _ROUTE
    if not st["steps"]:
        return
    os.makedirs(path, exist_ok=True)
    out = [(i.cpu(), w.cpu()) for i, w in st["steps"]]        # the only D2H
    torch.save(out, os.path.join(path, f"routing_{st['flushed']:05d}.pt"))
    st["flushed"] += 1; st["steps"].clear()


class MoeWNA16Method"""),
  ]),
  # ---- item 2: the PLE row fetch. numpy fancy-indexing a 51 GB memmap holds the
  #      GIL through every page fault (2.5-3.1 ms cold per token). os.pread releases
  #      it; 16 concurrent 160-byte reads measured 0.4 ms. Small N only - prefill
  #      keeps the numpy path, which the page cache serves well.
  "ple": (f"{SG}/models/qwen4_exp.py", [
    ("""        self._mm = np.memmap(path, dtype=np.uint8, mode="r",
                             shape=(self._rows, self._dim * itemsize))
        self._itemsize = itemsize
""",
     """        self._mm = np.memmap(path, dtype=np.uint8, mode="r",
                             shape=(self._rows, self._dim * itemsize))
        self._itemsize = itemsize
        # Parallel pread path for decode: one 160-byte read per row, GIL released.
        self._fd = os.open(path, os.O_RDONLY)
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=16)
        self._row_bytes = self._dim * itemsize
"""),
    ("""        rows = torch.from_numpy(self._mm[local.numpy()])          # uint8, (N, dim*b)
""",
     """        if local.numel() <= 64:
            rb = self._row_bytes
            def _rd(r):
                return os.pread(self._fd, rb, int(r) * rb)
            buf = b"".join(self._pool.map(_rd, local.tolist()))
            rows = torch.frombuffer(bytearray(buf), dtype=torch.uint8).view(-1, rb)
        else:
            rows = torch.from_numpy(self._mm[local.numpy()])      # uint8, (N, dim*b)
"""),
  ]),

  # ---- item 7: the single genuine CUDA-graph capture blocker on the decode path.
  #      Wrap the mmap embedding forward as an eager break: BCG runs it eagerly and
  #      copies its fresh output into the boundary buffer the next segment reads.
  #      Launch WITHOUT --disable-cuda-graph to use it.
  "bcg": (f"{SG}/models/qwen4_exp.py", [
    ("""    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.reduce(self.gather(input_ids))

    def reduce(self, output: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1 and not get_attn_tp_context().input_scattered:
""",
     """    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.reduce(self.gather(input_ids))

    def reduce(self, output: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1 and not get_attn_tp_context().input_scattered:
"""),
  ]),
  # ---- item 7b: the breakable decode backend walks Tensor / PPProxyTensors / tuple / list
  #      output structures only; this model's forward returns a LogitsProcessorOutput
  #      dataclass and capture died with "Unsupported BCG output type". Treat it as a
  #      structure of tensor fields (non-tensor fields pass through by reference).
  "bcg2": (f"{SG}/model_executor/runner_backend/breakable_cuda_graph_backend.py", [
    ("""    def _output_rows(self, output: Any, cap: int) -> int:
""",
     """    @staticmethod
    def _lpo_fields(output):
        \"\"\"(name, tensor) pairs of the tensor-valued fields of a LogitsProcessorOutput.\"\"\"
        return [(k, v) for k, v in vars(output).items() if torch.is_tensor(v)]

    @staticmethod
    def _is_lpo(output) -> bool:
        return type(output).__name__ == "LogitsProcessorOutput"

    def _output_rows(self, output: Any, cap: int) -> int:
"""),
    ("""        if isinstance(output, (list, tuple)) and output:
            return min(self._output_rows(o, cap) for o in output if o is not None)
        return cap
""",
     """        if isinstance(output, (list, tuple)) and output:
            return min(self._output_rows(o, cap) for o in output if o is not None)
        if self._is_lpo(output):
            rows = [t.shape[0] for _, t in self._lpo_fields(output)]
            return min([cap, *rows]) if rows else cap
        return cap
"""),
    ("""        if isinstance(output, list):
            return [self._alloc_full_buffer(o, size) for o in output]
        raise TypeError(f"Unsupported BCG output type: {type(output)}")
""",
     """        if isinstance(output, list):
            return [self._alloc_full_buffer(o, size) for o in output]
        if self._is_lpo(output):
            import copy as _copy
            buf = _copy.copy(output)
            for k, t in self._lpo_fields(output):
                setattr(buf, k, t.new_empty((size, *t.shape[1:])))
            return buf
        raise TypeError(f"Unsupported BCG output type: {type(output)}")
"""),
    ("""        if isinstance(output, list):
            return [self._slice_output(item, num_tokens) for item in output]
        raise TypeError(f"Unsupported BCG output type: {type(output)}")
""",
     """        if isinstance(output, list):
            return [self._slice_output(item, num_tokens) for item in output]
        if self._is_lpo(output):
            import copy as _copy
            out = _copy.copy(output)
            for k, t in self._lpo_fields(output):
                setattr(out, k, t[:num_tokens])
            return out
        raise TypeError(f"Unsupported BCG output type: {type(output)}")
"""),
    ("""        if torch.is_tensor(output) and torch.is_tensor(output_buffer):
            output_buffer[:num_tokens].copy_(output[:num_tokens])
            return
""",
     """        if torch.is_tensor(output) and torch.is_tensor(output_buffer):
            output_buffer[:num_tokens].copy_(output[:num_tokens])
            return
        if self._is_lpo(output) and self._is_lpo(output_buffer):
            for k, t in self._lpo_fields(output):
                b = getattr(output_buffer, k, None)
                if torch.is_tensor(b):
                    b[:num_tokens].copy_(t[:num_tokens])
                else:
                    setattr(output_buffer, k, t)
            for k, v in vars(output).items():
                if not torch.is_tensor(v):
                    setattr(output_buffer, k, v)
            return
"""),
  ]),
  # ---- diagnostics: dump MoE layer inputs/outputs for the first forwards
  #      (SGLANG_MOE_DUMP_IO=dir). Apply AFTER ncontig_gemv.py (it wraps the def line).
  "moedump": (f"{SG}/layers/quantization/moe_wna16.py", [
    ("""    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
""",
     """    def apply(self, layer, dispatch_output):
        out = self._apply_inner(layer, dispatch_output)
        _d = os.environ.get("SGLANG_MOE_DUMP_IO")
        if _d:
            _moe_dump(_d, layer, dispatch_output, out)
        return out

    def _apply_inner(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
"""),
    ("""class MoeWNA16Method""",
     """_MOE_DUMP_N = {}


def _moe_dump(d, layer, dispatch_output, out):
    \"\"\"Save (x, topk_ids, topk_weights, y) for the first 3 calls of every layer.\"\"\"
    lid = int(getattr(layer, "layer_id", -1))
    n = _MOE_DUMP_N.get(lid, 0)
    if n >= 3:
        return
    _MOE_DUMP_N[lid] = n + 1
    os.makedirs(d, exist_ok=True)
    tk = dispatch_output.topk_output
    torch.save({"layer": lid, "x": dispatch_output.hidden_states.detach().cpu(),
                "ids": tk.topk_ids.detach().cpu(), "w": tk.topk_weights.detach().float().cpu(),
                "y": out.hidden_states.detach().cpu()},
               os.path.join(d, f"L{lid:02d}_{n}.pt"))


class MoeWNA16Method"""),
  ]),
}
EDITS["dump"][1].append(("import torch\n", "import os\nimport torch\n"))
EDITS["moedump"][1].append(("import torch\n", "import os\nimport torch\n"))
# "bcg" is applied by appending a module-level wrap; keep the anchor edit a no-op and
# append the wrap line if missing.
_BCG_TAIL = """

# --- Breakable CUDA Graph: the mmap PLE lookup is the one eager break on the decode path.
try:
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import eager_on_graph as _eog
    Qwen4ExpMmapEmbedding.forward = _eog(True)(Qwen4ExpMmapEmbedding.forward)
except Exception as _e:  # pragma: no cover
    logger.warning("BCG wrap of Qwen4ExpMmapEmbedding.forward failed: %s", _e)
"""


def _bcg_state():
    p = EDITS["bcg"][0]
    return p, open(p, encoding="utf-8").read()


def _bcg_apply():
    p, t = _bcg_state()
    if _BCG_TAIL.strip() in t:
        print("  bcg: already applied"); return
    open(p, "w", encoding="utf-8").write(t.rstrip("\n") + "\n" + _BCG_TAIL)
    print("  bcg: applied (module-level wrap appended)")


def _bcg_revert():
    p, t = _bcg_state()
    if _BCG_TAIL.strip() not in t:
        print("  bcg: not applied"); return
    open(p, "w", encoding="utf-8").write(t.replace(_BCG_TAIL, "\n"))
    print("  bcg: reverted")


def state(item):
    path, pairs = EDITS[item]
    t = open(path, encoding="utf-8").read()
    applied = all(b in t for a, b in pairs)
    pristine = all(a in t for a, b in pairs)
    return path, pairs, t, applied, pristine


def apply(item):
    path, pairs, t, applied, pristine = state(item)
    if applied:
        print(f"  {item}: already applied"); return
    if not pristine:
        print(f"  [!] {item}: source does not match expected text; not touching"); return
    for a, b in pairs:
        t = t.replace(a, b, 1)
    open(path, "w", encoding="utf-8").write(t)
    print(f"  {item}: applied  ({os.path.relpath(path, SG)})")


def revert(item):
    path, pairs, t, applied, pristine = state(item)
    if pristine and not applied:
        print(f"  {item}: not applied"); return
    for a, b in pairs:
        t = t.replace(b, a, 1)
    open(path, "w", encoding="utf-8").write(t)
    print(f"  {item}: reverted")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--check":
        for k in EDITS:
            if k == "bcg":
                _, t = _bcg_state()
                print(f"  {k:<12} {'APPLIED' if _BCG_TAIL.strip() in t else 'clean'}")
                continue
            _, _, _, ap, pr = state(k)
            print(f"  {k:<12} {'APPLIED' if ap else ('clean' if pr else 'MISMATCH')}")
        sys.exit(0)
    cmd, items = sys.argv[1], sys.argv[2:]
    if items == ["all"]:
        items = [k for k in EDITS if k not in ("bcg", "dump")]   # opt-in ones stay explicit
    for it in items:
        if it == "bcg":
            (_bcg_apply if cmd == "apply" else _bcg_revert)()
        else:
            (apply if cmd == "apply" else revert)(it)
