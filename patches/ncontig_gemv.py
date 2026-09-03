#!/usr/bin/env python3
"""N-contiguous int32-word expert layout + batch-1 GEMV through a pointer table.

  layout  after loading, re-lay each 2-bit expert tensor from [E, N, K/4] uint8 to
          [E, K/16, N] int32 (same bytes; N contiguous; one word = 16 K-values). The tiled
          prefill kernel's int2 branch is swapped for the word-load variant (perf/int2_gemv/
          fused_moe_kernel_gptq_awq_wordload_variant.py, interpreter-verified identity), which
          reads coalesced 128-B lines instead of 32 distinct lines per warp-load.

          The layout is DERIVED FROM THE TENSOR DTYPE wherever it matters (int2 + int32 =
          word layout; int2 + uint8 = byte layout), so no flag has to be threaded through
          fused_experts, whose custom-op wrapper has a fixed schema.

  gemv    for decode (M <= 16) run the MoE as T independent 2-bit GEMVs that read each expert
          IN PLACE through an int64 address table - device or pinned host. Measured 320 GB/s from
          device, 51 GB/s (PCIe line rate) from pinned host. No gather, no staging, no renumbering.

  python3 ncontig_gemv.py --check | apply | revert
Env at runtime: SGLANG_MOE_NCONTIG=0 disables the layout change (and the GEMV),
                SGLANG_MOE_GEMV=0 keeps the layout but disables the GEMV dispatch.
"""
import os, shutil, sys
SG = os.path.join(os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")), "python/sglang")
HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_SRC = os.path.join(HERE, "..", "gemv", "moe_gemv_int2_tab.py")
KERNEL_DST = f"{SG}/srt/layers/moe/expert_gemv.py"
WORDLOAD_SRC = os.path.join(HERE, "..", "int2_gemv", "fused_moe_kernel_gptq_awq_wordload_variant.py")

M = f"{SG}/srt/layers/quantization/moe_wna16.py"
F = f"{SG}/srt/layers/moe/moe_runner/triton_utils/fused_moe.py"
K = f"{SG}/kernels/ops/moe/fused_moe_triton_kernels.py"
K_BACKUP = K + ".ncontig.orig"
MARK = "# [ncontig] word-load int2 branch"

EDITS = [
  # ---------------------------------------------------------------- fused_moe.py
  # _prepare_fused_moe_run: layout-independent config lookup (keeps the tuned JSON names)
  (F, """    config, (down_config, _) = try_get_optimal_moe_config(
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
""", """    # Config lookup keys on (E, N, K/pack). In the N-contiguous int32-word layout
    # ([E, K/16, N], see moe_wna16.process_weights_after_loading) present the shapes
    # as (E, N, K/4) so the tuned JSON names stay the same as for the byte layout.
    _ncontig = use_int2_w2a16 and w1.dtype == torch.int32
    if _ncontig:
        _w1s = (w1.shape[0], w1.shape[2], w1.shape[1] * 4)
        _w2s = (w2.shape[0], w2.shape[2], w2.shape[1] * 4 - padded_size)
    else:
        _w1s = w1.shape
        _w2s = (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size)
    config, (down_config, _) = try_get_optimal_moe_config(
        _w1s,
        _w2s,
"""),
  # _fused_moe_kernel_sequence: N and hidden from the right dims
  (F, """    num_tokens = hidden_states.shape[0]
    E, N, _ = w1.shape
    topk = topk_ids.shape[1]
""", """    num_tokens = hidden_states.shape[0]
    _ncontig = use_int2_w2a16 and w1.dtype == torch.int32
    E = w1.shape[0]
    N = w1.shape[2] if _ncontig else w1.shape[1]
    _w2_hidden = w2.shape[2] if _ncontig else w2.shape[1]
    topk = topk_ids.shape[1]
"""),
  (F, """    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, topk, w2.shape[1]),
""", """    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, topk, _w2_hidden),
"""),
  (F, """    intermediate_cache3 = torch.empty(
        (num_tokens, topk, w2.shape[1]),
""", """    intermediate_cache3 = torch.empty(
        (num_tokens, topk, _w2_hidden),
"""),
  # fused_experts_impl: constraint check
  (F, """    if use_int2_w2a16:
        # 2-bit: 4 values per byte, hence k/4 columns in the packed tensor
        assert hidden_states.shape[1] // 4 == w1.shape[2], "Hidden size mismatch"
""", """    if use_int2_w2a16:
        # 2-bit: 4 values per byte, hence k/4 columns in the packed tensor; in the
        # N-contiguous int32-word layout K/16 words sit in dim 1 instead
        if w1.dtype == torch.int32:
            assert hidden_states.shape[1] == w1.shape[1] * 16, "Hidden size mismatch"
        else:
            assert hidden_states.shape[1] // 4 == w1.shape[2], "Hidden size mismatch"
"""),
  # ---------------------------------------------------------------- kernels invoke
  (K, """            num_tokens_post_padded,
            B.shape[1],
            A.shape[1],
            sorted_token_ids.shape[0],
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(-2),
            C.stride(-1),
            B_scale.stride(0),
            B_scale.stride(2),
            B_scale.stride(1),
""", """            num_tokens_post_padded,
            B.shape[2] if _ncontig else B.shape[1],
            A.shape[1],
            sorted_token_ids.shape[0],
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1) if _ncontig else B.stride(2),
            B.stride(2) if _ncontig else B.stride(1),
            C.stride(-2),
            C.stride(-1),
            B_scale.stride(0),
            B_scale.stride(1) if _ncontig else B_scale.stride(2),
            B_scale.stride(2) if _ncontig else B_scale.stride(1),
"""),
  # grid N and the even_Ks check read B.shape[1]/B.shape[2]; in the word layout N is dim 2
  # and K must come from A (values), not from the packed dim
  (K, """    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )

    K = B.shape[2] - padded_size
    if K % config["BLOCK_SIZE_K"] == 0:
""", """    # int2 in the N-contiguous int32-word layout ([E, K/16, N]): N is dim 2 and the
    # K used for the even_Ks check is the activation width, not the packed dim
    _ncontig = use_int2_w2a16 and B.dtype == torch.int32
    _n_dim = B.shape[2] if _ncontig else B.shape[1]
    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(_n_dim, META["BLOCK_SIZE_N"]),
    )

    K = A.shape[1] if _ncontig else (B.shape[2] - padded_size)
    if K % config["BLOCK_SIZE_K"] == 0:
"""),
  # dispatch: word-load kernel for the int32 layout, original kernel otherwise
  (K, """        assert bias is None
        fused_moe_kernel_gptq_awq[grid](
""", """        assert bias is None
        (fused_moe_kernel_gptq_awq_word if _ncontig else fused_moe_kernel_gptq_awq)[grid](
"""),
  # ---------------------------------------------------------------- expert_stream.py
  # staging buffers were cached by (name, dtype, device) only; with two layouts in one
  # process a word-layout layer could receive a byte-layout-shaped scales buffer
  (f"{SG}/srt/layers/moe/expert_stream.py", """    key = (name, dtype, str(device))
    buf = _STAGING.get(key)
""", """    key = (name, dtype, str(device), tuple(rest))
    buf = _STAGING.get(key)
"""),
  # ---------------------------------------------------------------- moe_wna16.py
  (M, """    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

""", """    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        \"\"\"2-bit: re-lay each expert tensor as N-contiguous int32 words / [E, K/128, N] scales.

        Same bytes, re-arranged once. The tiled kernel's B loads become coalesced 128-B
        lines and the decode GEMV can read experts in place. Offloaded tensors live in
        pinned host memory at this point; re-pin after the copy.
        \"\"\"
        if self.quant_config.weight_bits != 2:
            return
        if os.environ.get("SGLANG_MOE_NCONTIG", "1") != "1":
            return
        # bisection aid: SGLANG_MOE_NCONTIG_LAYERS="0-16" converts only those layer ids
        _sel = os.environ.get("SGLANG_MOE_NCONTIG_LAYERS")
        if _sel:
            _ok = set()
            for part in _sel.split(","):
                lo, _, hi = part.partition("-")
                _ok.update(range(int(lo), int(hi or lo) + 1))
            if int(getattr(layer, "layer_id", -1)) not in _ok:
                return
        from sglang.srt.layers.moe.expert_gemv import to_word_ncontig
        for name in ("w13_qweight", "w2_qweight", "w13_scales", "w2_scales"):
            p = getattr(layer, name, None)
            if p is None or p.data.dim() != 3:
                continue
            t = p.data
            was_pinned = (not t.is_cuda) and t.is_pinned()
            if name.endswith("qweight"):
                t2 = to_word_ncontig(t)                    # [E, K/16, N] int32
            else:
                t2 = t.transpose(1, 2).contiguous()        # [E, K/128, N]
            if was_pinned:
                t2 = t2.pin_memory()
            p.data = t2
            del t
        layer._b_n_contig = True

    def _gemv_tables(self, layer):
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

    def _apply_gemv(self, layer, dispatch_output):
        \"\"\"Batch-1 path: T independent 2-bit GEMVs reading experts in place.\"\"\"
        from sglang.srt.layers.moe.expert_gemv import moe_gemv_int2_tab
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        x = dispatch_output.hidden_states
        topk = dispatch_output.topk_output
        ids = topk.topk_ids.reshape(-1)
        w = topk.topk_weights.reshape(-1).to(torch.float32)
        M_, top_k = topk.topk_ids.shape
        (wt13, st13, sw13, ss13), (wt2, st2, sw2, ss2), N13, K13, N2, K2, sbf = self._gemv_tables(layer)
        inter = N13 // 2
        c13 = moe_gemv_int2_tab(x, wt13, st13, ids, w, N13, K13, sw13, ss13,
                                top_k=top_k, mul_routed_weight=False, scale_bf16=sbf)
        h = torch.nn.functional.silu(c13[:, :inter]) * c13[:, inter:]
        c2 = moe_gemv_int2_tab(h, wt2, st2, ids, w, N2, K2, sw2, ss2,
                               top_k=1, mul_routed_weight=True, scale_bf16=sbf)
        out = c2.view(M_, top_k, N2).sum(dim=1)
        rsf = self.moe_runner_config.routed_scaling_factor
        if rsf is not None and rsf != 1.0:
            out = out * rsf
        return StandardCombineInput(hidden_states=out.to(x.dtype))

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        if (
            getattr(layer, "_b_n_contig", False)
            and self.quant_config.weight_bits == 2
            and dispatch_output.hidden_states.shape[0] <= 16
            and os.environ.get("SGLANG_MOE_GEMV", "1") == "1"
        ):
            return self._apply_gemv(layer, dispatch_output)

"""),
]
OS_IMPORT = ("import numpy as np\nimport torch\n", "import numpy as np\nimport os\nimport torch\n")


# ----------------------------------------------------------------- kernel function ADD
# The word-load variant is added as a second kernel, fused_moe_kernel_gptq_awq_word; the
# original stays for the byte layout and other bit widths. invoke dispatches by dtype, so
# byte- and word-layout layers can coexist in one process (bisection, partial rollouts).
def kernel_applied():
    return MARK in open(K, encoding="utf-8").read()


def kernel_apply():
    t = open(K, encoding="utf-8").read()
    if MARK in t:
        return
    v = open(WORDLOAD_SRC, encoding="utf-8").read()
    fn = v[v.index("def fused_moe_kernel_gptq_awq("):].rstrip("\n")
    fn = fn.replace("def fused_moe_kernel_gptq_awq(", "def fused_moe_kernel_gptq_awq_word(", 1)
    s = t.index("@triton.jit\ndef fused_moe_kernel_gptq_awq(")
    open(K, "w", encoding="utf-8").write(t[:s] + MARK + "\n@triton.jit\n" + fn + "\n\n\n" + t[s:])


def kernel_revert():
    if os.path.exists(K_BACKUP):
        shutil.copy(K_BACKUP, K); os.remove(K_BACKUP)


# ----------------------------------------------------------------- text edits
def state():
    out = []
    for p, a, b in EDITS:
        t = open(p, encoding="utf-8").read()
        out.append((p, a in t, b in t))
    return out


def check():
    bad = 0
    for i, (p, pr, ap) in enumerate(state()):
        st = "APPLIED" if ap else ("clean" if pr else "MISMATCH")
        bad += st == "MISMATCH"
        print(f"  {i:2} {st:<8} {os.path.relpath(p, SG)}: {EDITS[i][1].strip().splitlines()[0][:58]}")
    print(f"  kernel word-load branch: {'APPLIED' if kernel_applied() else 'clean'};  "
          f"gemv file: {'present' if os.path.exists(KERNEL_DST) else 'absent'};  "
          f"kernel backup: {'present' if os.path.exists(K_BACKUP) else 'absent'}")
    return bad == 0


def apply():
    st = state()
    if all(ap for _, _, ap in st) and kernel_applied():
        print("  already applied"); return
    if not all(pr or ap for _, pr, ap in st):
        print("  [!] source mismatch - refusing"); check(); return
    if not os.path.exists(K_BACKUP) and not kernel_applied():
        shutil.copy(K, K_BACKUP)          # pristine kernel file, taken before any text edit
    for (p, a, b), (_, pr, ap) in zip(EDITS, st):
        if ap:
            continue
        t = open(p, encoding="utf-8").read()
        open(p, "w", encoding="utf-8").write(t.replace(a, b, 1))
    t = open(M, encoding="utf-8").read()
    if "\nimport os\n" not in t:
        open(M, "w", encoding="utf-8").write(t.replace(OS_IMPORT[0], OS_IMPORT[1], 1))
    kernel_apply()
    shutil.copy(KERNEL_SRC, KERNEL_DST)
    print("  applied: layout + word-load tiled branch + gemv (srt/layers/moe/expert_gemv.py)")


def revert():
    # the kernel file is restored wholesale from the pristine backup; text edits elsewhere
    for (p, a, b), (_, pr, ap) in zip(EDITS, state()):
        if ap and p != K:
            t = open(p, encoding="utf-8").read()
            open(p, "w", encoding="utf-8").write(t.replace(b, a, 1))
    if os.path.exists(K_BACKUP):
        kernel_revert()
    else:
        for (p, a, b), (_, pr, ap) in zip(EDITS, state()):
            if ap and p == K:
                t = open(p, encoding="utf-8").read()
                open(p, "w", encoding="utf-8").write(t.replace(b, a, 1))
    if os.path.exists(KERNEL_DST):
        os.remove(KERNEL_DST)
    print("  reverted")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "apply": apply, "revert": revert}[cmd]()
