#!/usr/bin/env python3
"""
Work out the bit allocation before spending days of compute on it.

Takes the REAL tensor shapes (tensors.json, from 01_inspect_model.py or
get_safetensors_metadata) plus a policy, computes exactly how many GiB the
result occupies, and compares that against the budget from budget.py.

  python3 04_recipe.py                      # default policy
  python3 04_recipe.py --policy conservative
  python3 04_recipe.py --ram-reserve 8      # try a knob
  python3 04_recipe.py --emit-layer-config  # emit the AutoRound recipe

Why this tool exists: AutoScheme averages avg_bits over quantized layers only.
Tensors pinned to bf16 are added on top; here they are 3.3% of the parameters
but up to 22% of the budget. Without this calculation you discover the overrun
in 08_verify_quant.py, after the run, having lost the compute.
"""
import argparse, json, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B
from importlib.machinery import SourceFileLoader
_insp = SourceFileLoader("insp", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "01_inspect_model.py")).load_module()
GIB = 1024 ** 3


def bpw(scheme, gs=128):
    """Real bpw of a scheme, including scale and zero point."""
    if scheme == "BF16":
        return 16.0
    bits = int(scheme[1:].split("A")[0])
    return bits + (16 + 4) / gs


# ---------------------------------------------------------------- Policies
#
# What the allocation is anchored on:
#  - Qwen's own FP8 release (modules_to_not_convert) quantizes ONLY experts +
#    ngram. Attention, Gated DeltaNet, hyper-connections, the shared expert,
#    vision and the PLE projections all stay bf16 there. That is the best
#    available statement about which parts of this model are sensitive, and it
#    comes from the people who trained it.
#  - Community findings on MoE quantization: ffn_up/gate_exps tolerate few bits
#    well, ffn_down_exps less so, attention in hybrid architectures is
#    particularly delicate, and ssm_out should never go deep.
#  - The router is bf16, always. See the note at the router rule below.
#  - Keep the bit gradient flat; do not devalue whole layer blocks.
#
#
# LM_HEAD_NOTE - lm_head stays bf16, for two independent reasons:
#
#  1) Quality. AutoRound only quantizes lm_head on an explicit
#     --quant_lm_head, because the output projection is considered delicate.
#     With a 248,320-entry vocabulary every rounding error lands straight in
#     the logits. Qwen's own FP8 leaves lm_head untouched.
#
#  2) Memory. lm_head sits OUTSIDE the transformer blocks. The moment any
#     layer outside the blocks is quantized, AutoRound disables block-wise
#     saving whenever tuning is active (need_calib):
#       base.py:1224  has_qlayer_outside_block & need_calib -> inplace=False
#       base.py:1425  ... & inplace    -> is_immediate_packing stays False
#       base.py:1428  ... & need_calib -> is_immediate_packing=False
#       base.py:1446  low_cpu_mem_usage & is_immediate_packing -> is_immediate_saving
#     Consequence: all 48 finished blocks stay in RAM until the very end
#     (~0.73 GiB per block = 35.2 GiB, plus ~5 GiB of block inputs). On a
#     30 GiB machine that is a guaranteed OOM after hours of correct work.
#     Setting lm_head to bf16 makes has_qlayer_outside_block False and unlocks
#     the whole chain.
#
#     Price: 0.879 GiB (0.305 GiB at 4-bit g128 -> 1.184 GiB bf16). Cheap, and
#     it buys a quality improvement rather than costing one.
#
#
# MOE_UNIFORMITY_NOTE - why every expert projection shares one (bits, gs):
#
#     SGLang's FusedMoE fuses all experts of a layer into a single Triton
#     kernel launch, and that kernel takes ONE weight_bits and ONE group_size
#     for the whole layer. A recipe that gives down_proj a finer group size
#     than gate_up_proj, or gives the first N layers 3 bits, produces a
#     checkpoint that is arithmetically better and that no MoE inference
#     engine will load. Both refinements were tried and removed for this
#     reason - the finer grouping and the layer-wise W3 gradient. If you are
#     targeting a runtime that dequantizes per projection, put them back.
#
# (regex, scheme, group_size) - first match wins.

POLICIES = {
    # Fits the budget; follows Qwen's sensitivity ranking as far as it can.
    "default": [
        (r"ngram_embedding\.shard_",              "SKIP",  0),    # separate, on NVMe
        (r"^mtp\.",                                "SKIP",  0),    # optional
        (r"embed_tokens",                          "SKIP",  0),    # mmap
        (r"mlp\.gate(\.weight)?$|shared_expert_gate", "BF16",  0),    # router - never quantize
        (r"norm|layernorm",                        "BF16",  0),
        (r"linear_attn\.(A_log|dt_bias|conv1d)",  "BF16",  0),    # SSM parameters
        (r"ngram_heads_|layer_multipliers",        "BF16",  0),
        (r"\.ple\.(conv1d|key_proj|value_proj)",  "W8A16", 128),
        (r"hyper_connection",                      "W8A16", 128),  # gated residual
        (r"linear_attn\.out_proj",                 "W8A16", 128),  # "ssm_out" - never deep
        (r"linear_attn\.",                         "W4A16", 128),
        (r"self_attn\.",                           "W8A16", 128),  # only 12 layers, delicate
        (r"shared_expert\.",                       "W4A16", 128),  # runs on every token
        (r"^model\.visual\.",                      "W4A16", 128),
        (r"^lm_head",                              "BF16",  0),    # see LM_HEAD_NOTE
        # down_proj is more sensitive than up/gate and would warrant finer
        # groups; it cannot have them - see MOE_UNIFORMITY_NOTE.
        (r"experts\.down_proj",                    "W2A16", 128),
        (r"experts\.gate_up_proj",                 "W2A16", 128),  # same gs as down_proj
    ],
    # Like "quality", but halves the two pins that cost the most and hurt the
    # least: hyper_connection (a rank-320 bottleneck, where W8 is ample) and
    # linear_attn.out_proj. Everything flagged as particularly delicate -
    # self_attn in hybrid architectures, SSM parameters, the router - stays
    # bf16 exactly as in Qwen's own FP8.
    "balanced": [
        (r"ngram_embedding\.shard_", "SKIP", 0),
        (r"^mtp\.", "SKIP", 0),
        (r"embed_tokens", "SKIP", 0),
        (r"mlp\.gate(\.weight)?$|shared_expert_gate", "BF16", 0),
        (r"norm|layernorm", "BF16", 0),
        (r"linear_attn\.(A_log|dt_bias|conv1d)", "BF16", 0),
        (r"ngram_heads_|layer_multipliers", "BF16", 0),
        (r"\.ple\.(conv1d|key_proj|value_proj)", "BF16", 0),
        (r"self_attn\.", "BF16", 0),               # only 12 layers, Qwen pins them
        (r"hyper_connection", "W8A16", 128),
        (r"linear_attn\.out_proj", "W8A16", 128),
        (r"linear_attn\.", "W8A16", 128),
        (r"shared_expert\.", "W8A16", 128),
        (r"^lm_head", "BF16",  0),    # see LM_HEAD_NOTE
        (r"experts\.down_proj", "W2A16", 128),
        (r"experts\.gate_up_proj", "W2A16", 128),  # same gs - MOE_UNIFORMITY_NOTE
    ],
    # THE ONE THAT WAS SHIPPED. Follows Qwen's own sensitivity map (the
    # modules_to_not_convert list in the official FP8 repo) as far as the
    # budget allows: there, ONLY experts + ngram are quantized. That is not
    # affordable at this budget, but everything except the experts gets as
    # close to it as the arithmetic permits. Needs --ram-reserve 10, not 12.
    "quality": [
        (r"ngram_embedding\.shard_", "SKIP", 0),
        (r"^mtp\.", "SKIP", 0),
        (r"embed_tokens", "SKIP", 0),
        (r"mlp\.gate(\.weight)?$|shared_expert_gate", "BF16", 0),   # router
        (r"norm|layernorm", "BF16", 0),
        (r"linear_attn\.(A_log|dt_bias|conv1d)", "BF16", 0),      # SSM parameters
        # in_proj_a/b are [48, 2560] - 48 is not divisible by 32, so AutoRound
        # skips them. Without an explicit pin the base scheme still writes
        # 8 bits into the config, and the inference loader then fails with
        # "No compatible backend found for layer ...linear_attn.in_proj_a".
        # The tensor is bf16 either way; only the config entry is wrong.
        (r"linear_attn\.in_proj_(a|b)", "BF16", 0),
        (r"ngram_heads_|layer_multipliers|ple_embedding", "BF16", 0),
        (r"\.ple\.(conv1d|key_proj|value_proj)", "BF16", 0),     # Qwen pins these
        (r"hyper_connection", "BF16", 0),                        # Qwen pins these
        (r"self_attn\.", "BF16", 0),                             # Qwen pins these, 12 layers
        (r"linear_attn\.out_proj", "BF16", 0),                   # "ssm_out", never deep
        # Embeddings and convs are not Linear layers, so AutoRound never
        # touches them. They still have to appear as 16 bit in the config.
        (r"embed_tokens|patch_embed|pos_embed|fc_embedding", "BF16", 0),
        (r"^model\.visual\.", "BF16", 0),
        (r"linear_attn\.", "W8A16", 128),                        # rest of the GDN projections
        (r"shared_expert\.", "W8A16", 128),                      # runs on EVERY token
        (r"^lm_head", "BF16",  0),    # see LM_HEAD_NOTE
        (r"experts\.down_proj", "W2A16", 128),
        (r"experts\.gate_up_proj", "W2A16", 128),  # same gs - MOE_UNIFORMITY_NOTE
    ],
    # If the budget still overruns: everything one step sharper.
    "tight": [
        (r"ngram_embedding\.shard_", "SKIP", 0), (r"^mtp\.", "SKIP", 0),
        (r"embed_tokens", "SKIP", 0),
        (r"mlp\.gate(\.weight)?$|shared_expert_gate", "BF16", 0),
        (r"norm|layernorm", "BF16", 0),
        (r"linear_attn\.(A_log|dt_bias|conv1d)", "BF16", 0),
        (r"ngram_heads_|layer_multipliers", "BF16", 0),
        (r"\.ple\.(conv1d|key_proj|value_proj)", "W4A16", 128),
        (r"hyper_connection", "W4A16", 128),
        (r"linear_attn\.out_proj", "W4A16", 128),
        (r"linear_attn\.", "W4A16", 128),
        (r"self_attn\.", "W4A16", 128),
        (r"shared_expert\.", "W4A16", 128),
        (r"^model\.visual\.", "W4A16", 128),
        (r"^lm_head", "BF16",  0),    # see LM_HEAD_NOTE
        (r"experts\.down_proj", "W2A16", 128),
        (r"experts\.gate_up_proj", "W2A16", 128),  # same gs - MOE_UNIFORMITY_NOTE
    ],
    # If the RAM reserve is lowered and headroom appears: closer to Qwen's FP8.
    "conservative": [
        (r"ngram_embedding\.shard_", "SKIP", 0), (r"^mtp\.", "SKIP", 0),
        (r"embed_tokens", "SKIP", 0),
        (r"mlp\.gate(\.weight)?$|shared_expert_gate", "BF16", 0),
        (r"norm|layernorm", "BF16", 0),
        (r"linear_attn\.(A_log|dt_bias|conv1d)", "BF16", 0),
        (r"ngram_heads_|layer_multipliers", "BF16", 0),
        (r"\.ple\.(conv1d|key_proj|value_proj)", "BF16", 0),
        (r"hyper_connection", "BF16", 0),
        (r"linear_attn\.out_proj", "BF16", 0),
        (r"linear_attn\.", "W8A16", 128),
        (r"self_attn\.", "BF16", 0),
        (r"shared_expert\.", "W8A16", 128),
        (r"^model\.visual\.", "W8A16", 128),
        (r"^lm_head", "BF16",  0),    # see LM_HEAD_NOTE
        (r"experts\.down_proj", "W2A16", 128),
        (r"experts\.gate_up_proj", "W2A16", 128),  # same gs - MOE_UNIFORMITY_NOTE
    ],
}


def assign(name, policy):
    for rx, scheme, gs in policy:
        if re.search(rx, name):
            return scheme, gs
    return "W4A16", 128    # nothing falls through silently; shows as "!! unmatched"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "tensors.json"))
    ap.add_argument("--config",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "config.json"),
                    help="config.json (for the budget)")
    ap.add_argument("--policy", default="default", choices=list(POLICIES))
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--ram-reserve", type=float, default=None,
                    help="OS + other software")
    ap.add_argument("--ple-ring", type=float, default=0.0,
                    help="pinned hot cache for the PLE table, in GiB. "
                         "Comes out of the weight RAM.")
    ap.add_argument("--mem-fraction", type=float, default=None)
    ap.add_argument("--headless", action="store_true",
                    help="server with no desktop on the GPU (SSH). Frees the 1.3 GiB "
                         "the desktop holds and allows mem-fraction 0.95.")
    ap.add_argument("--safety-gib", type=float, default=0.8)
    ap.add_argument("--quant-vision", action="store_true",
                    help="quantize the vision tower (AutoRound: --quant_nontext_module). "
                         "Without this flag it stays bf16.")
    ap.add_argument("--no-skip-rule", action="store_true",
                    help="ignore the 'not divisible by 32' rule (for comparison only).")
    ap.add_argument("--emit-layer-config", action="store_true")
    a = ap.parse_args()

    if a.ram_reserve is not None:
        B.RAM_RESERVED_GIB = a.ram_reserve
    # The PLE ring is pinned and therefore not swappable - it comes straight
    # out of the weight RAM. The mmap page cache behind it costs nothing extra:
    # the kernel serves that elastically from what is already OS reserve.
    B.RAM_RESERVED_GIB += a.ple_ring
    if a.headless:
        B.VRAM_DESKTOP_GIB = 0.10
        if a.mem_fraction is None:
            B.MEM_FRACTION_STATIC = 0.95
    if a.mem_fraction is not None:
        B.MEM_FRACTION_STATIC = a.mem_fraction

    tensors = json.load(open(a.tensors))
    policy = POLICIES[a.policy]

    cfg = json.load(open(a.config)) if a.config and os.path.exists(a.config) else {}

    # -------------------------------------------------------- allocate
    rows = defaultdict(lambda: {"params": 0, "bytes": 0, "n": 0})
    skipped = defaultdict(int)
    unmatched = []
    forced_bf16 = defaultdict(lambda: [0, 0])
    for name, (shape, dt) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        scheme, gs = assign(name, policy)
        grp = _insp.classify(name)
        if scheme == "SKIP":
            skipped[grp] += n
            continue

        # AutoRound only quantizes tensors whose last two dimensions are both
        # divisible by 32. Everything else silently stays bf16:
        #   "some layers are skipped quantization (shape not divisible by 32)"
        # Here that catches linear_attn.in_proj_a/b and the whole vision MLP.
        # Not accounting for it means finding out in the size audit, after the
        # run - the tensors are larger than the recipe promised.
        if not a.no_skip_rule and scheme != "BF16" and len(shape) >= 2:
            if any(d % 32 for d in shape[-2:]):
                forced_bf16[grp][0] += n
                forced_bf16[grp][1] += 1
                scheme, gs = "BF16", 0

        # AutoRound only touches the vision tower with --quant_nontext_module.
        # Without that flag it stays entirely bf16, whatever the recipe says.
        if not a.quant_vision and grp == "vision" and scheme != "BF16":
            forced_bf16[grp][0] += n
            forced_bf16[grp][1] += 1
            scheme, gs = "BF16", 0

        key = (grp, scheme, gs)
        rows[key]["params"] += n
        rows[key]["bytes"] += n * bpw(scheme, gs) / 8
        rows[key]["n"] += 1

    print(f"\n=== Policy: {a.policy}"
          f"{' +vision' if a.quant_vision else ''} ===\n")
    print(f"  {'group':<13} {'scheme':<7} {'gs':>4} {'tensors':>9} {'params':>10} {'GiB':>8}")
    print("  " + "-" * 60)
    tot_p = tot_b = 0
    for (grp, scheme, gs), v in sorted(rows.items(), key=lambda x: -x[1]["bytes"]):
        gib = v["bytes"] / GIB
        tot_p += v["params"]; tot_b += v["bytes"]
        print(f"  {grp:<13} {scheme:<7} {gs if gs else '-':>4} {v['n']:>9} "
              f"{v['params']/1e9:>9.2f}B {gib:>8.3f}")
    print("  " + "-" * 60)
    print(f"  {'RESIDENT':<13} {'':<7} {'':>4} {'':>9} {tot_p/1e9:>9.2f}B {tot_b/GIB:>8.3f}")
    print(f"  effective bpw overall: {tot_b*8/tot_p:.3f}")

    if forced_bf16:
        print(f"\n  NOT quantizable by AutoRound -> forced to bf16:")
        for g, (n, cnt) in sorted(forced_bf16.items(), key=lambda x: -x[1][0]):
            print(f"    {g:<13} {cnt:>4} tensors   {n/1e9:>7.3f}B  "
                  f"= {n*2/GIB:.3f} GiB instead of {n*4.156/8/GIB:.3f} GiB")

    print(f"\n  Offloaded (not in the budget):")
    for g, n in sorted(skipped.items(), key=lambda x: -x[1]):
        print(f"    {g:<13} {n/1e9:>9.2f}B")

    # ------------------------------------------------------- Budget
    if cfg:
        bud = B.compute(cfg, tot_p / 1e9, max_model_len=a.max_model_len)
        limit = bud.total_weights_gib - a.safety_gib
        have = tot_b / GIB
        print(f"\n=== Budget @ {a.max_model_len} tokens, RAM reserve "
              f"{B.RAM_RESERVED_GIB} GiB, mem-fraction {B.MEM_FRACTION_STATIC} ===\n")
        print(f"  VRAM for weights : {bud.vram_weights_gib:7.2f} GiB")
        print(f"  RAM for weights  : {bud.ram_weights_gib:7.2f} GiB"
              + (f"  (after {a.ple_ring} GiB PLE ring)" if a.ple_ring else ""))
        print(f"  budget           : {bud.total_weights_gib:7.2f} GiB"
              f"  (- {a.safety_gib} safety = {limit:.2f})")
        print(f"  recipe needs     : {have:7.2f} GiB")
        head = limit - have
        mark = "[OK]" if head >= 0 else "[!!]"
        print(f"  {mark} {'headroom' if head>=0 else 'OVER BUDGET by'}: {abs(head):.2f} GiB")
        if head >= 0:
            extra = head * GIB * 8 / (tot_p)
            print(f"       = {extra:.3f} bpw of slack across all resident params")

        ple = B.ple_params_b(cfg)
        print(f"\n  PLE separately: {ple:.1f}B  -> fp8 {B.gib_for(ple,8):.1f} GiB on NVMe")

    if a.emit_layer_config:
        # CAREFUL - two different namespaces:
        #
        #   checkpoint : model...mlp.experts.gate_up_proj   [512, 1280, 2560]  (stacked)
        #   runtime    : model...mlp.experts.<N>.gate_proj  [640, 2560]
        #                model...mlp.experts.<N>.up_proj    [640, 2560]
        #                model...mlp.experts.<N>.down_proj  [2560, 640]
        #
        # auto-round calls prepare_model_for_moe_quantization(), which unstacks
        # the 3D parameters into three nn.Linear per expert. layer_config is
        # matched against the RUNTIME names - a regex on "experts.gate_up_proj"
        # matches nothing there and falls back to the default scheme without
        # saying so.
        RUNTIME = {
            r"experts\.gate_up_proj": r"experts\.\d+\.(gate|up)_proj",
            r"experts\.down_proj":    r"experts\.\d+\.down_proj",
        }
        lc = {}
        for rx, scheme, gs in policy:
            if scheme == "SKIP":
                continue
            rt = rx
            for chk, run in RUNTIME.items():
                rt = rt.replace(chk, run)
            key = rt if rt.startswith("^") else ".*" + rt
            lc[key] = ({"bits": 16, "act_bits": 16} if scheme == "BF16"
                       else {"scheme": scheme, "group_size": gs})
        print("\n=== layer_config for 05_quantize.py ===")
        print("    (runtime names, after unstacking - NOT checkpoint names)\n")
        print(json.dumps(lc, indent=2))

        print("\n=== What does NOT need to be in layer_config ===\n")
        print("  router  : Qwen4ExpTextTopKRouter is nn.Parameter + F.linear,")
        print("            not nn.Linear -> AutoRound never sees it and leaves")
        print("            it bf16. The rule satisfies itself.")
        print("  shared_expert_gate : nn.Linear(hidden, 1) -> 1 is not divisible")
        print("            by 32 -> skipped -> bf16.")


if __name__ == "__main__":
    main()
