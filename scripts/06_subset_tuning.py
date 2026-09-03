#!/usr/bin/env python3
"""
Tune experts in subsets, so AutoRound's learned rounding fits in 24 GB.

THE PROBLEM
-----------
A Qwen4-Exp block has 512 experts x 3 projections = 1536 Linear layers,
2.52B parameters together. AutoRound wraps them ALL at once and allocates a
`value` tensor per weight, the same size as the weight (the learned rounding
offset), plus its gradient; on top of that autograd holds the qdq
intermediates of all 1536 layers for the backward pass:

    weights bf16                  4.69 GiB
    value                         4.69 GiB   (fp16 after patch 3)
    gradient                      4.69 GiB
    qdq intermediates            ~15    GiB
    -------------------------------------
                                 ~29    GiB   on a 23.42 GiB card

WHY SUBSETS ARE LEGITIMATE HERE
-------------------------------
Experts are parallel, additive branches: expert i only affects the tokens the
router sends to it, and there is no interaction between experts within a
block. Tuning them one group after another is therefore not an approximation.

Better than that: `unwrapper_block` writes the quantized weights back after
each round. Round k+1 sees the experts from round k already quantized, and
tunes against the same unchanged reference output of the original block. That
is sequential error compensation, of the kind GPTQ performs across columns.

COST
----
Each round runs the full block forward but backpropagates through only 1/N of
the experts. The forward cost multiplies; the backward cost does not. Measure
one block before committing to a full run.

  python3 06_subset_tuning.py --self-test        # on a miniature model
  # in a run:  import ...; subset.install(n_groups=4)
"""
import functools, re, sys

EXP_RX = re.compile(r"(?:^|\.)experts\.(\d+)\.")
_state = {"allow": None, "first": True, "groups": 0}


def plan_groups(block, n_groups):
    """Split a block's quantizable layers into n groups.

    Group 0 additionally gets everything that is not an expert (attention,
    linear_attn, shared_expert) - that part is small and should run once.
    """
    from auto_round.wrapper import SUPPORTED_LAYER_TYPES
    from auto_round.utils import check_to_quantized

    experts, rest = [], []
    for n, m in block.named_modules():
        if type(m) not in SUPPORTED_LAYER_TYPES:
            continue
        if not check_to_quantized(m):
            continue
        mt = EXP_RX.search(n)
        (experts if mt else rest).append((int(mt.group(1)) if mt else -1, n))

    if not experts or n_groups <= 1:
        return [{n for _, n in experts + rest}]

    groups = [set() for _ in range(n_groups)]
    for i, n in experts:
        groups[i % n_groups].add(n)
    groups[0] |= {n for _, n in rest}
    return [g for g in groups if g]


def install(n_groups=4, verbose=True):
    """Patch AutoRound so that a block is tuned in n rounds."""
    import auto_round.algorithms.quantization.sign_round.quantizer as Q
    from auto_round.wrapper import SUPPORTED_LAYER_TYPES, WrapperLinear, NORM_MAPPING
    from auto_round.utils import check_to_quantized, set_module
    from auto_round.utils.common import logger

    if getattr(Q, "_subset_tuning_installed", False):
        if verbose:
            print("  [i] subset tuning already installed")
        return

    def wrapper_block_filtered(block, enable_minmax_tuning, enable_norm_bias_tuning,
                               enable_torch_compile=False, device="cpu",
                               wrapper_cls=WrapperLinear, **kwargs):
        allow = _state["allow"]
        quantized, unquantized = [], []
        for n, m in list(block.named_modules()):
            if type(m) in SUPPORTED_LAYER_TYPES:
                if not check_to_quantized(m):
                    unquantized.append(n)
                    continue
                if allow is not None and n not in allow:
                    continue            # belongs to a different round
                new_m = wrapper_cls(
                    m,
                    enable_minmax_tuning=enable_minmax_tuning,
                    enable_norm_bias_tuning=enable_norm_bias_tuning,
                    enable_torch_compile=enable_torch_compile,
                    device=device,
                    **kwargs,
                )
                set_module(block, n, new_m)
                quantized.append(n)
            # Norm/bias tuning in the FIRST round only, else applied repeatedly
            elif enable_norm_bias_tuning and _state["first"]:
                cls_name = m.__class__.__name__
                if "norm" in cls_name.lower():
                    key = cls_name if cls_name in NORM_MAPPING else (
                        "LlamaRMSNorm" if "RMSNorm" in cls_name else None)
                    if key:
                        set_module(block, n, NORM_MAPPING[key](m, device=device))
        return quantized, unquantized

    orig_quantize_block = Q.SignRoundQuantizer.quantize_block

    def quantize_block_subsets(self, ctx):
        groups = plan_groups(ctx.block, n_groups)
        if len(groups) <= 1:
            return orig_quantize_block(self, ctx)

        # Do NOT simply replace the wrapper that is already installed - adopt
        # it. With --enable_alg_ext, AutoRound attaches its own wrapper class
        # to the instance:
        #   sign_roundv2/quantizer.py:334
        #   self.wrapper_block = partial(wrapper_block,
        #                                wrapper_cls=SignRoundOptimizedWrapperLinear)
        # A blind self.wrapper_block = ... would silently revert that to the
        # standard WrapperLinear: alg_ext would then appear active and do
        # nothing. Extract the bound class and pass it through the name filter.
        prev = getattr(self, "wrapper_block", None)
        wcls = None
        if isinstance(prev, functools.partial):
            wcls = prev.keywords.get("wrapper_cls")
        if wcls is not None:
            filtered = functools.partial(wrapper_block_filtered, wrapper_cls=wcls)
            if verbose:
                logger.info(f"[subset tuning] adopting wrapper "
                            f"{wcls.__name__}")
        else:
            filtered = wrapper_block_filtered

        merged = {}
        for k, names in enumerate(groups):
            _state["allow"] = names
            _state["first"] = (k == 0)
            self.wrapper_block = filtered
            if verbose:
                logger.info(f"[subset tuning] round {k+1}/{len(groups)}: "
                            f"{len(names)} layers")
            res = orig_quantize_block(self, ctx)
            if res:
                merged.update(res)
        _state["allow"] = None
        _state["first"] = True
        if prev is not None:                 # leave the instance as we found it
            self.wrapper_block = prev
        return merged

    Q.wrapper_block = wrapper_block_filtered
    Q.SignRoundQuantizer.quantize_block = quantize_block_subsets
    Q._subset_tuning_installed = True
    _state["groups"] = n_groups
    if verbose:
        print(f"  subset tuning installed: {n_groups} rounds per block")


def _self_test():
    import torch, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextSparseMoeBlock)
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    cfg = Qwen4ExpTextConfig(hidden_size=64, moe_intermediate_size=32, num_experts=16,
                             num_experts_per_tok=2, shared_expert_intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=32, hc_count=4,
                             hc_lowrank=16, ple_layer_ids=[], eos_token_id=1,
                             vocab_size=128)
    blk = Qwen4ExpTextSparseMoeBlock(cfg)
    import auto_round.modeling.fused_moe.moe_experts_interface as mi
    mi.register_linear_loop_experts()
    mi._unfuse_experts_weights_inplace(blk.experts)
    # check_to_quantized() needs the attributes AutoRound normally sets
    from auto_round.wrapper import SUPPORTED_LAYER_TYPES
    for _n, _m in blk.named_modules():
        if type(_m) in SUPPORTED_LAYER_TYPES:
            _m.bits, _m.group_size, _m.sym = 2, 128, True
            _m.data_type, _m.act_bits = "int", 16
    for n in (1, 2, 4, 8):
        g = plan_groups(blk, n)
        sizes = [len(x) for x in g]
        total = sum(sizes)
        overlap = total - len({x for s in g for x in s})
        print(f"  n={n}: {len(g)} rounds, sizes {sizes}, "
              f"total {total}, overlap {overlap}")
    print("\n  Overlap must be 0 and the total must stay constant.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print(__doc__)
