#!/usr/bin/env python3
"""
Build a miniature Qwen4-Exp with the REAL tensor structure but tiny
dimensions. The point is to exercise the entire pipeline end to end before
committing days of compute to the real model.

Do not skip this because you have quantized a similar model before. A dense
model of the same family has neither stacked experts nor a PLE table - which
are precisely the two things that break here.

Peculiarities reproduced on purpose:
  - layer_types in the 3:1 pattern (linear_attention x3 -> full_attention)
  - PLE/n-gram at ONE layer, split into several shard_N
  - experts as stacked 3D parameters (gate_up_proj / down_proj)
  - hyper-connections (hc_count/hc_lowrank)
  - a vision MLP with a dimension NOT divisible by 32
    -> reproduces AutoRound skipping it silently
  - MTP head

  python3 make_mini_model.py -o ~/quant/mini-flashnext
"""
import argparse, json, os


def fuse_experts(path):
    """Entstapelte experts.<N>.{gate,up,down}_proj zu 3D-Tensoren zusammenfassen."""
    import glob, re, torch
    from collections import defaultdict
    from safetensors.torch import load_file, save_file

    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    tensors, origin = {}, {}
    for f in files:
        for k, v in load_file(f).items():
            tensors[k] = v
            origin[k] = f

    rx = re.compile(r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
    groups = defaultdict(dict)
    for k in list(tensors):
        m = rx.match(k)
        if m:
            groups[m.group(1)].setdefault(int(m.group(2)), {})[m.group(3)] = k

    if not groups:
        print("  [i] no unstacked experts found - nothing to fuse")
        return

    n_fused = 0
    for prefix, experts in groups.items():
        idx = sorted(experts)
        gate_up = torch.stack([torch.cat([tensors[experts[i]["gate_proj"]],
                                          tensors[experts[i]["up_proj"]]], dim=0)
                               for i in idx], dim=0)
        down = torch.stack([tensors[experts[i]["down_proj"]] for i in idx], dim=0)
        for i in idx:
            for p in ("gate_proj", "up_proj", "down_proj"):
                del tensors[experts[i][p]]
        tensors[f"{prefix}.gate_up_proj"] = gate_up.contiguous()
        tensors[f"{prefix}.down_proj"] = down.contiguous()
        n_fused += 1

    for f in files:
        os.remove(f)
    idxf = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idxf):
        os.remove(idxf)

    # neu schreiben, in Shards
    shard, shards, size, LIMIT = {}, [], 0, 200 * 1024 ** 2
    for k, v in tensors.items():
        b = v.numel() * v.element_size()
        if size + b > LIMIT and shard:
            shards.append(shard); shard, size = {}, 0
        shard[k] = v; size += b
    if shard:
        shards.append(shard)

    weight_map, total = {}, 0
    for i, sh in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, os.path.join(path, name), metadata={"format": "pt"})
        for k, v in sh.items():
            weight_map[k] = name
            total += v.numel() * v.element_size()
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(idxf, "w"), indent=2)
    print(f"  {n_fused} Experten-Bloecke gestapelt -> {len(shards)} Shards neu geschrieben")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=os.path.expanduser("~/quant/mini-flashnext"))
    ap.add_argument("--layers", type=int, default=8, help="multiple of 4")
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=256)
    a = ap.parse_args()

    import torch
    from transformers.models.qwen4_exp.configuration_qwen4_exp import (
        Qwen4ExpConfig, Qwen4ExpTextConfig, Qwen4ExpVisionConfig)
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpForConditionalGeneration)

    H = a.hidden
    # 3 x linear_attention -> 1 x full_attention, wie im echten Modell
    layer_types = [("full_attention" if (i + 1) % 4 == 0 else "linear_attention")
                   for i in range(a.layers)]

    text = Qwen4ExpTextConfig(
        # Must match the REAL tokenizer, otherwise token IDs run off the end
        # of the embedding ("index out of range in self"). Costs 2 x 248320 x H
        # parameters, but it is the only variant that works with the real chat
        # template and the MLLM calibration path.
        vocab_size=248320,
        hidden_size=H,
        num_hidden_layers=a.layers,
        layer_types=layer_types,
        full_attention_interval=4,
        # --- QSA ---
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        partial_rotary_factor=0.25,
        indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=32,
        indexer_budget=256, indexer_compress_ratio=4,
        # --- Gated DeltaNet ---
        linear_num_value_heads=8, linear_num_key_heads=4,
        linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        mamba_ssm_dtype="float32",
        # --- MoE, gestapelte Experten ---
        num_experts=a.experts,
        num_experts_per_tok=4,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        # --- Gated Residual ---
        hc_count=4, hc_lowrank=32,
        # --- PLE / N-Gram: EIN Layer, mehrere Shards ---
        ple_layer_ids=[2],
        ple_embed_dim=H,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=16384,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=8,
        # --- MTP ---
        mtp_num_hidden_layers=1,
        # The MTP head is only created with this dict, not from
        # mtp_num_hidden_layers alone - copied from the real config.json.
        mtp={"hybrid": True, "layer_types": ["full_attention"],
             "mtp_use_hidden_state_from_layer": None,
             "num_hidden_layers": 1, "rope_theta": 10000000},
        rope_parameters={"rope_type": "default", "rope_theta": 10000000,
                         "partial_rotary_factor": 0.25,
                         "mrope_interleaved": True, "mrope_section": [11, 11, 10]},
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        # Required as soon as ple_layer_ids is set: the n-gram context is
        # initialized with eos (NGramPool.reset_slots).
        bos_token_id=248044, eos_token_id=248044, pad_token_id=None,
    )
    # Vision: intermediate_size DELIBERATELY not divisible by 32 (the real
    # model has 4304). Reproduces AutoRound skipping it without a word.
    vision = Qwen4ExpVisionConfig(
        depth=2, hidden_size=128, intermediate_size=268,
        num_heads=4, out_hidden_size=H, patch_size=16,
        spatial_merge_size=2, temporal_patch_size=2,
        num_position_embeddings=64,
    )
    # Spezial-Token-IDs wie im echten Modell
    cfg = Qwen4ExpConfig(text_config=text, vision_config=vision,
                         image_token_id=248056, video_token_id=248057,
                         vision_start_token_id=248053, vision_end_token_id=248054,
                         tie_word_embeddings=False)

    print(f"\n=== Miniatur-Qwen4-Exp ===")
    print(f"  layers {a.layers}  ({layer_types.count('full_attention')} volle Attention)")
    print(f"  hidden {H}  experten {a.experts}  vocab {text.vocab_size}")
    print(f"  ngram  {text.ngram_vocab_size_base} x {H} in {text.split_ngram_parts} Shards")

    torch.manual_seed(0)
    model = Qwen4ExpForConditionalGeneration(cfg)
    model = model.to(torch.bfloat16)
    n = sum(p.numel() for p in model.parameters())
    print(f"  Parameter: {n/1e6:.1f} M")

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True, max_shard_size="200MB")
    cfg.save_pretrained(a.out)

    # Tokenizer placeholder, so AutoRound has something to load
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-Flash-Next")
        tk.save_pretrained(a.out)
        print("  tokenizer taken from the real model")
    except Exception as e:
        print(f"  [!] tokenizer not copied: {type(e).__name__}")

    # ---------------------------------------------------------------- fusion
    # save_pretrained writes the experts UNSTACKED (experts.<N>.gate_proj).
    # The real checkpoint has them STACKED as a 3D tensor:
    #     experts.gate_up_proj  [E, 2*inter, hidden]
    #     experts.down_proj     [E, hidden,  inter ]
    # Without this post-processing the dry run exercises everything except the
    # load path that actually matters. So rebuild it.
    fuse_experts(a.out)

    print(f"\n  -> {a.out}")
    print(f"\n=== Structure check ===")
    import glob
    from safetensors import safe_open
    names = []
    for f in glob.glob(os.path.join(a.out, "*.safetensors")):
        with safe_open(f, framework="pt") as h:
            names += list(h.keys())
    import re
    for pat, label in [(r"ngram_embedding\.shard_\d+", "PLE-Shards"),
                       (r"experts\.gate_up_proj$", "gestapelte gate_up"),
                       (r"experts\.down_proj$", "gestapelte down"),
                       (r"hyper_connection", "Hyper-Connections"),
                       (r"^mtp\.", "MTP"),
                       (r"visual\.blocks\.\d+\.mlp", "Vision-MLP"),
                       (r"linear_attn\.", "Gated DeltaNet"),
                       (r"self_attn\.indexer", "QSA-Indexer")]:
        hits = [n for n in names if re.search(pat, n)]
        print(f"  {label:<22} {len(hits):>4}   {hits[0] if hits else '!! FEHLT !!'}")
    print(f"  {'GESAMT':<22} {len(names):>4} Tensoren\n")


if __name__ == "__main__":
    main()
