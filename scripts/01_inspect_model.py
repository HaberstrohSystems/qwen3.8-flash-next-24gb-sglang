#!/usr/bin/env python3
"""
Step 1: discovery. Reads ONLY config.json plus the safetensors index (a few
MB), never the weights. It answers the four questions that decide the entire
budget before any compute is spent:

  1. What are the expert tensors called?  -> --cpu-offload-params
  2. Is there an n-gram table, how large, can it be offloaded?
  3. Hybrid attention? -> KV cache cost -> how much is left for weights
  4. Which avg_bits actually fits?

Writes discovered.json, which 05_quantize.py and the serve script read.

  python3 01_inspect_model.py Qwen/Qwen3.8-Flash-Next
  python3 01_inspect_model.py /path/to/local/checkpoint
"""
import argparse, json, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "F8_E5M2": 1,
               "I8": 1, "U8": 1, "I32": 4, "I64": 8, "BOOL": 1}

# Classification heuristics. Order matters - first match wins.
# Verified against the real model.safetensors.index.json (1658 tensors, 227
# distinct patterns). Regexes written for flat Llama-style names do not work
# here; they drop almost everything into "other", and the budget silently
# comes out wrong rather than failing.
#
# Real prefixes: model.language_model.layers.N.* / model.visual.* / mtp.*
GROUPS = [
    # PLE/n-gram first - 128 shards, all at layer 2, 51.2B params.
    # Source: shard_0..127 ; AutoRound output: a literal "shard_*" (collapsed)
    ("ngram",      re.compile(r"ngram_embedding\.shard_(\d+|\*)", re.I)),
    ("ngram_meta", re.compile(r"ngram_heads_(offsets|vocab_sizes)|layer_multipliers", re.I)),
    # MTP is entirely separate - optional, speculative decoding only.
    ("mtp",        re.compile(r"^mtp\.", re.I)),
    # Vision encoder separate - small, stays at high precision.
    ("vision",     re.compile(r"^model\.visual\.", re.I)),
    # Hyper-connections (gated residual, hc_count=4, rank=320) - tiny, bf16.
    ("hyperconn",  re.compile(r"hyper_connection", re.I)),
    # The rest of the PLE machinery (conv1d, key/value_proj) - small.
    ("ple_other",  re.compile(r"\.ple\.", re.I)),
    # Router/gates - bf16, NEVER quantize. A rounding error here does not
    # degrade an output, it routes the token to a different expert.
    ("router",     re.compile(r"mlp\.gate\.weight$|shared_expert_gate|(^|\.)router\.", re.I)),
    ("norm",       re.compile(r"norm|layernorm|\bln_", re.I)),
    ("token_embd", re.compile(r"embed_tokens|tok_embeddings|word_embeddings|token_embd", re.I)),
    ("lm_head",    re.compile(r"^lm_head|output\.weight$", re.I)),
    # Gated DeltaNet: A_log/dt_bias/conv1d are SSM parameters -> do not quantize.
    ("gdn_state",  re.compile(r"linear_attn\.(A_log|dt_bias|conv1d)", re.I)),
    ("linear_attn", re.compile(r"linear_attn\.", re.I)),
    # QSA full attention, including the indexer.
    ("attn",       re.compile(r"self_attn\.", re.I)),
    ("shared_exp", re.compile(r"shared_expert\.", re.I)),
    # Experts are STACKED 3D tensors [512, in, out] with NO ".weight" suffix.
    # Source is stacked (experts.down_proj); output is unstacked
    # (experts.7.down_proj), so both spellings have to match.
    ("exp_down",   re.compile(r"experts\.(\d+\.)?down_proj", re.I)),
    ("exp_upgate", re.compile(r"experts\.(\d+\.)?(gate_up|gate|up)_proj", re.I)),
    ("dense_mlp",  re.compile(r"mlp\.|feed_forward", re.I)),
]


def classify(name):
    for g, rx in GROUPS:
        if rx.search(name):
            return g
    return "other"


def load_json(src, fname):
    if os.path.isdir(src):
        p = os.path.join(src, fname)
        return json.load(open(p)) if os.path.exists(p) else None
    from huggingface_hub import hf_hub_download
    try:
        return json.load(open(hf_hub_download(src, fname)))
    except Exception as e:
        print(f"  [!] {fname} not loadable: {e}")
        return None


def tensor_shapes(src):
    """Return {name: (shape, dtype)} from the safetensors header, no weights."""
    out = {}
    if os.path.isdir(src):
        import glob
        from safetensors import safe_open
        for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
            with safe_open(f, framework="pt") as h:
                for k in h.keys():
                    sl = h.get_slice(k)
                    out[k] = (list(sl.get_shape()), sl.get_dtype())
        return out

    # Remote: get_safetensors_metadata() reads ONLY the shard headers.
    #
    # Do not hand-roll the range requests:
    #     requests.get(url, headers={"Range": "bytes=0-7"})
    # The HF resolve endpoint redirects to a CDN, and the Range header is
    # dropped across the redirect -> requests downloads the entire 3.5 GB
    # shard, .content materializes it, and the OOM killer takes the process
    # (exit 137). With 18 GiB free this happens on the first shard.
    #
    # get_safetensors_metadata does the same thing correctly, and in parallel.
    from huggingface_hub import get_safetensors_metadata
    try:
        meta = get_safetensors_metadata(src)
        for fname, fmeta in meta.files_metadata.items():
            for k, t in fmeta.tensors.items():
                out[k] = (list(t.shape), t.dtype)
        return out
    except Exception as e:
        print(f"  [!] get_safetensors_metadata failed: {e}")
        print(f"      Falling back to names from the index; shapes from the config.")

    # Fallback without shapes - enough to classify names, not enough to budget.
    idx = load_json(src, "model.safetensors.index.json")
    if idx:
        for k in idx.get("weight_map", {}):
            out[k] = ([], "BF16")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("-o", "--out", default="discovered.json")
    a = ap.parse_args()

    print(f"\n=== Discovery: {a.model} ===\n")

    cfg = load_json(a.model, "config.json") or {}
    arch = (cfg.get("architectures") or ["?"])[0]
    # Qwen4-Exp NESTS config.json: everything lives under "text_config".
    # Without unwrap() every cfg.get() returns None and the defaults take over.
    tc = B.unwrap(cfg)
    print(f"  architecture     : {arch}")
    print(f"  layers           : {tc.get('num_hidden_layers', '?')}")
    print(f"  hidden_size      : {tc.get('hidden_size', '?')}")
    print(f"  experts          : {tc.get('num_experts') or tc.get('n_routed_experts', '?')}")
    print(f"  active/token     : {tc.get('num_experts_per_tok') or tc.get('top_k', '?')}")

    n_layers = B.n_layers_of(cfg)
    full_attn = B._count_full_attention_layers(cfg, n_layers)
    hybrid = full_attn < n_layers
    print(f"  full attention   : {full_attn}/{n_layers} layers"
          f"{'  -> HYBRID, KV is cheap' if hybrid else '  -> all full'}")

    # ----------------------------------------------------------- tensors
    print("\n  Reading tensor headers ...")
    shapes = tensor_shapes(a.model)
    if not shapes:
        print("  [!] No tensors found. Is the model published yet?")
        sys.exit(1)

    per_group, names, dtypes = defaultdict(int), defaultdict(list), defaultdict(set)
    for name, (shape, dt) in shapes.items():
        n = 1
        for d in shape:
            n *= d
        g = classify(name)
        per_group[g] += n
        dtypes[g].add(dt)
        if len(names[g]) < 4:
            names[g].append(name)

    total = sum(per_group.values())
    print(f"\n  {'group':<12} {'params':>12} {'%':>7}  {'dtype':<8} example")
    print("  " + "-" * 76)
    for g, n in sorted(per_group.items(), key=lambda x: -x[1]):
        dt = ",".join(sorted(dtypes[g]))
        print(f"  {g:<12} {n/1e9:>10.2f}B {100*n/total:>6.1f}%  {dt:<8} "
              f"{names[g][0][:34] if names[g] else ''}")
    print(f"  {'TOTAL':<12} {total/1e9:>10.2f}B")

    # ------------------------------------------ offload parameter names
    # Experts are stacked 3D tensors with NO ".weight" suffix:
    #   model.language_model.layers.N.mlp.experts.gate_up_proj  [512, 2560, 1280]
    #   model.language_model.layers.N.mlp.experts.down_proj     [512, 640, 2560]
    # The offload matcher works on substrings, so "experts." MUST be part of
    # the token: a bare "down_proj" would also match shared_expert.down_proj,
    # and the shared expert runs on EVERY token - it belongs on the GPU.
    offload_tokens = set()
    for g in ("exp_down", "exp_upgate"):
        for full in names[g]:
            parts = full.split(".")
            if parts[-1] == "weight":
                parts = parts[:-1]
            offload_tokens.add(".".join(parts[-2:]))
    offload = sorted(offload_tokens)

    # ------------------------------------------------------------ Budget
    params_b = total / 1e9
    # What does NOT count against the weight budget:
    #   ngram      - 51.2B PLE table, sparse lookup, belongs offloaded (NVMe/CPU)
    #   token_embd - a lookup, mmap-able
    #   mtp        - optional; absent unless speculative decoding is enabled
    ple_b = per_group["ngram"] / 1e9
    mmap_b = ple_b + per_group["token_embd"] / 1e9
    mtp_b = per_group["mtp"] / 1e9
    resident_b = params_b - mmap_b - mtp_b

    bud = B.compute(cfg, resident_b, max_model_len=a.max_model_len)
    # Pinned to bf16: router, norms, hyper-connections, SSM parameters, PLE
    # metadata. All tiny, all sensitive - the cost of keeping them is noise.
    bf16_b = sum(per_group[g] for g in
                 ("router", "norm", "hyperconn", "gdn_state", "ngram_meta")) / 1e9
    target = B.autoscheme_target(bud, bf16_b)

    print("\n=== Budget ===\n")
    print(bud.pretty())
    print(f"  NOT in the budget (offloaded):")
    print(f"    PLE/N-Gram : {ple_b:6.2f}B  = bf16 {B.gib_for(ple_b,16):6.1f} GiB /"
          f" fp8 {B.gib_for(ple_b,8):5.1f} GiB / 4bit {B.gib_for(ple_b,4.156):5.1f} GiB")
    print(f"    token_embd : {per_group['token_embd']/1e9:6.2f}B")
    print(f"    MTP        : {mtp_b:6.2f}B  (optional, speculative decoding only)")
    print(f"  pinned bf16  : {bf16_b:6.3f}B  "
          f"(router+norms+hyperconn+SSM) = {B.gib_for(bf16_b,16):.2f} GiB\n")
    print(f"  >>> AutoScheme avg_bits      : {target}")
    print(f"  >>> --cpu-offload-params     : {' '.join(offload) if offload else '!! NOT DETECTED !!'}")
    print(f"  >>> --cpu-offload-gb start   : {bud.ram_weights_gib:.0f}")
    print(f"  >>> max context at {bud.kv_cache_gib} GiB KV: "
          f"{B.max_context_for_kv(cfg, bud.kv_cache_gib):,} tokens")

    # ---------------------------------------------------------- warnings
    warn = []
    if not offload:
        warn.append("Expert tensors not recognized -> adjust the GROUPS regexes in this "
                    "script, otherwise --cpu-offload-params cannot be set.")
    if per_group["ngram"] == 0:
        warn.append("No n-gram table found. Either it is named differently (check the "
                    "tensor list by hand) or it is folded into token_embd.")
    else:
        ple_layers = sorted({int(m.group(1)) for n in shapes
                             if "ngram_embedding" in n
                             for m in [re.search(r"layers\.(\d+)\.", n)] if m})
        if len(ple_layers) > 1:
            warn.append(f"PLE table sits at MULTIPLE layers {ple_layers} - then it is read "
                        "densely and the budget no longer holds. Check this.")
        ple_fp8 = B.gib_for(ple_b, 8)
        ram_free = B.RAM_TOTAL_GIB - B.RAM_RESERVED_GIB
        if ple_fp8 > ram_free:
            warn.append(f"The PLE table needs {ple_fp8:.1f} GiB even at fp8, but only "
                        f"{ram_free:.1f} GiB of RAM is free. SGLang's --ple-offload-embedding "
                        "uses CPU-PINNED memory and therefore does NOT fit. It has to be "
                        "mmap-ed from NVMe -> see WRITEUP.md, finding 1.")
    if target < 2.1:
        warn.append(f"avg_bits {target} is below W2A16-g128 (2.156). Reduce KV cache/context "
                    "first, then raise group_size. Do NOT simply keep lowering it.")
    if per_group["other"] / total > 0.05:
        warn.append(f"{100*per_group['other']/total:.1f}% of parameters unclassified - "
                    "check the GROUPS regexes.")
    if warn:
        print("\n=== WARNINGS ===")
        for w in warn:
            print(f"  [!] {w}")

    out = {
        "model": a.model, "architecture": arch, "config": cfg,
        "params_b_total": round(params_b, 3),
        "params_b_resident": round(resident_b, 3),
        "params_by_group_b": {k: round(v / 1e9, 4) for k, v in per_group.items()},
        "example_names": dict(names),
        "hybrid_attention": hybrid,
        "full_attn_layers": full_attn, "n_layers": n_layers,
        "budget": bud.__dict__,
        "autoscheme_avg_bits": target,
        "cpu_offload_params": offload,
        "cpu_offload_gb_start": round(bud.ram_weights_gib),
        "max_context_at_kv_budget": B.max_context_for_kv(cfg, bud.kv_cache_gib),
        "warnings": warn,
    }
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n  -> wrote {a.out}\n")


if __name__ == "__main__":
    main()
