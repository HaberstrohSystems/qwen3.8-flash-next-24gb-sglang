"""
Memory budget calculator for Qwen3.8-Flash-Next.

Verified against the actual config.json, not estimated.

UNITS: hardware figures (nvidia-smi, free, RAM sticks) are GiB = 1024^3, and
this module works in GiB throughout.

Four traps this module exists to avoid, all of which produce a plausible but
wrong number rather than an error:

  1. config.json is NESTED. Everything relevant lives under "text_config".
     Reading cfg["num_hidden_layers"] at the top level returns None and falls
     back to defaults (n_kv=8, head_dim=128, all 48 layers full attention),
     which overstates KV per token by 8x. Always go through `unwrap()`.

  2. Hybrid detection fails for the same reason: "layer_types" is also under
     text_config. The truth is 12 of 48 layers with full attention (interval
     4); the other 36 are Gated DeltaNet with a constant state rather than a
     growing KV cache.

  3. Hardware has to be measured, not assumed. On the reference machine the
     GPU reports 24467 MiB = 23.89 GiB (not 24.0) and the host has 30 GiB
     (not 32). Both shrink the budget; the hybrid correction in (2) grows it.

  4. The n-gram/PLE table (51.2B parameters) is a SEPARATE budget. It sits
     entirely at one layer, in 128 shards, and is read sparsely. It does not
     belong in total_weights_gib. See ple_gib().
"""
from dataclasses import dataclass

GIB = 1024 ** 3
GB = 10 ** 9

# --------------------------------------------- Hardware (measure these!)
VRAM_TOTAL_GIB = 23.89     # nvidia-smi: 24467 MiB
VRAM_DESKTOP_GIB = 1.30    # what the desktop already holds (nvidia-smi: 1293 MiB)
MEM_FRACTION_STATIC = 0.90 # SGLang knob; fraction of TOTAL VRAM
RAM_TOTAL_GIB = 30.0       # free -g: 30
RAM_RESERVED_GIB = 12.0    # OS + other software, set by the operator

# ------------------------------------------------------- fixed VRAM costs
CUDA_CTX_GIB = 1.0
ACTIVATIONS_EAGER_GIB = 1.2
ACTIVATIONS_GRAPH_GIB = 1.8
RAM_SLACK_GIB = 1.0


def unwrap(cfg: dict) -> dict:
    """
    Return the text half of the config. Qwen4-Exp nests everything under
    "text_config"; older models keep it flat. Skip this and every KV formula
    silently computes with defaults instead of real values.
    """
    if not isinstance(cfg, dict):
        return {}
    tc = cfg.get("text_config")
    return tc if isinstance(tc, dict) and tc else cfg


def real_bpw(bits: int, group_size: int = 128) -> float:
    """Real bpw including the fp16 scale (16 bit) and zero point (4 bit) per group."""
    return bits + (16 + 4) / group_size


def gib_for(params_b: float, bpw: float) -> float:
    return params_b * 1e9 * bpw / 8 / GIB


def bpw_for(params_b: float, gib: float) -> float:
    return gib * GIB * 8 / (params_b * 1e9) if params_b else 0.0


# ============================================================== KV / State

def _count_full_attention_layers(cfg: dict, n_layers: int) -> int:
    """
    Only real attention layers have a cache that GROWS with the context.
    In Qwen4-Exp that is 12 of 48 (layer_types, interval 4). The other 36 are
    Gated DeltaNet: constant state per sequence, see linear_state_gib().
    """
    c = unwrap(cfg)
    for key in ("layer_types", "layers_block_type", "block_types"):
        lt = c.get(key)
        if isinstance(lt, list) and lt:
            return sum(1 for t in lt
                       if "full" in str(t).lower() or str(t).lower() == "attention")
    for key in ("full_attention_interval", "attn_every_n_layers", "full_attn_interval"):
        v = c.get(key)
        if isinstance(v, int) and v > 1:
            return max(1, n_layers // v)
    return n_layers   # conservative


def n_layers_of(cfg: dict) -> int:
    c = unwrap(cfg)
    return c.get("num_hidden_layers") or c.get("n_layer") or 48


def kv_cache_per_token_bytes(cfg: dict, dtype_bytes: float = 1.0) -> float:
    """
    KV per token in bytes; dtype_bytes=1.0 for fp8, 2.0 for bf16.

    Two costs per full-attention layer:
      - QSA K/V:      2 * num_key_value_heads * head_dim
      - indexer cache: indexer_kv_heads * indexer_head_dim / indexer_compress_ratio
        (compressed addressing full_slot // ratio, see SGLang qsa/config.py)

    Qwen4-Exp: 12 layers * (2*2*256 + 1*128/4) = 12 * 1056 = 12,672 B/token at
    fp8. Reading the config unnested instead yields 98,304 B/token -- an 8x
    overstatement that looks entirely reasonable until you check it.
    """
    c = unwrap(cfg)
    n_full = _count_full_attention_layers(cfg, n_layers_of(cfg))
    n_kv = c.get("num_key_value_heads") or c.get("num_attention_heads") or 8
    head_dim = (c.get("head_dim")
                or c.get("hidden_size", 4096) // max(c.get("num_attention_heads", 32), 1))

    per_layer = 2 * n_kv * head_dim * dtype_bytes

    ix_h = c.get("indexer_kv_heads")
    ix_d = c.get("indexer_head_dim")
    if ix_h and ix_d:
        ratio = c.get("indexer_compress_ratio") or 1
        per_layer += ix_h * ix_d * dtype_bytes / max(ratio, 1)

    return n_full * per_layer


def linear_state_gib(cfg: dict, batch: int = 1) -> float:
    """
    Gated DeltaNet state: CONSTANT per sequence, does not grow with context.
    This is why a 262k context is affordable on this architecture at all.

      recurrent state: n_v_heads * k_head_dim * v_head_dim, in mamba_ssm_dtype
      conv state     : conv_kernel * (2*n_k_heads*k_dim + n_v_heads*v_dim)

    Qwen4-Exp: 36 layers * 48*128*128 * 4 bytes (fp32) = 113 MB per sequence.
    """
    c = unwrap(cfg)
    n_layers = n_layers_of(cfg)
    n_lin = n_layers - _count_full_attention_layers(cfg, n_layers)
    if n_lin <= 0:
        return 0.0

    nv = c.get("linear_num_value_heads") or 0
    nk = c.get("linear_num_key_heads") or 0
    kd = c.get("linear_key_head_dim") or 0
    vd = c.get("linear_value_head_dim") or 0
    if not (nv and kd and vd):
        return 0.0

    ssm_bytes = 4 if str(c.get("mamba_ssm_dtype", "float32")).endswith("32") else 2
    recurrent = nv * kd * vd * ssm_bytes
    conv_k = c.get("linear_conv_kernel_dim") or 0
    conv = conv_k * (2 * nk * kd + nv * vd) * 2   # conv state is bf16

    return n_lin * (recurrent + conv) * batch / GIB


def max_context_for_kv(cfg: dict, kv_gib: float, kv_dtype_bytes: float = 1.0) -> int:
    per_tok = kv_cache_per_token_bytes(cfg, kv_dtype_bytes)
    return int(kv_gib * GIB / per_tok) if per_tok else 0


# ============================================================== PLE / N-Gram

def ple_params_b(cfg: dict) -> float:
    """
    Parameter count of the n-gram/PLE table, in billions.
    Here: ngram_vocab_size_base 20,000,000 * ple_embed_dim 2560 = 51.2B.
    It sits entirely at ple_layer_ids (=[2]), split into split_ngram_parts
    (=128) shards. The lookup is sparse, which is what makes it offloadable
    and mmap-able.
    """
    c = unwrap(cfg)
    vocab = c.get("ngram_vocab_size_base")
    dim = c.get("ple_embed_dim") or c.get("hidden_size")
    if not (vocab and dim):
        return 0.0
    div = c.get("make_ngram_vocab_size_divisible_by") or 1
    vocab = ((vocab + div - 1) // div) * div
    return vocab * dim / 1e9


def ple_gib(cfg: dict, bpw: float = 16.0) -> float:
    """Footprint of the PLE table at a given bit width."""
    return gib_for(ple_params_b(cfg), bpw)


# ============================================================== Budget

@dataclass
class Budget:
    kv_cache_gib: float
    linear_state_gib: float
    vram_weights_gib: float
    ram_weights_gib: float
    total_weights_gib: float
    params_b: float
    avg_bits_all: float
    max_model_len: int

    def pretty(self) -> str:
        vram_avail = VRAM_TOTAL_GIB * MEM_FRACTION_STATIC - VRAM_DESKTOP_GIB
        return (
            f"  VRAM {VRAM_TOTAL_GIB} x {MEM_FRACTION_STATIC} - {VRAM_DESKTOP_GIB} desktop"
            f" = {vram_avail:6.2f} GiB\n"
            f"    - CUDA context               -{CUDA_CTX_GIB:5.2f}\n"
            f"    - activations                -{ACTIVATIONS_EAGER_GIB:5.2f}\n"
            f"    - KV cache @ {self.max_model_len:>7} tok    -{self.kv_cache_gib:5.2f}\n"
            f"    - GDN state (constant)       -{self.linear_state_gib:5.2f}\n"
            f"    = VRAM for weights            {self.vram_weights_gib:6.2f} GiB\n\n"
            f"  RAM {RAM_TOTAL_GIB} - {RAM_RESERVED_GIB} reserved   = "
            f"{RAM_TOTAL_GIB-RAM_RESERVED_GIB:6.2f} GiB\n"
            f"    - allocator slack            -{RAM_SLACK_GIB:5.2f}\n"
            f"    = RAM for weights             {self.ram_weights_gib:6.2f} GiB\n\n"
            f"  WEIGHT BUDGET (without PLE)     {self.total_weights_gib:6.2f} GiB"
            f"  (= {self.total_weights_gib*GIB/GB:.1f} GB decimal)\n"
            f"  at {self.params_b:.1f}B resident params   -> {self.avg_bits_all:.3f} bpw\n"
        )


def compute(cfg: dict, params_b: float, max_model_len: int = 32768,
            kv_dtype_bytes: float = 1.0, eager: bool = True,
            kv_cap_gib: float = 3.5, batch: int = 1) -> Budget:
    """
    params_b = RESIDENT parameters in billions, i.e. WITHOUT the PLE table and
    without token_embd (both are offloaded or memory-mapped).
    """
    per_tok = kv_cache_per_token_bytes(cfg, kv_dtype_bytes)
    kv_gib = min(per_tok * max_model_len / GIB, kv_cap_gib)
    lin_gib = linear_state_gib(cfg, batch)
    act = ACTIVATIONS_EAGER_GIB if eager else ACTIVATIONS_GRAPH_GIB

    vram_w = (VRAM_TOTAL_GIB * MEM_FRACTION_STATIC - VRAM_DESKTOP_GIB
              - CUDA_CTX_GIB - act - kv_gib - lin_gib)
    ram_w = RAM_TOTAL_GIB - RAM_RESERVED_GIB - RAM_SLACK_GIB
    total = vram_w + ram_w

    return Budget(round(kv_gib, 2), round(lin_gib, 3), round(vram_w, 2),
                  round(ram_w, 2), round(total, 2), params_b,
                  round(bpw_for(params_b, total), 3), max_model_len)


def autoscheme_target(budget: Budget, bf16_params_b: float,
                      safety_gib: float = 0.8) -> float:
    """
    AutoScheme averages `avg_bits` over quantized layers ONLY. Tensors pinned
    to bf16 (router, norms, hyper-connections) drop out of that average but
    still occupy space. This returns the value AutoScheme has to be given so
    the result lands on the real budget.
    """
    bf16_gib = gib_for(bf16_params_b, 16.0)
    quant_gib = budget.total_weights_gib - bf16_gib - safety_gib
    quant_params_b = budget.params_b - bf16_params_b
    return round(bpw_for(quant_params_b, quant_gib), 2)


if __name__ == "__main__":
    import json, os, sys
    p = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    if not os.path.exists(p):
        print(f"  config.json not found ({p}) - usage: python3 budget.py <config.json>")
        sys.exit(1)
    cfg = json.load(open(p))
    c = unwrap(cfg)
    nl = n_layers_of(cfg)
    nf = _count_full_attention_layers(cfg, nl)

    print(f"\n=== {cfg.get('architectures',['?'])[0]} ===")
    print(f"  {nl} layers, {nf} with full attention (hybrid={nf<nl})")
    print(f"  KV/token fp8: {kv_cache_per_token_bytes(cfg,1.0):,.0f} B"
          f"   bf16: {kv_cache_per_token_bytes(cfg,2.0):,.0f} B")
    print(f"  GDN state (constant, batch=1): {linear_state_gib(cfg)*1024:.0f} MiB")
    print(f"  PLE table: {ple_params_b(cfg):.1f}B params ="
          f" bf16 {ple_gib(cfg,16):.1f} GiB / fp8 {ple_gib(cfg,8):.1f} GiB /"
          f" 4bit {ple_gib(cfg,4.156):.1f} GiB / 2bit {ple_gib(cfg,2.156):.1f} GiB\n")

    for ctx in (32768, 65536, 131072, 262144):
        b = compute(cfg, params_b=124.7, max_model_len=ctx)
        print(f"  ctx {ctx:>7}: KV {b.kv_cache_gib:5.2f} GiB -> "
              f"weight budget {b.total_weights_gib:5.2f} GiB -> {b.avg_bits_all:.3f} bpw")
