#!/usr/bin/env python3
"""
The quantization run.

  python3 05_quantize.py --dry-run          # show the recipe, preflight, compute nothing
  python3 05_quantize.py                    # the real run (hours to days)
  python3 05_quantize.py --policy balanced --ram-reserve 12

Three things here are not negotiable. Each was paid for once:

  1. EXACTLY ONE --format. AutoRound only enables immediate packing when
     len(formats)==1. With two formats it accumulates finished layers in host
     memory - a run that grows steadily until the OOM killer ends it, hours in.

  2. The auto-round patch (07_patch_autoround.py). Without it the run dies on
     the first calibration forward, because AutoRound deletes `group_size`
     from Qwen4ExpTextRMSNorm - where it is an architecture parameter, not a
     quantization setting.

  3. The mmap PLE (03_split_ple.py). Without it transformers tries to
     dequantize 51.2B parameters to bf16: 95 GB on a 30 GB machine.
"""
import argparse, json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import budget as B
from importlib.machinery import SourceFileLoader

_recipe = SourceFileLoader("recipe", os.path.join(HERE, "04_recipe.py")).load_module()
_ple = SourceFileLoader("ple", os.path.join(HERE, "03_split_ple.py")).load_module()

# Checkpoint name -> runtime name. AutoRound unstacks the experts before
# quantizing, and layer_config is matched against the RUNTIME names.
RUNTIME = {
    r"experts\.gate_up_proj": r"experts\.\d+\.(gate|up)_proj",
    r"experts\.down_proj":    r"experts\.\d+\.down_proj",
}


def build_layer_config(policy_name):
    """layer_config in runtime names, plus the base scheme.

    The base scheme is the fallback for anything no rule matches. It MUST be
    the expert scheme: the experts are 96% of the quantized parameters, so if
    a rule fails to match, the result should not silently blow the budget.
    """
    policy = _recipe.POLICIES[policy_name]
    lc = {}
    # REVERSED order - not cosmetic, required.
    # The policy is written "first match wins" (specific rules first).
    # AutoRound applies rules in insertion order and lets the LAST match win:
    #   compressors/utils.py:506  for name in list(layer_config.keys()):
    #   compressors/utils.py:536      layer_config[match] = val   <- overwrites
    # Without the reversal the general rule overwrites the specific one:
    #   linear_attn.in_proj_(a|b) BF16  <- by  linear_attn.  W8A16
    #   linear_attn.out_proj      BF16  <- by  linear_attn.  W8A16
    #   layers.([0-7])...down_proj W3   <- by  experts.down_proj W2
    # Confirmed on disk in a miniature run: layers.0 down_proj came out with
    # qweight [4,256] / qzeros [1,16], i.e. pack factor 16 = 2 bits, not 3.
    # The W8 cases are worse than a quality loss: AutoRound cannot quantize
    # in_proj_a/b at all, so the config and the tensors on disk contradict
    # each other and the model does not load ("No compatible backend found").
    for rx, scheme, gs in reversed(policy):
        rt = rx
        for chk, run in RUNTIME.items():
            rt = rt.replace(chk, run)
        key = rt if rt.startswith("^") else ".*" + rt
        # NO backslash escapes: auto_round's to_standard_regex() doubles them
        # ("\\." instead of "\."), which in a regex means "literal backslash
        # followed by any character". The pattern then matches nothing and the
        # layer falls back to the base scheme without a word. That is how
        # self_attn ended up at 2 bits instead of bf16 in an early run. A bare
        # dot is unambiguous here - tensor names always have a real dot there.
        key = key.replace("\\.", ".")
        # SKIP means "not in the weight budget" - the config still has to say
        # 16 bit, otherwise the base scheme applies there.
        if scheme in ("BF16", "SKIP"):
            lc[key] = {"bits": 16, "act_bits": 16}
        else:
            bits = int(scheme[1:].split("A")[0])
            lc[key] = {"bits": bits, "group_size": gs, "act_bits": 16}
    # Base = the rule for the stacked experts (down_proj, no layer prefix)
    base = None
    for rx, scheme, gs in policy:
        if scheme not in ("SKIP", "BF16") and rx.startswith("experts"):
            base = (int(scheme[1:].split("A")[0]), gs)
    if base is None:
        base = (2, 128)
    return lc, base


def coverage_check(lc):
    """Which quantizable modules does no rule match?

    Checks against the REAL tensor names (tensors.json), translated into the
    runtime names that exist after unstacking. A module with no rule falls
    back to the base scheme; that is rarely intended and should be visible.
    """
    import re
    tj = os.path.join(HERE, "tensors.json")
    if not os.path.exists(tj):
        return None
    names = set()
    for k in json.load(open(tj)):
        if not k.endswith((".weight", "gate_up_proj", "down_proj")):
            continue
        # stacked experts -> runtime names
        if k.endswith("mlp.experts.gate_up_proj"):
            base = k[: -len("gate_up_proj")]
            names.add(base + "0.gate_proj"); names.add(base + "0.up_proj")
        elif k.endswith("mlp.experts.down_proj"):
            names.add(k[: -len("down_proj")] + "0.down_proj")
        else:
            names.add(k[:-7] if k.endswith(".weight") else k)
    rx = [re.compile(p) for p in lc]
    uncovered = sorted({n for n in names if not any(r.search(n) for r in rx)})
    # Without --quant_nontext_module AutoRound never touches vision/embeddings
    ignorable = [n for n in uncovered if ".visual." in n or "embed_tokens" in n
                 or "ngram_embedding" in n or n.startswith("mtp.")]
    real = [n for n in uncovered if n not in ignorable]
    return real, ignorable



def resolution_check(lc, policy_name):
    """Resolve every layer the way AutoRound does, and compare.

    The bug this catches: the policy is written "first match wins", but
    AutoRound applies "last match wins". Without the reversed insertion order
    in build_layer_config() the general rule overwrites the specific one -
    silently, with no warning, visible only hours later on disk, or not until
    the model fails to load.

    Returns a list of (name, expected, actual); empty means correct.
    """
    import re
    tj = os.path.join(HERE, "tensors.json")
    if not os.path.exists(tj):
        return None
    names = set()
    for k in json.load(open(tj)):
        if not k.endswith((".weight", "gate_up_proj", "down_proj")):
            continue
        if k.endswith("mlp.experts.gate_up_proj"):
            base = k[: -len("gate_up_proj")]
            names.add(base + "0.gate_proj"); names.add(base + "0.up_proj")
        elif k.endswith("mlp.experts.down_proj"):
            names.add(k[: -len("down_proj")] + "0.down_proj")
        else:
            names.add(k[:-7] if k.endswith(".weight") else k)

    # Rules in POLICY order, translated the same way as build_layer_config
    ordered = []
    for rx, scheme, gs in _recipe.POLICIES[policy_name]:
        rt = rx
        for chk, run in RUNTIME.items():
            rt = rt.replace(chk, run)
        key = rt if rt.startswith("^") else ".*" + rt
        key = key.replace("\\.", ".")
        cfg = ({"bits": 16, "act_bits": 16} if scheme in ("BF16", "SKIP")
               else {"bits": int(scheme[1:].split("A")[0]),
                     "group_size": gs, "act_bits": 16})
        ordered.append((key, cfg))

    comp = [(re.compile(k), c) for k, c in ordered]
    lc_comp = [(re.compile(k), lc[k]) for k in lc]     # insertion order

    bad = []
    for n in sorted(names):
        expected = next((c for r, c in comp if r.search(n)), None)   # first match
        actual = None
        for r, c in lc_comp:                                         # last match
            if r.search(n):
                actual = c
        if expected != actual:
            bad.append((n, expected, actual))
    return bad


def preload_model(model_path, ple_dir):
    """Load the model BEFORE auto_round is imported. That ordering is the point.

    Measured on the real checkpoint, each inside a 26 GB cgroup:

        plain                                        OK   anon 0.43 GB  FP8Experts
        + torch.cuda initialized                     OK   anon 0.45 GB  FP8Experts
        + import auto_round                          OK   anon 0.48 GB  FP8Experts
        + import auto_round.context.model            OOM
        + import ...fused_moe.moe_experts_interface  OOM

    Importing that module registers AutoRound's "linear_loop" implementation
    with transformers' experts interface. After that `replace_with_fp8_linear`
    finds nothing to replace ("no linear modules were found"), the fp8 weights
    land in bf16 parameters and are cast on load: twice the size, anonymous
    memory, OOM after roughly a minute.

    Load first and the model already exists with FP8Experts and mmap-backed
    weights; AutoRound is then handed something finished.
    """
    import torch
    from transformers import AutoProcessor, AutoTokenizer
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpForConditionalGeneration)

    _ple.install(ple_dir)
    model = Qwen4ExpForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        processor = None

    # --- set is_transposed explicitly --------------------------------------
    # auto_round's unstacking path guesses the memory layout:
    #     is_transposed = getattr(module, "is_transposed", None)
    #     if is_transposed is None: is_transposed = dim1 < dim2
    # gate_up_proj is (512, 1280, 2560) -> 1280 < 2560 -> guessed "transposed".
    # Wrong: this IS [experts, out, in], exactly what F.linear expects
    # (modeling_qwen4_exp: F.linear(x, self.gate_up_proj[i])).
    # Acting on the wrong guess triggers transpose(1,2).contiguous() - a full
    # 134 GB copy, and an OOM 13 seconds after loading.
    n_marked = 0
    for name, mod in model.named_modules():
        if name.endswith(".experts") and hasattr(mod, "gate_up_proj"):
            mod.is_transposed = False
            n_marked += 1
    print(f"  is_transposed=False set on {n_marked} expert modules")

    e = model.model.language_model.layers[0].mlp.experts
    print(f"  {type(e).__name__}  gate_up {tuple(e.gate_up_proj.shape)} "
          f"{e.gate_up_proj.dtype}")
    import torch as _t
    if e.gate_up_proj.dtype == _t.float8_e4m3fn and type(e).__name__ != "FP8Experts":
        print("  [!] WARNING: fp8 weights without FP8Experts - cast on load, OOM likely")
    elif e.gate_up_proj.dtype == _t.bfloat16:
        print("  bf16 checkpoint, stacked - AutoRound can tune directly")
    return model, processor, tokenizer


def hand_over(model, processor, tokenizer):
    """Hand AutoRound the finished model instead of a path.

    From here on auto_round may be imported - the model already exists.
    """
    import auto_round.utils.model as arm
    import auto_round.context.model as arc
    def _preloaded(path, **kw):
        return model, processor, tokenizer, None
    arm.mllm_load_model = _preloaded
    arc.mllm_load_model = _preloaded

    # Register linear_loop in the FP8 interface too.
    # auto_round's register_linear_loop_experts() only enters the implementation
    # into ALL_EXPERTS_FUNCTIONS. These experts are FP8Experts, built with
    # experts_interface=ALL_FP8_EXPERTS_FUNCTIONS (finegrained_fp8.py:830).
    # Without this entry the first forward dies with
    #   KeyError: '`linear_loop` is not a valid experts implementation'
    from auto_round.modeling.fused_moe.moe_experts_interface import (
        linear_loop_experts_forward, LINEAR_LOOP_IMPL, register_linear_loop_experts)
    register_linear_loop_experts()
    try:
        from transformers.integrations.finegrained_fp8 import ALL_FP8_EXPERTS_FUNCTIONS
        m = ALL_FP8_EXPERTS_FUNCTIONS._global_mapping
        if LINEAR_LOOP_IMPL not in m:
            m[LINEAR_LOOP_IMPL] = linear_loop_experts_forward
            print(f"  '{LINEAR_LOOP_IMPL}' also registered in the FP8 experts interface")
    except Exception as e:
        print(f"  [!] FP8 interface not reachable: {type(e).__name__}")


def preflight(a, out_dir):
    ok = True
    def chk(label, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{'OK' if good else '!!'}] {label:<44} {detail}")

    print("\n=== Preflight ===")
    # 1. auto-round patch
    patched = False
    try:
        # deliberately NOT imported - merely importing auto_round's MoE
        # interface destroys the FP8 replacement (see preload_model)
        import importlib.util
        spec = importlib.util.find_spec("auto_round")
        p = os.path.join(os.path.dirname(spec.origin), "compressors", "utils.py")
        patched = "[qwen4-exp] group_size collision" in open(p).read()
        chk("auto-round patched (group_size)", patched,
            "" if patched else "-> python3 07_patch_autoround.py")
    except Exception as e:
        chk("auto-round importable", False, str(e)[:40])

    # 2. transformers knows the architecture
    try:
        from transformers.models import qwen4_exp  # noqa
        import transformers
        chk("transformers knows qwen4_exp", True, transformers.__version__)
    except Exception:
        chk("transformers knows qwen4_exp", False, "-> pip install -U 'transformers>=5.16.0'")

    # 3. PLE manifest
    man = os.path.join(a.ple, "ple.json")
    have_ple = os.path.exists(man)
    detail = ""
    if have_ple:
        m = json.load(open(man))
        binp = os.path.join(a.ple, m["file"])
        sz_ok = os.path.exists(binp) and os.path.getsize(binp) == m["bytes"]
        have_ple &= sz_ok
        detail = f"{m['rows']:,} x {m['dim']} {m['dtype']}" + ("" if sz_ok else "  FILE INCOMPLETE")
    chk("PLE offloaded (03_split_ple.py)", have_ple,
        detail or "-> python3 03_split_ple.py split <src> -o <ple>")

    # 4. source checkpoint
    src_ok = os.path.exists(os.path.join(a.model, "model.safetensors.index.json"))
    n = len([f for f in os.listdir(a.model) if f.endswith(".safetensors")]) if src_ok else 0
    inc = len([f for f in os.listdir(a.model) if f.endswith(".incomplete")]) if src_ok else 0
    chk("source checkpoint complete", src_ok and inc == 0, f"{n} shards, {inc} incomplete")

    # 5. disk
    st = os.statvfs(os.path.dirname(out_dir) or ".")
    free = st.f_bavail * st.f_frsize
    chk("disk space for the output", free > 60e9, f"{free/1e9:.0f} GB free (>= 60 needed)")

    # 6. GPU
    try:
        import torch
        f, t = torch.cuda.mem_get_info()
        chk("GPU visible", True, f"{torch.cuda.get_device_name(0)}, {f/2**30:.1f}/{t/2**30:.1f} GiB free")
        chk("enough free VRAM (>= 18 GiB)", f > 18 * 2**30, "otherwise close the desktop session")
    except Exception as e:
        chk("GPU visible", False, str(e)[:40])

    # 7. RAM
    with open("/proc/meminfo") as fh:
        mi = {l.split(":")[0]: int(l.split()[1]) for l in fh}
    avail = mi["MemAvailable"] / 1024**2
    chk("free RAM (>= 18 GiB)", avail >= 18, f"{avail:.1f} GiB available")

    return ok



def enforce_immediate_saving():
    """Abort if AutoRound would NOT write the blocks out one at a time.

    Why this exists: as long as even ONE layer outside the transformer blocks
    is quantized, AutoRound disables block-wise saving whenever tuning is
    active (base.py:1224/1425/1428/1446). All 48 finished blocks then
    accumulate in RAM - around 35 GiB - and the run dies to the OOM killer an
    hour in, rather than to an error message a second in. This guard inverts
    that: the failure happens BEFORE the expensive work, and it names both the
    offending layers and the cause.

    The lever, if this fires: set the named layers to BF16 in 04_recipe.py
    (see LM_HEAD_NOTE there).
    """
    import auto_round.compressors.base as B
    from auto_round.compressors.utils import check_to_quantized

    orig = B.BaseCompressor._adjust_immediate_packing_and_saving

    def checked(self):
        r = orig(self)
        cc = self.compress_context
        ok = bool(getattr(cc, "is_immediate_saving", False))
        print(f"\n  immediate saving : {'ON' if ok else 'OFF'}"
              f"   (packing={getattr(cc,'is_immediate_packing','?')}, "
              f"inplace={getattr(self,'inplace','?')}, "
              f"low_cpu_mem={getattr(cc,'low_cpu_mem_usage','?')})", flush=True)
        if ok:
            return r
        offenders = [n for n, c in self.layer_config.items()
                     if not c.get("in_blocks", False) and check_to_quantized(c)]
        print("\n  [ABORT] AutoRound would keep every finished block in RAM")
        print("          until the end (~35 GiB across 48 blocks).")
        print("          On a 30 GiB machine that is a certain OOM.")
        if offenders:
            print(f"\n  Cause: {len(offenders)} quantized layer(s) "
                  f"OUTSIDE the blocks:")
            for n in offenders[:20]:
                print(f"    - {n}  (bits={self.layer_config[n].get('bits')})")
            if len(offenders) > 20:
                print(f"    ... and {len(offenders)-20} more")
            print("\n  Fix: set these rules to BF16 in 04_recipe.py.")
        else:
            print("\n  Cause: no layer outside the blocks - so it is one of")
            print("         the other switches (format/inplace/dtype).")
        sys.exit(3)

    B.BaseCompressor._adjust_immediate_packing_and_saving = checked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/quant/src-bf16"),
                    help="restacked checkpoint (02_prepare_source.py), NOT the original")
    ap.add_argument("--ple", default=os.path.expanduser("~/quant/ple"))
    ap.add_argument("-o", "--out", default=os.path.expanduser("~/quant/out/autoround-w2g128"))
    ap.add_argument("--policy", default="quality", choices=list(_recipe.POLICIES))
    ap.add_argument("--ram-reserve", type=float, default=10.0)
    ap.add_argument("--max-model-len", type=int, default=32768)
    # Calibration
    ap.add_argument("--iters", type=int, default=200)
    # 128 rather than 512: AutoRound caches the block INPUTS, and in Qwen4-Exp
    # those are hc_count x hidden = 4 x 2560 = 10240 wide, not 2560.
    #   128 x 2048 x 10240 x 2 bytes =  5.0 GiB   <- fits
    #   512 x 2048 x 10240 x 2 bytes = 20.0 GiB   <- OOM
    # On a 30 GiB host even 128 thrashes; see the README on --nsamples 48.
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    # bs 1 x ga 8 rather than 4 x 2: same effective batch size, a quarter of
    # the activation memory. Needed because the gated residual widens the
    # residual stream to hc_count x hidden = 10240 - activations are 4x larger
    # than in an ordinary model of the same size. At 4 x 2 the GPU fills up:
    #   torch.OutOfMemoryError: 23.42 GiB total, 283 MiB free
    # For reference, on a dense 27B: bs1/ga8 269 s/layer, peak 8.63 GB
    #                                bs4/ga2 113 s/layer, peak 10.22 GB
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--subset-groups", type=int, default=0,
                    help="tune the experts in N rounds per block (0 = off). "
                         "Only meaningful without --model-free.")
    ap.add_argument("--alg-ext", action="store_true",
                    help="SignRoundV2 (--enable_alg_ext). AutoRound recommends it "
                         "for bits<=2 and lists W2A16 as validated, which is "
                         "exactly what the experts are. Costs compute time.")
    ap.add_argument("--model-free", action="store_true",
                    help="AutoRound's model_free path: works directly on the "
                         "checkpoint files and never loads the model. Uses "
                         "optimized RTN instead of tuned rounding.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="start despite a failed preflight")
    a = ap.parse_args()

    lc, base = build_layer_config(a.policy)
    bits, gs = base

    print(f"\n=== policy: {a.policy} ===")
    print(f"  source    : {a.model}")
    print(f"  PLE       : {a.ple}   (mmap, not resident)")
    print(f"  output    : {a.out}")
    print(f"  base      : W{bits}A16 g{gs}   + {len(lc)} layer_config rules")
    print(f"  calib     : iters {a.iters}, nsamples {a.nsamples}, seqlen {a.seqlen}, "
          f"bs {a.batch_size} x ga {a.grad_accum}")

    # Check it against the budget
    print()
    subprocess.run([sys.executable, os.path.join(HERE, "04_recipe.py"),
                    "--policy", a.policy, "--ram-reserve", str(a.ram_reserve),
                    "--ple-ring", "0.0625", "--headless",
                    "--max-model-len", str(a.max_model_len)],
                   check=False)

    print("\n=== layer_config (runtime names) ===")
    for k, v in lc.items():
        d = "bf16" if v["bits"] == 16 else f"W{v['bits']}A16 g{v['group_size']}"
        print(f"  {d:<12} {k}")

    cov = coverage_check(lc)
    if cov is not None:
        real, ign = cov
        print(f"\n=== Coverage ===")
        print(f"  skipped by AutoRound regardless : {len(ign)} modules "
              f"(vision, embeddings, MTP)")
        if real:
            print(f"  [!] NO RULE, falls back to W{build_layer_config(a.policy)[1][0]}"
                  f"A16 g{build_layer_config(a.policy)[1][1]}: {len(real)}")
            for n in real[:8]:
                print(f"        {n}")
        else:
            print(f"  [OK] every quantizable module has a rule")

    res = resolution_check(lc, a.policy)
    if res is not None:
        print(f"\n=== Rule resolution (AutoRound: last match wins) ===")
        if res:
            print(f"  [ABORT] {len(res)} layer(s) resolve DIFFERENTLY from what "
                  f"the policy says.")
            print(f"  The cause is almost always the insertion order in "
                  f"build_layer_config().")
            for n, e, t in res[:10]:
                print(f"    {n}\n      policy says      {e}\n      AutoRound does   {t}")
            if len(res) > 10:
                print(f"    ... and {len(res)-10} more")
            sys.exit(4)
        print(f"  [OK] every layer resolves the way the policy intends")

    ok = preflight(a, a.out)
    if not ok and not a.dry_run and not a.force:
        print("\n  [!] Preflight failed. Fix it, or pass --force.\n")
        sys.exit(2)

    argv = [
        "--model", a.model,
        "--scheme", f"W{bits}A16", "--group_size", str(gs),
        "--layer_config", json.dumps(lc),
        "--format", "auto_round",            # EXACTLY ONE - see the module docstring
        "--iters", str(a.iters), "--nsamples", str(a.nsamples),
        "--seqlen", str(a.seqlen),
        "--batch_size", str(a.batch_size),
        "--gradient_accumulate_steps", str(a.grad_accum),
        # REQUIRED: without it AutoRound uses device_map="auto", and accelerate
        # materializes half the checkpoint as anonymous memory while sharding
        # -> OOM at 26 GB. get_device_and_parallelism("cuda") -> ("cuda", False)
        # loads the model mmap-backed on the CPU, and AutoRound moves the
        # blocks to the GPU one at a time. Measured: anon 0.43 GB, not 26 GB.
        "--device_map", "cuda",
        "--low_gpu_mem_usage", "--low_cpu_mem_usage",
        # no --max_shard_size: it does not exist in auto-round 0.14.2, and
        # passing it is silently ignored rather than rejected.
        "--output_dir", a.out,
    ]

    if a.alg_ext:
        argv += ["--enable_alg_ext"]

    if a.model_free:
        # iters is ignored; RTN runs without a calibration loop.
        # NO --quant_lm_head: lm_head is bf16 in every policy (see LM_HEAD_NOTE
        # in 04_recipe.py). layer_config says bits=16; if AutoRound quantized
        # lm_head anyway, the config and the tensors on disk would contradict
        # each other and loading fails with
        # "No compatible backend found for layer ...".
        argv += ["--model_free"]
        for drop in ("--iters", "--nsamples", "--seqlen", "--batch_size",
                     "--gradient_accumulate_steps"):
            if drop in argv:
                i = argv.index(drop); del argv[i:i+2]

    if a.dry_run:
        print("\n=== Command (dry run, nothing computed) ===\n")
        print("  python -m auto_round \\")
        for i in range(0, len(argv), 2):
            v = argv[i + 1] if i + 1 < len(argv) else ""
            if v.startswith("{"):
                v = "'<layer_config>'"
            print(f"    {argv[i]} {v} \\")
        print("\n  (with ple.install() called first, in the same process)\n")
        return

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.model_free:
        print(f"\n=== model_free: no loading needed ===")
        print("  AutoRound reads the shards directly and writes them back")
        print("  quantized. No unstacking, no autograd, no GPU pressure.")
    else:
        print(f"\n=== Loading the model (before importing auto_round!) ===")
        model, processor, tokenizer = preload_model(a.model, a.ple)
        hand_over(model, processor, tokenizer)
        print("  handed over to AutoRound")

    # Sanity guard: more rounds than experts would be absurdly slow
    if a.subset_groups > 32:
        print(f"\n  [!] --subset-groups {a.subset_groups} is implausible "
              f"(512 experts). Allowed range is 2..32.")
        sys.exit(2)
    if not a.model_free:
        enforce_immediate_saving()

    if a.subset_groups > 1 and not a.model_free:
        print(f"\n=== Subset tuning ===")
        _subset = SourceFileLoader("subset",
            os.path.join(HERE, "06_subset_tuning.py")).load_module()
        _subset.install(n_groups=a.subset_groups)

    print(f"\n=== Start {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"  Extrapolate after block 3. If it projects past 100 h, stop and rethink.\n")
    from auto_round.cli.main import start
    start(argv=argv)
    print(f"\n=== End {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"  Next step: python3 08_verify_quant.py {a.out}/<subdirectory>\n")


if __name__ == "__main__":
    main()
