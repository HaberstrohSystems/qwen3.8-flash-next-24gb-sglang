#!/usr/bin/env python3
"""Phase 1 driver: try each change, measure, keep or revert. State in phase1_state.json beside it.

  python3 phase1.py            run all pending steps
  python3 phase1.py STEP       run one step by name
  python3 phase1.py --status   show accepted state

Acceptance rule: a step is KEPT if (a) the greedy diff is exact (for steps that must be
exact) and (b) mean decode tok/s over the two long contexts is not worse than the accepted
state by more than 2 %. Flags accumulate into the accepted set; patches likewise.

Layout note: as run, this driver sat in one working directory with sweep.sh, patches/,
logprob_diff.py, greedy_diff.py, logprob/ and phase1_state.json. In this repository those are
scripts/sweep.sh, patches/, tools/ (logprob_diff.py, greedy_diff.py, logprob/) and
assets/phase1_state.json; to run it, symlink them into one directory or set the paths at
lines 22-25, 52, 55, 72, 96 and 98 (STATE, PY, PYS, CAMP, the patches/ and sweep.sh joins, the
logprob/ glob and the logprob_diff.py path). Interpreter defaults are ~/quant/venv*/bin/python3.
"""
import glob, json, os, re, subprocess, sys, time

H = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(H, "phase1_state.json")
PY = os.path.expanduser("~/quant/venv/bin/python3")
PYS = os.path.expanduser("~/quant/venv-sglang/bin/python3")
CAMP = os.path.join(H, "CAMPAIGN.md")


def load():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"drop": [], "add": [], "patches": [], "env": "", "accepted": None, "done": []}


def save(s):
    json.dump(s, open(STATE, "w"), indent=2)


def log(line):
    with open(CAMP, "a") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} {line}\n")
    print(f"  LOG {line}")


def patch(cmd, item):
    if item in ("ncontig", "placement", "elastic", "kv_lazy", "kv_fp8", "ple_random", "kv_stats", "kv_fakeq", "kv_int8",
                "kv_int4", "ngram_ple", "kv_tiers", "kv_paged_prefix"):
        script = {"ncontig": "ncontig_gemv.py", "placement": "placement.py", "elastic": "elastic.py",
                  "kv_lazy": "kv_lazy.py", "kv_fp8": "kv_fp8.py", "ple_random": "ple_random.py",
                  "kv_stats": "kv_stats.py", "kv_fakeq": "kv_fakeq.py", "kv_int8": "kv_int8.py",
                  "kv_int4": "kv_int4.py", "ngram_ple": "ngram_ple.py", "kv_tiers": "kv_tiers.py",
                  "kv_paged_prefix": "kv_paged_prefix.py"}[item]
        r = subprocess.run([PYS, os.path.join(H, "patches", script), cmd],
                           capture_output=True, text=True)
    else:
        r = subprocess.run([PYS, os.path.join(H, "patches", "host_fixes.py"), cmd, item],
                           capture_output=True, text=True)
    print(r.stdout.strip())
    return r.returncode == 0


def revert_patches(patches):
    """Revert a step's patches in reverse apply order (layered patches: e.g. kv_tiers before kv_int4).
    A refused revert (non-zero exit, e.g. an out-of-order guard) leaves the tree inconsistent: stop the run."""
    for p in reversed(patches):
        if not patch("revert", p):
            log(f"revert of {p} REFUSED/failed -> tree inconsistent; aborting (repair by hand, see patches/{p}.py)")
            raise SystemExit(f"revert of {p} refused")


def restart_and_bench(name, drop, add, env):
    e = dict(os.environ, DROP=" ".join(drop), EXTRA_ENV=env)
    r = subprocess.run(["bash", os.path.join(H, "sweep.sh"), name] + add,
                       env=e, capture_output=True, text=True)
    out = r.stdout + r.stderr
    print(out[-1500:])
    if "FAILED" in out or "TIMEOUT" in out:
        return None
    rows = re.findall(r"^\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+\d+\s*$", out, re.M)
    if len(rows) < 3:
        return None
    dec = [float(x[3]) for x in rows]
    pre = [float(x[2]) for x in rows]
    return {"decode_long": sum(dec[-2:]) / 2, "decode_all": dec, "prefill_10k": pre[-1]}


def greedy_check():
    """Teacher-forced logprob oracle (perf/logprob_diff.py) against reference lp0.

    Same-config noise floor measured 2026-09-02: MAX 0.09 nats, MEAN 0.002.
    Returns (ok, (max, mean)). A crash returns (False, None).
    """
    # reference: lp0 = baseline byte layout; lp1 = accepted state after S9 (word layout). Each
    # kernel-class acceptance re-bases the reference so host-class steps are judged against
    # the state they actually change.
    # newest lpN.json is the reference (each accepted approximation / kernel-class change re-bases it)
    refs = sorted(glob.glob(os.path.join(H, "logprob", "lp[0-9]*.json")), key=lambda f: int(os.path.basename(f)[2:-5]))
    ref = os.environ.get("LP_REF", os.path.basename(refs[-1])[:-5] if refs else "lp0")
    r = subprocess.run([PY, os.path.join(H, "logprob_diff.py"), "check", ref],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.stderr.strip():
        print("  logprob_diff stderr:", r.stderr.strip()[-600:])
    m = re.search(r"LOGPROB_MAX=([\d.]+) LOGPROB_MEAN=([\d.]+)", r.stdout)
    if not m:
        return False, None
    mx, mean = float(m.group(1)), float(m.group(2))
    return (mean <= LP_MEAN_TOL and mx <= LP_MAX_TOL), (mx, mean)


# Numerical-equivalence thresholds: ~5x the measured run-to-run noise. A wrong
# kernel or a mis-set flag lands orders of magnitude above these.
LP_MEAN_TOL = 0.01
LP_MAX_TOL = 0.5


STEPS = [
  # name,            drop flags,                          add flags,                          patches, env,   must_be_exact
  ("S0_baseline",    [],                                  [],                                 [],      "",    False),
  # tuned int2 MoE configs (BLOCK_SIZE_N=16 at M=1): bit-identical, the biggest single item
  ("S0b_int2cfg",    [],                                  [],                                 [],
      "SGLANG_MOE_CONFIG_DIR=" + os.path.join(H, "moe_configs"),                                        True),
  ("S1_overlap",     ["--disable-overlap-schedule"],      [],                                 [],      "",    True),
  ("S2_contdecode",  [],                                  ["--num-continuous-decode-steps","4"],[],    "",    True),
  # mamba reclaim first (frees ~0.86 GiB); chunk 2048 OOMed at mem-fraction 0.95, try 1024
  ("S4_mamba2",      ["--max-mamba-cache-size"],          ["--max-mamba-cache-size","2"],     [],      "",    True),
  ("S3_prefill1024", ["--chunked-prefill-size"],          ["--chunked-prefill-size","1024",
                                                           "--max-prefill-tokens","32768"],   [],      "",    True),
  ("S5_hostfixes",   [],                                  [],                     ["hook","skipgather","memo","rope","ple"], "", True),
  ("S6_bcg",         ["--disable-cuda-graph"],            [],                                 ["bcg"], "",    True),
  ("S7_ngram",       [],                                  ["--speculative-algorithm","NGRAM"],[],      "",    True),
  # kernel-class: tuned int2 configs change the reduction order; judged with TOL_PREFIX
  ("S8_int2cfg_tol", [],                                  [],                                 [],
      "SGLANG_MOE_CONFIG_DIR=" + os.path.join(H, "moe_configs"),                                        "tol"),
  # N-contiguous layout + in-place decode GEMV (idea 1). Kernel-class -> tolerance rule.
  ("S9_ncontig_gemv", [],                                 [],                                 ["ncontig"], "", "tol"),
  # isolation: layout + word-load tiled path only, GEMV dispatch off
  ("S9a_layout_only", [],                                 [],                                 ["ncontig"], "SGLANG_MOE_GEMV=0", "tol"),
  # bisection: convert only resident layers (0-16) / only host layers (17-47)
  ("S9b_layers_res",  [],                                 [],                                 ["ncontig"], "SGLANG_MOE_GEMV=0 SGLANG_MOE_NCONTIG_LAYERS=0-16", "tol"),
  ("S9c_layers_host", [],                                 [],                                 ["ncontig"], "SGLANG_MOE_GEMV=0 SGLANG_MOE_NCONTIG_LAYERS=17-47", "tol"),
  # S6 failed because the DECODE graph backend defaults to "full" (classic capture); the
  # eager_on_graph wrap only takes effect under the breakable backend.
  ("S6b_bcg",        ["--disable-cuda-graph"],            ["--cuda-graph-backend-decode","breakable"], ["bcg"], "", True),
  # S7 failed for VRAM (speculative worker needs headroom); retry once the staging buffers are gone.
  ("S7b_ngram",      [],                                  ["--speculative-algorithm","NGRAM"],[],      "",    True),
  # S6b failed: the breakable decode backend does not walk LogitsProcessorOutput; bcg2 teaches it.
  ("S6c_bcg2",       ["--disable-cuda-graph"],            ["--cuda-graph-backend-decode","breakable"], ["bcg","bcg2"], "", True),
  ("S7c_ngram",      [],                                  ["--speculative-algorithm","NGRAM"],[],      "",    True),
  # NGRAM still short of VRAM at 32k tokens of KV; try with a 16k token budget to see if it pays
  ("S7d_ngram16k",   ["--max-total-tokens"],              ["--max-total-tokens","16384",
                                                           "--speculative-algorithm","NGRAM"],[],      "",    True),
  # frequency-based expert placement (exact: values unchanged, only their location)
  ("S10_placement",  [],                                  [],                                 ["placement"],
      "SGLANG_MOE_PLACEMENT=" + os.path.join(H, "expert_freq.pt") + " SGLANG_MOE_PLACEMENT_S=184", True),
  # APPROXIMATION: dense groups requantized to INT8 g128 (perf/requant_int8.py output dir).
  # No oracle gate (values change by design); quality is judged with nll_eval.py afterwards.
  ("S11_int8dense",  ["--model-path"],                    ["--model-path", open(os.path.join(H, "requant_out.txt")).read().strip()
                                                           if os.path.exists(os.path.join(H, "requant_out.txt")) else "MISSING"],
                                                                                              [],      "",    False),
  # S11 failed at load: exact extra_config entries beat the regexes, and lm_head resolved to
  # group_size -1. config.json rewritten (requant_int8.py --config-only); same step again.
  ("S11b_int8dense", ["--model-path"],                    ["--model-path", open(os.path.join(H, "requant_out.txt")).read().strip()
                                                           if os.path.exists(os.path.join(H, "requant_out.txt")) else "MISSING"],
                                                                                              [],      "",    False),
  # 1.6 GB came back with INT8 dense; NGRAM without touching S first
  ("S12a_ngram",     [],                                  ["--speculative-algorithm","NGRAM"],[],      "",    True),
  # NGRAM fails on the mamba intermediate-state reserve (0.16 GB/req x 2 x 12 draft tokens = 3.8 GB),
  # not on weights. ReplaySSM drops that scratch (needs a linear draft chain: topk 1).
  ("S12b_ngram_replay", [],                                ["--speculative-algorithm","NGRAM",
                                                           "--enable-linear-replayssm-spec",
                                                           "--speculative-eagle-topk","1",
                                                           "--speculative-ngram-max-bfs-breadth","1"], [], "", True),
  # fallback: shrink the reserve instead (4 draft tokens -> 1.3 GB)
  ("S12c_ngram_d4",   [],                                  ["--speculative-algorithm","NGRAM",
                                                           "--speculative-num-draft-tokens","4"], [],  "", True),
  # elastic expert residency: rank-ordered VMM arenas + host slot pool; S=184 at start must be
  # exact vs v3 (same rows on the GPU, different arrangement). Live S changes via elastic.ctl.
  ("S13_elastic",     [],                                 [],                                 ["elastic"],
      "SGLANG_MOE_ELASTIC=1 SGLANG_MOE_ELASTIC_PIN_MB=512 SGLANG_MOE_ELASTIC_CTL=" + os.path.join(H, "elastic.ctl"),                    True),
  # S12b: "Qwen QSA requires speculative_num_draft_tokens <= the QSA compress ratio (4)". So: 4
  # draft tokens (reserve 1.3 GB) AND replayssm (reserve 0) with a linear chain.
  ("S12d_ngram_d4_replay", [],                            ["--speculative-algorithm","NGRAM",
                                                           "--speculative-num-draft-tokens","4",
                                                           "--enable-linear-replayssm-spec",
                                                           "--speculative-eagle-topk","1",
                                                           "--speculative-ngram-max-bfs-breadth","1"], [], "", True),
  # lazy KV backing: the 32k KV pool becomes VA + on-demand physical prefix (floor 4096 tokens).
  # Exact by construction (same buffers, different backing); frees ~0.65 GB at short context.
  ("S14_kvlazy",      [],                                 [],                                 ["kv_lazy"],
      "SGLANG_KV_LAZY=1",                                                                              True),
  # 128k context on the same VRAM: KV reserved virtually, physical pages borrowed from the expert
  # cache during long requests. Judged by the oracle at short context + longctx_test.py afterwards.
  ("S15_ctx128k",     ["--max-total-tokens", "--context-length"],
                                                           ["--max-total-tokens","131072","--context-length","131072"], [],
      "SGLANG_KV_LAZY_TOKENS=131072 SGLANG_MOE_ELASTIC_FILL_MB=2048",                               True),
  # APPROXIMATION: fp8_e4m3 KV (12 KB/token) with the QSA read-path patch. No oracle gate; quality by
  # nll_eval afterwards (and a 10k logprob check for gross errors). Reclaims: triton attention backend
  # (drops the 384 MiB FlashInfer workspace), mamba cache 1 (max_running_requests is 1 anyway).
  ("S16_fp8kv",       ["--attention-backend", "--max-mamba-cache-size"],
                                                           ["--kv-cache-dtype","fp8_e4m3","--attention-backend","triton",
                                                            "--max-mamba-cache-size","1"], ["kv_fp8"], "",    False),
  # Stage A0 of the own KV scheme: fake quantization on the write path (accuracy only, bf16 storage).
  ("FQ_int8_g64",     [],                                 ["--kv-cache-dtype","auto"],                                 ["kv_fakeq"], "SGLANG_KV_FAKEQ=int8_g64", False),
  ("FQ_int8_tok",     [],                                 ["--kv-cache-dtype","auto"],                                 ["kv_fakeq"], "SGLANG_KV_FAKEQ=int8_tok", False),
  ("FQ_int8_g32",     [],                                 ["--kv-cache-dtype","auto"],                                 ["kv_fakeq"], "SGLANG_KV_FAKEQ=int8_g32", False),
  ("FQ_e4m3",         [],                                 ["--kv-cache-dtype","auto"],                                 ["kv_fakeq"], "SGLANG_KV_FAKEQ=e4m3",     False),
  # OWN KV SCHEME stage A: int8 K/V with per-(token, head, 64-channel group) fp16 scales, 12.4 KB/token.
  ("S17_int8kv",      ["--attention-backend", "--max-mamba-cache-size"],
                                                           ["--kv-cache-dtype","int8_g64","--attention-backend","triton",
                                                            "--max-mamba-cache-size","1"], ["kv_int8"], "", False),
  # ceiling search with int8 KV: 256k virtual context; the safety cap / headroom rule decide the real limit
  ("S18_ctx256k",     ["--max-total-tokens", "--context-length"],
                                                           ["--max-total-tokens","262144","--context-length","262144"], [],
      "SGLANG_KV_LAZY_TOKENS=262144",                                                                  False),
  ("FQ_int4_g32",     [],                                 ["--kv-cache-dtype","auto"],                                 ["kv_fakeq"], "SGLANG_KV_FAKEQ=int4_g32", False),
  ("FQ_int4_g32_sm",  [],                                 ["--kv-cache-dtype","auto"],                                 ["kv_fakeq"], "SGLANG_KV_FAKEQ=int4_g32_sm", False),
  ("FQ_int8_g64_sm",  [],                                 ["--kv-cache-dtype","auto"],                                 ["kv_fakeq"], "SGLANG_KV_FAKEQ=int8_g64_sm", False),
  # NGRAM speculation with PLE support (chain draft, ReplaySSM, eager verify: decode graphs disabled).
  ("S12e_ngram_chain", [],                                ["--speculative-algorithm","NGRAM","--speculative-num-draft-tokens","4",
                                                           "--speculative-ngram-min-bfs-breadth","1","--speculative-ngram-max-bfs-breadth","1",
                                                           "--enable-linear-replayssm-spec","--disable-cuda-graph",
                                                           "--cuda-graph-backend-decode","disabled"], ["ngram_ple"], "", False),
  # OWN KV SCHEME stage C: int4 K/V (nibble-packed) with per-(token, head, 32-channel group) fp16 scales, 6.75 KB/token.
  ("S19_int4kv",      ["--attention-backend", "--max-mamba-cache-size"],
                                                           ["--kv-cache-dtype","int4_g32","--attention-backend","triton",
                                                            "--max-mamba-cache-size","1"], ["kv_int4"], "", False),
  ("FQ_int3_g16",     [],                                 ["--kv-cache-dtype","auto"],                                 ["kv_fakeq"], "SGLANG_KV_FAKEQ=int3_g16", False),
  ("FQ_int2_g16",     [],                                 ["--kv-cache-dtype","auto"],                                 ["kv_fakeq"], "SGLANG_KV_FAKEQ=int2_g16", False),
  ("FQ_int2_g8",      [],                                 ["--kv-cache-dtype","auto"],                                 ["kv_fakeq"], "SGLANG_KV_FAKEQ=int2_g8",  False),
  ("FQ_zero",         [],                                 ["--kv-cache-dtype","auto"],                                 ["kv_fakeq"], "SGLANG_KV_FAKEQ=zero", False),
  ("FQ_noise",        [],                                 ["--kv-cache-dtype","auto"],                                 ["kv_fakeq"], "SGLANG_KV_FAKEQ=noise", False),
  # plain bf16 KV on the accepted set (reference for KV-quantization measurements; int8_g64 is the default)
  ("BF16KV",          [],                                 ["--kv-cache-dtype","auto"],        [],      "",    False),
  # OWN KV SCHEME stage D ("compost"): int8 ring for the last W tokens over the int4 full-context pool (dual-write).
  ("S21_tiers",       ["--attention-backend", "--max-mamba-cache-size"],
                                                           ["--kv-cache-dtype","int8ring_int4","--attention-backend","triton",
                                                            "--max-mamba-cache-size","1"], ["kv_int4","kv_tiers"], "SGLANG_KV_TIERS_W=8192", False),
  # paged prefix-chunk prefill kernel (KV_PAGED_PREFIX_PLAN.md): no materialised bf16 prefix, same numerics within 2 ulps
  ("S22_paged",       ["--attention-backend", "--max-mamba-cache-size"],
                                                           ["--kv-cache-dtype","int8ring_int4","--attention-backend","triton",
                                                            "--max-mamba-cache-size","1"], ["kv_int4","kv_tiers","kv_paged_prefix"], "SGLANG_KV_TIERS_W=8192", False),
  # VRAM as a dial: S=128 hot experts per layer frees ~1.9 GB; enough for the NGRAM worker?
  ("S12_ngram_S128", [],                                  ["--speculative-algorithm","NGRAM"],[],
      "SGLANG_MOE_PLACEMENT_S=128",                                                                   True),
]


def run_step(s, step):
    name, drop, add, patches, env, exact = step
    if name in s["done"]:
        print(f"  {name}: done"); return
    print(f"\n════════ {name} ════════")
    for p in patches:
        patch("apply", p)
    d = s["drop"] + drop
    a = s["add"] + add
    res = restart_and_bench(name, d, a, s["env"] + " " + env)
    if res is None:
        log(f"{name}: FAILED to start or bench -> reverted")
        revert_patches(patches)
        s["server_matches_accepted"] = False      # the failed start left no server behind
        s["done"].append(name); save(s); return
    ok = True
    if exact and s["accepted"] is not None:
        ok, vals = greedy_check()
        if vals is None:
            log(f"{name}: logprob oracle crashed -> treated as not equivalent")
        else:
            # kernel-class ("tol"): a different fp32 accumulation order flips routing for a
            # few tokens and the deviation amplifies (measured: layer-0 identical-input error
            # 1e-4, end-to-end mean 0.012). Verified per layer once; accept a wider band here.
            if exact == "tol":
                ok = vals[1] <= 0.05 and vals[0] <= LP_MAX_TOL
            log(f"{name}: logprob max {vals[0]:.3f} mean {vals[1]:.4f} -> {'equivalent' if ok else 'NOT equivalent'}"
                + (" (kernel-class band)" if exact == "tol" else ""))
    acc = s["accepted"]
    better = acc is None or res["decode_long"] >= acc["decode_long"] * 0.98
    if name == "S3_prefill2048":     # judged on prefill, not decode
        better = acc is None or res["prefill_10k"] >= acc["prefill_10k"] * 0.98
    if ok and better:
        s["drop"], s["add"], s["patches"] = d, a, s["patches"] + patches
        s["env"] = (s["env"] + " " + env).strip()
        s["accepted"] = res
        log(f"{name}: KEPT  decode {res['decode_long']:.1f} tok/s, prefill {res['prefill_10k']:.0f} tok/s"
            + ("" if acc is None else f"  (was {acc['decode_long']:.1f} / {acc['prefill_10k']:.0f})"))
    else:
        why = "not exact" if not ok else "not better"
        log(f"{name}: REVERTED ({why})  decode {res['decode_long']:.1f}, prefill {res['prefill_10k']:.0f}")
        revert_patches(patches)
        s["server_matches_accepted"] = False
    s["done"].append(name); save(s)
    if ok and better:
        s["server_matches_accepted"] = True; save(s)


def main():
    s = load()
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        print(json.dumps(s, indent=2)); return
    if len(sys.argv) > 2 and sys.argv[1] == "--bringup":     # accepted set + one step's flags/patches, no verdict
        step = next(x for x in STEPS if x[0] == sys.argv[2])
        name, drop, add, patches, env, _ = step
        for p in patches:
            patch("apply", p)
        restart_and_bench(name, s["drop"] + drop, s["add"] + add, s["env"] + " " + env)
        s["server_matches_accepted"] = False; save(s); return
    if len(sys.argv) > 1 and sys.argv[1] == "--restart":     # bring up the accepted set, nothing else
        restart_and_bench("accepted", s["drop"], s["add"], s["env"])
        s["server_matches_accepted"] = True; save(s); return
    want = sys.argv[1:] or [x[0] for x in STEPS]
    for step in STEPS:
        if step[0] in want:
            run_step(s, step)
    # leave the server running in the accepted configuration
    if s.get("accepted") and not s.get("server_matches_accepted", True):
        print("\n  last step was reverted - restarting with the accepted set")
        restart_and_bench("accepted", s["drop"], s["add"], s["env"])
        s = load()                                   # the state file may have been edited meanwhile
        s["server_matches_accepted"] = True; save(s)
    print("\n  accepted:", json.dumps(s["accepted"]))


if __name__ == "__main__":
    main()
