"""Lossless gate for NGRAM speculation -- SPEC_NGRAM_PLAN.md section 4, step 3 (a/b/c). HTTP only.

Run against the RUNNING spec server (S12e_ngram_chain with perf/patches/ngram_ple.py applied); the
script never starts or stops a server.

  python3 spec_lossless.py [--tag TAG] [--ref lp_prespec] [--greedy-ref prespec] [--n 200]
                           [--near-tie 0.01] [--noise-band 0.1] [--spec-mean 0.003] [--lp-mean 0.01]
                           [--lp-max 0.5] [--draft-tokens N] [--force-reject] [--skip-a]

  0. mode check (exit 2 on mismatch): /server_info must report speculative_algorithm NGRAM and
     speculative_num_draft_tokens == --draft-tokens (default 2 for --tag d2, else 4).  With
     --force-reject (default for --tag force_reject) every prompt must additionally show
     meta_info.spec_verify_ct == tokens-1 (one committed token per verify step; a draft padding
     token 0 can be accepted when it IS the argmax, so the exact form is
     spec_verify_ct >= tokens-1-count(0 in ids), == tokens-1 when no 0 was generated).  For every
     tag spec_verify_ct must be present (spec active) and consistent with the draft width.
  a. teacher-forced prefill oracle: same measurement as `logprob_diff.py check <ref>` (150 forced tokens
     of perf/greedy/oa.json per prompt); the mean must stay at the run-to-run floor (~0.002; gate
     --lp-mean / --lp-max = the phase1 thresholds).  Confirms Edit 1 did not disturb the prefill path.
  b. per prompt (greedy_diff.PROMPTS: prose / reasoning / code): greedy-generate n tokens with
     return_logprob (spec path: the accepted rows' log_softmax, compute_spec_logprobs), then
     teacher-force prompt+generated with logprob_start_len=len(prompt)-1, top_logprobs_num=2.  Every
     generated token must equal the teacher-forced top-1 or be the top-2 within the noise band
     (a near-tie: counted and reported, not failed).  The band is tied to the measured noise of this
     very measurement, not a fixed constant: the campaign's same-config floor is MAX 0.09 nats
     (phase1.greedy_check), so a token whose top-1/top-2 gap is below the per-token logprob
     perturbation can legitimately flip.  Per token: near-tie iff generated == top-2 and
     gap <= max(--near-tie, --noise-band, part (a) max |dlogprob| of this run, 2*|spec - forced
     logprob| at that token).  A token that is neither top-1 nor a within-band top-2 is a MISMATCH
     (hard fail).  mean |spec logprob - forced logprob| <= --spec-mean, max reported.  This compares
     the verify-row logits (12 QSA + 36 GDN + PLE, GDN/PLE rollback every step) with the prefill
     path without depending on run-to-run token reproducibility.
  c. = (b) again on a server launched with SGLANG_NGRAM_FORCE_REJECT=1 (every step commits one token:
     rollback exercised every step) and on one launched with --speculative-num-draft-tokens 2.  Those are
     launch-time settings, so: relaunch, then re-run this script with --tag force_reject / --tag d2.
  d. greedy_diff-style divergence positions vs perf/greedy/<greedy-ref>.json (report only): they must
     look like noise (first divergence typically > 100 as with the bf16 x2 runs), not systematic early
     divergence.

Results: perf/spec_lossless/<tag>.json.  Exit 1 if (a) or (b) fails, 2 if the server is not in the
tagged mode (so a label-only rerun can never be recorded as evidence).
"""
import argparse
import json
import os
import sys
import time
import urllib.request

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import greedy_diff  # noqa: E402
import logprob_diff  # noqa: E402

URL = greedy_diff.URL
OUT = os.path.join(H, "spec_lossless")


def post(payload, timeout=1800):
    req = urllib.request.Request(URL, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d[0] if isinstance(d, list) else d


def health():
    try:
        with urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def server_info():
    with urllib.request.urlopen("http://127.0.0.1:30000/server_info", timeout=30) as r:
        return json.load(r)


# ----------------------------------------------------------------------------- 0. mode check
SKIP_SPEC_CHECK = False


def part_0(draft_tokens):
    """The server under test must be the NGRAM server with the expected draft width (HTTP only)."""
    info = server_info()
    algo = info.get("speculative_algorithm")
    d = info.get("speculative_num_draft_tokens")
    got = {
        "speculative_algorithm": algo,
        "speculative_num_draft_tokens": d,
        "speculative_ngram_max_bfs_breadth": info.get("speculative_ngram_max_bfs_breadth"),
        "enable_linear_replayssm_spec": info.get("enable_linear_replayssm_spec"),
        "model_path": info.get("model_path"),
        "version": info.get("version"),
    }
    ok = str(algo).upper() == "NGRAM" and d == draft_tokens
    print(f"  (0) server: {got}")
    print(f"  (0) expected NGRAM / {draft_tokens} draft tokens -> {'ok' if ok else 'MODE MISMATCH'}")
    return ok, got


def check_spec_ct(ids, mi, draft_tokens, force_reject):
    """spec_verify_ct consistency for one prompt; returns (ok, dict)."""
    ct = mi.get("spec_verify_ct")
    n = len(ids)
    row = {"spec_verify_ct": ct, "tokens": n, "zeros": ids.count(0)}
    if SKIP_SPEC_CHECK:                      # baseline on a non-speculative server
        row["reason"] = "baseline (no spec check)"
        return True, row
    if ct is None:
        row["reason"] = "meta_info.spec_verify_ct missing (spec inactive?)"
        return False, row
    # n-1 tokens come from verify steps, each commits 1..draft_tokens tokens
    lo = -(-(n - 1) // draft_tokens)
    row["mean_accept"] = (n - 1) / ct if ct else None
    if not (lo <= ct <= n - 1):
        row["reason"] = f"spec_verify_ct {ct} outside [{lo}, {n - 1}] for {draft_tokens} drafts"
        return False, row
    if force_reject:
        # every step commits the bonus token only, unless the target argmax is the padding token 0
        need = n - 1 if 0 not in ids else n - 1 - ids.count(0)
        if ct < need or (0 not in ids and ct != n - 1):
            row["reason"] = f"force_reject: spec_verify_ct {ct} != tokens-1 = {n - 1} (zeros {ids.count(0)})"
            return False, row
    return True, row


# ----------------------------------------------------------------------------- a. prefill oracle
def part_a(ref, lp_mean_tol, lp_max_tol):
    p = os.path.join(H, "logprob", f"{ref}.json")
    if not os.path.exists(p):
        refs = sorted(
            (f for f in os.listdir(os.path.join(H, "logprob")) if f.startswith("lp") and f.endswith(".json")),
        )
        print(f"  [!] reference {p} missing; available: {refs}")
        return {"ok": False, "reason": "missing reference"}
    ref_lp = json.load(open(p))
    cur = logprob_diff.collect()
    worst, tot, n, rows = 0.0, 0.0, 0, []
    for i, (a, b) in enumerate(zip(ref_lp, cur)):
        m = min(len(a), len(b))
        d = [abs(x - y) for x, y in zip(a[:m], b[:m])]
        mx, mean = max(d), sum(d) / m
        worst = max(worst, mx)
        tot += sum(d)
        n += m
        rows.append({"prompt": i, "max": mx, "mean": mean, "tokens": m})
        print(f"  prompt {i}: max |dlogprob| {mx:.4f}   mean {mean:.5f}   over {m} forced tokens")
    mean_all = tot / n
    ok = mean_all <= lp_mean_tol and worst <= lp_max_tol
    print(f"  (a) prefill oracle vs {ref}: MAX {worst:.4f}  MEAN {mean_all:.5f}  -> {'ok' if ok else 'FAIL'}")
    return {"ok": ok, "ref": ref, "max": worst, "mean": mean_all, "rows": rows}


# ----------------------------------------------------------------------------- b. spec path vs teacher forcing
def generate(text, n):
    t0 = time.time()
    d = post({
        "text": text,
        "sampling_params": {"max_new_tokens": n, "temperature": 0, "ignore_eos": True},
        "return_logprob": True,
    })
    dt = time.time() - t0
    mi = d["meta_info"]
    otl = mi["output_token_logprobs"]                      # [logprob, token_id, text|None] per output token
    ids = [int(t[1]) for t in otl]
    lps = [float(t[0]) for t in otl]
    ref_ids = mi.get("output_token_ids") or d.get("output_ids")
    if ref_ids is not None and list(ref_ids) != ids:
        print(f"  [!] output_token_ids ({len(ref_ids)}) differ from output_token_logprobs ids ({len(ids)})")
    return ids, lps, dt, mi


def teacher_force(pids, cids):
    d = post({
        "input_ids": pids + cids,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "return_logprob": True,
        "logprob_start_len": len(pids) - 1,
        "top_logprobs_num": 2,
    })
    mi = d["meta_info"]
    itl, itop = mi["input_token_logprobs"], mi["input_top_logprobs"]
    if len(itl) != len(itop):
        raise RuntimeError(f"input_token_logprobs ({len(itl)}) and input_top_logprobs ({len(itop)}) misaligned")
    pairs = [(t, top) for t, top in zip(itl, itop) if t[0] is not None and top]
    pairs = pairs[-len(cids):]
    got = [int(t[1]) for t, _ in pairs]
    if got != cids:
        raise RuntimeError(f"teacher-forced ids misaligned: {got[:5]}... vs {cids[:5]}...")
    out = []
    for t, top in pairs:
        top = sorted(((float(x[0]), int(x[1])) for x in top), reverse=True)
        out.append((float(t[0]), top[0], top[1] if len(top) > 1 else (float("-inf"), -1)))
    return out


def part_b(n, band, spec_mean_tol, greedy_ref, draft_tokens, force_reject):
    ref_path = os.path.join(H, "greedy", f"{greedy_ref}.json")
    gref = json.load(open(ref_path)) if os.path.exists(ref_path) else None
    if gref is None:
        print(f"  (d) no greedy reference {ref_path}: divergence report skipped")
    print(f"  near-tie band: gap <= max({band:.4f}, 2*|spec-forced| per token)")
    ok_all, mode_ok, rows, n_ties, n_tok = True, True, [], 0, 0
    for i, text in enumerate(greedy_diff.PROMPTS):
        pids = logprob_diff.prompt_ids(text)
        ids, spec_lp, dt, mi = generate(text, n)
        forced = teacher_force(pids, ids)
        mism, ties, deltas = [], [], []
        for j, (g, slp, (flp, top1, top2)) in enumerate(zip(ids, spec_lp, forced)):
            d = abs(slp - flp)
            deltas.append(d)
            if g != top1[1]:
                gap = top1[0] - top2[0]
                tok_band = max(band, 2.0 * d)
                if g == top2[1] and gap <= tok_band:
                    ties.append((j, g, top1[1], gap, tok_band))
                else:
                    mism.append((j, g, top1[1], top2[1], gap))
        mean_d, max_d = sum(deltas) / len(deltas), max(deltas)
        ok = not mism and mean_d <= spec_mean_tol
        ok_all &= ok
        n_ties += len(ties)
        n_tok += len(ids)
        ct_ok, ct_row = check_spec_ct(ids, mi, draft_tokens, force_reject)
        mode_ok &= ct_ok
        div = None
        if gref is not None and not isinstance(gref[i], str):
            a = gref[i]
            m = min(len(a), len(ids))
            div = next((k for k in range(m) if a[k] != ids[k]), None)
            if div is None and len(a) != len(ids):
                div = m
        print(
            f"  prompt {i}: {len(ids)} tokens in {dt:.1f} s ({len(ids) / dt:.1f} tok/s incl. prefill)  "
            f"mean|d| {mean_d:.5f} max {max_d:.4f}  mismatches {len(mism)}  near-ties {len(ties)}  "
            f"spec_verify_ct {ct_row.get('spec_verify_ct')} (mean accept {ct_row.get('mean_accept') or 0:.2f})"
            + (f"  greedy-ref divergence @{div}" if gref is not None else "")
            + f"  -> {'ok' if ok else 'FAIL'}" + ("" if ct_ok else f"  MODE MISMATCH: {ct_row.get('reason')}")
        )
        for j, g, t1, t2, gap in mism[:10]:
            print(f"      MISMATCH pos {j}: generated {g}, forced top1 {t1} top2 {t2} (gap {gap:.4f})")
        for j, g, t1, gap, tb in ties[:10]:
            print(f"      near-tie pos {j}: generated {g} == top2, top1 {t1}, gap {gap:.4f} <= band {tb:.4f}")
        if div is not None and div < 100:
            print(f"      [!] first divergence vs {greedy_ref} at {div} (< 100: check that this is not systematic)")
        rows.append({
            "prompt": i, "tokens": len(ids), "seconds": dt, "mean_delta": mean_d, "max_delta": max_d,
            "mismatches": mism, "near_ties": ties, "greedy_divergence": div, "ok": ok,
            "ids": ids, "spec_logprobs": spec_lp, "forced_logprobs": [f[0] for f in forced],
            "spec_ct": ct_row, "mode_ok": ct_ok,
        })
    if n_tok and n_ties > 0.05 * n_tok:
        print(f"  [!] {n_ties}/{n_tok} tokens are near-ties ({100 * n_ties / n_tok:.1f} %): the band is doing "
              f"real work; inspect the gaps before calling this lossless")
    print(f"  (b) spec path vs teacher forcing -> {'ok' if ok_all else 'FAIL'}   near-ties {n_ties}/{n_tok}"
          + ("" if mode_ok else "   MODE MISMATCH (spec_verify_ct)"))
    return {"ok": ok_all, "mode_ok": mode_ok, "band": band, "near_ties": n_ties, "tokens": n_tok, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="spec", help="label: spec | force_reject | d2 | prespec ...")
    ap.add_argument("--ref", default="lp_prespec", help="logprob reference name (perf/logprob/<ref>.json)")
    ap.add_argument("--greedy-ref", default="prespec", help="greedy reference name (perf/greedy/<ref>.json)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--no-spec-check", action="store_true", help="baseline on a non-speculative server: skip part (0) and the spec_verify_ct checks")
    ap.add_argument("--near-tie", type=float, default=0.01, help="floor of the near-tie band (nats)")
    ap.add_argument("--noise-band", type=float, default=0.1,
                    help="near-tie band from the measured same-config noise (MAX 0.09 nats, phase1.greedy_check); "
                         "the effective band is max(--near-tie, --noise-band, part (a) max, 2*|d| per token)")
    ap.add_argument("--spec-mean", type=float, default=0.003)
    ap.add_argument("--lp-mean", type=float, default=0.01)
    ap.add_argument("--lp-max", type=float, default=0.5)
    ap.add_argument("--draft-tokens", type=int, default=None,
                    help="expected speculative_num_draft_tokens (default: 2 for --tag d2, else 4)")
    ap.add_argument("--force-reject", action="store_true", default=None,
                    help="expect SGLANG_NGRAM_FORCE_REJECT=1 behaviour (default: on for --tag force_reject)")
    ap.add_argument("--skip-a", action="store_true")
    args = ap.parse_args()
    if args.draft_tokens is None:
        args.draft_tokens = 2 if args.tag == "d2" else 4
    if args.force_reject is None:
        args.force_reject = args.tag == "force_reject"
    if not health():
        print("  server not healthy at 127.0.0.1:30000"); sys.exit(2)
    os.makedirs(OUT, exist_ok=True)
    res = {"tag": args.tag, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args)}
    p = os.path.join(OUT, f"{args.tag}.json")
    global SKIP_SPEC_CHECK
    SKIP_SPEC_CHECK = bool(args.no_spec_check)
    print("=== (0) server mode ===")
    mode_ok, res["server"] = (True, server_info()) if args.no_spec_check else part_0(args.draft_tokens)
    if not mode_ok:
        res["mode_ok"] = False
        json.dump(res, open(p, "w"))
        print(f"\n  MODE MISMATCH [{args.tag}]: the running server is not the tagged configuration; nothing measured. {p}")
        sys.exit(2)
    band = max(args.near_tie, args.noise_band)
    if not args.skip_a:
        print("=== (a) teacher-forced prefill oracle ===")
        res["a"] = part_a(args.ref, args.lp_mean, args.lp_max)
        if res["a"].get("max") is not None:
            band = max(band, res["a"]["max"])
    print("=== (b) greedy spec generation vs teacher forcing ===")
    res["b"] = part_b(args.n, band, args.spec_mean, args.greedy_ref, args.draft_tokens, args.force_reject)
    res["mode_ok"] = res["b"]["mode_ok"]
    ok = res["b"]["ok"] and (args.skip_a or res["a"]["ok"])
    json.dump(res, open(p, "w"))
    if not res["mode_ok"]:
        print(f"\n  MODE MISMATCH [{args.tag}]: spec_verify_ct inconsistent with the tagged mode "
              f"(force_reject={args.force_reject}, draft_tokens={args.draft_tokens}); result NOT valid as evidence. {p}")
        sys.exit(2)
    print(f"\n  {'LOSSLESS' if ok else 'NOT LOSSLESS'} [{args.tag}]  near-ties {res['b']['near_ties']}/{res['b']['tokens']} "
          f"(band {band:.4f})  results: {p}")
    print("  (c) re-run with --tag force_reject on a server launched with SGLANG_NGRAM_FORCE_REJECT=1,\n"
          "      and with --tag d2 on one launched with --speculative-num-draft-tokens 2")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
