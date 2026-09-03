"""Teacher-forced logprob oracle - replaces the greedy token diff.

Greedy token diffs are chaotic: a 1-ulp change at a near-tie flips a token and every
token after it, and this server is not bitwise reproducible run to run (same config,
back to back: divergences at 104-169 of 200). So compare what the model actually
computes: the log-probability it assigns to a FIXED continuation. Numerically equivalent
changes move these by run-to-run noise; a wrong kernel moves them by orders more.

  python3 logprob_diff.py save NAME     # record per-token logprobs of the fixed continuations
  python3 logprob_diff.py check NAME    # compare against NAME; prints max/mean |delta|

Continuations come from perf/greedy/oa.json (token ids). Prompt ids are obtained from the
server itself (first call with logprob_start_len=0 returns the input token ids).
"""
import json, os, sys, urllib.request

URL = "http://127.0.0.1:30000/generate"
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logprob")
from greedy_diff import PROMPTS  # noqa: E402

CONT = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "greedy", "oa.json")))


def post(payload):
    req = urllib.request.Request(URL, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return d[0] if isinstance(d, list) else d


def prompt_ids(text):
    d = post({"text": text, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
              "return_logprob": True, "logprob_start_len": 0})
    return [t[1] for t in d["meta_info"]["input_token_logprobs"]]


def forced_logprobs(pids, cids):
    ids = pids + cids
    d = post({"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
              "return_logprob": True, "logprob_start_len": len(pids) - 1})
    lp = [t[0] for t in d["meta_info"]["input_token_logprobs"]]
    lp = [x for x in lp if x is not None]
    return lp[-len(cids):]


def collect():
    out = []
    for text, cont in zip(PROMPTS, CONT):
        pids = prompt_ids(text)
        out.append(forced_logprobs(pids, cont[:150]))
    return out


def main():
    cmd, name = sys.argv[1], sys.argv[2]
    os.makedirs(D, exist_ok=True)
    p = os.path.join(D, f"{name}.json")
    cur = collect()
    if cmd == "save":
        json.dump(cur, open(p, "w")); print(f"  saved {p}  ({[len(c) for c in cur]} tokens)"); return
    ref = json.load(open(p))
    worst = 0.0; tot = 0.0; n = 0
    for i, (a, b) in enumerate(zip(ref, cur)):
        m = min(len(a), len(b))
        d = [abs(x - y) for x, y in zip(a[:m], b[:m])]
        mx = max(d); mean = sum(d) / m
        worst = max(worst, mx); tot += sum(d); n += m
        print(f"  prompt {i}: max |dlogprob| {mx:.4f}   mean {mean:.5f}   over {m} forced tokens")
    print(f"\n  MAX {worst:.4f}  MEAN {tot/n:.5f}")
    print(f"  LOGPROB_MAX={worst:.6f} LOGPROB_MEAN={tot/n:.6f}")


if __name__ == "__main__":
    main()
