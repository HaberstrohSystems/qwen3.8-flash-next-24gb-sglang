"""KV-cache read-path quality on LONG text (the short nll_eval never reads the cache: one chunk).

  python3 nll_long.py save NAME      teacher-forced NLL over ~6k tokens (prefix-chunk path) + mean logprob of a
                                     300-token greedy continuation after a 3k prompt (decode gather path)
  python3 nll_long.py check NAME     same measurements now vs the saved NAME

Text: perf/CAMPAIGN.md (technical English/German prose the model has not seen).
"""
import json, os, sys, urllib.request

URL = "http://127.0.0.1:30000/generate"
H = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(H, "nll")


def post(body):
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800) as r:
        d = json.load(r)
    return d[0] if isinstance(d, list) else d


def measure():
    text = open(os.environ.get("NLL_LONG_TEXT", os.path.join(H, "..", "docs", "CAMPAIGN.md"))).read()[:26000]
    # logits for every input position would be ~7k x 248k x 4 B = 7 GB: score only the last 512 positions
    probe = post({"text": text[:2000], "sampling_params": {"max_new_tokens": 1, "temperature": 0}})
    n_est = int(len(text) / 2000 * probe["meta_info"]["prompt_tokens"])
    if os.environ.get("NLL_LONG_ALL"):                       # every position (needs ~1 GB free per logit chunk: ctl 'S 184')
        d = post({"text": text, "sampling_params": {"max_new_tokens": 1, "temperature": 0}, "return_logprob": True,
                  "logprob_start_len": 1024})
        lp = [t[0] for t in d["meta_info"]["input_token_logprobs"] if t[0] is not None]
    else:
        d = post({"text": text, "sampling_params": {"max_new_tokens": 1, "temperature": 0}, "return_logprob": True,
                  "logprob_start_len": max(0, n_est - 700)})
        lp = [t[0] for t in d["meta_info"]["input_token_logprobs"] if t[0] is not None][-512:]
    n_in = d["meta_info"]["prompt_tokens"]
    prompt = text[:12000]
    g = post({"text": prompt, "sampling_params": {"max_new_tokens": 300, "temperature": 0, "ignore_eos": True}, "return_logprob": True})
    out_lp = [t[0] for t in g["meta_info"]["output_token_logprobs"] if t[0] is not None]
    return {"tf_tokens": n_in, "tf_lp": lp, "gen_prompt_tokens": g["meta_info"]["prompt_tokens"], "gen_lp": out_lp, "gen_text": g["text"][:400]}


def main():
    cmd, name = sys.argv[1], sys.argv[2]
    os.makedirs(D, exist_ok=True)
    cur = measure()
    tf = -sum(cur["tf_lp"]) / len(cur["tf_lp"]); gen = -sum(cur["gen_lp"]) / len(cur["gen_lp"])
    print(f"  teacher-forced: {cur['tf_tokens']} tokens, NLL {tf:.4f} nats/token over the last {len(cur['tf_lp'])} positions (their chunks read the cache)")
    print(f"  greedy after {cur['gen_prompt_tokens']} tokens: 300 tokens, mean NLL {gen:.4f} (decode reads the cache)")
    p = os.path.join(D, f"long_{name}.json")
    if cmd == "save":
        json.dump(cur, open(p, "w")); print(f"  saved {p}"); return
    tag = os.environ.get("SGLANG_KV_FAKEQ") or os.environ.get("NLL_LONG_TAG") or "cur"
    json.dump(cur, open(os.path.join(D, f"long_{name}_vs_{tag}.json"), "w"))
    ref = json.load(open(p))
    m = min(len(ref["tf_lp"]), len(cur["tf_lp"]))
    a, b = ref["tf_lp"][:m], cur["tf_lp"][:m]
    dl = sum(abs(x - y) for x, y in zip(a, b)) / m
    print(f"  vs {name}: NLL delta {(-sum(b) / m) - (-sum(a) / m):+.4f}, mean|dlogprob| {dl:.4f}, max {max(abs(x - y) for x, y in zip(a, b)):.3f} over the last {m} positions")
    rg = -sum(ref["gen_lp"]) / len(ref["gen_lp"])
    print(f"  greedy mean NLL: ref {rg:.4f} now {gen:.4f}; first divergence: "
          f"{next((i for i, (x, y) in enumerate(zip(ref['gen_text'], cur['gen_text'])) if x != y), 'none in 400 chars')}")


if __name__ == "__main__":
    main()
