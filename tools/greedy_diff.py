"""Exactness check: greedy-decode a fixed prompt set and compare token ids to a saved reference.

  python3 greedy_diff.py save  NAME      # record reference from the running server
  python3 greedy_diff.py check NAME      # compare running server against NAME

Three prompts (German prose, English reasoning, code), 200 tokens each, temperature 0.
A systems change that is exact must reproduce every token id. Small numeric drift from
kernel reordering shows up as a divergence position; report it rather than hide it.
"""
import json, sys, os, urllib.request

URL = "http://127.0.0.1:30000/generate"
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "greedy")
PROMPTS = [
    "Erkläre in einem zusammenhängenden Absatz, warum Mixture-of-Experts-Modelle bei gleicher "
    "Anzahl aktiver Parameter mehr Wissen speichern können als dichte Modelle, und nenne einen Nachteil.",
    "A train leaves city A at 9:00 travelling at 80 km/h. A second train leaves city B, 300 km away, "
    "at 10:00 travelling toward A at 120 km/h. Reason step by step: when and where do they meet?",
    "Write a Python function that parses an ISO-8601 duration string like 'P1DT2H30M' into total "
    "seconds, with tests. Then explain one edge case it does not handle.",
]


def gen(text, n=200):
    body = json.dumps({"text": text, "sampling_params": {
        "max_new_tokens": n, "temperature": 0, "ignore_eos": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    d = d[0] if isinstance(d, list) else d
    return d["meta_info"].get("output_token_ids") or d.get("output_ids") or d["text"]


def main():
    cmd, name = sys.argv[1], sys.argv[2]
    os.makedirs(D, exist_ok=True)
    p = os.path.join(D, f"{name}.json")
    outs = [gen(t) for t in PROMPTS]
    if cmd == "save":
        json.dump(outs, open(p, "w"))
        print(f"  reference saved: {p}")
        return
    ref = json.load(open(p))
    ok = True
    for i, (a, b) in enumerate(zip(ref, outs)):
        if isinstance(a, str) or isinstance(b, str):
            same = a == b
            print(f"  prompt {i}: {'identical' if same else 'DIFFERS'} (text compare)")
        else:
            n = min(len(a), len(b))
            div = next((j for j in range(n) if a[j] != b[j]), None)
            if div is None and len(a) == len(b):
                print(f"  prompt {i}: identical ({len(a)} tokens)")
                same = True
            else:
                print(f"  prompt {i}: DIVERGES at token {div if div is not None else n} of {len(a)}")
                same = False
        ok &= same
    print(f"\n  {'EXACT' if ok else 'NOT EXACT'} vs {name}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
