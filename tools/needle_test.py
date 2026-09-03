"""Needle-in-a-haystack retrieval test against a running server.

  python3 needle_test.py 60000 [TAG]     ~N-token prompt, 5 needles at 10/30/50/70/90 % depth

The haystack is pseudo-random prose (fixed seed) so n-gram matching cannot help; each needle is a
sentence with a unique code ("The access code for project Kestrel is 4821-QV."). One request asks
for all five codes at the end; the score is the number of codes reproduced. Prints per-needle hits,
the answer, prefill/decode timings, and appends a line to perf/needle_results.tsv.
"""
import json, os, random, re, sys, time, urllib.request

URL = "http://127.0.0.1:30000/generate"
H = os.path.dirname(os.path.abspath(__file__))
WORDS = ("river valley harbour lantern quiet measure ancient signal garden copper thread window silent motor "
         "ledger orchard basalt canvas meadow saddle tunnel velvet anchor pigment glacier compass marble "
         "furnace sparrow cinder ribbon granite hollow beacon timber pollen quartz sable willow crater").split()
NEEDLES = [("Kestrel", "4821-QV"), ("Marlin", "7093-TX"), ("Basalt", "2617-RH"), ("Juniper", "9354-LM"), ("Tamarind", "1178-ZK")]


def haystack(n_words, seed=7):
    rng = random.Random(seed); out = []
    while len(out) < n_words:
        k = rng.randint(6, 14)
        out.extend(rng.choice(WORDS) for _ in range(k)); out[-1] = out[-1] + "."
    return out


def build(target_tokens):
    words = haystack(int(target_tokens * 0.55))          # ~1.8 tokens per word (measured)
    depths = [0.1, 0.3, 0.5, 0.7, 0.9]
    for (proj, code), d in zip(NEEDLES, depths):
        i = int(len(words) * d)
        words.insert(i, f"Note: the access code for project {proj} is {code}.")
    text = " ".join(words)
    q = ("\n\nQuestion: list the access codes for projects " + ", ".join(p for p, _ in NEEDLES) +
         " exactly as written in the notes above. Answer with one line per project, no thinking.\nAnswer:")
    return text + q


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    tag = sys.argv[2] if len(sys.argv) > 2 else "cur"
    text = build(target)
    body = json.dumps({"text": text, "stream": True,
                       "sampling_params": {"max_new_tokens": 400, "temperature": 0, "ignore_eos": True}}).encode()
    t0 = time.time(); times, toks, prompt, out, fin = [], [], 0, "", None
    with urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            d = json.loads(line[5:]); m = d["meta_info"]
            times.append(time.time()); toks.append(m["completion_tokens"]); prompt = m["prompt_tokens"]; out = d["text"]
            fin = m.get("finish_reason")
    print(f"  finish: {fin if times else 'no events'}")
    dec = (times[-1] - times[0]) / max(1, toks[-1] - toks[0]) if len(times) > 1 else 0
    pre = times[0] - t0 - dec if times else 0
    hits = [code in out for _, code in NEEDLES]
    print(f"  prompt {prompt} tokens, prefill {pre:.1f} s, decode {1 / dec if dec else 0:.1f} tok/s")
    print("  needles: " + " ".join(f"{p}:{'OK' if h else '--'}" for (p, _), h in zip(NEEDLES, hits)) + f"   score {sum(hits)}/5")
    print(f"  answer: {out.strip()[:300]!r}")
    with open(os.path.join(H, "needle_results.tsv"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{tag}\t{prompt}\t{sum(hits)}/5\t{''.join('1' if h else '0' for h in hits)}\t{pre:.1f}\t{1 / dec if dec else 0:.1f}\n")


if __name__ == "__main__":
    main()
