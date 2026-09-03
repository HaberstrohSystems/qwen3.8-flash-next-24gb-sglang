"""Measure prefill and decode throughput against a running SGLang server.

One streamed generation per context length. Decode speed is the token rate between the
first and the last streamed event (inter-token time, measured directly); prefill time is
the time to the first token minus one decode step. Nothing is inferred from a second
request, so prefill jitter does not leak into the decode number.

  python3 bench_speed.py [tokens]        default 200 decode tokens per context
"""
import json, sys, time, urllib.request

URL = "http://127.0.0.1:30000/generate"


def stream(text, n):
    body = json.dumps({"text": text, "stream": True, "sampling_params":
                       {"max_new_tokens": n, "temperature": 0, "ignore_eos": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); times, toks, prompt = [], [], 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            m = json.loads(payload)["meta_info"]
            times.append(time.time()); toks.append(m["completion_tokens"]); prompt = m["prompt_tokens"]
    return prompt, t0, times, toks


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    filler = "Quantizing large language models reduces their memory footprint. "
    print(f"  {'context':>8s} {'prefill s':>10s} {'prefill t/s':>12s} {'decode t/s':>11s} {'tokens':>6s}")
    print("  " + "-" * 54)
    for target in (128, 512, 2048, 8192, 12000):
        p = filler * max(1, target // 12)
        try:
            prompt, t0, times, toks = stream(p, n)
        except Exception as e:
            print(f"  {target:8d}  error: {e}"); continue
        if len(times) < 3 or toks[-1] - toks[0] < 2:
            print(f"  {prompt:8d}  too few events ({len(times)})"); continue
        dec = (times[-1] - times[0]) / (toks[-1] - toks[0])       # s per token, steady state
        pre = max(times[0] - t0 - dec, 1e-3)                      # TTFT minus one decode step
        print(f"  {prompt:8d} {pre:10.2f} {prompt / pre:12.0f} {1 / dec:11.1f} {toks[-1]:6d}")


if __name__ == "__main__":
    main()
