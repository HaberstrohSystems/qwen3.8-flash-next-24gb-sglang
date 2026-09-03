"""Long-context functional test against a running server.

  python3 longctx_test.py 60000        prompt of ~60k tokens, 64 new tokens

Prints prefill time / rate, decode rate, the answer, and (if SGLANG_MOE_ELASTIC_CTL is set)
the elastic status before, during (polled every 5 s) and after the request, so a shrink of the
expert cache under a long KV commit and its regrowth afterwards are visible.
"""
import json, os, sys, threading, time, urllib.request

URL = "http://127.0.0.1:30000/generate"
H = os.path.dirname(os.path.abspath(__file__))
STATUS = os.environ.get("SGLANG_MOE_ELASTIC_CTL", os.path.join(H, "elastic.ctl")) + ".status"


def status():
    try:
        kv = dict(l.split(" ", 1) for l in open(STATUS).read().strip().splitlines() if " " in l)
        return f"S {kv.get('S_min')}-{kv.get('S_max')} arena {float(kv.get('arena_GB', 0)):.2f} GB free {float(kv.get('vram_free_GB', 0)):.2f} GB"
    except Exception:
        return "(no status)"


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    filler = ("The quick brown fox jumps over the lazy dog. Section %d discusses memory hierarchies, "
              "virtual address spaces and the cost of page faults in accelerators. ")
    text = "".join(filler % i for i in range(target // 30)) + "\nQuestion: what animal jumps over the dog? Answer in one word:"
    print(f"  before: {status()}")
    stop = False

    def watch():
        while not stop:
            time.sleep(5); print(f"  during: {status()}", flush=True)
    th = threading.Thread(target=watch, daemon=True); th.start()
    body = json.dumps({"text": text, "stream": True,
                       "sampling_params": {"max_new_tokens": 64, "temperature": 0, "ignore_eos": True}}).encode()
    t0 = time.time(); times, toks, prompt, out = [], [], 0, ""
    try:
        with urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=3600) as r:
            for line in r:
                if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                    continue
                d = json.loads(line[5:])
                if "meta_info" not in d:
                    print(f"  server message: {str(d)[:300]}"); continue
                m = d["meta_info"]
                times.append(time.time()); toks.append(m["completion_tokens"]); prompt = m["prompt_tokens"]; out = d["text"]
    except Exception as ex:
        body_txt = ex.read()[:400] if hasattr(ex, "read") else ""
        print(f"  request failed after {time.time() - t0:.1f} s: {ex} {body_txt}")
    stop = True
    if len(times) < 2:
        print(f"  no streamed tokens (events {len(times)}); after: {status()}"); return
    dec = (times[-1] - times[0]) / max(1, toks[-1] - toks[0]); pre = times[0] - t0 - dec
    print(f"  prompt {prompt} tokens: prefill {pre:.1f} s ({prompt / pre:.0f} tok/s), decode {1 / dec:.1f} tok/s")
    print(f"  answer: {out[:80]!r}")
    time.sleep(3); print(f"  after:  {status()}")


if __name__ == "__main__":
    main()
