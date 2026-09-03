"""Live S sweep against a running elastic server: no restarts, one bench per S.

  python3 elastic_sweep.py 184 200 216 232 248     explicit S values
  python3 elastic_sweep.py fill                      grow to fill VRAM (512 MB reserve), bench once

Writes the command to SGLANG_MOE_ELASTIC_CTL (default perf/elastic.ctl), pokes the server
with a tiny request so the poll runs, waits for the status file, then runs the streaming bench.
"""
import json, os, subprocess, sys, time, urllib.request

H = os.path.dirname(os.path.abspath(__file__))
CTL = os.environ.get("SGLANG_MOE_ELASTIC_CTL", os.path.join(H, "elastic.ctl"))
BENCH = os.path.join(H, "..", "release", "tools", "bench_speed.py")
PY = os.path.expanduser("~/quant/venv/bin/python3")
URL = "http://127.0.0.1:30000/generate"


def poke(n=1):
    body = json.dumps({"text": "Hi", "sampling_params": {"max_new_tokens": n, "temperature": 0}}).encode()
    urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=600).read()


def command(cmd, timeout=600):
    st = CTL + ".status"
    before = os.path.getmtime(st) if os.path.exists(st) else 0
    with open(CTL, "w") as f:
        f.write(cmd + "\n")
    t0 = time.time()
    while time.time() - t0 < timeout:
        poke()                                   # prefill runs apply() -> poll()
        time.sleep(1.0)
        if os.path.exists(st) and os.path.getmtime(st) > before:
            return open(st).read()
    raise TimeoutError(f"no status update for '{cmd}'")


def main():
    args = sys.argv[1:] or ["184"]
    print(f"  {'cmd':>8s} {'S':>9s} {'arena GB':>9s} {'free GB':>8s} {'mass':>6s} | decode t/s @ 100 / 1.7k / 6.8k / 10k")
    for a in args:
        cmd = "fill 512" if a == "fill" else f"S {int(a)}"
        status = command(cmd)
        kv = dict(l.split(" ", 1) for l in status.strip().splitlines() if " " in l)
        out = subprocess.run([PY, BENCH, "160"], capture_output=True, text=True).stdout
        rows = [l.split() for l in out.splitlines() if l.strip() and l.split()[0].isdigit()]
        dec = [r[3] for r in rows]
        print(f"  {cmd:>8s} {kv.get('S_min','?')+'-'+kv.get('S_max','?'):>9s} {float(kv.get('arena_GB','0')):9.2f} "
              f"{float(kv.get('vram_free_GB','0')):8.2f} {float(kv.get('mass_covered','0')):6.3f} | " + " / ".join(dec))


if __name__ == "__main__":
    main()
