"""Quality metric for APPROXIMATE changes: teacher-forced mean NLL on held-out text.

  python3 nll_eval.py save NAME      # record per-token logprobs of the held-out texts
  python3 nll_eval.py check NAME     # mean NLL now vs NAME, plus mean |delta| per token

Three ~700-token passages (German prose, English technical, Python code). A requantization
that is quality-neutral changes the mean NLL by a few 1e-3 nats; the sealed int2 experts
themselves cost roughly 0.1-0.3 nats over bf16 on such text, for scale.
"""
import json, os, sys, urllib.request

URL = "http://127.0.0.1:30000/generate"
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nll")
TEXTS = [
    ("de", "Die Geschichte der Dampfmaschine beginnt nicht mit James Watt, sondern mit Thomas Newcomen, "
           "dessen atmosphärische Maschine ab 1712 Wasser aus englischen Kohlebergwerken pumpte. Sie war "
           "ineffizient, weil der Zylinder bei jedem Hub abgekühlt und wieder erhitzt werden musste. Watts "
           "entscheidende Idee war der getrennte Kondensator: Der Dampf kondensierte in einem eigenen, "
           "kalten Gefäß, während der Zylinder heiß blieb. Das senkte den Kohleverbrauch um etwa drei "
           "Viertel und machte die Maschine für Fabriken interessant, die weit von Kohlefeldern entfernt "
           "lagen. Matthew Boulton finanzierte die Entwicklung und drängte auf die Umsetzung der "
           "Drehbewegung, denn Pumpen allein reichte den Textilfabrikanten nicht. Mit dem Sonnen-und-"
           "Planeten-Getriebe, das Watt entwarf, weil das Kurbelprinzip patentiert war, ließen sich "
           "Spinnmaschinen und Webstühle antreiben. Innerhalb von zwei Jahrzehnten entstanden in "
           "Manchester und Birmingham Fabrikviertel, deren Schornsteine das Stadtbild prägten. Die "
           "sozialen Folgen waren gewaltig: Landarbeiter zogen in die Städte, Kinderarbeit nahm zu, und "
           "erst die Fabrikgesetze des neunzehnten Jahrhunderts setzten dem Grenzen. Zugleich sanken die "
           "Preise für Tuch so stark, dass sich auch einfache Haushalte Baumwollkleidung leisten konnten, "
           "was wiederum die Nachfrage nach Rohbaumwolle aus den Südstaaten Amerikas anheizte und dort "
           "die Sklaverei wirtschaftlich zementierte. Technikgeschichte ist deshalb nie nur eine Geschichte "
           "von Erfindungen, sondern immer auch eine von Märkten, Institutionen und Machtverhältnissen, "
           "die eine Erfindung erst zur Umwälzung machen oder sie jahrzehntelang in der Werkstatt "
           "verkümmern lassen, wie es dem Heron von Alexandria mit seiner Aeolipile ergangen war."),
    ("en", "A Kalman filter maintains a Gaussian belief over the state of a linear dynamical system and "
           "updates it in two steps. In the prediction step the mean is propagated through the state "
           "transition matrix and the covariance grows by the process noise: P becomes F P F^T + Q. In "
           "the update step a measurement z with noise covariance R is incorporated by computing the "
           "innovation z - H x, its covariance S = H P H^T + R, and the Kalman gain K = P H^T S^{-1}. The "
           "posterior mean is x + K (z - H x) and the posterior covariance is (I - K H) P. The gain "
           "balances trust between the model and the sensor: when R is large relative to P the gain is "
           "small and the filter ignores noisy measurements; when the prediction is uncertain the gain "
           "approaches H^{-1} and the estimate snaps to the measurement. For a one-dimensional object "
           "tracked with position measurements, the state is position and velocity, F encodes constant "
           "velocity over the sampling interval, and H selects the position. With dt = 0.1 s, a process "
           "noise of 0.01 and a measurement noise of 1.0, the steady-state gain settles near 0.3 for "
           "position and 0.5 for velocity, so the filter effectively averages about three measurements "
           "while still tracking accelerations within a few samples. The derivation assumes linearity and "
           "Gaussian noise; the extended and unscented variants relax the first assumption by "
           "linearizing around the current estimate or by propagating sigma points, and particle filters "
           "relax both at considerably higher computational cost."),
    ("py", "import heapq\nfrom collections import defaultdict\n\n\ndef dijkstra(graph, source):\n"
           "    \"\"\"Shortest path distances from source in a weighted directed graph.\n\n"
           "    graph: dict mapping node -> list of (neighbor, weight); weights must be non-negative.\n"
           "    Returns (dist, prev) where prev reconstructs paths.\n    \"\"\"\n"
           "    dist = defaultdict(lambda: float('inf'))\n    prev = {}\n    dist[source] = 0.0\n"
           "    heap = [(0.0, source)]\n    visited = set()\n    while heap:\n"
           "        d, u = heapq.heappop(heap)\n        if u in visited:\n            continue\n"
           "        visited.add(u)\n        for v, w in graph.get(u, ()):\n            nd = d + w\n"
           "            if nd < dist[v]:\n                dist[v] = nd\n                prev[v] = u\n"
           "                heapq.heappush(heap, (nd, v))\n    return dist, prev\n\n\n"
           "def path(prev, target):\n    out = [target]\n    while out[-1] in prev:\n"
           "        out.append(prev[out[-1]])\n    return out[::-1]\n\n\n"
           "if __name__ == '__main__':\n    g = {'a': [('b', 1), ('c', 4)], 'b': [('c', 2), ('d', 5)], "
           "'c': [('d', 1)]}\n    dist, prev = dijkstra(g, 'a')\n    assert dist['d'] == 4\n"
           "    assert path(prev, 'd') == ['a', 'b', 'c', 'd']\n    print('ok', dict(dist))\n"),
]


def logprobs(text):
    body = json.dumps({"text": text, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                       "return_logprob": True, "logprob_start_len": 0}).encode()
    with urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}),
                                timeout=900) as r:
        d = json.load(r)
    d = d[0] if isinstance(d, list) else d
    return [t[0] for t in d["meta_info"]["input_token_logprobs"] if t[0] is not None]


def main():
    cmd, name = sys.argv[1], sys.argv[2]
    os.makedirs(D, exist_ok=True)
    p = os.path.join(D, f"{name}.json")
    cur = {tag: logprobs(t) for tag, t in TEXTS}
    if cmd == "save":
        json.dump(cur, open(p, "w"))
        for tag, lp in cur.items():
            print(f"  {tag}: {len(lp)} tokens, mean NLL {-sum(lp)/len(lp):.4f}")
        print(f"  saved {p}"); return
    ref = json.load(open(p))
    tot_ref = tot_cur = n = 0.0
    for tag in cur:
        a, b = ref[tag], cur[tag]; m = min(len(a), len(b))
        nll_a = -sum(a[:m]) / m; nll_b = -sum(b[:m]) / m
        dl = sum(abs(x - y) for x, y in zip(a[:m], b[:m])) / m
        print(f"  {tag}: NLL ref {nll_a:.4f}  now {nll_b:.4f}  delta {nll_b-nll_a:+.4f}   mean|dlogprob| {dl:.4f}")
        tot_ref += -sum(a[:m]); tot_cur += -sum(b[:m]); n += m
    print(f"\n  overall NLL ref {tot_ref/n:.4f}  now {tot_cur/n:.4f}  delta {(tot_cur-tot_ref)/n:+.4f} nats/token")
    print(f"  NLL_DELTA={(tot_cur-tot_ref)/n:.6f}")


if __name__ == "__main__":
    main()
