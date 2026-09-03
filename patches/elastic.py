#!/usr/bin/env python3
"""Elastic expert residency (on top of ncontig_gemv.py + placement.py v3).

Installs perf/gemv/row_arena.py and perf/gemv/expert_elastic.py into sglang/srt/layers/moe/
and adds two hooks to moe_wna16.py: the deferred placement pass builds an ExpertElastic
instead of the static v3 placement when SGLANG_MOE_ELASTIC=1, and apply() polls the control
file so S can be changed while the server runs.

  python3 elastic.py --check | apply | revert
"""
import os, shutil, sys
SG = os.path.join(os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")), "python/sglang")
M = f"{SG}/srt/layers/quantization/moe_wna16.py"
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [(os.path.join(HERE, "..", "gemv", "row_arena.py"), f"{SG}/srt/layers/moe/row_arena.py"),
         (os.path.join(HERE, "..", "gemv", "expert_elastic.py"), f"{SG}/srt/layers/moe/expert_elastic.py")]

EDITS = [
  (M, """        if len(cls._PLACE_LAYERS) == n_layers:
            self._run_placement()
            cls._PLACE_LAYERS = []
""", """        if len(cls._PLACE_LAYERS) == n_layers:
            if os.environ.get("SGLANG_MOE_ELASTIC") == "1":
                from sglang.srt.layers.moe.expert_elastic import ExpertElastic
                inst = ExpertElastic(cls._PLACE_FREQ, int(os.environ.get("SGLANG_MOE_PLACEMENT_S", "184")))
                inst.place_all(cls._PLACE_LAYERS)
                cls._ELASTIC = inst
            else:
                self._run_placement()
            cls._PLACE_LAYERS = []
"""),
  (M, """        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        if (
            getattr(layer, "_b_n_contig", False)
""", """        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."
        _el = getattr(type(self), "_ELASTIC", None)
        if _el is not None:
            _el.poll(int(dispatch_output.hidden_states.shape[0]), int(getattr(layer, "layer_id", 0)))

        if (
            getattr(layer, "_b_n_contig", False)
"""),
]


def state():
    return [(p, a in open(p, encoding="utf-8").read(), b in open(p, encoding="utf-8").read())
            for p, a, b in EDITS]


def check():
    for i, (p, pr, ap) in enumerate(state()):
        print(f"  {i} {'APPLIED' if ap else ('clean' if pr else 'MISMATCH'):<8} "
              f"{os.path.relpath(p, SG)}: {EDITS[i][1].strip().splitlines()[0][:60]}")
    for src, dst in FILES:
        print(f"  {'installed' if os.path.exists(dst) else 'absent':<9} {os.path.relpath(dst, SG)}")


def apply():
    st = state()
    if not all(pr or ap for _, pr, ap in st):
        print("  [!] mismatch (are ncontig_gemv.py and placement.py applied?)"); check(); return
    for (p, a, b), (_, pr, ap) in zip(EDITS, st):
        if not ap:
            t = open(p, encoding="utf-8").read()
            open(p, "w", encoding="utf-8").write(t.replace(a, b, 1))
    for src, dst in FILES:
        shutil.copy2(src, dst)
    print("  applied (elastic expert residency; SGLANG_MOE_ELASTIC=1 to activate)")


def revert():
    for (p, a, b), (_, pr, ap) in zip(EDITS, state()):
        if ap:
            t = open(p, encoding="utf-8").read()
            open(p, "w", encoding="utf-8").write(t.replace(b, a, 1))
    for _, dst in FILES:
        if os.path.exists(dst):
            os.remove(dst)
    print("  reverted")


if __name__ == "__main__":
    {"--check": check, "apply": apply, "revert": revert}[sys.argv[1] if len(sys.argv) > 1 else "--check"]()
