#!/usr/bin/env python3
"""
Three patches to auto-round, without which this architecture cannot be
quantized on this hardware. Each is a comment-marked, revertible edit to the
installed package.

PATCH 1 - the group_size name collision
---------------------------------------
auto_round/compressors/utils.py clears "stale" attributes from ALL modules
before a run:

    for n, m in model.named_modules():
        for key in scheme_keys:              # bits, group_size, sym, ...
            if hasattr(m, key):
                delattr(m, key)

`group_size` is a quantization key. Qwen4-Exp uses the same name as an
ARCHITECTURE parameter on Qwen4ExpTextRMSNorm - the grouped norm over the
residual stream widened by the hyper-connections (hc_count=4 -> 10240 instead
of 2560, normalized in groups of 2560).

    class Qwen4ExpTextRMSNorm(nn.Module):
        def __init__(self, dim, group_size=None, eps=1e-6):
            self.group_size = group_size          # <- architectural
        def _norm(self, x):
            if self.group_size is not None:       # <- AttributeError

AutoRound deletes it and the next forward dies with

    AttributeError: 'Qwen4ExpTextRMSNorm' object has no attribute 'group_size'

100 modules are affected in the real model (2x48 hyper_connection.hc_norm,
1x hyper_connection_mixer.hc_norm, 3x ple.norm_*). It fires on the FIRST
calibration forward, i.e. after the model has loaded - with a 185 GB source
that is roughly 40 minutes before the crash.

The fix: only clear those attributes on modules that are actually
quantization targets. On anything else, an attribute with one of these names
is architectural by definition and none of AutoRound's business.

PATCH 2 - a needless 134 GB copy while unstacking (see MARK2 below)
PATCH 3 - fp32 tuning state that does not fit in 24 GB (see MARK3 below)

  python3 07_patch_autoround.py            # apply
  python3 07_patch_autoround.py --revert   # undo
  python3 07_patch_autoround.py --check    # check only

Worth reporting upstream: github.com/intel/auto-round
"""
import argparse, os, shutil, sys

MARK = "# [qwen4-exp] group_size collision"
MARK2 = "# [qwen4-exp] no 3D contiguous"
MARK3 = "# [qwen4-exp] p_dtype fp16"

ORIG = """        # cleanup stale attributes
        for key in scheme_keys:
            # `rotation_config` on the root model carries the active
            # Hadamard rotation state (weights + hooks)
            if n == "" and key == "rotation_config":
                continue
            if hasattr(m, key):
                delattr(m, key)"""

PATCHED = f"""        # cleanup stale attributes
        {MARK}: in Qwen4-Exp `group_size` is an
        # architecture parameter on Qwen4ExpTextRMSNorm, not a scheme attribute.
        # Only clean up on real quantization targets (and the root module),
        # otherwise the grouped norm of the hyper-connections is destroyed.
        _ar_is_quant_target = (
            type(m) in supported_types or m.__class__.__name__ in inner_supported_types
        )
        for key in scheme_keys:
            # `rotation_config` on the root model carries the active
            # Hadamard rotation state (weights + hooks)
            if n == "" and key == "rotation_config":
                continue
            if n != "" and not _ar_is_quant_target:
                continue
            if hasattr(m, key):
                delattr(m, key)"""


def target():
    import auto_round
    return os.path.join(os.path.dirname(auto_round.__file__), "compressors", "utils.py")


def target3():
    import auto_round
    return os.path.join(os.path.dirname(auto_round.__file__), "wrapper.py")


ORIG3 = "        p_dtype = torch.float32  ##parameter dtype"

PATCHED3 = f"""        {MARK3}: the tuning state does not fit in 24 GB otherwise.
        # AutoRound allocates a `value` tensor per weight, the same size as the
        # weight (the learned rounding offset), plus its gradient. One
        # Qwen4-Exp block holds 512 experts = 2.52B parameters:
        #     weights bf16       4.69 GiB
        #     value   fp32       9.38 GiB
        #     grad    fp32       9.38 GiB
        #     ------------------------
        #                       23.44 GiB   on a 23.42 GiB GPU  -> OOM
        # In fp16 it is 14.06 GiB. The offset lies in [-0.5, 0.5], where fp16
        # resolves 0.0005, and the learning rate is ~0.005 - a factor of 10 in
        # hand. SignSGD only uses the sign of the gradient anyway.
        p_dtype = torch.float16  ##parameter dtype"""


def target2():
    import auto_round
    return os.path.join(os.path.dirname(auto_round.__file__), "modeling",
                        "fused_moe", "moe_experts_interface.py")


ORIG2 = """        # Ensure contiguous layout so unbind produces contiguous 2D slices.
        # This is needed when the param comes from a chunk split (Phase 1)
        # which produces non-contiguous views even on CPU.
        if not weights_cpu.is_contiguous():
            weights_cpu = weights_cpu.contiguous()
        # Unbind into a tuple of 2D tensors (zero-copy views since contiguous)
        weight_slices = weights_cpu.unbind(0)"""

PATCHED2 = f"""        {MARK2}: unbind first, then check.
        # gate_up_proj is split beforehand with chunk(2, dim=1), which makes the
        # 3D chunk non-contiguous - but EVERY individual 2D expert slice is
        # contiguous (stride (in,1)). Calling .contiguous() on the 3D tensor
        # therefore copies the entire expert stack for no reason: measured at
        # 1.57 GB per layer, 75 GB across the model -> OOM.
        weight_slices = weights_cpu.unbind(0)
        if any(not w.is_contiguous() for w in weight_slices):
            weight_slices = tuple(w.contiguous() for w in weight_slices)"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    import auto_round
    print(f"\n  auto-round {auto_round.__version__}")
    jobs = [("group_size collision", target(), MARK, ORIG, PATCHED),
            ("3D contiguous while unstacking", target2(), MARK2, ORIG2, PATCHED2),
            ("tuning state fp16 instead of fp32", target3(), MARK3, ORIG3, PATCHED3)]
    for label, p, mark, o, n in jobs:
        print(f"  {label:<36} {'patched' if mark in open(p).read() else 'open'}")
    if a.check:
        sys.exit(0 if all(m in open(p).read() for _, p, m, _, _ in jobs) else 1)

    for label, p, mark, o, n in jobs:
        src = open(p).read()
        bak = p + ".qwen4-exp.bak"
        if a.revert:
            if mark not in src:
                print(f"  {label}: nothing to revert"); continue
            if os.path.exists(bak):
                shutil.copy2(bak, p)
            else:
                open(p, "w").write(src.replace(n, o))
            print(f"  {label}: reverted"); continue
        if mark in src:
            print(f"  {label}: already patched"); continue
        if o not in src:
            print(f"  [!] {label}: expected code block not found ({p})")
            sys.exit(2)
        if not os.path.exists(bak):
            shutil.copy2(p, bak)
        open(p, "w").write(src.replace(o, n))
        print(f"  {label}: -> patched  (backup {os.path.basename(bak)})")
    print()


if __name__ == "__main__":
    main()
