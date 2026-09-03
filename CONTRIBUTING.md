# Contributing

This repository is the code side of a measured result. Contributions are welcome; the second half of
this document, the measurement rules, is what makes a change comparable with the numbers in the README.

## Issues

Open a GitHub issue with: the SGLang commit and whether the serving patch applied cleanly
(`git apply --check`), the GPU and host RAM, the launch line (ideally `scripts/serve.sh` with your
variables), the server log around the failure, and for a numbers question the tool output
(`tools/bench_speed.py`, `tools/nll_long.py`, ...) together with the flag set it was measured on.

## Pull requests

* Keep one change per pull request and say in the description which layer (`patches/*.py`), tool or
  document it touches.
* Before opening it, run what applies: `python3 -m py_compile` over changed Python files; the unit
  tests of the touched layer (table below); for a serving change, the validation protocol below with
  its numbers in the description; for a documentation change, check that every path in the text
  exists in the repository and every number cites its source.
* Patch scripts edit the checkout named by `$SGLANG` by exact string replacement and must keep
  `apply` / `revert` / `--check` symmetric. Revert layered patches in reverse order; the layering and
  the expected `MISMATCH` lines under overlays are in `patches/README.md`.
* Measurement code (`tools/`, `gemv/`, `scripts/`) is kept as it ran; path changes go through
  environment variables or `os.path.expanduser`, and code that produced a published number is not
  refactored.
* No fixed formatter is enforced; match the style of the file you edit.

## License of contributions

By contributing you agree that your contribution is licensed under the Apache License 2.0 of this
repository (`LICENSE`). Changes to SGLang code carried in the patch are subject to SGLang's Apache-2.0
license as well.

## Upstream policy

Changes to SGLang itself belong upstream. `sglang/UPSTREAM.md` is the plan for that: which parts are
general and which are Qwen4-Exp-specific, the split into reviewable commits, and what an upstream
version still needs; the reviewable series, the RFC issue text and the PR descriptions are in
`sglang/upstream/`. A fix that lands in SGLang should be removed from the patch here, with a note in
`sglang/PATCH_NOTES.md`.

## Measurement rules

1. **Every change is measured against the previous accepted state with the same benchmark**
   (`tools/bench_speed.py`, streamed inter-token rate over 200 tokens at five contexts; the older
   two-request bench leaked prefill jitter into the decode number, CAMPAIGN.md:306).
2. **Exact changes are verified with the teacher-forced logprob oracle** (`tools/logprob_diff.py`,
   450 continuation tokens, ~9 s). Same-config noise is MAX 0.09 / MEAN 0.002 nats (CAMPAIGN.md:247).
   Thresholds in `scripts/phase1.py` (`LP_MEAN_TOL = 0.01`, `LP_MAX_TOL = 0.5`) for host-side changes;
   kernel-class changes are accepted at MEAN <= 0.05 only after a per-layer identical-input A/B shows
   agreement below 1e-3 relative (CAMPAIGN.md:92-96). Greedy token diffs are not an oracle: the server
   is not bitwise reproducible run to run (CAMPAIGN.md:244).
3. **Approximate changes need a quality check and are flagged as such** — short held-out NLL
   (`tools/nll_eval.py`) for weight changes; the long-text tests below for anything that touches the KV
   cache, because prompts shorter than one prefill chunk (1,024 tokens) never read it (CAMPAIGN.md:353).
4. **A change that does not improve the number is reverted**, not kept for later.
5. **Numbers in documents are traceable**: quote the dated log line, the log file or the state file.
   `CAMPAIGN.md` is the verbatim campaign log and is not extended by this repository; new measurements
   go into a new dated log under `docs/logs/` and are cited from there.
6. **Host memory rules**: never pin host memory at runtime on a 32 GB host with 24 GB already pinned
   (CAMPAIGN.md:346-347); keep `MemoryMax=30G` as the scope's backstop and do not exempt the scope from
   `systemd-oomd` (`docs/HISTORY.md`, host-RAM facts). No server start while a patch workflow may be
   applying or reverting (CAMPAIGN.md:420, :435-436).

## Unit tests (`gemv/`)

All tests are plain scripts (`python3 gemv/<test>.py`); most need the GPU and a window in which the
server is down or has the stated headroom. The patch-dependent ones read the patched SGLang tree
(the checkout named by `$SGLANG`, default `~/quant/sglang`) and must run with its virtualenv interpreter.
The GEMV tests read real expert tensors from the checkpoint directory named by `$Q` (default
`~/quant/model`).

| Test | Needs | What it checks |
|---|---|---|
| `test_gemv.py` | GPU, real expert tensors, server down (>1.5 GiB free) | `moe_gemv_int2` vs a torch fp32 reference; bandwidth from device and pinned host |
| `test_gemv_ncontig.py` | GPU, real tensors | N-contiguous int2 GEMV in the checkpoint's native orientation; GB/s device / host |
| `test_gemv_tab.py` | GPU, real tensors | `to_word_ncontig()` bit-exact vs native int32 words; full one-layer decode MoE vs fp32 |
| `test_tiled_word.py` | GPU; `ref` on a pristine tree, `cmp` after `ncontig_gemv.py apply` | word-load int2 branch through the real `invoke_fused_moe_kernel` (the `tiled_ref*.pt` dumps, 14 MB each, are not in the repo; regenerate with `ref`) |
| `row_arena.py` (self-test in `__main__`) | GPU, idle | VMM arena: 2 MiB granularity, shrink returns memory to the driver, CUDA graph replays after shrink and regrow |
| `test_kv_fp8.py` | GPU, `kv_fp8.py` applied, ~50 MB VRAM | fp8 gather-dequant bit-exact vs `to(fp8).to(bf16)`; write-path saturation |
| `test_kv_int8.py` | GPU, `kv_int8.py` applied, ~50 MB | quantize+scatter, scale index, compact gather, prefix row gather — bit-exact vs torch reference |
| `test_kv_int4.py` | GPU, `kv_int4.py` applied, ~30 MB | nibble pack/unpack round trip, the three INT4 kernels, fp16 scale clamp, trtllm strided compact |
| `test_kv_tiers.py` | GPU, `kv_int4.py` + `kv_tiers.py` applied, ~40 MB | ring R = 64 over a 4,096-slot pool: owner table, hot/cold tier dispatch, same-launch aliasing |
| `test_kv_paged_prefix.py` | GPU, `kv_tiers.py` + `kv_paged_prefix.py` applied, <200 MB | the rejected paged kernel vs the materialised path (2 row-ulps), timing |
| `test_ngram_chain.py` | CPU only | `_linearize_chain` of `ngram_ple.py` against a Python replica of the trie |

The kernel A/B that settled the S9 numerics question is `docs/logs/MOE-IO-DIFF.txt` (per-layer MoE
input/output dumps, `host_fixes.py` item `moedump`; CAMPAIGN.md:92-96), not a unit test.

## Validation protocol for a serving change

The sequence below is what `tools/int8_validate.sh` ran for every KV mode (CAMPAIGN.md:369-370, :413).
The individual tools (`bench_speed.py`, `nll_eval.py`, `nll_long.py`, `logprob_diff.py`,
`needle_test.py`, `longctx_test.py`, `elastic_sweep.py`) run as they are against a server started
with `scripts/serve.sh` (plus the flags of the change). The harness drivers (`scripts/phase1.py`,
`scripts/sweep.sh`, `tools/int8_validate.sh`, `tools/nll_series.sh`) are kept as they ran and expect
the working layout described in the docstring of `scripts/phase1.py` (the driver, `sweep.sh`,
`patches/`, the oracle files and `phase1_state.json` in one directory); to use them, symlink those
files into one directory or set the paths named there.

1. **Bring-up** with the accepted flag set plus the step's flags/patches (`scripts/phase1.py --bringup
   STEP`, or `scripts/serve.sh` with the added flags); read the "KV lazy backing" lines of the server
   log.
2. **Keep the processes warm** (`tools/keepalive.sh`) — on the reference host (32 GB RAM) idle
   scheduler pages get swapped out and a request minutes later refaults them (CAMPAIGN.md:368).
3. **Shrink the expert cache to S = 184** (`tools/elastic_sweep.py`, control file `S 184`) before any
   input-logprob scoring: SGLang computes input logprobs per prefill chunk (1,024 x 248,320 x 4 B = 1 GB).
4. **Short NLL** — `tools/nll_eval.py check int8dense` (write path only; the reference is saved with
   `save int8dense` on the accepted state first).
5. **Long-text NLL vs the bf16 pool** — `tools/nll_long.py check bf16` (last 512 of 9,586 tokens plus a
   300-token greedy continuation after 3,759 tokens). Noise floor: NLL +-0.008, mean abs. dlogprob 0.099.
6. **All-position series** — `tools/nll_series.sh REF STEP...` with `NLL_LONG_ALL=1` (8,561 positions,
   ~0.001-nat resolution; noise +0.0002 / 0.059). Fake-quant steps (`SGLANG_KV_FAKEQ`) must run with
   `--kv-cache-dtype auto`: the hook lives in the bf16 pool's `set_kv_buffer` and is bypassed by the
   int8/int4/tiered subclasses (ERRATUM, CAMPAIGN.md:394). Always include a destructive control
   (`SGLANG_KV_FAKEQ=zero`) when the metric looks too good.
7. **Logprob oracle** — `tools/logprob_diff.py check lp2` (lp2 is the reference saved on the INT8-dense
   server; references must be re-saved whenever an approximation is accepted, CAMPAIGN.md:317).
8. **Streaming bench** — `tools/bench_speed.py`; the rule is "not worse than the accepted state by more
   than 2 %" on the two long contexts (`scripts/phase1.py` docstring). Re-measure after the startup
   transient (CAMPAIGN.md:336).
9. **Long context** — `tools/longctx_test.py N` (prints the elastic status before, during and after) and
   `tools/needle_test.py N [TAG]` (appends to `needle_results.tsv`; use `max_new_tokens` >= 400, the bf16
   3/5 was a truncation, CAMPAIGN.md:401-402).

A KV mode is accepted only with all of: short NLL within noise, all-position NLL within the ladder
expectation for its precision, oracle mean at the floor (~0.002), bench at parity, a long prompt with a
correct answer and a live server afterwards, needles 5/5.

## Documentation rules

* Every number names its source: the log file under `docs/logs/`, the `CAMPAIGN.md` line, or the
  state file.
* Documents written at an earlier state keep their text and carry a status note at the top; the
  authoritative numbers are the README and the log.
