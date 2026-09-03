# Tools

Measurement and benchmark tools. All of them talk to a running server on `127.0.0.1:30000` over HTTP
(`/generate`, `/v1/chat/completions`) and never start or stop one, except where noted. State files
(`nll/`, `greedy/`, `logprob/`, `spec_lossless/`, `needle_results.tsv`, `elastic.ctl`) are created next
to the scripts and are git-ignored, except the two oracle inputs listed below.

| Tool | Purpose |
|---|---|
| `bench_speed.py` | Streaming prefill/decode bench: one streamed generation per context (101 .. 10,001 tokens), decode = inter-token rate over 200 tokens, prefill = time to first token minus one step (CAMPAIGN.md:306). |
| `logprob_diff.py` | Teacher-forced logprob oracle for exactness: `save NAME` / `check NAME`, prints `LOGPROB_MAX` / `LOGPROB_MEAN` (thresholds in `CONTRIBUTING.md`). Needs `greedy_diff.py` and `greedy/oa.json`. |
| `greedy_diff.py` | The three-prompt set (German prose, English reasoning, code; the German passage is a deliberate held-out domain, not a leftover) the oracle uses; on its own a greedy token diff, which is not an oracle (the server is not bitwise reproducible run to run, CAMPAIGN.md:244). |
| `greedy/oa.json` | The fixed continuations the oracle scores: 3 x 200 greedy token ids, of which `logprob_diff.py` teacher-forces the first 150 per prompt (450 tokens). Tracked in git. |
| `logprob/lp2.json` | The oracle reference saved on the accepted INT8-dense state (`logprob_diff.py check lp2`, CAMPAIGN.md:317). Tracked in git; other references are regenerated with `logprob_diff.py save NAME`. |
| `nll_eval.py` | Short held-out NLL (three ~700-token passages: German prose, English, Python; German is a deliberate held-out domain) for approximations; blind to the KV cache. |
| `nll_long.py` | Long-text NLL on the cache read paths: last 512 (or with `NLL_LONG_ALL=1` all >= 1,024) positions of a 9,586-token text plus a 300-token greedy continuation. The text is the first 26,000 characters of `docs/CAMPAIGN.md` (stable because the log is append-only); `NLL_LONG_TEXT=<file>` overrides it. |
| `nll_series.sh` | Drives an all-position series over several bring-ups via `scripts/phase1.py` (restarts servers). Harness driver, see below. |
| `int8_validate.sh` | The validation sequence for a KV mode (bring-up, keepalive, S 184, short NLL, long NLL, oracle, bench, long context). Restarts a server. Harness driver, see below. |
| `needle_test.py` | Needle-in-a-haystack: five codes at 10-90 % depth of seeded random prose; appends to `needle_results.tsv`. |
| `longctx_test.py` | Long prompt with elastic-status polling before/during/after (watch the expert cache shrink and regrow). |
| `elastic_sweep.py` | Live S sweep through the control file (`S n`, `fill`), one bench per S, no restarts. |
| `keepalive.sh` | One tiny request every 5 s to keep the server processes from being swapped out during long measurements. |
| `expert_freq.py` | Builds the routing-mass histogram (`assets/expert_freq.pt`) from a `SGLANG_ROUTE_DUMP` directory (`host_fixes.py` item `dump`). Imports torch. |
| `spec_lossless.py` | Lossless gate for NGRAM speculation (spec-path logprobs vs teacher forcing, near-tie rule). |
| `toolcall_smoke.py` | One chat request with the mini-swe-agent bash tool; checks the reasoning and tool-call parsers. |
| `final_bench.sh` | GPQA Diamond through `sglang.test.run_eval` (local or OpenRouter; thinking sampling T 1.0 / top_p 0.95 / top_k 20, 16,384 max tokens). Reads the OpenRouter key from `OPENROUTER_API_KEY`. No score of the 2-bit model is published (`docs/HISTORY.md`). |
| `deepswe_run.sh` | DeepSWE 1.1 driver: one `pier run` per task from the seed-0 task list `deepswe/tasklist_seed0.txt` (`TASKLIST` overrides; `TASKS` names the task directory), images deleted after each task; local side through `fwd.sh`. |
| `fwd.sh` | `socat` forwarder `172.17.0.1:30001 -> 127.0.0.1:30000` so Docker task containers reach the server. |
| `make_mini_model.py`, `nvme_probe.py`, `test_unpack_2bit.py`, `test_group_layout.py`, `test_vs_bf16.py` | Pipeline helpers and 2-bit kernel checks from the pre-campaign state (`docs/WRITEUP.md` section 8). |

The individual tools run as they are against a server started with `scripts/serve.sh`. The harness
drivers (`int8_validate.sh`, `nll_series.sh`, together with `scripts/phase1.py` and `scripts/sweep.sh`)
are kept as they ran and expect the working layout described in the docstring of `scripts/phase1.py`
(driver, `sweep.sh`, `patches/`, the oracle files and `phase1_state.json` in one directory; a saved
`nll/int8dense` reference; `int8_validate.sh` runs the `bench_speed.py` next to it, or the one named by
`BENCH`, and writes `logs/` relative to that directory). Symlink the files into one directory or set the
paths named there.

Interpreter paths default to `~/quant/venv*/bin/python3` (override `PY=` for the shell drivers);
`test_unpack_2bit.py` reads its mini-model from `~/quant/out/mini-sub4/` via `os.path.expanduser`,
`test_vs_bf16.py` reads the checkout named by `$SGLANG_SRC`.
