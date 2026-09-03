# Documents

Documents written during the work, kept as written. Placeholders in the plan documents: `$SGLANG` = the
patched SGLang checkout, `perf/` (where it still appears in quoted log lines) = the working directory
the tools ran in (its scripts are `tools/`, `patches/` and `scripts/` in this repository; its data
files `assets/`), `~/quant/` = model and venv directory. `CAMPAIGN.md` and everything under `logs/` are verbatim except that host-specific paths are abbreviated to `~` (log quotes otherwise keep their paths).
Documents written before the final state carry a status note at the top; the authoritative numbers
are in the top-level [README](../README.md) and in `CAMPAIGN.md`. The complete serving patch and its
notes live in [`../sglang/`](../sglang/). What was rejected, and why, is collected in
[`HISTORY.md`](HISTORY.md).

| File | What it is | Written at |
|---|---|---|
| [`WRITEUP.md`](WRITEUP.md) | The quantization recipe, the two AutoRound traps, the nine SGLang findings, the base patch, the first measurements (15.5 tok/s, 32k) | after the base patch, before the performance campaign |
| [`ELASTIC_MEMORY.md`](ELASTIC_MEMORY.md) | The three memory mechanisms, the KV quality protocol and precision ladder, speculation, where long-prefill time goes | 2026-09-02, last updated after S21 |
| [`HISTORY.md`](HISTORY.md) | The rejected steps with their reasons, the ERRATUM, the host-RAM facts, the benchmark status | 2026-09-03 |
| [`TIMELINE.md`](TIMELINE.md) | The dated timeline of the work, in chronological order, every entry citing the log | 2026-09-03 |
| [`MODELCARD.md`](MODELCARD.md) | Pointer to the model card, which lives with the weights on the Hub | 2026-09-03 |
| [`KV_INT8_PLAN.md`](KV_INT8_PLAN.md) | Design panel synthesis and implementation plan for the INT8-G64 KV cache (stage A implemented as `patches/kv_int8.py`) | 2026-09-02 15:50 |
| [`KV_INT4_PLAN.md`](KV_INT4_PLAN.md) | Stage C plan, INT4-G32 (`patches/kv_int4.py`) | 2026-09-02 |
| [`KV_TIERS_PLAN.md`](KV_TIERS_PLAN.md) | The tiered KV cache plan (the plan's own working name is "compost"), layout (B) dual-write (`patches/kv_tiers.py`, the default) | 2026-09-02 19:56 |
| [`KV_PAGED_PREFIX_PLAN.md`](KV_PAGED_PREFIX_PLAN.md) | Paged prefix-chunk kernel plan, with its **Outcome** section: built, verified, rejected on timing | 2026-09-02 22:18 / outcome 2026-09-03 |
| [`SPEC_NGRAM_PLAN.md`](SPEC_NGRAM_PLAN.md) | Speculative decoding verdict (MTP infeasible, NGRAM with PLE) and the patch list for `patches/ngram_ple.py` | 2026-09-02 17:08 |
| [`DECODE_PERF_PLAN.md`](DECODE_PERF_PLAN.md) | The ranked decode-performance plan synthesised from six investigation threads before the campaign | before the campaign |
| [`CAMPAIGN.md`](CAMPAIGN.md) | The append-only engineering log; the single source of truth for every number. `CAMPAIGN.md:N` in any document is line N of this file; line numbers stay valid as the log grows | 2026-09-01 .. 2026-09-03 |
| [`logs/tiers-validate.log`](logs/tiers-validate.log) | S21 validation: lazy-backing lines, short NLL, 512-window NLL, oracle, streaming bench, 258k prompt, all-position NLL, needles | 2026-09-02 20:33-21:05 |
| [`logs/night.log`](logs/night.log) | Overnight validation chain of 2026-09-02/03: tree check, smoothing A/B, restart #20 bench, 258k re-measure (165.3 s / 52.3), tool-call smoke | 2026-09-02 23:40-23:52 |
| [`logs/night2.log`](logs/night2.log), [`night3.log`](logs/night3.log), [`night4.log`](logs/night4.log) | Post-restart 10k benches of restarts #21-#23 (`night4.log`: restart #23 with `--max-running-requests 4 --max-mamba-cache-size 8`, not the published configuration) | 2026-09-02 23:55 .. 2026-09-03 00:36 |
| [`logs/elastic.ctl.status`](logs/elastic.ctl.status) | The elastic expert cache's status file at 00:36:50 (restart #23, 8 GDN state slots): S = 184 on all 48 layers, arena 11.062 GiB (11.88 GB), 2.084 GiB free, 84.3 % routing mass covered; the file's `_GB` suffix means bytes / 2**30 | 2026-09-03 00:36 |
| [`logs/needle_results.tsv`](logs/needle_results.tsv) | Needle test rows: time, tag, tokens, score, hits, prefill s, decode tok/s | 2026-09-02 |
| [`logs/MOE-IO-DIFF.txt`](logs/MOE-IO-DIFF.txt) | The per-layer MoE input/output A/B that settled the S9 numerics question (word layout vs byte layout; CAMPAIGN.md:92-96) | 2026-09-02 11:24 |
| [`logs/probe_trtllm_sm120.log`](logs/probe_trtllm_sm120.log) | Output of `tools/probe_trtllm_sm120.py`: FlashInfer's `trtllm_batch_decode_with_kv_cache` (the SM120 sparse-decode route of `qwen4-main-squashed` since #36806) at the QSA backend's shapes against a torch softmax reference; max relative error 3.9e-3 / 2.5e-3 / 2.5e-3, no NaN (`../sglang/UPSTREAM.md`) | 2026-09-03 10:29 |

`night2-4.log` show the startup transient after a restart (CAMPAIGN.md:306, :336); the accepted numbers
are those of `tiers-validate.log`.
