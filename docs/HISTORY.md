# History: what was rejected, corrected or left open

The engineering record behind the numbers in the top-level README. Every line cites
[`CAMPAIGN.md`](CAMPAIGN.md) (`:N` = line N of the append-only log).

## Steps tried and reverted

* **Overlap schedule** (S1) — numerically equivalent (logprob max 0.075 / mean 0.0016) but not
  faster (14.8 vs 15.2 tok/s); reverted (CAMPAIGN.md:253-254). **Continuous decode steps** (S2) —
  decode 9.7 tok/s (-28 %) for +25 % prefill; reverted as harmful (CAMPAIGN.md:240, :250).
* **fp8 KV cache as a speed step** (S16) — 55.1 vs 56.9 tok/s, reverted as a speed step; kept as a mode
  at parity after re-measurement (55.9-57.2 / 2,303-2,339) (CAMPAIGN.md:351-353). A 1-token prompt
  still crashes in that mode (:355).
* **NGRAM speculation** — lossless within the decode-vs-prefill floor (0 mismatches in 600 tokens), but
  own-history n-gram drafts accept only 1.11-1.27 tokens per step, so decode lands at 22-25 tok/s with
  eager verify vs 56 without; opt-in only (`patches/ngram_ple.py`, CAMPAIGN.md:387). MTP is infeasible
  on the reference host: the head is bf16 (4.85 GiB, `WRITEUP.md` section 3; its experts alone 4.7 GiB,
  :379).
* **Paged prefix-chunk kernel** (S22) — built, correct within 2 row-ulps, rejected on timing: 6.1 ms
  (tiered) / 3.2 (int8) / 4.2 (int4) vs 1.4 ms materialised per head-layer at prefix 60k; the gather it
  removes is ~2.5 % of chunk time (CAMPAIGN.md:439; [`KV_PAGED_PREFIX_PLAN.md`](KV_PAGED_PREFIX_PLAN.md)
  "Outcome"). The O(prefix) slope of long prefills is the QSA indexer, i.e. the model.
* **Static per-channel smoothing for INT4** (stage B) — mean NLL better by ~0.005 nats, per-token
  deviation worse (+0.0022 / 0.160 vs +0.006..0.009 / 0.137-0.139); not implemented in the real kernels
  (CAMPAIGN.md:444).
* **Byte-shuffled table GEMV** — 157 / 14 GB/s, refuted in favour of the int32-word layout (320 / 51
  GB/s) (CAMPAIGN.md:218-228, :196-204).

## ERRATUM in the KV precision ladder

Every fake-quant run after S17 had silently executed on the real INT8 pool, so the first all-position
ladder (including an "INT8-G64 +0.001 / 0.059" row quoted in early documents) is invalid; it was caught
by the K/V := 0 control (CAMPAIGN.md:394). The corrected ladder on the bf16 pool: noise +0.0002 /
0.059, fake INT4-G32 +0.0078 / 0.139, real INT4-G32 +0.0088 / 0.138, fake INT2-G16 +0.3004 / 0.619,
K/V := 0 +3.699 / 3.923 (:406). No valid all-position INT8-vs-bf16 figure exists; the INT8 evidence is
the 512-window comparison (mean abs. dlogprob 0.094 vs noise 0.099, NLL +0.010, :370). The tiered
default measures -0.0001 / 0.0743 on the all-position test (:413).

## Host-RAM facts

* After loading, 31 offloaded expert layers pin ~24 GB of the 32 GB (1.33 MB per expert,
  CAMPAIGN.md:305); ~2 GB stay free.
* A runtime `cudaHostAlloc` of 4 MB took 22 s and triggered `systemd-oomd`; a 315 MB pinned reserve at
  startup killed the server at load (:346-347). Policy: no runtime pinning; the expert-cache floor is
  S = 184; prompts whose KV cannot be backed are refused at admission.
* `MemoryMax=27G` caused cgroup reclaim storms at ~55k-token prefills; the scope runs with
  `MemoryMax=30G` (:345). `systemd-oomd` kills on pressure, not on limits.
* Concurrency multiplies the host expert-row traffic; eight concurrent requests were killed by
  `systemd-oomd` (:454). The published configuration serves one request at a time
  (`scripts/serve.sh`). Do not exempt the server scope from `systemd-oomd`: under pressure oomd then
  kills other units of the session instead (:460-464; the one-line warning is in ELASTIC_MEMORY.md).
* Idle server processes get swapped; `tools/keepalive.sh` keeps them warm (:368).
* `SGLANG_LOG_GC=1` is mentioned as "added to the accepted env" in CAMPAIGN.md:343 but is in neither
  `scripts/sweep.sh` nor `assets/phase1_state.json`; it is not part of the accepted environment.

## Benchmarks

Standard-benchmark runs (GPQA Diamond, DeepSWE 1.1) were started on 2026-09-03 and not completed; no
score is published (CAMPAIGN.md:422-431, :465-468). The internal quality protocol (NLL ladder, logprob
oracle, needle retrieval) is in [`../CONTRIBUTING.md`](../CONTRIBUTING.md) and
[`ELASTIC_MEMORY.md`](ELASTIC_MEMORY.md); its numbers are in the README. The drivers
`tools/final_bench.sh`, `tools/deepswe_run.sh` (with its task list `tools/deepswe/tasklist_seed0.txt`)
and `tools/fwd.sh` are kept for anyone who wants to run that comparison.
