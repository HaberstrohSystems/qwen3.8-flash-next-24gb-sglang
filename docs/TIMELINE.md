# Timeline

Dated milestones, taken from the append-only engineering log [`CAMPAIGN.md`](CAMPAIGN.md) (`CAMPAIGN.md:N`
is line N of that file; timestamps are as recorded there, local time, in chronological order). Speeds are
decode / prefill tok/s from `tools/bench_speed.py`; entries before 2026-09-02 12:58 used the older
two-request bench, later ones the streamed measurement (see the "BENCH FIX" entry). What was rejected,
and why, is collected in [`HISTORY.md`](HISTORY.md); the release changelog is
[`../CHANGELOG.md`](../CHANGELOG.md).

## Before the performance campaign

* **2026-08-26** — PLE n-gram table extracted: `ple.f8_e4m3.bin`, 51,200,245,760 bytes, 320,001,536
  rows x 160 fp8, 128 shards (`ple.json`; file dated 2026-08-26).
* **2026-08-27** — AutoRound 2-bit run, 3 days 8 hours: 48 blocks, four subset rounds over the 512
  experts each, `--enable_alg_ext`, iters 200, nsamples 48, seqlen 2048. Result 2.572 bpw, 38.60 GiB
  without the MTP head; the sealed AutoRound output (`WRITEUP.md` section 2).
* **Base SGLang patch** — 2-bit `moe_wna16` path, nine findings addressed (eight fixes, the whole-layer
  offload replaced by the `ExpertStreamer`), mmap PLE: decode 2.24 -> 15.5 tok/s, prefill ~500 -> 1,158
  tok/s, PCIe 26 GB -> 0.31 GB per token, context 13,824 -> 32,768 (`WRITEUP.md` results table).
* **2026-09-01** — the performance campaign starts (CAMPAIGN.md:3).

## 2026-09-02 — the campaign

* **08:29-08:31** — GEMV measurements on real layer-5 tensors: `[N, K/4]` row-major in-place GEMV 73 GB/s
  device / 5 GB/s pinned host (refuted); N-contiguous int32-word GEMV 320 / 51 GB/s (PCIe line rate).
  CAMPAIGN.md:196-204.
* **08:35** — S0 baseline: 13.4 / 1,189. CAMPAIGN.md:237.
* **09:00-09:08** — the greedy-diff oracle is found to crash on identical outputs and GPU bf16 to be
  non-reproducible run to run; replaced by a teacher-forced logprob oracle (450 tokens, noise MAX 0.09 /
  MEAN 0.002, thresholds MEAN <= 0.01, MAX <= 0.5). CAMPAIGN.md:242-247.
* **09:13** — S0b tuned `int2_w2a16` Triton configs KEPT: 15.2 / 1,190. CAMPAIGN.md:252.
* **09:22** — S4 `--max-mamba-cache-size 2` KEPT: 15.3 / 1,221. CAMPAIGN.md:256.
* **09:26** — S3 chunked prefill 1024 KEPT: 15.4 / 1,662. CAMPAIGN.md:258.
* **09:30** — S5 host fixes (hook, skipgather, memo, rope, ple) KEPT: 19.4 / 1,443. CAMPAIGN.md:260.
* **09:45-10:00** — byte-shuffled table GEMV rejected (157 / 14 GB/s); int32-word table GEMV end to end
  for one layer: 184 GB/s device, 45-46 GB/s pinned host. CAMPAIGN.md:218-228.
* **09:48-09:53** — routing probe (2,496 decode tokens x 48 layers): top 32/64/128/171/256 experts cover
  37/53/73/82/93 % of routing mass, Zipf exponent 0.51; adjacent-token recall@10 0.36; dynamic-k dead
  (90 % of gate mass needs 8.7 of 10 experts). CAMPAIGN.md:206-216.
* **10:03-11:14** — S9 (N-contiguous layout + GEMV) integration attempts. CAMPAIGN.md:63-91.
* **11:25** — per-layer MoE input/output A/B: the word layout is numerically equivalent (bf16 output
  rounding, ~0.1 % of elements one ulp); kernel-class threshold set to MEAN <= 0.05. CAMPAIGN.md:92-96.
* **11:30** — S9 N-contiguous layout + in-place GEMV KEPT: 21.8 / 2,249. CAMPAIGN.md:286.
* **11:48** — S6c breakable decode CUDA graphs (PLE eager break, `LogitsProcessorOutput` support) KEPT:
  40.0 / 2,249; capture 3.97 s / 0.12 GB. CAMPAIGN.md:291, :102.
* **12:10** — S10 frequency placement v3 (memory-neutral, S = 184) KEPT: 48.4 / 2,334. CAMPAIGN.md:296.
* **12:12** — INT8 dense re-pack build started; short held-out NLL reference on the exact state: de 1.512,
  en 1.278, py 0.414 nats/token. CAMPAIGN.md:133-137.
* **12:28** — S11 root cause: exact `extra_config` entries win over regexes in SGLang's AutoRound config
  resolution; `lm_head` group size. CAMPAIGN.md:298.
* **12:40** — S11b INT8 dense (85 tensors, RTN g128) KEPT: 48.8 / 2,319; 3.47 GB VRAM free after graph
  capture (was 1.88); NLL de -0.016, en +0.010, py +0.005, overall -0.001. CAMPAIGN.md:300.
* **12:49** — host memory attributed: Shmem 24-26 GB = pinned offloader tensors (layers 0-30 x 0.767 GB,
  1.33 MB per expert). CAMPAIGN.md:305.
* **12:58** — BENCH FIX: streamed inter-token measurement; accepted reference re-based to 56.0 / 2,362.
  CAMPAIGN.md:306.
* **13:17** — S13 elastic expert cache KEPT: 56.2 / 2,335 (after a slot-pool fix for the split layer 30 and
  an oracle re-base on the INT8 server). CAMPAIGN.md:310-319.
* **13:53** — S15 131,072-token context KEPT (56.9 / 2,340 after the startup transient). CAMPAIGN.md:336.
* **14:12** — `ple_random.py`: MADV_RANDOM / POSIX_FADV_RANDOM on the PLE table. CAMPAIGN.md:341.
* **14:20** — live S sweep: S = 184 / 200 / 216 -> decode 56-57.6 / 55.6-57.9 / 57-58.1, i.e. +2-3 % per
  ~1.7 GB. CAMPAIGN.md:320.
* **14:38-14:49** — 60k-token deaths root-caused: `MemoryMax=27G` reclaim storms (raised to 30G), then
  runtime pinned allocations by the expert cache (a 4 MB pin took 22 s); a 315 MB startup reserve killed
  the server at load. Final policy: no runtime pinning, floor S = 184, refuse KV at admission.
  CAMPAIGN.md:345-347.
* **14:45** — S14 lazy VMM KV backing KEPT (attempt 4): 55.4 / 2,323. CAMPAIGN.md:330.
* **14:54** — long context works on bf16 KV: 68,905 tokens, 48.9 s (1,408 tok/s), decode 53.3.
  CAMPAIGN.md:348.
* **14:58** — admission cap min(requested, SAFETY x profiled). CAMPAIGN.md:350.
* **15:01-15:12** — S16 fp8_e4m3 KV read path: reverted as a speed step (55.1 vs 56.9), then measured at
  parity as a mode (55.9-57.2 / 2,303-2,339); short tests shown to be blind to the KV cache -> `nll_long.py`.
  CAMPAIGN.md:351-353.
* **15:26** — open bug: 1-token prompt crashes in fp8 mode. CAMPAIGN.md:355.
* **15:49** — long-text bf16 reference (TF NLL 2.0388 over the last 512 of 9,586 tokens) and its noise
  floor (NLL -0.008, mean abs. dlogprob 0.099). CAMPAIGN.md:358-359.
* **15:50** — KV design panel: INT8-G64 chosen from measured K/V statistics (simulated relative RMS error
  e4m3 2.66 % vs int8 g64 0.9 %). CAMPAIGN.md:356; `KV_INT8_PLAN.md`.
* **16:08-16:26** — INT8-G64 implemented (`kv_int8.py`, `int8_kv_pool.py`, `test_kv_int8.py`); unit tests
  bit-exact; review fixes. CAMPAIGN.md:365, :369.
* **16:17** — fake-quant series (512 positions): int8_g64 0.099, int8_tok 0.100, int8_g32 0.106, e4m3
  0.110 mean abs. dlogprob vs noise 0.099. CAMPAIGN.md:368.
* **16:45** — S17 INT8-G64 KV VALIDATED and accepted: 12.4 KB/token; 115,560-token prompt 71.8 s
  (1,610 tok/s), decode 50.8. CAMPAIGN.md:370.
* **17:00** — S18 ceiling with INT8 KV: 150,560 tokens 92.8 s (1,623 tok/s); 162,215 tokens 95.8 s
  (1,694 tok/s), decode 51.2, VRAM free min 1.18 GB; ~179k refused. Context flags set to 262,144 (VA
  only) with safety 0.77. CAMPAIGN.md:372.
* **17:08** — speculation research: MTP infeasible (the MTP head is bf16, 4.85 GiB, `WRITEUP.md` section
  3; its experts alone 4.7 GiB, CAMPAIGN.md:379); NGRAM planned. `SPEC_NGRAM_PLAN.md`.
* **18:50** — NGRAM verdict: lossless within the decode-vs-prefill floor, accept length 1.11-1.27, 22-25
  tok/s with eager verify -> opt-in only. CAMPAIGN.md:387.
* **19:10-19:19** — needle test at 41,370 tokens: bf16 pool 3/5 (answer truncated at 120 new tokens),
  INT8 5/5, fake INT4 5/5. CAMPAIGN.md:401-404.
* **19:34** — 256k reached on INT4-G32: 257,905 tokens, 182.2 s (1,415 tok/s), decode 51.9.
  CAMPAIGN.md:407.
* **19:35** — ERRATUM: every fake-quant run after S17 had executed on the real INT8 pool; the
  all-position ladder measured to that point is invalid. Fake-quant steps now force the bf16 pool.
  CAMPAIGN.md:394.
* **19:40** — corrected ladder (8,561 positions, bf16 pool): noise +0.0002 / 0.059, fake int4_g32
  +0.0078 / 0.139, int2_g16 +0.30 / 0.62, K/V := 0 +3.70 / 3.92. S19 real INT4-G32: 6.8 KB/token,
  all-position +0.0088 / 0.138, prefill 1,493 at 10k, 162k prompt 112.6 s (1,441 tok/s). CAMPAIGN.md:406.
* **19:56-20:31** — the tiered KV cache planned and implemented: INT8 ring of W = 8,192 slots over the
  INT4 pool, dual write, owner table, device-side tier test. CAMPAIGN.md:410-411; `KV_TIERS_PLAN.md`.
* **20:00** — needle test at 247,629 tokens on INT4-G32: 5/5, prefill 269 s (920 tok/s on random text),
  decode 51.4, after a PLE page-cache fix (`POSIX_FADV_DONTNEED` after bulk gathers). CAMPAIGN.md:408-409.
* **21:05** — S21 tiered KV VALIDATED and ACCEPTED as the default (`--kv-cache-dtype int8ring_int4`):
  short NLL -0.0002; all-position -0.0001 / 0.074; oracle 0.0019; 54-57 / 2,271; 257,905-token prompt
  171 s (1,508 tok/s), decode 51.8; needles 5/5 at 41k and 248k. CAMPAIGN.md:413.
* **22:04** — fp16 scale clamp (65504) added to the INT8 kernel; tree re-layered kv_int8 -> kv_int4 ->
  kv_tiers. CAMPAIGN.md:415.
* **22:18-22:19** — long-prefill cost model at 258k: chunk_ms = 615 + 0.487 us x prefix; ~28 % of the
  slowdown is the O(prefix) gather, 64 % prompt-content base cost, 8 % periodic spikes; `empty_cache`
  rate-limited below the watermark. CAMPAIGN.md:417-418.
* **22:58** — rule: no server starts while a patch workflow may be editing the tree (two restarts died on
  that race). CAMPAIGN.md:420, :435-436.
* **23:30** — the server gains `--reasoning-parser qwen3 --tool-call-parser qwen3_coder` (thinking out
  of content, the chat template's tool-call format). CAMPAIGN.md:430-432.
* **~23:45** — restart #20 (accepted set plus the parsers); 258k re-measured with the rate-limited
  `empty_cache`: 165.3 s (1,560 tok/s), decode 52.3. `logs/night.log` (between its 23:40 and 23:52
  marks; CAMPAIGN.md:448 records an estimated 00:55).

## 2026-09-03

* **00:05** — S22 paged prefix-chunk kernel: built, correct within 2 row-ulps, REJECTED on timing (6.1 ms
  vs 1.4 ms per head-layer at prefix 60k; the gather it removes is ~2.5 % of chunk time). Tree stays at
  the S21 state. CAMPAIGN.md:439; `KV_PAGED_PREFIX_PLAN.md` "Outcome".
* **00:33** — restart #23: the accepted S21 flag set (CAMPAIGN.md:413) plus the parsers (:430-432), run
  with `--max-running-requests 4 --max-mamba-cache-size 8` and an oomd exemption for a concurrent
  benchmark; systemd-oomd ended the session (:454-464), the exemption was withdrawn, and the published
  configuration is single-request (`scripts/serve.sh`). `logs/night4.log`, `logs/elastic.ctl.status`.
* **00:40** — stage-B static per-channel smoothing for INT4 closed (mean NLL better by ~0.005, per-token
  deviation worse: +0.0022 / 0.160). Not implemented in the real kernels. CAMPAIGN.md:444.
* **2026-09-03** — the serving patch `sglang/qwen4exp-serving-73a255206f.patch` (34 files,
  +4,155 / -89) produced from the served tree and verified byte for byte against a clean worktree of
  `73a255206f` (`sglang/PATCH_NOTES.md` section 2).
* **2026-09-03** — published serving configuration fixed at `--max-running-requests 1
  --max-mamba-cache-size 1` (`scripts/serve.sh`).
* **2026-09-03 — release.** Text-only checkpoint (222,856 tensors, 38,755,352,600 B; vision tower and
  MTP head removed) and the 51,200,245,760-B PLE table published on the Hub as
  `HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang`
  (https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang);
  this repository published. CAMPAIGN.md:469-487.
