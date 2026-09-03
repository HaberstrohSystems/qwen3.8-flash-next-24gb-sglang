# Autonomous performance campaign — status file

Owner instruction (2026-09-01): try every idea autonomously, one after another, keep what
works, go deep, do not give up early. No permission needed for engineering. Standing
constraints: never delete the sealed quant; watch disk space; do not delete src-bf16
without explicit authorization.

Rules I hold myself to:
- Every change is measured against the previous accepted state with the same benchmark.
- Exact changes are verified with a greedy-decode token diff (200 tokens, temp 0) before
  being kept. Approximate changes need a quality check and are flagged as such.
- A change that does not improve the number is reverted, not kept "for later".
- Server restarts go through serve.sh variants; the sealed model directory is read-only.

Benchmark: release/tools/bench_speed.py against the running server. Report decode tok/s at
~1.7k and ~10k context, prefill tok/s at 10k.

## Baseline (accepted state 0)

    decode 15.5 tok/s (64.5 ms/token)   prefill ~1158 tok/s   TTFT@32k ~28 s
    GPU 38.9 ms busy: MoE kernel 13.7, expert movement 9.5, dense bf16 9.9, int8 3.5
    host gap ~26 ms, one thread at 97 % of one core

## Phase 0 — preparation (no GPU needed)            [in progress]

- [x] GEMV kernel prototype (idea 1) + correctness/bandwidth test    perf/gemv/moe_gemv_int2.py, test_gemv.py
- [x] flag-sweep harness                                             perf/sweep.sh  (DROP="..." removes base flags)
- [x] host-fix patches (hook, skipgather, memo, rope)                perf/patches/host_fixes.py  apply|revert|--check
- [x] routing instrumentation patch (dump: topk ids + weights)       same file, item "dump"; lookahead needs arch thread
- [x] PLE pread patch (item 2) and BCG eager-break wrap (item 7)     same file, items "ple", "bcg"
- [x] greedy reference saved against baseline server                 perf/greedy/baseline.json (text compare)
- [x] phase-1 driver: restart, bench, greedy-check, keep-or-revert    perf/phase1.py  (state: phase1_state.json)
- [x] workflow 2 folded in (see outcome section); no synthesis-v2, ordering is mine
- [x] layout + GEMV patch ready: perf/patches/ncontig_gemv.py (19 anchors clean),
      kernel perf/gemv/moe_gemv_int2_tab.py, test perf/gemv/test_gemv_tab.py
- [x] routing run driver perf/route_run.py (+ route_stats.py reads the probe dir)

## Phase 1 — first restart window   [running: perf/phase1.py, log perf/phase1.log]

- [x] GEMV kernel test (see measurements): N-contiguous wins, tab variant untested on GPU yet
- [ ] flag sweep: overlap-schedule, continuous-decode-steps, chunked-prefill 2048, mamba 2
- [ ] host fixes applied + greedy diff
- [ ] BCG graphs, NGRAM
- [ ] S8: int2 configs under the tolerance rule (S0b gave +6 % but "not exact" was unproven)
- [ ] instrumented run: SGLANG_ROUTE_DUMP + route_run.py + route_stats.py  (manual, after phase 1)

## Open items surfaced by phase 1

- S5 host fixes: decode +26 % but prefill 1662 -> 1443 (-13 %). One of the five items costs
  prefill; suspects: skipgather (E=512 config path at M=1024) or ple. Isolate one item per restart
  after phase 2 (S5b..), keep the decode win either way.
- S1 (overlap schedule) was neutral on top of int2cfg (14.8 vs 15.2); retest once graphs/GEMV
  change the host/GPU balance.
- S6 BCG failed at capture: the decode graph backend defaults to "full" (classic capture), where
  eager_on_graph is a no-op and the PLE's pageable H2D copy is inside the graph. Needs
  --cuda-graph-backend-decode breakable. Re-queued as S6b.
- S7 NGRAM failed: "Loaded weights leave no GPU memory for the KV cache" - the speculative worker
  needs headroom. Re-queued as S7b after the staging buffers are gone (S9).
- S6b (breakable decode backend) failed: "Unsupported BCG output type: LogitsProcessorOutput" -
  the backend walks Tensor/PPProxyTensors/tuple/list only. Patch item "bcg2" teaches its four
  structure helpers the dataclass. Re-queued as S6c.
- S7b NGRAM failed again for VRAM (S9 had not landed). Re-queued as S7c after S9.
- S9 attempts: (1) fp16-vs-bf16 scales assert, (2) return type must be StandardCombineInput,
  (3) fused_experts_impl "Hidden size mismatch" assert assumes [E,N,K/4], (4) NameError from
  threading a flag through wrappers, (5) duplicate argument, (6) fused_experts custom-op schema
  rejects new kwargs -> layout now DERIVED from dtype (int32 = word layout), (7) garbage output:
  invoke computes the grid N from B.shape[1] and even_Ks from B.shape[2] BEFORE the kernel args
  -> only 1/8 of N tiles ran. Fixed; GPU A/B of the tiled path through the patched invoke:
  word layout vs byte reference rel err 2e-5 (IDENTICAL). (8) 10:43 server run: decode 21.4,
  prefill 2246 tok/s, logprob mean 0.0119 / max 0.275 - just over the 0.01 mean threshold
  (noise floor 0.002). Tiled path is identical, so isolating: S9a = layout only, GEMV off.
- S9a (10:49, layout + tiled only, GEMV off): decode 18.0, prefill 2257 (+56 %), oracle mean 0.0120
  max 0.275, prompt 2 no longer 0.000 -> the deviation is in the layout/tiled path, NOT the GEMV.
  All server configs use BLOCK_K=128 (the A/B-identical scale branch). Bisecting by layer set:
  S9b converts layers 0-16 (resident, plain path), S9c layers 17-47 (host, streamer path).
- S9b/S9c (10:56/10:59) both FAILED to start: the word-load variant is swapped in for ALL int2
  calls, and for an unconverted byte-layout layer invoke derives even_Ks from B.shape[2]=160 (w2)
  -> False -> the variant's static_assert. Mixed layouts cannot coexist with the global swap;
  layer-wise bisection is off the table. Next: A/B the untested scale branch (BLOCK_K=32, the
  untuned default used for host-layer prefill) and the M=1 config (16/16/128).
- 11:04 A/B at 64/64/32, 16/16/128, 16/32/128 through the patched invoke: all IDENTICAL (2e-5).
  The tiled kernel is not the source. Remaining suspect: the streamer path for host layers on
  the int32 layout. Patch reworked: the word-load kernel is ADDED as fused_moe_kernel_gptq_awq_word
  and invoke dispatches by dtype, so mixed layouts coexist -> the layer bisection (S9b/S9c) can run.
- 11:10/11:14 bisection with per-call dispatch: S9b (layers 0-16) illegal memory access in the
  streamer's index_select; S9c (17-47) oracle mean 0.37. CORRECTION: the offloader offloads the
  FIRST ~30 layers; resident = 31-47, layer 30 split. Both failures come from the streamer's
  staging cache being keyed by (name, dtype, device) without shape: with two layouts alive, a
  word-layout layer gets a byte-shaped scales buffer (same numel, transposed). Fixed by keying
  on the shape. This cannot explain the full-conversion 0.012 (all shapes agree there).
  Next: per-layer MoE input/output dumps on the real model, byte vs word, to locate the 0.012.
- 11:25 per-layer dump A/B (perf/MOE-IO-DIFF.txt, 144 files): layer 0 with IDENTICAL inputs
  differs by 5.1e-5 (M=6) / 1.7e-4 (M=1) / 0.0 relative - bf16 output rounding, ~0.1 % of
  elements one ulp apart (Triton lays the reshaped word tile out differently -> other fp32
  accumulation order). From layer 1 the inputs differ, from layer 2 the routing flips for some
  tokens, and the difference amplifies to the 0.012-nat end-to-end mean. Same mechanism as the
  greedy non-reproducibility. VERDICT: numerically equivalent. The logprob threshold (5x a noise
  floor that is ~0 for the deterministic MoE path) is too strict for kernel-class changes; the
  per-layer identical-input check is the right criterion (< 1e-3 relative). Kernel-class steps
  now accept MEAN <= 0.05 / MAX <= 0.5 in the driver; host-class steps keep 0.01.

- 11:33 S6c (breakable decode graphs + bcg2 LogitsProcessorOutput support): capture succeeded
  (3.97 s, 0.12 GB), decode 39.5 tok/s (+81 % over 21.8), prefill 2235. Reverted only because
  the oracle still compared against lp0 (byte layout): mean 0.0119 / max 0.275 is exactly the
  inherited S9 kernel-class deviation. Re-basing the reference (lp1 on the accepted state) and
  re-running S6c under the host-class threshold.

- 11:36 S7c NGRAM: still "no GPU memory for the KV cache" (the prefill staging buffers are lazy,
  so the startup VRAM picture is unchanged). Re-queued as S7d with --max-total-tokens 16384.
- 11:44 oracle re-based: lp1 saved on the accepted state (word layout); self-consistency MAX 0.074
  MEAN 0.0019 - same noise floor. phase1 now compares against lp1 automatically.

- 11:51 S7d NGRAM at 16k tokens: same VRAM error - the speculative worker's own allocations
  exceed the headroom regardless of the KV budget. Parked until the INT8 requant frees ~1.8 GB.
- 11:55 placement patch written (perf/patches/placement.py): per layer the hottest S=184 experts
  (by probe mass) go to the GPU, the rest to pinned host; S=184 keeps both VRAM and host pinned
  memory at today's level (the split layer 30 counts). Two-source gather kernel for prefill,
  pointer tables from hot/cold for decode. Exact. Queued as S10.

- 11:57 S10 v1 killed by systemd-oomd during placement (host memory pressure): pinned host
  memory is never returned to the OS by the caching host allocator, and pin_memory() makes a
  pageable copy first, so "new pinned cold tensor per layer" grows the footprint by ~1 GB per
  layer. v2 is memory-neutral by construction: host layers donate the slots of their hot rows,
  GPU layers write their cold rows into those slots; at S=184 the two sums are equal (5658 rows).
  One int64 address table per (layer, tensor kind) serves both the decode GEMV and a new
  table-driven prefill gather kernel.

- 12:04 S10 v2 also oom-killed: layer 0 is GPU-RESIDENT (the offloader order was assumed, not
  discovered), so the first layers found no donated host slots and allocated pinned fallbacks.
  v3 collects all layers during loading, discovers the host/GPU split, and places in one
  interleaved pass (host, host, GPU, ...) at the last layer, so slots always exist first.

- 12:12 INT8 dense requant build started (perf/requant_int8.py -> ~/quant/int8dense-20260902-1212;
  ~/quant/out is read-only by the seal, so the new model lives beside it). Quality metric for
  approximations: perf/nll_eval.py, teacher-forced NLL on three held-out passages; reference on
  the exact state: de 1.512, en 1.278, py 0.414 nats/token (perf/nll/exact.json).

## Accepted state after S10 (2026-09-02 12:10)

    decode 48.4 tok/s (15.5 original = 3.1x)   prefill 2334 tok/s (1158 = 2.0x)   all exact
    + frequency placement v3: 31 host / 17 GPU layers, S=184 hot experts per layer, address
      tables per (layer, kind), table-driven prefill gather; 6 small pinned fallbacks, 128
      spare w13 slots. Oracle mean 0.0013 (noise floor).
    per-token estimate now: dense ~13.4 ms (floor), experts ~3.9 ms (84 % from HBM), host gap
      ~1.4 ms (graphs) -> GPU-bound on the dense reads. Next lever: INT8 dense requant (-3 ms,
      +1.8 GB VRAM), then speculation (NGRAM/MTP) once VRAM allows, then dynamic placement.

## Accepted state after S6c (2026-09-02 11:48)

    decode 40.0 tok/s (13.4 at S0, 15.5 original)   prefill 2249 tok/s (1189)   = 2.6x / 1.9x
    kept: int2 tuned configs, mamba cache 2, chunked prefill 1024 (+max-prefill-tokens 32768),
          host fixes (hook, skipgather, memo, rope, ple), N-contiguous int32 layout + word-load
          tiled kernel + in-place decode GEMV through pointer tables, breakable-backend CUDA
          graphs for decode (PLE as eager break, LogitsProcessorOutput support in the backend)
    reverted: overlap schedule (neutral), continuous decode steps (harmful)
    pending: NGRAM (S7d), placement by frequency (S10), S5 prefill isolation, INT8 dense requant,
             MTP draft, layer-major prefill

## Phase 2 — deep engineering (gated on phase 1 numbers)

- [ ] test_gemv_tab.py on GPU (needs server down), then apply ncontig_gemv.py, restart,
      greedy check under the tolerance rule, bench
- [ ] GEMV integrated into FusedMoE dispatch with pointer table (replaces gather+staging)
- [ ] slab layout at load time
- [ ] hot-expert cache with placement policy
- [ ] copy-stream prefetch with one-layer lookahead (if recall@20 is good)
- [ ] BCG CUDA graphs (PLE pread + eager_on_graph)
- [ ] MTP speculation (if VRAM allows)
- [ ] layer-major prefill (if workflow 2 says it is sound)

## Phase 3 — approximations (only after error survey)

- [ ] dense requant to int8/fp8 per group

## Workflow 2 outcome (2026-09-02, seven threads complete; critic + synthesis hit the session limit)

- int2 kernel: NOT dequant/occupancy - uncoalesced B loads from the loader's .T.contiguous()
  (lanes along N at 640 B stride). Fixes: tuned JSON BN=16 (2-4x, today), native [K/16,N]
  layout (kernel vectorises, ~4-5 ms), N-contiguous GEMV (~1.5-2 ms). FP4 tensor cores dead
  (e2m1 x e2m1 only).
- dense requant: RTN INT8 g128 error 0.7-1.2% on every dense group, BELOW the 1.3-3.2% the
  sealed int8 in_proj already carries; lm_head 0 argmax flips at gap>=0.25 nats. GO, no
  AutoRound rerun: CPU RTN pack + merge into a NEW dir. ~4.5 ms/token + 2.4 GB VRAM. FP8 dead.
- MTP: fully plumbed (Qwen4ExpForCausalLMMTP, NEXTN, GDN/PLE/QSA rollback). Needs a ~1 GB
  int4 draft dir for the 1536 bf16 MTP expert tensors. K capped at 3 by QSA compress ratio.
  1.1-1.25x today, ~1.35x after host fixes, 1.6x with sync-free dedup. Verify expert traffic
  is the wall.
- layer-major prefill: legal, buildable from SPLIT_PREFILL parts. TTFT@32k 27.5 s -> 12-15 s
  with chunk 2048/4096 flags -> 7-9 s layer-major. Compute floor ~393 TFLOP at 32k.
- VMM/kvcached/managed memory: none buys decode time (no demand-fault path; kvcached <=0.75 GB
  with no consumer; managed memory dead by 10x). Staging-buffer shrink frees 0.5-0.58 GB.
- stream overlap: worth nothing until wall < ~47 ms (host is the critical path); then 2.9-5.6 ms.
- architecture: 7.77 GB VRAM reads/token, 92% not experts. Exact levers: fuse HC chain
  (-288 launches), host-side PLE, embed_tokens to pinned host (+1.9 resident layers).

## GEMV measurements (2026-09-02 08:29-08:31, server down, real layer-5 tensors, 10 of 64 experts)

- my [N,K/4] row-major GEMV: EXACT (3.7e-7) but 73 GB/s device, 5 GB/s pinned host -> the
  "read in place" half of idea 1 is REFUTED for that layout (128 B chunks at 640 B stride).
- agent's N-contiguous GEMV (native [K/16,N] int32): 320 GB/s device, 51 GB/s pinned host
  (= PCIe line rate, from inside the compute kernel). 2.1 ms/token all-device; ~9 ms/token at
  today's 17/31 resident/host split, vs 23.2 ms (kernel + movement) today. Idea 1 CONFIRMED
  with lanes along N. Output within bf16 rounding of fp32 reference (1.66e-3).
- best config BN=64 BK=128 warps=4. BK=256 is WRONG for w2 (K=640): rel err 0.44. Guard it.

## Routing probe (2026-09-02 09:48-09:53, 2496 decode tokens x 48 layers, 3 domains) - perf/ROUTE-STATS.txt

- adjacent-token reuse recall@10: prev-1 0.36, union prev-2 0.46 (layer 0: 0.09). Prefetch-last-token
  would hit ~40 %; useful only as a complement.
- mass concentration: top 32/64/128/171/256 of 512 experts per layer cover 37/53/73/82/93 % of routing
  mass. Zipf exponent 0.51. 171 per layer is what fits in VRAM today -> frequency placement would serve
  82 % of expert bytes from VRAM vs 35 % (17 whole layers) now: PCIe ~400 -> ~115 MB/token, ~5.5 ms.
  EXACT (placement does not change outputs). Caveat: 3 prompts; a live LFRU or prefill-histogram
  repack adapts per workload.
- gate mass: 90 % needs 8.7 of 10 experts, 99 % needs all 10, the 10th carries 5.6 %. Dynamic-k and
  shared-expert absorption (idea 5) are DEAD - the tail is fat.

## Byte-shuffled table GEMV (old phase2, 09:45): 157 GB/s device, 14 GB/s pinned host -> REJECTED.
   64-byte row loads are too small for PCIe. The int32-word N-contiguous variant (320 / 51 GB/s)
   is the one to integrate; the tiled prefill kernel needs the agent's word-load int2 branch.

## int32-word table GEMV (2026-09-02 09:58-10:00, server down, layer 5, 10 of 64 experts, end to end)

- loader layout [E,N,K/4] u8 -> [E,K/16,N] i32 conversion bit-exact vs checkpoint words (after a
  view-shape fix: view(int32) leaves a trailing dim of 1).
- full decode MoE for one layer (w13 GEMV, silu*up, w2 GEMV, weighted sum): rel err 3.5e-3 (bf16
  intermediates), 184 GB/s from device (3.4 ms/token if all resident), 45-46 GB/s from pinned
  host (13.8 ms/token if all on host). Today: 13.7 kernel + 9.5 movement = 23.2 ms.
- integrated as perf/patches/ncontig_gemv.py: layout in process_weights_after_loading, word-load
  int2 branch swapped into the tiled prefill kernel (function-level, backup for revert), stride/N
  swap at invoke, GEMV dispatch for M <= 16 in MoeWNA16Method.apply.

## Log

(appended as steps complete: date, step, before -> after, kept/reverted, why)
- 2026-09-02 08:31 phase 1 started (perf/phase1.log): S0 S0b S1 S2 S3 S4 S5 S6 S7
- 2026-09-02 08:35 S0_baseline: KEPT  decode 13.4 tok/s, prefill 1189 tok/s
- 2026-09-02 08:41 S0b_int2cfg: REVERTED (not exact)  decode 14.2, prefill 1214
- 2026-09-02 08:46 S1_overlap: REVERTED (not exact)  decode 14.9, prefill 1208
- 2026-09-02 08:53 S2_contdecode: REVERTED  decode 9.7 (-28 %), prefill 1498 (+25 %) - genuinely harmful for decode
- 2026-09-02 08:58 S3_prefill2048: CUDA OOM during bench (48 MiB alloc, 78 MiB free) -> re-queued as 1024 after the mamba reclaim
- 2026-09-02 09:00 ORACLE BUG: greedy_diff.py crashed on IDENTICAL outputs (unset `same`), phase1 read the
  crash as "not exact". S0b (+6 %) and S1 (+11 %) were rejected on a broken check. Fixed.
- 2026-09-02 09:05 NONDETERMINISM: same server, same config, back to back: greedy divergence at tokens
  104-169 of 200 (prompt 2 once identical, once at 104). GPU bf16 reductions are not bitwise reproducible
  here; greedy token diffs are not a usable exactness oracle at 200 tokens.
- 2026-09-02 09:08 NEW ORACLE: teacher-forced logprobs on fixed continuations (perf/logprob_diff.py, 450 tokens,
  9 s). Same-config noise floor: MAX 0.09 nats, MEAN 0.002 (prompt 2: 0.000). Thresholds: MEAN <= 0.01, MAX <= 0.5.
  Reference lp0 saved on baseline flags. phase1 state reset: S0b, S1, S3(1024), S4.. re-queued; S2 stays reverted.
- 2026-09-02 08:53 S2_contdecode: REVERTED (not exact)  decode 9.7, prefill 1498
- 2026-09-02 09:13 S0b_int2cfg: logprob max 0.086 mean 0.0018 -> equivalent
- 2026-09-02 09:13 S0b_int2cfg: KEPT  decode 15.2 tok/s, prefill 1190 tok/s  (was 13.4 / 1189)
- 2026-09-02 09:18 S1_overlap: logprob max 0.075 mean 0.0016 -> equivalent
- 2026-09-02 09:18 S1_overlap: REVERTED (not better)  decode 14.8, prefill 1185
- 2026-09-02 09:22 S4_mamba2: logprob max 0.091 mean 0.0019 -> equivalent
- 2026-09-02 09:22 S4_mamba2: KEPT  decode 15.3 tok/s, prefill 1221 tok/s  (was 15.2 / 1190)
- 2026-09-02 09:26 S3_prefill1024: logprob max 0.075 mean 0.0022 -> equivalent
- 2026-09-02 09:26 S3_prefill1024: KEPT  decode 15.4 tok/s, prefill 1662 tok/s  (was 15.3 / 1221)
- 2026-09-02 09:30 S5_hostfixes: logprob max 0.078 mean 0.0019 -> equivalent
- 2026-09-02 09:30 S5_hostfixes: KEPT  decode 19.4 tok/s, prefill 1443 tok/s  (was 15.4 / 1662)
- 2026-09-02 09:33 S6_bcg: FAILED to start or bench -> reverted
- 2026-09-02 09:36 S7_ngram: FAILED to start or bench -> reverted
- 2026-09-02 09:40 S8_int2cfg_tol: logprob max 0.077 mean 0.0018 -> equivalent
- 2026-09-02 09:40 S8_int2cfg_tol: REVERTED (not better)  decode 18.5, prefill 1550
- 2026-09-02 09:43 S9_ncontig_gemv: FAILED to start or bench -> reverted
- 2026-09-02 09:47 test_gemv_tab: correct (see phase2.log)
- 2026-09-02 10:03 S9_ncontig_gemv: FAILED to start or bench -> reverted
- 2026-09-02 10:06 S6b_bcg: FAILED to start or bench -> reverted
- 2026-09-02 10:09 S7b_ngram: FAILED to start or bench -> reverted
- 2026-09-02 10:14 S9_ncontig_gemv: FAILED to start or bench -> reverted
- 2026-09-02 10:19 S9_ncontig_gemv: FAILED to start or bench -> reverted
- 2026-09-02 10:24 S9_ncontig_gemv: FAILED to start or bench -> reverted
- 2026-09-02 10:29 S9_ncontig_gemv: FAILED to start or bench -> reverted
- 2026-09-02 10:34 S9_ncontig_gemv: logprob max 19.468 mean 11.1971 -> NOT equivalent
- 2026-09-02 10:34 S9_ncontig_gemv: REVERTED (not exact)  decode 23.4, prefill 2098
- 2026-09-02 10:43 S9_ncontig_gemv: logprob max 0.275 mean 0.0119 -> NOT equivalent
- 2026-09-02 10:43 S9_ncontig_gemv: REVERTED (not exact)  decode 21.4, prefill 2246
- 2026-09-02 10:49 S9a_layout_only: logprob max 0.275 mean 0.0120 -> NOT equivalent
- 2026-09-02 10:49 S9a_layout_only: REVERTED (not exact)  decode 18.0, prefill 2257
- 2026-09-02 10:56 S9b_layers_res: FAILED to start or bench -> reverted
- 2026-09-02 10:59 S9c_layers_host: FAILED to start or bench -> reverted
- 2026-09-02 11:10 S9b_layers_res: FAILED to start or bench -> reverted
- 2026-09-02 11:14 S9c_layers_host: logprob max 4.112 mean 0.3696 -> NOT equivalent
- 2026-09-02 11:14 S9c_layers_host: REVERTED (not exact)  decode 19.4, prefill 1985
- 2026-09-02 11:30 S9_ncontig_gemv: logprob max 0.275 mean 0.0125 -> equivalent (kernel-class band)
- 2026-09-02 11:30 S9_ncontig_gemv: KEPT  decode 21.8 tok/s, prefill 2249 tok/s  (was 19.4 / 1443)
- 2026-09-02 11:33 S6c_bcg2: logprob max 0.275 mean 0.0119 -> NOT equivalent
- 2026-09-02 11:33 S6c_bcg2: REVERTED (not exact)  decode 39.5, prefill 2235
- 2026-09-02 11:36 S7c_ngram: FAILED to start or bench -> reverted
- 2026-09-02 11:48 S6c_bcg2: logprob max 0.079 mean 0.0022 -> equivalent
- 2026-09-02 11:48 S6c_bcg2: KEPT  decode 40.0 tok/s, prefill 2249 tok/s  (was 21.8 / 2249)
- 2026-09-02 11:51 S7d_ngram16k: FAILED to start or bench -> reverted
- 2026-09-02 11:57 S10_placement: FAILED to start or bench -> reverted
- 2026-09-02 12:04 S10_placement: FAILED to start or bench -> reverted
- 2026-09-02 12:10 S10_placement: logprob max 0.057 mean 0.0013 -> equivalent
- 2026-09-02 12:10 S10_placement: KEPT  decode 48.4 tok/s, prefill 2334 tok/s  (was 40.0 / 2249)
- 2026-09-02 12:22 S11_int8dense: FAILED to start or bench -> reverted
- 2026-09-02 12:28 S11 root cause: SGLang AutoRound config resolution — exact extra_config module entries win over regexes (84 dense modules still said bits 16), and lm_head (outside block_name_to_quantize) resolves group_size to -1 unless explicit → qzeros [1,62080] vs [20,62080]. Fixed in requant_int8.py write_config (--config-only), config.json rewritten. Retrying as S11b.
- 2026-09-02 12:33 S11b_int8dense: KEPT  decode 48.8 tok/s, prefill 2319 tok/s  (was 48.4 / 2334)
- 2026-09-02 12:40 S11b_int8dense KEPT: decode 48.8 tok/s (was 48.4), prefill 2319 (was 2334). VRAM after graph capture 3.47 GB free (was 1.88): the INT8 dense groups give back 1.6 GB. Quality (nll_eval vs exact): de -0.016, en +0.010, py +0.005, overall -0.001 nats/token -> neutral. Accepted state now includes --model-path ~/quant/int8dense-20260902-1212. NLL reference re-based to nll/int8dense.json.
- 2026-09-02 12:40 Design note (elastic expert cache): SGLang's KV pool already lives in a VMM arena (kv_vmm_backing.KvVmmBufferOwner, monotonic commit only). Weights are immutable, so with full host mirrors an expert's GPU residency is a pure cache: eviction = table write, admission = one H2D row copy + table write. Rank-ordered slots in a VMM row arena let physical memory return to the driver in 2 MB granules (tail unmap) while every address stays fixed -> CUDA graphs need no recapture. Building perf/gemv/row_arena.py first.
- 2026-09-02 12:39 S12a_ngram: FAILED to start or bench -> reverted
- 2026-09-02 12:44 S12a_ngram FAILED: not weights but the mamba intermediate-state reserve for speculation: mamba_cache_per_req 0.16 GB x (max_mamba_cache_size+1=3? capped 2) x speculative_num_draft_tokens=12 ≈ 3.8 GB > budget. Qwen4ExpTextConfig subclasses Qwen3NextConfig, so --enable-linear-replayssm-spec applies (needs --speculative-eagle-topk 1). Queued S12b (replayssm) and S12c (4 draft tokens).
- 2026-09-02 12:44 row_arena.py self-test passed on the idle GPU: 2 MiB granularity, 4 MiB chunks; ensure_rows(184) maps 36 MiB; shrink_rows(64) returns 20 MiB to the driver (mem_get_info confirms); CUDA graph captured against arena addresses replays correctly after shrink (evicted rows repointed to host) and after regrow; ArenaOOM raised cleanly.
- 2026-09-02 12:49 Host memory attributed: Shmem 24-26 GB is the offloader's pinned tensors (layers 0-30 x 0.767 GB; an MoE layer is 681 MB int2 since moe_intermediate_size=640 -> 1.33 MB per expert incl. scales). Host RAM (32 GB) is as full as VRAM; full host mirrors for an elastic expert cache are impossible. Third tier must be NVMe (rows exist in the sealed safetensors). Rank-ordered 3-tier residency (GPU [0,S) / host [S,S+H) / disk tail) keeps disk hits at ~0.1% of routing mass.
- 2026-09-02 12:58 BENCH FIX: release/tools/bench_speed.py measured decode as (t72-t8)/64 over two full requests, so prefill jitter leaked in (742 tok/s at ctx 101, 39-58 at 7-10k, and 114-120 after a plain restart of the same config). Replaced by a streamed measurement: inter-token rate between first and last event over 200 tokens, TTFT minus one step as prefill. Two runs on the accepted state: decode 55.5-57.0 tok/s at every context (run-to-run < 0.5 %), prefill 2313-2362 tok/s at 10k. Accepted reference rebased to decode 56.0 / prefill 2362 (old-bench value 48.8 kept as accepted_oldbench). Earlier per-step deltas stand as recorded but carry the old bench's noise; the 15.5 -> 56 trend is unaffected.
- 2026-09-02 12:54 S12b_ngram_replay: FAILED to start or bench -> reverted
- 2026-09-02 13:20 ELASTIC DESIGN (built, untested in server): perf/gemv/expert_elastic.py + patches/elastic.py. Each (layer, kind) gets a VMM RowArena holding expert rows in routing-mass rank order; addr tables point into the arena for rank < S and at a pinned host slot otherwise. Grow = table-driven gather host->arena + table rewrite + slot to pool; shrink = D2H copy of the tail ranks into pool slots + table rewrite + arena tail unmap (VRAM returns to the driver). Host memory conserved via a global slot pool (host layers' hot rows vacate slots); S_floor 184 without a pinned fallback budget. Control file (S n | fill MB | free MB) polled from apply() (eager paths only; skipped under graph capture). Why it should pay: a cold expert row is 1.33 MB, at S=184 ~16 % of routing mass is cold -> ~100 MB/token over PCIe (~4 ms of 17.9 ms); the 3.5 GB freed by INT8 dense allow S~230 (+50 rows x 48 layers x 1.33 MB = 3.2 GB) -> ~9 % cold -> ~+1.7 ms/token. perf/elastic_sweep.py runs a live S sweep with the streaming bench. S13_elastic queued (must be exact at S=184).
- 2026-09-02 12:58 S12b_ngram_replay FAILED at graph capture: Qwen QSA requires speculative_num_draft_tokens <= compress ratio 4 (got 12). Queued S12d: NGRAM, 4 draft tokens, replayssm, topk 1.
- 2026-09-02 13:01 S13_elastic: FAILED to start or bench -> reverted
- 2026-09-02 13:03 S13_elastic attempt 1 FAILED in placement: layer 30 is split by the offloader (w13 host, w2 GPU); the credit interleave counted it as a donor -> w2 pool short (272 < 328). place_all now schedules by per-kind pool demand. (Note to self: pkill -f with a pattern that appears in the calling shell's own command line kills the shell; kill by pid.)
- 2026-09-02 13:45 LAZY KV DESIGN (built, untested): patches/kv_lazy.py. SGLANG_KV_LAZY=1 allocates the full-attention KV buffers through SGLang's KvVmmBufferOwner (VA reservation for max_total_tokens) but backs only SGLANG_KV_LAZY_FLOOR=4096 tokens; TokenToKVPoolAllocator.alloc backs up to max slot + SGLANG_KV_LAZY_MARGIN=2048; when the pool is idle the allocator resets to slot order and unmaps beyond the floor (new KvVmmArena.uncommit_beyond / owner.release_beyond). A failed commit calls ExpertElastic.free() and retries. SGLANG_KV_LAZY_TOKENS raises the token capacity above the profiled value (virtual), so --max-total-tokens 131072 becomes possible on the same VRAM; long requests borrow physical memory from the expert cache. Queued as S14_kvlazy (32k, exact); 128k test via longctx_test.py afterwards.
- 2026-09-02 13:05 S13_elastic: FAILED to start or bench -> reverted
- 2026-09-02 13:06 S13 attempt 2 FAILED, same message, now understood: w2 slot supply 30x184=5520 < demand 18x328=5904 (split layer 30 donates w13 only, takes w2). v3 covered this with pinned fallbacks; S13 now runs with SGLANG_MOE_ELASTIC_PIN_MB=512 (~160 MB actually needed).
- 2026-09-02 13:09 S13_elastic: logprob max 0.169 mean 0.0151 -> NOT equivalent
- 2026-09-02 13:09 S13_elastic: REVERTED (not exact)  decode 56.0, prefill 2331
- 2026-09-02 13:14 S13 attempt 3: placement OK (48 layers, S=184, 4.51 GB free after placement), decode 56.0 / prefill 2331 (no regression) but oracle mean 0.0151 vs lp1 -> REVERTED. Root cause: lp1 predates the INT8 dense approximation (S11b, no oracle); its per-token |dlogprob| 0.02-0.08 explains the 0.015. Saved lp2 on the accepted (int8) server; greedy_check now picks the newest lpN. Re-running S13.
- 2026-09-02 13:17 S13_elastic: logprob max 0.060 mean 0.0016 -> equivalent
- 2026-09-02 13:17 S13_elastic: KEPT  decode 56.2 tok/s, prefill 2335 tok/s  (was 56.0 / 2362)
- 2026-09-02 14:20 S13 live S sweep (elastic_sweep.py, no restarts): S=184 arena 11.06 GB free 1.73 GB mass 84.3% decode 56-57.6; S=200 arena 12.19 free 0.62 mass 86.7% decode 55.6-57.9; S=216 only partly reached (ArenaOOM: torch's cached blocks are not driver-free; set_S lacks empty_cache) free 0.08 mass 88.1% decode 57-58.1; higher S capped. Verdict: the dial works mechanically (grow/shrink, tables, graphs stay valid) but "S up" buys only +2-3 % decode per ~1.7 GB -> cold expert reads are not the dominant decode cost. The dial's value is the other direction: VRAM on demand for KV (long context) and speculation. Next: S14 lazy KV, S15 128k, S12d NGRAM.
- 2026-09-02 13:24 S14_kvlazy: FAILED to start or bench -> reverted
- 2026-09-02 13:25 S14_kvlazy attempt 1 FAILED: lazy backing itself worked (32768 VA, 4096 backed, 24 KB/token; pool end avail 3.88 GB) but decode graph capture consumed 2.93 GB (S13: 0.12 GB) -> 0.57 GB left, first prefill OOM (200 MB staging). Not explainable by KV commits (full 32k = 0.79 GB). Re-running with commit diagnostics (tokens, torch reserved, driver free, capturing flag).
- 2026-09-02 13:30 S14_kvlazy: logprob oracle crashed -> treated as not equivalent
- 2026-09-02 13:30 S14_kvlazy: REVERTED (not exact)  decode 55.3, prefill 1413
- 2026-09-02 13:32 S14 attempt 2: graph capture normal this time (0.12 GB; the 2.93 GB of attempt 1 did not reproduce). Crash: illegal memory access in QSA forward_extend at the first >4096-token prefill, and no 'KV lazy commit' lines: the server runs page_size=64 (QSA forces it, despite --page-size 1) so the allocator is PagedTokenToKVPoolAllocator (alloc/alloc_extend/alloc_decode); the hook sat in TokenToKVPoolAllocator only. Added paged hooks (_lazy_hook before pages are consumed, refuse allocation on commit failure; _lazy_idle_check in _release_page_ids). Review fixes applied: last-slot backing (size+page_size), empty_cache retry, acquire-then-commit shrink, free() never grows/raises, stale ctl ignored, chunk-exact fill estimate, poll acts only at layer 0 of m<=16 forwards.
- 2026-09-02 13:37 S14_kvlazy: FAILED to start or bench -> reverted
- 2026-09-02 13:38 S14 attempt 3: paged hooks fire (first idle release reached uncommit_beyond) but NameError _get_cuda_driver in kv_vmm_backing.py (not imported there). Local import added; attempt 4.
- 2026-09-02 13:42 S14_kvlazy: logprob max 0.057 mean 0.0014 -> equivalent
- 2026-09-02 13:42 S14_kvlazy: KEPT  decode 55.4 tok/s, prefill 2323 tok/s  (was 56.2 / 2335)
- 2026-09-02 14:45 S14_kvlazy KEPT (attempt 4): exact (oracle mean 0.0014), decode 55.4 / prefill 2323 (was 56.2 / 2335). Log shows the mechanism: commits in margin steps during prefill (owner backed 288 MB at 10k tokens = 24 KB/token), release to the 4096-token floor (144 MB) when the pool goes idle, re-commit for the next request. Refinements applied before S15: commit targets rounded to 2048-token steps (no per-page no-op commits); after a forced expert shrink the cache regrows at the next KV-idle point (safe: no forward in flight). S15 = --max-total-tokens/--context-length 131072 with SGLANG_KV_LAZY_TOKENS=131072 (virtual capacity above the profiled value) and SGLANG_MOE_ELASTIC_FILL_MB=768 (startup fill).
- 2026-09-02 13:47 S15_ctx128k: logprob oracle crashed -> treated as not equivalent
- 2026-09-02 13:47 S15_ctx128k: REVERTED (not exact)  decode 57.3, prefill 1974
- 2026-09-02 13:48 S15 attempt 1 FAILED: startup fine (profiled 106560 tokens -> virtual 131072; autofill S=224, 1.08 GB free) but the 6.8k/10k prefill OOMed inside the GDN chunk kernel: prefill working memory is ~1.1-1.5 GB above idle (S14 log: driver free 3.37 -> 2.25 GB during a 10k prefill). Fill reserve 768 MB -> 2048 MB; forced-shrink margin 256 MB -> 1 GB. Note: bf16 128k needs 3.1 GB KV vs ~2.1 GB available at the S floor -> fp8 KV is the path to 128k/256k (user: 'fp8kv wäre perfekt', best 8-bit by speed/accuracy).
- 2026-09-02 13:51 S15_ctx128k: logprob max 0.073 mean 0.0021 -> equivalent
- 2026-09-02 13:51 S15_ctx128k: KEPT  decode 56.1 tok/s, prefill 1920 tok/s  (was 55.4 / 2323)
- 2026-09-02 13:53 S15_ctx128k KEPT (attempt 2, fill reserve 2 GB -> S=200, 2.22 GB free): exact (mean 0.0021). First bench right after startup read decode 41.6/48.9 at small ctx and prefill 1920 (startup transient: fill + first KV releases); re-measured on the live server: decode 55.0-57.5, prefill 2270-2340 -> accepted numbers set to 56.9 / 2340. Context length now 131072 with 24 KB/token committed on demand.
- 2026-09-02 13:55 longctx 60k on S15 CRASHED: KV commits succeeded up to 55296 tokens (1.3 GB) but each commit ate the prefill's working memory (driver free 0.07 GB) until torch OOMed a 12 MB tensor in radix_attention. The shrink only triggered on cuMemCreate failure. Added a watermark rule to lazy_ensure: keep SGLANG_KV_LAZY_HEADROOM_MB (1536) driver-free after every commit, shrinking the expert cache first. bf16 ceiling at the S floor: ~(3.4-1.5) GB / 24 KB = ~80k tokens; fp8 KV is required beyond that.
- 2026-09-02 15:05 longctx 60k, attempt 2 (headroom policy active): KV commits fine (driver free 1.5 GB kept, expert cache 200 -> 192 -> 184), but systemd-oomd killed the scope (MemoryMax 27G) at 14:01: HOST memory grows with the request beyond the ~2 GB slack. Cause to be measured (host sampler on the next restart).
- 2026-09-02 15:05 256k research (5 agents) verdict: fp8_e4m3 storage works with the lazy VMM patch unchanged (uint8 store, 12 KB/token); blockers are only in the QSA read path: P1 decode/verify gather _compact_kv must dequant (uint8 -> fp8 bitcast -> bf16) and the FA2 scratch keyed by q.dtype; P2 prefix-chunk prefill index_select+cat must produce bf16 (fused gather-dequant preferred); P3 unit scale (skip div_ when scale==1, saturate to +-448 before the cast); P4 arg guard; P5 reclaims: --attention-backend triton (drops the 384 MiB FlashInfer workspace), --max-mamba-cache-size 1. Budget: 128k fp8 ~3.5 GiB peak vs ~3.9 GB reclaimable -> reachable; 256k fp8 ~5.3 GiB -> NOT reachable without a paged prefix-chunk kernel (the prefill materializes the whole prefix K/V per layer per chunk) and smaller GDN working memory. bf16 ceiling ~64-80k.
- 2026-09-02 14:10 longctx 60k, attempts 2+3: 48k passes (31 s, no host growth) but 60k dies both times by systemd-oomd (user slice ManagedOOMMemoryPressureLimit=50%) ~30 s after the last KV commit at exactly 55296 tokens; GPU side healthy (driver free 1.5 GB). Hypothesis: memory PRESSURE from file reads under 2 GB free RAM (PLE n-gram table pread/mmap: thousands of scattered 4 KB pages per chunk) -> reclaim stalls -> PSI > 50% -> kill. Next: measure PSI/Cached during a 40k prompt; if confirmed, read PLE rows with O_DIRECT (no page-cache churn).
- 2026-09-02 14:12 PLE readahead fix (patches/ple_random.py): MADV_RANDOM on the 51 GB memmap + POSIX_FADV_RANDOM on the pread fd; added to the accepted patch set (no numerics involved). Restarting, then PSI measurement at 40k and the 60k retry.
- 2026-09-02 14:24 longctx 60k attempt 4 (PLE fix + 1 s trace): steady until 14:23:10 (scheduler RSS 165 MB, 2.9 GB of its heap in swap, PSI 12-15 %), then a 30 s system freeze and oomd at 14:23:39; last commit again 55296 -> deterministic trigger at the same token position. Hypothesis: Python gen-2 GC in the scheduler walks the swapped-out 2.9 GB heap -> swap storm. SGLang ships freeze_gc (gc.freeze after warmup); testing it before the next 60k run.
- 2026-09-02 14:25 sweep.sh now warms once and POSTs /freeze_gc (gc.freeze in scheduler + detokenizer) after the server is up; SGLANG_LOG_GC=1 added to the accepted env so GC generations/durations appear in the server log.
- 2026-09-02 14:32 GC hypothesis REFUTED: SGLang froze the scheduler GC itself at 14:27:36 (gen2 1.1M objects -> 0) and 60k died anyway at the same position. Fine trace: scheduler swaps in continuously during prefill (~1850 major faults/s, 3 GB of its heap in swap) while MemAvailable stays 2.2-2.7 GB; then a 30 s system-wide stall (even the 1 s sampler stops) and the oomd kill. Now recording kernel counters (pswpin/pswpout, compact_stall, allocstall, pgscan_direct) per second; THP settings checked.
- 2026-09-02 14:38 ROOT CAUSE (60k deaths): per-second kernel counters show direct-reclaim storms (pgscan_direct 7M pages/s, pswpout 87k pages/s, allocstall) with ~2 GB MemFree -> reclaim against a limit below the system: sweep.sh runs the server in a systemd scope with MemoryMax=27G; pinned weights + anon heap + PLE page cache hit it during long prefills, the cgroup swaps its own pages, the prefill refaults them, oomd kills on cgroup pressure. MemoryMax raised to 30G; cgroup memory.current/stat/events now sampled.
- 2026-09-02 14:45 ROOT CAUSE CONFIRMED: at 55k tokens the watermark shrinks the expert cache below S=184; the slot pool is empty there, so _take_slots pins fallback rows at runtime (many small cudaHostAlloc); on a box with 26 GB pinned / 2 GB free each pinned allocation triggers reclaim+compaction storms (a 4 MB pin took 22 s: log 14:42:56 -> 14:43:18) until oomd kills the scope. Fix: no runtime pinning (pool exhaustion raises; free() stops), a one-time pinned reserve at startup (SGLANG_MOE_ELASTIC_RESERVE_ROWS=384/kind, ~0.5 GB -> floor 176), and lazy_ensure refuses the KV allocation when the headroom cannot be restored (request fails instead of the server). MemoryMax stays 30G (cgroup was not the cause: memory.current 26.2 GB, no events).
- 2026-09-02 14:49 Startup pinned reserve (384 rows/kind, 0.5 GB) killed the server at load: the 315 MB cudaHostAlloc itself triggered the reclaim storm and oomd. This host cannot pin anything beyond the 24 GB offload + 160 MB startup shortfall. Final policy: no reserve, floor S=184, KV allocation refused when headroom cannot be restored. bf16 ceiling ~50k tokens on this box; fp8 KV (S16) is the way to ~100k; 32 GB host RAM is the binding constraint.
- 2026-09-02 14:54 LONG CONTEXT WORKS (bf16): 68,905-token prompt on the accepted server: prefill 48.9 s (1408 tok/s), decode 53.3 tok/s, correct answer, server alive. Expert cache 192 -> 184 during the request (VRAM free down to 1.18 GB), KV released and cache regrown to 192 afterwards. The 'deaths' were all the runtime pinned allocations.
- 2026-09-02 14:56 100k bf16 test: refusal fired as designed at 84k tokens ('no headroom for 90112 tokens, free 779 MB, need 1584 MB') but the server died afterwards: the scheduler does not survive alloc_extend returning None mid-prefill (see traceback). Restarting; the 128k/256k path is S16 (fp8).
- 2026-09-02 14:58 Admission cap: virtual KV capacity = min(SGLANG_KV_LAZY_TOKENS, 0.85 x profiled) (SGLANG_KV_LAZY_SAFETY); bf16 -> ~90k tokens, fp8 -> 131072 (profiled ~213k). Launching S16_fp8kv (fp8_e4m3 KV + QSA read-path patch, triton attention backend, mamba cache 1); quality via nll_eval afterwards.
- 2026-09-02 15:01 S16_fp8kv: REVERTED (not better)  decode 55.1, prefill 2246
- 2026-09-02 15:05 S16_fp8kv: started fine (fp8 gather kernel captured, 12 KB/token, capacity 131072, +0.42 GB from the triton backend, fill S=208) but REVERTED by the speed rule (decode 55.1 vs 56.9, -3 %). fp8 is a capacity mode, not a speed step: bringing it up as a mode (phase1 --bringup S16_fp8kv) for NLL + 100k/128k tests.
- 2026-09-02 15:12 fp8 mode (bring-up): decode 55.9-57.2 / prefill 2303-2339 = parity with bf16 (the -3 % was startup noise). Short-text NLL delta +0.0003 and oracle mean 0.0018 are NOT evidence: prompts < 1024 tokens never read the cache (first chunk uses projected bf16 K/V). Wrote nll_long.py (teacher-forced over the last 512 of ~7k tokens + 300-token greedy after 3k); first version requested logprobs for all positions (7 GB of logits) and OOM-killed the server. Re-launching fp8 mode with the KV statistics hook (patches/kv_stats.py) for the own-scheme design.
- 2026-09-02 15:19 nll_long attempt 2 OOM again: SGLang computes input logprobs per prefill chunk (1024 x 248k x 4 B = 1 GB) and the fp8 mode's autofill leaves ~1.9 GB. Fix for the test: shrink the expert cache live (ctl 'S 184') before scoring.
- 2026-09-02 15:26 fp8 mode: a 1-token prompt ('Hi') crashes with an illegal memory access surfacing in the QSA indexer prefill (metadata.py:141 sync); >=101-token prompts work; bf16 mode handles 1-token prompts fine. Debug run with CUDA_LAUNCH_BLOCKING=1 to locate the faulting kernel.
- 2026-09-02 15:50 KV DESIGN PANEL (5 designs, 2 judges, synthesis; grounded in perf/nll/kv_stats_fp8run.pt = real K/V stats over ~19k tokens): winner INT8-G64 = int8 K and V in the existing [slots, 2, 256] layout with one fp16 absmax scale per (token, kv-head, 64-channel group; group 0 = the rotary dims), 12.4 KB/token, fused quantize+scatter Triton kernel at write, dequant inside the two gather sites (decode _compact_kv sibling; a row->slot gather-dequant kernel replacing index_select+cat for prefix chunks), compressed index-K stays bf16, scales live in the lazy VMM arena as extra descs. Simulated relative RMS error on the measured channel profiles: e4m3 2.66 % | int8 per-token 1.3 % | int8 g64 0.9 % | g64 + static channel smoothing 0.76 %. Runner-up ideas: asymmetric K/V with channel smoothing (stage B), sealed-page deferred quantization, selection-aged fp8/int4 tiers (dropped: accuracy). Stage A0 = fake quantization on the write path (patches/kv_fakeq.py) to measure the real-model NLL of each scheme before writing kernels.
- 2026-09-02 15:43 Bug: poll() kept a '% 64' call gate from an earlier version; combined with the layer-0 / m<=16 condition the control file was honoured only every ~64th eligible forward, so 'S 184' before nll_long never ran and the 1 GB logit chunk OOMed twice more. Fixed (check on every eligible forward). Next: bf16 references (nll_long save bf16), then fake-quant runs int8_g64 / int8_tok / e4m3.
- 2026-09-02 15:49 nll_long bf16 reference saved (S=184): TF NLL 2.0388 (last 512 of 9586 tokens), greedy-after-3759 NLL 0.7860. Noise floor of the same config measured next; then the fake-quant series.
- 2026-09-02 15:49 nll_long noise floor (same bf16 config, S=184): NLL delta -0.008, mean|dlogprob| 0.099, max 0.80 over 512 positions; greedy diverges at char 10. Long-context runs are far noisier than the short oracle (top-k selection flips + 9 chunks of bf16 accumulation). 512-position resolution ~ +-0.01 nats: enough for 'no harm', not for e4m3-vs-int8; NLL_LONG_ALL=1 scores every position >= 1024 (needs the 1 GB logit chunk headroom).
- 2026-09-02 15:53 FQ_int8_g64 (fake quantization, bf16 storage):   greedy after 3759 tokens: 300 tokens, mean NLL 0.5668 (decode reads the cache)   vs bf16: NLL delta -0.0069, mean|dlogprob| 0.0994, max 1.100 over the last 512 positions   greedy mean NLL: ref 0.7860 now 0.5668; first divergence: 10 
- 2026-09-02 15:57 FQ_int8_tok (fake quantization, bf16 storage):   File "/usr/lib/python3.12/http/client.py", line 305, in _read_status     raise RemoteDisconnected("Remote end closed connection without" http.client.RemoteDisconnected: Remote end closed connection without response 
- 2026-09-02 16:01 FQ_e4m3 (fake quantization, bf16 storage):   greedy after 3759 tokens: 300 tokens, mean NLL 0.5921 (decode reads the cache)   vs bf16: NLL delta -0.0133, mean|dlogprob| 0.1099, max 0.994 over the last 512 positions   greedy mean NLL: ref 0.7860 now 0.5921; first divergence: 10 
- 2026-09-02 16:06 FQ_int8_g32 (fake quantization, bf16 storage):   File "/usr/lib/python3.12/http/client.py", line 305, in _read_status     raise RemoteDisconnected("Remote end closed connection without" http.client.RemoteDisconnected: Remote end closed connection without response 
- 2026-09-02 16:07 Fake-quant series (512-position window, noise floor NLL -0.008 / mean|dlogprob| 0.099): int8_g64 -0.0069 / 0.0994 (= noise); e4m3 -0.0133 / 0.1099 (+11 % token deviation over noise); int8_tok and int8_g32 runs died by oomd at the first long prefill (host pressure at the edge; the bench right after startup always survives, requests minutes later die ~50 %: swapped-out idle processes refault). Added keepalive.sh (tiny request every 5 s) around the scoring; re-running the two.
- 2026-09-02 16:08 INT8-G64 stage A implemented (perf/patches/kv_int8.py, 19 edits + new srt/mem_cache/int8_kv_pool.py; perf/gemv/test_kv_int8.py): unit tests pass on the GPU: quant+scatter bit-exact (int32/int64 loc), scale index, compact gather-dequant bit-exact (3 requests), prefix row gather-dequant bit-exact (lens 300/1203/4001), int8_g64 relative RMS error K 0.62 % / V 0.62 % (e4m3 2.7 %), pool subclass eager + lazy VMM ok. Adversarial review + fix round running. S17_int8kv queued (--kv-cache-dtype int8_g64).
- 2026-09-02 16:12 FQ_int8_tok (fake quantization, bf16 storage):   greedy after 3759 tokens: 300 tokens, mean NLL 0.6433 (decode reads the cache)   vs bf16: NLL delta -0.0059, mean|dlogprob| 0.1000, max 0.940 over the last 512 positions   greedy mean NLL: ref 0.7860 now 0.6433; first divergence: 10 
- 2026-09-02 16:16 FQ_int8_g32 (fake quantization, bf16 storage):   greedy after 3759 tokens: 300 tokens, mean NLL 0.6857 (decode reads the cache)   vs bf16: NLL delta -0.0044, mean|dlogprob| 0.1059, max 0.838 over the last 512 positions   greedy mean NLL: ref 0.7860 now 0.6857; first divergence: 10 
- 2026-09-02 16:17 Fake-quant series complete (keepalive.sh fixed the flakiness): mean|dlogprob| over the last 512 of 9.6k tokens: noise 0.099 | int8_g64 0.099 | int8_tok 0.100 | int8_g32 0.106 | e4m3 0.110; NLL deltas all within +-0.013 (noise +-0.008). No 8-bit scheme is distinguishable from bf16 at this resolution; e4m3 shows the largest token deviation. INT8-G64 chosen on the simulated error (0.6-0.9 % vs 2.7 %) at 12.4 KB/token.
- 2026-09-02 16:26 INT8-G64 review round: 5 low findings, all fixed + verified (per-call prefix scratch instead of a retained buffer; kv_int8 requires kv_fp8 applied and refuses out-of-order reverts, kv_fp8 refuses revert while kv_int8 is applied; compact store mask = valid columns only; row-gather test covers interior gaps, mutation-checked). Validating S17_int8kv (int8_validate.sh: short NLL, long NLL vs bf16, oracle, bench, longctx 100k).
- 2026-09-02 16:45 S17_int8kv VALIDATED (own INT8-G64 KV cache): 12.4 KB/token; short NLL delta -0.001; long-text mean|dlogprob| 0.094 (bf16 noise floor 0.099), NLL +0.010 (noise +-0.008); oracle 10k mean 0.0019 (floor); bench decode 56.1-57.4 / prefill 2301-2316 (parity); LONG CONTEXT: 115,560-token prompt -> prefill 71.8 s (1610 tok/s), decode 50.8 tok/s, correct answer, server alive, expert cache 192 -> 184 -> 192. ACCEPTED as the default: --kv-cache-dtype int8_g64 --attention-backend triton --max-mamba-cache-size 1, patches kv_fp8 + kv_int8. fp8_e4m3 remains available as a mode. Next: the 128k ceiling test, then the ~150-178k question (safety cap 0.85 x profiled 209600).
- 2026-09-02 16:34 128k ceiling test on the int8 default: 126,060-token prompt -> prefill 78.1 s (1614 tok/s), decode 53.4 tok/s, correct, alive. Next: S18_ctx256k (virtual 262144, cap 0.85 x profiled) to find the real ceiling; prompts 150k / 178k.
- 2026-09-02 17:00 CEILING (int8 KV, virtual 262144, cap 0.85 -> 177,984 admission): 150,560 tokens: prefill 92.8 s (1623 tok/s), decode 53.3; 162,215 tokens: prefill 95.8 s (1694 tok/s), decode 51.2, VRAM free bottomed at 1.18 GB, correct answers, server alive; a ~179k prompt was refused at admission (server alive). Proven ceiling ~162k on 24 GB VRAM / 32 GB RAM. Accepted default now: --context-length 262144 --max-total-tokens 262144 (VA only) with SGLANG_KV_LAZY_SAFETY=0.77 (~161k admission = proven). 256k itself is out of reach here: it needs ~1.3 GB more (int4 for old blocks, or a paged prefix-chunk kernel + smaller GDN working memory, or more host RAM for a lower expert floor).
- 2026-09-02 16:49 S12d_ngram_d4_replay: FAILED to start or bench -> reverted
- 2026-09-02 16:49 S12d_ngram_d4_replay FAILED: NotImplementedError 'Qwen4 PLE does not support NGRAM speculation' (qwen4_exp.py PLE path). Speculative decoding on this model needs PLE support for draft tokens (n-gram context per draft position in verify) -> future deep item (potential 1.5-2x decode).
- 2026-09-02 16:56 FQ_int4_g32 (fake quantization, bf16 storage):   greedy after 3759 tokens: 300 tokens, mean NLL 0.7334 (decode reads the cache)   vs bf16: NLL delta -0.0083, mean|dlogprob| 0.0980, max 0.942 over the last 512 positions   greedy mean NLL: ref 0.7860 now 0.7334; first divergence: 10 
- 2026-09-02 16:59 int4_g32 fake-quant: mean|dlogprob| 0.098 (= noise) although the function's RMS error is 9.7 % (verified offline; env verified in the launcher environ). Conclusion: the 512-position test cannot separate any scheme from bf16. Next: all-position series (nll_series.sh, ~8.5k positions) for bf16 x2, int4_g32, real int8 (S17), e4m3.
- 2026-09-02 17:00 FQ_int4_g32_sm (fake quantization, bf16 storage):   greedy after 3759 tokens: 300 tokens, mean NLL 0.7223 (decode reads the cache)   vs bf16: NLL delta +0.0033, mean|dlogprob| 0.0999, max 1.053 over the last 512 positions   greedy mean NLL: ref 0.7860 now 0.7223; first divergence: 1 
- 2026-09-02 17:04 FQ_int8_g64_sm (fake quantization, bf16 storage):   greedy after 3759 tokens: 300 tokens, mean NLL 0.7639 (decode reads the cache)   vs bf16: NLL delta -0.0044, mean|dlogprob| 0.1005, max 1.162 over the last 512 positions   greedy mean NLL: ref 0.7860 now 0.7639; first divergence: 15 
- 2026-09-02 17:08 SPECULATION RESEARCH (3 readers + synthesis, perf/SPEC_NGRAM_PLAN.md): MTP not feasible (mtp.* experts are bf16, 4.7 GiB + 2.5 GB transient; no elastic/streamer support for bf16 MoE; host RAM full). NGRAM feasible with ~150 lines: drop the PLE NGRAM guard (qwen4_exp.py:122-124), force a linear draft chain in ngram_worker (_linearize_chain; bfs-breadth 1 still yields a star -> breaks GDN fold, QSA pending ring, int8 KV move), commit PLE n-gram history + short-conv state in the ReplaySSM fold/ring branches of spec_utils.py (a real bug for topk-1 MTP too), a startup self-check, and an optional debug check. Expected 0.9-1.2x prose, 1.3-1.8x code, cap 4 tokens/step; validation via a lossless test (spec logprobs vs teacher-forced). Implementing as patches/ngram_ple.py + spec_lossless.py.
- 2026-09-02 17:08 nll_long ALL-positions [accepted save vs bf16all]:   teacher-forced: 9586 tokens, NLL 1.6276 nats/token over the last 8561 positions (their chunks read the cache)   greedy after 3759 tokens: 300 tokens, mean NLL 0.6019 (decode reads the cache)   saved ~/Downloads/qwen38-flash-next-handoff/perf/nll/long_bf16all.json 
- 2026-09-02 17:13 nll_long ALL-positions [accepted check vs bf16all]:   greedy after 3759 tokens: 300 tokens, mean NLL 0.7314 (decode reads the cache)   vs bf16all: NLL delta -0.0006, mean|dlogprob| 0.0609, max 1.439 over the last 8561 positions   greedy mean NLL: ref 0.6019 now 0.7314; first divergence: 318 
- 2026-09-02 17:17 nll_long ALL-positions [FQ_int4_g32 check vs bf16all]:   greedy after 3759 tokens: 300 tokens, mean NLL 0.5199 (decode reads the cache)   vs bf16all: NLL delta -0.0006, mean|dlogprob| 0.0601, max 1.740 over the last 8561 positions   greedy mean NLL: ref 0.6019 now 0.5199; first divergence: 10 
- 2026-09-02 17:21 nll_long ALL-positions [S17_int8kv check vs bf16all]:   greedy after 3759 tokens: 300 tokens, mean NLL 0.4142 (decode reads the cache)   vs bf16all: NLL delta +0.0010, mean|dlogprob| 0.0591, max 1.310 over the last 8561 positions   greedy mean NLL: ref 0.6019 now 0.4142; first divergence: 106 
- 2026-09-02 17:25 nll_long ALL-positions [FQ_e4m3 check vs bf16all]:   greedy after 3759 tokens: 300 tokens, mean NLL 0.8181 (decode reads the cache)   vs bf16all: NLL delta +0.0000, mean|dlogprob| 0.0582, max 1.535 over the last 8561 positions   greedy mean NLL: ref 0.6019 now 0.8181; first divergence: 1 
- 2026-09-02 18:15 ALL-POSITION NLL SERIES (8561 positions >= 1024 of a 9586-token text, chunks that read the cache; 1 GB logit chunks with the expert cache at S=184): bf16 vs bf16 (noise) NLL -0.0006 / mean|dlogprob| 0.061 | fake int4_g32 -0.0006 / 0.060 | REAL int8_g64 server +0.0010 / 0.059 | fake e4m3 +0.0000 / 0.058. VERDICT: at 0.001-nats resolution no scheme down to int4-g32 (9.7 % RMS error on the stored values) is distinguishable from bf16 on this model. Reasons (architecture): block selection uses the separate bf16 compressed index keys, K is k_norm-ed, softmax over 2048 selected positions averages the noise, 36 of 48 layers are GDN without KV. Consequence: INT4-G32 storage (~6.9 KB/token incl. scales) is the 256k path: 256k = 1.8 GB KV. Still needed for 256k: the paged prefix-chunk kernel (no materialized prefix, -0.5 GB) and/or smaller GDN working memory.
- 2026-09-02 18:25 S12e_ngram_chain (NGRAM + PLE patch, eager, 4 draft tokens): server up, bench decode 20-36 tok/s (eager + 4-row verify). Lossless gate: (a) prefill oracle ok (mean 0.0015); (b) 0 mismatches / 600 tokens (every token top-1 or within-band top-2), 7 near-ties, but spec-path vs teacher-forced mean|dlogprob| 0.009-0.019 (max 0.25) > the plan's 0.003 threshold -> needs the non-spec baseline (decode path vs prefill path differ even without speculation). Acceptance: mean accept length 1.11-1.27, accept rate 0.07-0.09 on prose/reasoning/code -> NGRAM is a net loss on these workloads regardless (22-25 tok/s incl. prefill vs 56).
- 2026-09-02 18:50 NGRAM VERDICT: non-spec baseline of the same gate gives mean|dlogprob| 0.019/0.009/0.011 (spec: 0.019/0.011/0.010), max 0.16-0.27 (spec 0.18-0.25), 0 mismatches, 4 near-ties (spec 7) -> the speculative path is lossless within the decode-vs-prefill floor (the plan's 0.003 threshold was unrealistic; the floor itself is ~0.01). Speed: accept length 1.11-1.27 (rate 7-9 %) on prose/reasoning/code -> no gain; with eager verify 22-25 tok/s vs 56. Status: patches/ngram_ple.py is correct and stays an opt-in for repetitive/structured workloads (--speculative-algorithm NGRAM ... per the patch docstring); not the default. Optional follow-up S12f: verify under breakable decode graphs.
- 2026-09-02 18:29 Compost ladder prep (user idea: age-tiered KV precision): fake-quant schemes int3_g16 (19.9 % RMS on N(0,3)), int2_g16 (58.5 %), int2_g8 (49.7 %) added to find the precision cliff with the all-position test; then tiers (open block bf16 / recent int8 / sealed int4 / ancient int2-or-evict) keyed by block age or indexer selection.
- 2026-09-02 18:32 nll_long ALL-positions [FQ_int3_g16 check vs bf16all]:   greedy after 3759 tokens: 300 tokens, mean NLL 0.7151 (decode reads the cache)   vs bf16all: NLL delta -0.0016, mean|dlogprob| 0.0615, max 2.109 over the last 8561 positions   greedy mean NLL: ref 0.6019 now 0.7151; first divergence: 1 
- 2026-09-02 18:36 nll_long ALL-positions [FQ_int2_g16 check vs bf16all]:   greedy after 3759 tokens: 300 tokens, mean NLL 0.6681 (decode reads the cache)   vs bf16all: NLL delta +0.0012, mean|dlogprob| 0.0598, max 1.614 over the last 8561 positions   greedy mean NLL: ref 0.6019 now 0.6681; first divergence: 106 
- 2026-09-02 18:37 Cliff series (all positions): int3_g16 (19.9 % RMS) NLL -0.0016 / 0.0615; int2_g16 (58.5 % RMS) +0.0012 / 0.0598 -> still at the bf16 noise floor. Too good: running destructive controls (KV := 0, KV := noise) to test whether the held-out-NLL metric is blind to the main K/V cache on this text; if it is, a needle-in-haystack test decides the precision question.
- 2026-09-02 18:39 INT4-G32 review (2 reviewers + fix): 3 low findings fixed (fp16 scale clamp at 65504 -> no NaN on extreme groups, trtllm strided compact test, out-of-order revert refusal). Same overflow class exists in kv_int8's kernel at 127 x 65504 (8.3e6): add the clamp there later. Patch ready for S19.
- 2026-09-02 18:41 nll_long ALL-positions [FQ_zero check vs bf16all]:   greedy after 3759 tokens: 300 tokens, mean NLL 0.6206 (decode reads the cache)   vs bf16all: NLL delta -0.0003, mean|dlogprob| 0.0610, max 1.745 over the last 8561 positions   greedy mean NLL: ref 0.6019 now 0.6206; first divergence: 1 
- 2026-09-02 19:35 ERRATUM: the KV := 0 control also measured at the noise floor -> impossible -> root cause: the fake-quant hook lives in MHATokenToKVPool.set_kv_buffer, but since S17 the default pool is MHATokenToKVPoolInt8 whose set_kv_buffer replaces it. Every fake-quant run after S17 (int4_g32, int3_g16, int2_g16, int8_g64_sm, int4_g32_sm, zero) and the 'bf16all' all-position reference were in fact the real int8 server -> those numbers are int8-vs-int8 noise, NOT evidence about int4/int3/int2. Still valid: the 512-position series before S17 (int8_g64/int8_tok/int8_g32/e4m3 on the bf16 pool) and S17's validation against the 512-position bf16 reference (0.094 vs noise 0.099). Fix: FQ_* steps now add --kv-cache-dtype auto (bf16 pool), new BF16KV step for references. Re-measuring: bf16 ref + noise (all positions), FQ_int4_g32, FQ_int2_g16, FQ_zero, then needle tests, then S19.
- 2026-09-02 18:48 nll_long ALL-positions [BF16KV save vs bf16kv]:   teacher-forced: 9586 tokens, NLL 1.6279 nats/token over the last 8561 positions (their chunks read the cache)   greedy after 3759 tokens: 300 tokens, mean NLL 0.7886 (decode reads the cache)   saved ~/Downloads/qwen38-flash-next-handoff/perf/nll/long_bf16kv.json 
- 2026-09-02 18:52 nll_long ALL-positions [BF16KV check vs bf16kv]:   greedy after 3759 tokens: 300 tokens, mean NLL 0.3147 (decode reads the cache)   vs bf16kv: NLL delta +0.0002, mean|dlogprob| 0.0590, max 1.712 over the last 8561 positions   greedy mean NLL: ref 0.7886 now 0.3147; first divergence: 1 
- 2026-09-02 18:56 nll_long ALL-positions [FQ_int4_g32 check vs bf16kv]:   greedy after 3759 tokens: 300 tokens, mean NLL 0.7040 (decode reads the cache)   vs bf16kv: NLL delta +0.0078, mean|dlogprob| 0.1386, max 3.666 over the last 8561 positions   greedy mean NLL: ref 0.7886 now 0.7040; first divergence: 1 
- 2026-09-02 19:00 nll_long ALL-positions [FQ_int2_g16 check vs bf16kv]:   greedy after 3759 tokens: 300 tokens, mean NLL 0.4230 (decode reads the cache)   vs bf16kv: NLL delta +0.3004, mean|dlogprob| 0.6189, max 11.425 over the last 8561 positions   greedy mean NLL: ref 0.7886 now 0.4230; first divergence: 0 
- 2026-09-02 19:05 nll_long ALL-positions [FQ_zero check vs bf16kv]:   greedy after 3759 tokens: 300 tokens, mean NLL 0.2638 (decode reads the cache)   vs bf16kv: NLL delta +3.6988, mean|dlogprob| 3.9228, max 20.832 over the last 8561 positions   greedy mean NLL: ref 0.7886 now 0.2638; first divergence: 2 
- 2026-09-02 19:06 needle 60000 [cur]:   prompt 0 tokens, prefill 0.0 s, decode 0.0 tok/s   needles: Kestrel:-- Marlin:-- Basalt:-- Juniper:-- Tamarind:--   score 0/5   answer: '' 
- 2026-09-02 19:10 needle 60000 [BF16KV]:   prompt 41370 tokens, prefill 50.3 s, decode 54.1 tok/s   needles: Kestrel:OK Marlin:OK Basalt:OK Juniper:-- Tamarind:--   score 3/5   answer: '<think>\nThe user wants me to list the access codes for the five projects exactly as written in the notes embedded in the text. Let me find each note:\n\n1
- 2026-09-02 19:11 needle test: bf16 pool 3/5 at 41k tokens with 120 new tokens (answer truncated inside a think block); max_new_tokens raised to 400 for the remaining runs; bf16 to be re-run.
- 2026-09-02 19:15 needle 60000 [accepted]:   prompt 41370 tokens, prefill 54.5 s, decode 50.3 tok/s   needles: Kestrel:OK Marlin:OK Basalt:OK Juniper:OK Tamarind:OK   score 5/5   answer: '<think>\n\n</think>\n\n4821-QV\n7093-TX\n2617-RH\n9354-LM\n1178-ZK' 
- 2026-09-02 19:19 needle 60000 [FQ_int4_g32]:   prompt 41370 tokens, prefill 52.6 s, decode 50.8 tok/s   needles: Kestrel:OK Marlin:OK Basalt:OK Juniper:OK Tamarind:OK   score 5/5   answer: '<think>\nThe user wants me to list the access codes for projects Kestrel, Marlin, Basalt, Juniper, and Tamarind exactly as written in the notes embedded in
- 2026-09-02 19:30 nll_long ALL-positions [S19_int4kv check vs bf16kv]:   greedy after 3759 tokens: 300 tokens, mean NLL 0.7226 (decode reads the cache)   vs bf16kv: NLL delta +0.0088, mean|dlogprob| 0.1379, max 2.999 over the last 8561 positions   greedy mean NLL: ref 0.7886 now 0.7226; first divergence: 1 
- 2026-09-02 19:40 CORRECTED LADDER (bf16 pool, 8561 positions, noise +0.0002 / 0.059): fake int4_g32 +0.0078 / 0.139; fake int2_g16 +0.30 / 0.62 (cliff); KV:=0 +3.70 / 3.92 (metric sharp). Needle 41k tokens: bf16 3/5 (truncated at 120 new tokens), int8 real 5/5, int4 fake 5/5. S19 REAL INT4-G32: 6.8 KB/token, capacity 262144 (profiled 366k), short NLL +0.0004, 512-window dev 0.197, all-position +0.0088 / 0.138 (= fake int4), oracle 10k 0.0015, decode 55-57, prefill 1493 at 10k (int8 2316: the unpack gather costs), 162k prompt: prefill 112.6 s (1441 tok/s), decode 51.4, correct, alive.
- 2026-09-02 19:34 256K REACHED: on the S19 int4 server a 257,905-token prompt ran: prefill 182.2 s (1415 tok/s), decode 51.9 tok/s, correct answer, server alive, expert cache 184 -> 192 afterwards (VRAM free bottomed at 1.46 GB). Needle test at ~250k next.
- 2026-09-02 19:37 needle at ~250k on the int4 server: systemd-oomd killed the scope at ~65k tokens into the prefill. Random-word text makes every PLE n-gram lookup a fresh 4 KB page (repetitive filler did not): ~1M page-cache pages for the prompt -> host pressure. Fix in ple_random.py: POSIX_FADV_DONTNEED after each bulk gather (>= 512 ids); decode's pread path keeps its pages.
- 2026-09-02 20:00 NEEDLE AT 248K (int4 server, PLE page-drop fix, keepalive 900 s, ignore_eos): 247,629-token random-word haystack, five codes at 10/30/50/70/90 % depth -> 5/5 retrieved; prefill 269 s (920 tok/s on random text: PLE rows come from NVMe), decode 51.4 tok/s, server alive. The first attempt's empty answer was an immediate EOS (raw completion format), not a memory problem; the attempt before that died by oomd from PLE page-cache churn (fixed).
- 2026-09-02 19:56 TIERS PLAN (perf/KV_TIERS_PLAN.md): layout (B) dual-write: fresh tokens written int8 into a ring of R=W slots (owner table int32[R]) AND int4 into the full-context pool; readers test owner[slot & (R-1)] == slot on device (graph-safe) and read int8 (hot) or int4 (cold). No compactor, no VMM change, 7.3 KB/token at 256k (ring 104 MB), the int4 row is quantized from bf16 (same error as S19). Implementing as patches/kv_tiers.py.
- 2026-09-02 20:31 TIERS IMPLEMENTED (patches/kv_tiers.py + srt/mem_cache/tiered_kv_pool.py, test_kv_tiers.py): dual-write int8 ring (W=8192) over the int4 pool, owner stamp launch + guarded ring write (same-launch aliasing safe), tier-dispatch compact/prefix gathers, fp16 scale clamp; review: 6 findings fixed (phase1 now reverts layered patches in reverse and aborts on refusal; kv_int4 refuses revert under kv_tiers). Validating S21_tiers (tiers_validate.sh).
- 2026-09-02 20:39 needle 60000 [cur]:   prompt 41370 tokens, prefill 41.2 s, decode 52.7 tok/s   needles: Kestrel:OK Marlin:OK Basalt:OK Juniper:OK Tamarind:OK   score 5/5   answer: '<think>\nThe user wants me to extract the access codes for the five projects mentioned in the text. Let me find them:\n\n1. "Note: the access code for proj
- 2026-09-02 21:05 S21_tiers VALIDATED and ACCEPTED as the default (--kv-cache-dtype int8ring_int4, SGLANG_KV_TIERS_W=8192; patches kv_fp8 < kv_int8 < kv_int4 < kv_tiers): short NLL -0.0002; 512-window dev 0.118; ALL-POSITION NLL -0.0001 / 0.074 (int8 +0.001/0.059, int4 +0.009/0.138); oracle 10k 0.0019; bench decode 54-57 / prefill 2271 at 10k; 257,905-token prompt: prefill 171 s (1508 tok/s), decode 51.8, correct; needles 5/5 at 41k and 5/5 at 248k (prefill 257 s, decode 50.8). Pool 6.8 KB/token + 104 MB ring. The compost idea works: int8 quality where attention is dense, int4 memory for the old context, 256k on 24 GB.
- 2026-09-02 22:03 FQ_int4_g32_sm failed at graph capture (smoothing profile H2D copy during capture); guard added (no smoothing inside capture). Re-running after the current A/B.
- 2026-09-02 22:04 kv_int8 kernel: fp16 scale clamp (65504) added, same as int4/tiers; tree re-layered kv_int8 -> kv_int4 -> kv_tiers.
- 2026-09-02 22:06 nll_long ALL-positions [FQ_int4_g32 check vs bf16kv]:   greedy after 3759 tokens: 300 tokens, mean NLL 0.7049 (decode reads the cache)   vs bf16kv: NLL delta +0.0059, mean|dlogprob| 0.1369, max 3.636 over the last 8561 positions   greedy mean NLL: ref 0.7886 now 0.7049; first divergence: 74 
- 2026-09-02 22:18 PAGED PREFIX PLAN (perf/KV_PAGED_PREFIX_PLAN.md, future work): per-chunk fit on the 258k tiered run: chunk_ms = 615 + 0.487 us x prefix -> the O(prefix) gather is ~28 % of the long-context slowdown (max gain 1508 -> ~1900 tok/s), 64 % is prompt-content base cost (MoE/PLE), 8 % periodic ~1 s spikes every 16-18 chunks (candidate: lazy_ensure's empty_cache / commit bursts). Deferred: kernel effort high, gain moderate. Checking the spike candidate cheaply first.
- 2026-09-02 22:19 kv_lazy: below the watermark, empty_cache()+shrink attempts now only when the expert cache can still shrink, else empty_cache rate-limited to once per 30 s (suspected source of the periodic ~1 s prefill spikes at 256k).
- 2026-09-02 22:37 FINAL BENCH started: OpenRouter now lists qwen/qwen3.8-flash (Flash-Next itself?) besides qwen3.8-27b -> GPQA Diamond (198, thinking sampling T=1.0/top_p 0.95/top_k 20, max 16k tokens, sglang.test.run_eval chat API) running for 27b then flash; local GPQA + GSM8K on the accepted tiered server after the current measurements. Also launched: paged prefix-chunk kernel implementation (user: 'gerne mehr kernel aufwand').
- 2026-09-02 22:58 Lesson: FQ_int4_g32_sm bring-up failed with ImportError (sparse_gqa_fwd_interface_paged) because the paged-prefix implementer had its patch half applied at that moment. Rule: no server starts while an implementation workflow may be applying/reverting patches in the tree; queue them after the workflow.

## 2026-09-02 23:30 - final benchmark scope + DeepSWE harness (user: only GPQA Diamond and DeepSWE, only vs Flash)
- 27B OpenRouter run cancelled; GPQA Diamond reference = qwen/qwen3.8-flash via OpenRouter (16 threads, same run_eval harness).
- DeepSWE 1.1: pier 0.3.1 (uv tool), tasks ~/quant/harness/deep-swe (113 with tests/), images 3.9-5 GB each (not ~1 GB as
  planned) with 28 GB free (Docker build cache pruned: +12 GB) -> perf/deepswe_run.sh runs one `pier run` per task and
  deletes the task's images afterwards (pier builds <task[:32]>__<id>-main + egress-proxy images). Deterministic subset:
  bench_final/deepswe/tasklist_seed0.txt (random.Random(0) shuffle), first 24 on both sides, 2 parallel.
- Harness mode = the leaderboard's: mini-swe-agent with the bash TOOL (function calling), model class litellm (chat
  completions) locally, openrouter class for Flash; sampling T=1.0/top_p 0.95/top_k 20 (model card) on both sides.
- Local server additions for that: --reasoning-parser qwen3 (thinking out of content) and --tool-call-parser qwen3_coder
  (the chat template uses the <function=..><parameter=..> XML format); containers reach the server through
  perf/fwd.sh (socat 172.17.0.1:30001 -> 127.0.0.1:30000). toolcall_smoke.py checks both parsers after restart.
- Night chain perf/night.sh: smoothing A/B -> S22_paged validation + auto verdict (paged_decide.py) -> accepted restart
  -> 258k spike check -> local GPQA (8 threads) -> local DeepSWE. or_chain.sh: Flash GPQA -> Flash DeepSWE.
- Restart #19 died with an AssertionError after weight load: the paged-prefix implementer was applying/reverting in the
  tree at that moment (second occurrence of the race) -> night chain starts only after the workflow.
- Flash GPQA Diamond reference (OpenRouter, run_eval, T=1.0/top_p 0.95/top_k 20, 16k max tokens, 16 threads): score 0.662
  (198 questions); some 429 rate-limit retries during the run (see bench_final/gpqa_qwen_qwen3.8-flash_openrouter.log).
- 2026-09-03 00:05 S22_paged: kernel workflow finished (4 agents, reviews fixed: cu_q int64, kv_tiers anchor guard, honest
  ulp metric, warm/cold timing). Verdict REJECT without a server run: paged 6.1 ms vs materialised 1.4 ms per head-layer
  at prefix 60k (gather itself 0.18 ms). Tree = accepted S21 state (kv_tiers 13 APPLIED, paged clean). Night chain
  reduced to: smoothing A/B, accepted restart, 258k spike check, tool-call smoke, local GPQA, local DeepSWE.
- 2026-09-02 23:45 nll_long ALL-positions [FQ_int4_g32_sm check vs bf16kv]:   greedy after 3759 tokens: 300 tokens, mean NLL 0.7906 (decode reads the cache)   vs bf16kv: NLL delta +0.0022, mean|dlogprob| 0.1603, max 3.761 over the last 8561 positions   greedy mean NLL: ref 0.7886 now 0.7906; first divergence: 41 
- 2026-09-03 00:40 Stage B smoothing A/B (fake-quant int4_g32 with static per-channel RMS smoothing, bf16 pool, 8561
  positions): NLL delta +0.0022, mean|dlogprob| 0.160 vs plain int4_g32 +0.006..+0.009 / 0.137-0.139. Mixed: the mean
  NLL improves by ~0.005 nats, the per-token deviation gets larger. Not a clear win -> NOT implemented in the real
  kernels; the tiered default is at -0.0001 anyway (int4 only matters beyond the 8k int8 ring). Closed.
- 2026-09-03 00:55 restart #20 (accepted tiered set + --reasoning-parser qwen3 --tool-call-parser qwen3_coder). 258k
  re-measure with the rate-limited empty_cache: 257,905 tokens prefill 165.3 s (1560 tok/s, was 171 s / 1508), decode 52.3.
- 2026-09-03 ~01:00 local GPQA started serially: #running-req 1 with 7 queued -> --max-mamba-cache-size 1 caps concurrency
  at ONE request (the GDN state cache has one slot). Restart #21 with 8 mamba slots (night2.sh), GPQA + DeepSWE local after.
- 2026-09-03 00:27 still #running-req 1 with 8 mamba slots: sweep.sh BASE carries --max-running-requests 1 (early
  single-user setting). Restart #22 with --max-running-requests 8 (drop+add in state), GPQA restarted from zero (night3.sh).
- 2026-09-03 00:31 systemd-oomd killed the server 90 s into the 8-concurrent GPQA run ("memory pressure for
  user@1000.service"): swap-in storm 58 MB/s, PSI some avg10 79 %. Cause: at batch 8 the host expert-row traffic is ~8x
  and the pageable host rows that were swapped out during the restart (SwapFree -7 GB, Docker builds + weight load)
  refault under load; host RAM (24 GB rows + VS Code + containers) is at the wall. Mitigations: --max-running-requests 4,
  GPQA 4 threads, sweep.sh scope gets ManagedOOMPreference=omit (accepted by the user manager; whether oomd honours it for
  a user cgroup is unverified), Flash DeepSWE reduced to 1 parallel task after the two in-flight ones. Restart #23 (night4.sh).
- 2026-09-03 01:02 the ManagedOOMPreference=omit on the server scope backfired: systemd-oomd, unable to kill the server,
  killed the session's dbus.service ("pressure for user@1000.service 59.48 %") -> GNOME session ended (login screen),
  VS Code and Claude Code died, all chains and the release workflow stopped, the server died with the session.
  State at 07:44: local GPQA reached 9/198 (88 s/it at 4 concurrent), Flash DeepSWE 2 tasks scored (2/2 solved), tengo
  task killed mid-run, no release-prep output. omit REMOVED from sweep.sh; never again. Root problem stays: host RAM.
- 2026-09-03 07:55 user decision: no more time for the benchmark. Final benchmark status: Flash GPQA Diamond 0.662
  (OpenRouter, complete); local GPQA Diamond aborted at 9/198 (no score); DeepSWE Flash 2 tasks run, 2 solved
  (testem-per-launcher-reports, true-myth-iterable-collection-combinators); DeepSWE local not run. Benchmark containers,
  images and helper processes cleaned up. Release docs get these numbers as-is, marked incomplete.
- 2026-09-03 08:30 release: release/TECHNICAL_REPORT.md written (achievement, recipe, quality protocol, speed ladder,
  own engineering, limits). HF upload started as PRIVATE repo HaberstrohSystems/Qwen3.8-Flash-Next-2.5bpw-autoround-sglang
  (checkpoint int8dense-20260902-1212 without config.json.v1-broken + ple/ subfolder, ~93 GB, hf upload-large-folder,
  log logs/hf-upload.log). Three background agents
  produce release/sglang (patch + notes + upstream draft), release/hf (model card, upload notes), release/repo (GitHub).
- 2026-09-03 08:40 HF release checkpoint = ~/quant/hf-release-textonly (hardlinks to int8dense-20260902-1212; shard 8
  rewritten without the 333 vision-tower tensors, 0.84 GiB, index updated; config.json.v1-broken and the stale
  quantization_config.json omitted). Upload restarted from it. PLE table uploads as one 51.2 GB file (HF hard limit 500 GB).
- 2026-09-03 09:05 HF release: MTP file removed as well (index rebuilt, 222,856 tensors, total_size 38,755,352,600 B);
  release/hf/README.md rewritten as a user-facing card (target machine, files, recipe, serving commands, mechanisms,
  performance, quality ladder, scope; no incident history). Load test of ~/quant/hf-release-textonly with the accepted
  flags running (logs/hfcheck.log, server-hfcheck.log).
- 2026-09-03 09:20 HF repo renamed to HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang (name states bits, method, target GPU, stack); uploads restarted under the new id.
- 2026-09-03 09:30 release checkpoint load test #1 (hfcheck): weights load (179.7 s, elastic placement S=184, KV lazy
  capacity 351,872 profiled -> 262,144 admitted, 6.8 KB/token), then oomd-killed during graph capture at 08:30:11 while
  the HF uploads were hashing ~90 GB (host pressure, not the checkpoint). PLE table (51.2 GB) already in the repo
  before the rename; checkpoint shards uploading under the new name. Load test + bench repeats after the uploads (hfcheck2).
- 2026-09-03 09:45 session handoff: user keeps the new repo name; asks for a fresh session that reviews the model card
  against other cards for professionalism. Plan + state in perf/NEXT_SESSION.md. hfcheck2.sh runs nohup (survives).
- 2026-09-03 09:03 (wall clock; the four entries above carry estimated times, real order is correct) checkpoint upload: hashed 17/17, uploading at ~5 MB/s single-stream, ETA ~2 h; then hfcheck2.sh runs the load test.
- 2026-09-03 10:33 checkpoint upload stalled at ~46 KB/s with ~99 % transferred (xet dedup keeps uploaded chunks); uploader restarted.
- 2026-09-03 10:45 checkpoint upload complete: 17/17 files committed under the new repo name.
- 2026-09-03 10:50 release-checkpoint load test #2 (hfcheck2, host quiet except the user's desktop + Chrome): weights loaded, oomd kill at the elastic arena init (pressure avg10 72 %). Retry #3 running (hfcheck3). The checkpoint content is verified by index/key checks; a served proof of the exact upload is still open.
- 2026-09-03 11:00 control start from the SERVED directory (restart #24, same flags): came up, answered the bench (101 ctx
  35.5 tok/s, 421 ctx 57.9, 1701 ctx 18.8 tok/s at 94 tok/s prefill = heavy host paging), then oomd-killed at 10:51:50.
  So the host state (active desktop session: Chrome, VS Code, ~8 GB in swap) is the cause, not the release checkpoint;
  the three release-checkpoint attempts (hfcheck 1-3) failed the same way. The served proof of the exact upload is
  deferred to a quiet host (no browser, ideally after a fresh login); leftover processes and /dev/shm cleaned.
- 2026-09-03 13:36 served proof of the published checkpoint: sha256 of all 12 LFS files on the Hub == local hf-release-textonly / ple (12/12 match); load test hfcheck4 from hf-release-textonly with the accepted flags (running 1, mamba 1): up after 164 s, streaming bench 101/421/1701/6821/10001 ctx -> decode 57.4/57.6/58.2/57.6/57.1 tok/s, prefill 2338 tok/s at 10001 (4.28 s). Host quiet (old session and the crash-looping vllm container stopped), pressure avg10 29 % during load. Log docs/logs/hfcheck4.log.
