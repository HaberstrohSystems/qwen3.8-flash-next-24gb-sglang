# Qwen3.8-Flash-Next decode: one ranked engineering plan

> **Status:** pre-campaign synthesis, kept verbatim including its first-person notes ("verified
> this session" refers to 2026-09-01, "the owner" to the maintainer). What was done, and what each
> step measured, is in [CAMPAIGN.md](CAMPAIGN.md) and [TIMELINE.md](TIMELINE.md).

*Synthesis of six investigation threads plus the completeness critique. Everything below is grounded in the dossier's evidence; where I re-verified something myself this session I say so.*

---

## 0. The baseline you must stand on (three corrections before anything else)

Three numbers in the brief are wrong or misleading, and every downstream estimate changes if you keep them.

**"GPU IDLE 78% of wall time" — set aside, run-specific.** The trace (`/tmp/1788290168.2759433-TP-0.trace.json.gz`, still present, verified this session: 22 MB, 6 decode steps at bs=1) was taken with `with_stack=1, record_shapes=1` and recorded 139,699 `python_function` events per step, stretching step wall to 146–197 ms. The kernel durations in it are device-side and trustworthy; the idle fraction is not. Steady state is **~38.9 ms GPU busy against a 64.5–71 ms wall (14.1–15.5 tok/s per `server.log`), i.e. ~40–46% idle**, consistent with WRITEUP's independently-sampled 43–58% busy. Two threads burned pages arguing over this; one adversarial verdict settled it. Use 38.9 ms busy / ~26–32 ms host gap.

**"PCIe rx ~4 GB/s of 55" — a duty-cycle artifact, but the correction cuts both ways.** The gather moves 399.9 MB/token in 7.9 ms = 50.6 GB/s, which is ~83% of the 61 GB/s this machine has actually sustained (not 92% of an assumed 55). So *each individual gather is at line rate and cannot be made faster*. But it runs at a **12% duty cycle** — the link is idle for the other 56.6 ms. Both statements are true and the dossier's threads each held only one of them. Consequence: there is no bandwidth to win on the existing transfers, and there *is* ~8× headroom for more concurrent transfers, which is only reachable with a second stream and double-buffering that the code does not have (`expert_stream.py` contains no `Stream`, no events; the gather is issued synchronously on the default stream at `moe_wna16.py:412` immediately before `runner.run()` at `:430`).

**"A quarter of GPU time is spent moving expert weights" — true, but the experts are 8.5% of the byte budget.** This is the single largest omission in the dossier and it reorders the whole plan. Per decode token, attributed from the trace's recorded shapes and cross-checked against the sealed checkpoint headers (I re-parsed all 9 safetensors files this session):

| what | MB/token | kernel | ms | achieved GB/s |
|---|---|---|---|---|
| `linear_attn.out_proj` + `self_attn.o_proj` BF16 | 1510 | gemvx | 2.98 | 507 |
| hyper-connection input mix BF16 | 1258 | `_hc_mix_persistent_kernel` | 2.86 | 441 |
| `lm_head` BF16 [248320,2560] | 1271 | gemvx (**one launch**) | 2.08 | 610 |
| `self_attn.{q,k,v}_proj` BF16 | 818 | gemvx | 1.41 | 578 |
| `mlp.gate` routers BF16 | 126 | gemvx | 0.54 | 233 |
| **dense BF16 subtotal** | **5023** | | **9.87** | **509** |
| GDN `in_proj_*` + shared expert INT8 | 1745 | Marlin | 3.53 | 494 |
| **MoE experts INT2** | **627** | `fused_moe_kernel_gptq_awq` | **13.72** | **46** |
| expert movement | (same 627) | gather + index_select | 9.47 | 50.6 / 283 |

That is 36.6 of 38.87 ms. I confirmed the structural premise from `quantization_config.json`: `extra_config` has 789 entries and pins `hyper_connection` (291 keys), `self_attn` (61), `linear_attn` (111), `mlp.gate`, `embed_tokens`, `lm_head` all to **16 bits**, and zero entries touch `.experts.`. Checkpoint census: `HYPERCONN_BF16` 1.193 GiB, `LMHEAD_BF16` 1.184, `EMBED_BF16` 1.184, `SELFATTN_BF16` 1.150, `LINATTN_BF16` 1.074, `ROUTER_BF16` 0.117.

**Two facts fall straight out of this table.** (1) The dense BF16 groups run at 440–610 GB/s against a ~672 GB/s roofline — genuinely bandwidth-bound, so halving their bytes roughly halves their time. (2) The int2 MoE kernel runs at **46 GB/s on the same SMs in the same step** — an 11–13× per-byte efficiency gap. That kernel, not the gather, is the biggest single line item in the model.

Also worth stating plainly: the trace shows **zero kernel overlap** and everything on the default stream, so wall ≈ GPU busy + host gap, serially. Every millisecond saved on either side is 1:1 in wall time. At 64.5 ms baseline, 1 ms ≈ +0.24 tok/s.

**Regime caveat that applies to the entire plan:** the trace was taken at `#full token: 512` — 1.5% of the 32768-token window. `flash_fwd_splitkv` is 0.145 ms and the QSA indexer scan is negligible at that length; both grow with context, and full-context KV reads add ~805 MB/token (~+1.3 ms). GDN is O(1) in context so 36 of 48 layers are safe, but nobody has checked the other 12. If the deployment actually runs long contexts, re-derive the table before acting on it.

---

## 1. The ranked plan

Ranked by (expected ms/token saved) ÷ (risk × effort). Effort: **S** ≤ 1 day, **M** 1–3 days, **L** > 1 week. "Exact" = bit-identical output; "Approx" = changes model output and needs quality validation.

| # | Change | Kind | Expected saving | Effort | Risk | Confidence |
|---|---|---|---|---|---|---|
| 1 | Delete the offloader forward hook when all offloaded params are streamed experts | Exact | 1.5–4 ms host | S | Low | High |
| 2 | Parallelise the PLE row fetch (`os.pread` ×16) | Exact | 2.2–2.7 ms host | S | Low | Med-High |
| 3 | Skip the gather for fully-GPU-resident layers | Exact | ~1.7 ms (1.34 GPU + 68 launches); **prefill much larger** | S | Low | High |
| 4 | Flag sweep: overlap schedule, continuous decode steps, chunked prefill, mamba cache | Exact | up to 2.4 ms decode; 2–4× prefill | S | Low-Med | Med |
| 5 | Memoize `arange`/dtype-cast; hoist `positions.max().item()` | Exact | ~0.5–0.7 ms + 12 pipeline drains | S | Low | High |
| 6 | Fix the batch-1 int2 MoE kernel (config sweep first, then a GEMV kernel) | Exact | **5–11.6 ms GPU** | S→L | Med-High | Med |
| 7 | CUDA graphs (full-graph route, PLE hoisted) | Exact | most of the remaining host gap | L | High | Med |
| 8 | Requantize the five BF16 dense groups to INT8 | **Approx** | **4.5–5 ms GPU** | L (+3 days compute) | Med | Med-High |
| 9 | `int32` view in the gather / index_select kernels | Exact | ≤0.8 ms decode; +8–12% prefill claimed | S | Low | Low |
| 10 | Frequency-ranked expert residency | Exact | unknown, possibly large | L | High | Unmeasured |
| 11 | VRAM reclaim (mamba cache, embed_tokens to host) → lower `--cpu-offload-gb` | Exact | ~0.2 ms per GiB | S–M | Low-Med | High |
| 12 | Speculative decoding with the checkpoint's MTP head | **Approx** | up to ~30% (sketch only) | L | High | Speculative |

### 1. Delete the offloader forward hook — do this first

`offloader.py:148-172` wraps every module that had ≥1 parameter offloaded. Under `SGLANG_MOE_EXPERT_STREAM=1` the *only* offloaded parameters are streamed expert params (`offloader.py:121-127`), and the hook's `device_state` comprehension **excludes exactly those** (`if not _is_streamed_expert_param(k)`, `:158`). I read this code this session: the wrapper therefore does a `state_dict()`, a pile of provably-no-op `.to(device)` calls, and a `functional_call` reparametrization — every layer, every token, for nothing.

The trace measures the real shape (which corrects the cuda-graph thread's mock-based estimate in both directions): the wrapper fires **31 times per token, not 48** (only modules that actually offloaded something are wrapped), but per call it is heavier than the stand-in — **1077 of the token's 1440 `aten::to` calls, 898 `state_dict()` calls and 1263 `_is_streamed_expert_param` calls originate here**.

Fix: track whether every offloaded parameter of a module was a streamed-expert param; if so, do not install the wrapper. ~3–10 lines at `offloader.py:148`.

*Experiment before implementing (30 min, no restart):* attach to the running process or run a CPU-only repro of `offloader.py:150-172` against a real decoder layer's `state_dict()` and time it ×31. If it is under 1 ms, deprioritise. Cost: near zero.

### 2. Parallelise the PLE row fetch

`Qwen4ExpMmapEmbedding.gather` (`qwen4_exp.py:846-853`) does a blocking D2H, then numpy fancy-indexing into a 51 GB memmap, then a pageable H2D — once per token (`ple_layer_ids=[2]` → layer_id 1 only; the prefetch stream is dead because `--no-ple-offload-embedding` nulls it at `qwen4_exp.py:1066-1068`). Two independent measurements against the real table agree: **2.56–3.09 ms cold, 0.013 ms warm**, and cold is the norm (51 GB table, ~2 GiB available RAM, mmap resident set 0.22 GiB). numpy holds the GIL through the page fault so threading it does nothing; `os.pread` releases it and 16 concurrent 160-byte reads measured **0.404 ms**.

This is ~2.4 ms of pure NVMe latency per token, removable with ~20 lines, and it is *also* the single genuine CUDA-graph capture blocker — so it is on the critical path of item 7 regardless.

*Experiment first (one restart):* instrument `qwen4_exp.py:849` with a wall-clock histogram in a live run. If the real n-gram distribution has page-cache locality the out-of-process uniform-random measurement overstates it. Cost: one restart + 5 minutes of decode.

### 3. Skip the gather for fully-GPU-resident layers

The one claim that survived adversarial review essentially intact, with two corrections. For layers whose expert tensors are *all* on the GPU, `expert_stream.py:139-140` does `torch.index_select` purely to renumber ids onto a compact staging buffer the Triton kernel does not need.

Corrections the dossier converged on: it is **17 layers, not 17.7** — layer 30 is split (its `w13_qweight` is on the host, the other three tensors on GPU) and cannot use the fast path, because `E = w1.shape[0]` (`fused_moe.py:411`) and one `moe_align_block_size` result (`:452`) feeds both the up-GEMM (`:621`) and down-GEMM (`:832`), so all four tensors of a layer must share one numbering. So the gate must be **per-layer all-CUDA, never per-tensor**. And the saving is **68 launches / ~1.34–1.57 ms**, not 71 / 1.6 ms — the trace resolves `indexSelectSmallIndex` cleanly into 1.335 ms (uint8 qweight, 35 launches) + 0.234 ms (bf16 scales, 36 launches), with only 2 launches of contamination from other callers.

The change is provably safe: `moe_align_block_size.py:87-88` caps `max_num_tokens_padded` at `numel * block_size`, so E=512 produces the *same* 160-row sorted_ids and the *same* 400/800-CTA grid as E=10; `get_default_config` picks the same block sizes because `M=1 ≤ E` either way; the kernel has no atomics (plain `tl.store` at `:352`) and `off_experts` is int64 so E=512 does not overflow strides. It is byte-for-byte the path `moe_wna16.py:404-406` already takes when the streamer is None.

Two honest deductions: part of the 1.34 ms migrates into the MoE GEMM, which loses an L2-warm 13 MB staging slice and reads the 512-row original instead (net DRAM traffic still falls). And the **prefill** win is much larger — it removes ~12 GB of pointless device-to-device copying per 512-token chunk.

*Experiment first:* none needed beyond a correctness A/B — generate 200 tokens greedily before and after and diff the token ids. Cost: one restart.

### 4. The flag sweep — highest gain per unit of engineering effort in the entire plan

Four launch-flag changes, zero code risk, each mapping to a specific measured quantity. I verified the live cmdline this session.

- **Drop `--disable-overlap-schedule`.** No justification for it exists in the codebase (`server_args.py` forces it off only for MPS, `pp_size>1`, and PD-multiplexing — none apply). The trace shows what it costs: `process_batch_result` 1.91 ms + `get_next_batch_to_run` 0.42 ms + `recv_requests` 0.08 ms = **~2.4 ms of GPU-idle host work between every pair of decode steps**, including one blocking `cudaMemcpyAsync` (1749 µs host duration) draining the pipeline for the sampled token.
- **`--num-continuous-decode-steps 4`.** Amortises the same inter-step gap without touching the overlap scheduler. Currently 1. Nobody in the dossier mentioned it.
- **`--chunked-prefill-size 512 → 2048`.** The gather thread computed that prefill is ~100% gather-bound at 46.3 GB/s and that a 512-token chunk already dedups to ~all 512 experts, then drew no conclusion. The conclusion is that **per-chunk PCIe cost is nearly constant above ~256 tokens** — E[unique] = 512(1−(1−1/512)^(10C)) is 470 at C=128, 508 at C=256, 512 at C=512, and cannot exceed 512. Doubling or quadrupling the chunk should approach 2–4× prefill throughput for the same 20.5 GB/layer, bounded by activation VRAM (2048×10240×2 B ≈ 42 MB, affordable) and MoE compute. At 32k context that is time-to-first-token going from ~28 s to ~8 s.
- **`--max-mamba-cache-size 10 → 2`.** Frees 0.86 GiB (36 GDN layers × 11 slots × 48 × 128 × 128 × 4 B = 1.16 GB, matching `server.log:33`; `mamba usage: 0.10` every step). Safe because `--disable-radix-cache` forces slots-per-request to 1 (`kv_cache_configurator.py:1866-1867`) and `--max-running-requests` is 1. Also drop `--page-size 1`: `server.log:11` shows it is overridden to 64 anyway.

**One flag I would *not* touch on the critique's reasoning: `--max-running-requests 1`.** The critique calls it "the largest untouched lever" and computes ~2.4× aggregate throughput at bs=4 by amortising the batch-invariant 13.4 ms. The arithmetic is right, but it only pays if the workload has concurrency. For a single interactive stream, raising it does *nothing* and slightly worsens per-token latency. Name the tradeoff to the owner; do not change it blind. If concurrency does exist, note the hard ceiling: `expert_stream.py:63 _NO_DEDUP_LIMIT = 64` means bs ≤ 6 keeps the sync-free branch; bs ≥ 7 introduces a `torch.unique` + `int(uniq.numel())` device sync per layer per step.

*Experiment:* four restarts, one flag each, running the existing benchmark. Cost: ~2 hours total. This should be done **before** any code change, because it moves the baseline.

### 5. Two micro-fixes worth their line count

- `expert_stream.py:128` launches a `torch.arange(10)` and `:150` an int64→int32 cast, **per MoE layer, per token** — 96 launches for two loop-invariant constants. Memoize on `(n, device, dtype)`. Guard: confirm nothing downstream writes `topk_ids` in place.
- `qsa_indexer.py:186` does `int(positions.max().item())` — a full device sync — once per QSA layer, i.e. **12 `cudaStreamSynchronize` + 12 `cudaMemcpyAsync` per decode token**. The cuda-graph thread filed this as a NON-BLOCKER because it is guarded by `not get_is_capture_mode()`; that is right for capture and exactly backwards for production, where the server runs eager and the guard is *open*. Their measured host cost is only 95 µs precisely because the CPU is already behind — but they hard-cap how far the CPU can run ahead, which is the mechanism that would otherwise absorb launch latency. All 12 layers get the same `positions`, and at bs=1 decode `positions.max()` is `seq_len-1`, known on the host. Hoist it.

### 6. The int2 MoE kernel — the biggest single item, and nobody attacked it

`fused_moe_kernel_gptq_awq` moves 627 MB in 13.72 ms = **46 GB/s**, in the same step where cuBLAS GEMVs on the same SMs hit 507–610 GB/s and Marlin INT8 hits 494 GB/s. The diagnosis is already in the dossier, in pieces, from two threads that were arguing with each other and both turned out to be describing the same misconfiguration:

- `BLOCK_SIZE_M=16` with exactly one valid token row per expert block wastes 15/16 of every `HMMA` — the kernel issues 75.5 GFLOP to do 4.72 GFLOP of useful work.
- Grid is 400 CTAs (w13) / 800 (w2) on 70 SMs → **48% SM fill**, which is precisely the profile's "SM ~50%".
- The B tile issues 16 scalar `ld.global.b8` per thread per K-iteration with 4× lane redundancy from `(offs_k//4)`; SASS shows 228 instructions/thread/K-iteration, 76.7 G warp-instructions/s = **11% of issue roof**.
- The config comes from the untuned fallback: zero int2 tuned JSONs and zero RTX_PRO_4000 JSONs exist, so `get_default_config`'s generic else-branch runs and `M ≤ E` flips it to the small-M config (`fused_moe_triton_config.py:243-260`). The same heuristic gives `BLOCK_SIZE_M=16` even at M=512 during prefill, because E=512 — almost certainly leaving prefill on the table too.

At 300 GB/s this kernel would take 2.1 ms instead of 13.7 — **11.6 ms, 30% of GPU busy**, larger than everything else in the plan combined.

*Experiment before implementing, in two stages.* **Stage 1 (a few hours, config only):** drop a tuned `int2_w2a16` JSON into `SGLANG_MOE_CONFIG_DIR` and sweep `BLOCK_SIZE_N` / `num_warps` / `num_stages` to raise SM fill from 48% toward 100%. This has a known ceiling — a config cannot fix the 16× M-padding, because 16 is Triton's MMA minimum — but it directly targets the occupancy half of the diagnosis and costs almost nothing. **Stage 2 (only if Stage 1 cannot get below ~10 ms):** at batch 1 this is not a GEMM, it is ten independent 2-bit GEMVs per layer, and it wants a GEMV kernel (no MMA, no M padding, split-K for SM fill), which is an **L** effort.

Two hard constraints for whoever writes that kernel, both established by adversarial verification: `BLOCK_SIZE_K` cannot exceed **128** for `down_proj` (K=640, and `down_config` falls through to the shared `config` at `fused_moe.py:855`, so one block size drives both GEMMs); and `fused_moe_triton_kernels.py:278`'s `b = tl.load(b_ptrs)` is **unmasked** — benign today only because VRAM tolerates OOB reads that a masked `b_scale` later zeroes.

### 7. CUDA graphs — real, but smaller than advertised and gated on items 1, 2, 5

The dossier's headline "1.5× → 24 tok/s, floor 25.7" was refuted and should not be quoted. What survives:

- **The only genuine capture blocker on the decode path is the PLE mmap gather** (`qwen4_exp.py:846-853`). Everything else is clean: the expert streamer takes the sync-free branch at bs·top_k ≤ 64; the offloader hook is pure Python; attention/mamba metadata planning already runs out-of-graph; the QSA `.item()` calls are capture-guarded; no `@triton.autotune` anywhere.
- **BCG cannot be enabled for decode as shipped.** The decode runner captures the *outer* `model.forward`, which returns a `LogitsProcessorOutput` dataclass; `BreakableCudaGraphBackend._alloc_full_buffer` (`breakable_cuda_graph_backend.py:160-177`) understands only Tensor/PPProxyTensors/tuple/list and raises `TypeError` at `:177`. BCG is a prefill-only backend in this checkout.
- **The better route needs no BCG at all.** The PLE row indices depend only on `input_ids` and pool n-gram state (`qwen4_exp.py:706-724`, `:1760-1769`) — *not* on any layer's activations. So the whole hash → 16 row ids → NVMe read → H2D chain can be hoisted into the pre-replay host step and written into a static device buffer. One `cudaGraphLaunch` per token, no mid-model break, no GPU bubble.
- **Capture changes the GPU schedule in an unmeasured direction.** `qwen3_5.py:631-642` splits the GDN input projections across two streams *only* under `get_is_capture_mode()` (all 36 DeltaNet layers) and `qwen4_exp.py:1621-1626` runs the QSA indexer on `alt_stream` *only* under capture (all 12 QSA layers). These paths have never executed on this build. So "38.9 ms is the floor" is not established.
- **There is no VRAM for the graph pool.** 460 MiB free right now (verified this session). Capture will OOM unless `--mem-fraction-static` drops, which costs KV or expert residency.

And critically: **items 1, 2 and 5 are not additive with graphs — graphs subsume them.** Do the cheap host fixes first because they are hours of work with immediate payoff; but when sizing the graph project, size it against the *post-fix* host gap (~20 ms, or ~18 ms after the overlap-schedule flag), not the current ~26–32 ms.

*Experiment first (2 minutes):* run `--cuda-graph-backend-decode breakable --cuda-graph-bs-decode 1` and observe whether the predicted `TypeError` at `breakable_cuda_graph_backend.py:177` actually fires. That single test either confirms BCG is prefill-only or invalidates the whole route-A analysis.

### 8. Requantize the dense BF16 groups — the largest approximation on the table

Five groups totalling 5023 MB/token read at 440–610 GB/s are genuinely bandwidth-bound, so halving the bytes roughly halves the time: **~4.5–5 ms of 38.9 ms GPU busy (12–13%)** at INT8, ~7.4 ms at INT4. The Marlin INT8 path is already in the build and already achieving 494 GB/s on the GDN in-projections, so the kernel risk is near zero.

The effort and risk are not in the kernel — they are in the recipe. `extra_config` pins these to 16 bits *deliberately*, and that decision was made against a 2-bit expert baseline for accuracy reasons, never revisited against decode cost, because decode cost was never computed. Requantizing means a new AutoRound run against a **new** output directory (`~/quant/out/` is sealed and must not be touched) — roughly 3 days of compute — plus perplexity validation.

Priority order within the group, by MB/token per unit of accuracy risk: `linear_attn.out_proj` + `o_proj` (1510 MB) and the hyper-connection mixes (1258 MB) first; `lm_head` (1271 MB in a single 2.08 ms GEMV — the largest individual kernel invocation in the entire step, at 610 GB/s and therefore at roofline) is a big prize but output-layer quantization is the most quality-sensitive; `mlp.gate` routers (126 MB) I would leave at 16 bits, since the measured top-10 margin is only 0.045–0.065 sd and 55–70% of samples sit below 0.05 sd — routing is already razor-thin and does not want more noise.

*Experiment first (a day, CPU only):* simulate INT8/INT4 round-trip on each group's tensors from the sealed checkpoint and measure relative Frobenius error per group; then run perplexity on the *existing* deployment with one group at a time replaced by its fake-quantized version. Kill any group whose degradation is disproportionate before spending 3 days of GPU.

### 9–11. Smaller items, honestly sized

**int32 view in the gather.** AOT compilation showed the kernel emits `ld.global.b8`/`st.global.b8` (32 B per warp request); an int32 view yields `b32` (128 B). All four `row_bytes` are divisible by 4 and by BLOCK=1024, and pointers are ≥512 B aligned, so it is safe and it is three lines. But the kernel is already at ~83% of the demonstrated link ceiling, so decode gain is capped at ~0.8 ms and is **unmeasured**. Free A/B: prefill is ~100% gather-bound, so if prefill tok/s does not move, the link — not the kernel — is the wall, and this dies.

**Frequency-ranked expert residency.** The offload split has no policy at all: `offloader.py:96` walks layers 0→47 and offloads until a byte counter trips (`:107-108`, `:128-132`), so layers 0–29 stream from host and 31–47 sit in VRAM, chosen by *position*. Residency by router hotness instead of layer index would be strictly better if routing is skewed — but nobody measured the skew, and it cannot be measured without instrumenting a live run. Note the built-in tool is broken for this model: `expert_distribution.py:397` hardcodes `_TOP_K_NUM = 8` and this model is top-10, so `_DetailSinglePassGatherer` raises. Prerequisite measurement: per-layer expert frequency histogram and the coverage curve at M = 64/128/209/256 resident experts (209 is the real budget: 13.1 GB ÷ 1.3056 MB ÷ 48 layers). That single curve decides whether the gather-kernel rewrite (per-row `(base_ptr, index)` indirection) is worth it. Also requires a mixed-residency gather, which is a real kernel change.

**VRAM reclaim → lower `--cpu-offload-gb`.** Each GiB freed buys ~0.75 more resident layers ≈ 10 MB/token less over PCIe ≈ ~0.2 ms. The mamba flag (0.86 GiB) is in item 4. `embed_tokens` (1.184 GiB BF16 [248320,2560], untied, read once per forward for 5 KB/token) is real but was **refuted as a "clean reclaim"** — see §3. The 668 MB of MoE staging is genuinely needed at full 512-row width during prefill.

**Speculative decoding with the MTP head — nobody raised it.** The checkpoint contains a complete MTP block (4.852 GiB, 1571 tensors, verified this session), skipped unconditionally at load by `qwen4_exp.py:2088-2089 if "mtp" in name: continue`, and `qwen4_exp_mtp.py` exists in the codebase. Because ~13.4 ms of every step is batch-invariant (dense BF16 + Marlin + all 1846 launches), accepting *k* tokens per step amortises it exactly the way batching would — but for a *single* stream, which `--max-running-requests 1` cannot. Rough sketch at 2 accepted tokens/step: (13.4 + 2×23.2)/2 ≈ 30 ms/token vs 38.9, ~30% at perfect acceptance, less at realistic 60–70%. The blocker is that the MTP experts are unquantized BF16 (4.69 GiB) with no VRAM and no host RAM to hold them, so this needs its own quantization pass. Filed as an unexplored direction with a plausible sizing, not a recommendation.

---

## 2. Dependencies — what must happen first because it moves the baseline

1. **Re-profile without `with_stack` / `record_shapes`.** One `/start_profile` call on the live server, no restart, no GPU allocation. Settles the idle-fraction argument three threads spent pages on and confirms the kernel split holds without 139,699 Python events per step. *Everything* in the plan is sized off this. (Flag to the owner: this is a POST, not a GET — it needs sign-off under the do-not-disturb rule.)
2. **The flag sweep (item 4)** — four restarts. These change decode wall time by up to 2.4 ms and prefill by up to 4×, so measuring any code change against the pre-sweep baseline gives the wrong number.
3. **Items 1, 2, 5 before sizing CUDA graphs.** Graphs subsume them; the graph project must be justified against the residual host gap, not the current one.
4. **The int2 config sweep (item 6, stage 1) before committing to a GEMV kernel.** If a tuned JSON gets the MoE kernel from 13.7 to 8 ms, the kernel rewrite's incremental value halves.
5. **A server-down window** is required for exactly one thing: the PCIe read-efficiency-vs-run-length microbenchmark (§4). Do not spend it on anything else.
6. **Expert-frequency instrumentation before item 10.** No skew measurement, no residency policy.

---

## 3. Exact vs approximate

**Purely systems changes — output must be bit-identical, and that is the acceptance test.** Items 1, 2, 3, 4, 5, 6, 7, 9, 11. Validate each with a greedy 200-token generation diffed against the pre-change ids at fixed seed and prompt. Two carry a subtlety: item 3 changes the expert *numbering* passed to the kernel (E=512 with original ids instead of E=10 with renumbered ids) — verified equivalent by the `moe_align` cap, no atomics, and int64 stride arithmetic, but it is the one "exact" change where a diff is genuinely load-bearing. Item 6, if it becomes a new GEMV kernel, changes floating-point accumulation order and will *not* be bit-identical; hold it to a tight numerical tolerance against the Triton reference instead, per-layer.

**Approximation — needs quality validation, not a diff.** Item 8 (dense INT8/INT4 requantization) and item 12 (MTP speculative decoding, which is exact in expectation only if verification is exact — check whether SGLang's acceptance test is exact or thresholded before claiming it). Also `--mamba-ssm-dtype bfloat16` (0.16 GiB, recurrent-state precision across 36 layers — I would not) and `--context-length 16384` (capability loss, not approximation). Bit-plane residency was approximation *and* a dead end; see below.

---

## 4. Dead ends — say these out loud so nobody re-derives them

**Bit-plane expert residency (MSB in VRAM, LSB on host).** Rejected on three independent grounds, any one sufficient. The decisive one is economic and does not depend on accuracy at all: **MSB and LSB of the same expert have identical VRAM cost (0.6144 MB) and identical marginal traffic saving, so the allocation problem is a *linear* knapsack whose value density is the same for both bits** — whole-expert residency is never worse. Measured identical (398.5 MB/token) under uniform routing and 92–275 MB/token *worse* under Zipf skew, because skew rewards concentration. Second, the exact "never degrade" variant does not fit: MSB-of-all-experts needs 15.82 GiB against a 10.88 GiB budget, short by 4.94 GiB, and the MTP weights that would have covered it are already skipped at load. Third, the degrading variant costs **54% relative weight error**, which is a *floor*: the MSB split {0,1}|{2,3} scores 0.5398 against 0.8421 and 0.6214 for the other two threshold partitions, i.e. it already coincides with the Lloyd–Max 2-level solution. The algebra was correct and elegant — at 4 bits this would be worth building; at 2 bits the low bit is half the model.

**Replacing the gather with `cudaMemcpyAsync` / `cudaMemcpy2DAsync` / `cudaMemcpyBatchAsync`.** All need the expert ids on the *host*, costing ~30 D2H syncs per token; the per-row variant is 1220 launches/token at 2–5 µs each; 2D copy cannot express arbitrary non-contiguous row selection. The existing docstring at `expert_stream.py:41-48` already got this right.

**Chasing pinning.** The host memory is provably pinned: `/proc/vmstat` shows `nr_foll_pin_acquired − nr_foll_pin_released` = 6,410,057 pages = 24.45 GiB outstanding, matching the 24.45 GiB of `/dev/zero (deleted) rw-s` mappings exactly. (`VmLck=0` is not counter-evidence — the driver pins via `pin_user_pages`, not `mlock`.) There is no one-line fix here.

**Chasing gather bandwidth.** 50.6 GB/s decode and 46.3 GB/s prefill against a demonstrated 61 GB/s ceiling. The kernel is not the wall.

**Cross-layer expert correlation tables (ST-MoE CCT, FATE's cross-layer rule).** Measured: feeding the same input to two different layers' routers gives top-10 overlap of **0.019–0.020 at every layer distance d=1..12, against a chance level of 10/512 = 0.0195**. Expert indices carry no cross-layer identity structure in this model. The free version is worthless.

**"Predict all 48 layers' routing from layer 0."** Dominated. Drift to layer 40+ is near-total; at ε=1.4 even fetching 256 of 512 experts reaches only 0.93 recall.

**Reversing the offload order (resident = layers 0–17) as a standalone win.** Refuted: with no prefetch stream, host-resident layers cost identical bytes and identical time regardless of index. It matters only *after* someone implements prefetch.

**"Delete the gather and read experts direct from host at the current BLOCK_SIZE_K."** The direction is unresolved, not settled (see §5), but the specific prescription — raise `BLOCK_SIZE_K` to 256/512 — is dead: 640 % 256 = 128, so `down_proj` (a third of the traffic) cannot use it, one config drives both GEMMs, and the unmasked B load would go out of bounds into pinned host memory. The accompanying "transfer hides under compute" argument was circular — both sides of it are the same 399.9 MB divided by rates 1.2% apart.

**Two claims that were "wins" but are too small to schedule.** Restoring E=512 re-enables the `moe_align_small_numel` fast path — but the currently-selected kernel costs **0.128 ms total across 48 launches**, so that is the entire ceiling. And sampling: the actual sampler is an `ArgMax` reduce at **5.5 µs** — `lm_head` is 380× more expensive than sampling, so there is nothing there.

---

## 5. Load-bearing unknowns, ranked by cheapness of resolution

1. **Idle fraction and the kernel split without profiler distortion.** One `/start_profile` with stack/shapes off. Resolves an argument that consumed three threads. Prediction to check: kernel count and ~38.9 ms busy hold steady while step wall drops from ~150 ms toward ~65 ms.
2. **Does BCG actually die at `breakable_cuda_graph_backend.py:177`?** Two minutes. Either confirms BCG is prefill-only or invalidates a whole recommendation.
3. **Real PLE row-fetch cost in situ.** One restart + a wall-clock histogram at `qwen4_exp.py:849`. Decides whether item 2 is worth 2.4 ms or 0.2 ms.
4. **The PCIe read-efficiency-vs-run-length curve.** The *only* thing needing a server-down window. Four threads produced ~15 GB/s, ~31 GB/s, ~400 ms and "flat" as candidate answers for what a direct-from-host B read would cost, and the entire spread between "direct-from-host wins 9.5 ms" and "loses 8.8 ms" rests on it. Run it standalone: a Triton kernel reading pinned host memory at a 640-byte stride with 1/16/32/64/128/256/640-byte contiguous runs. If bandwidth is flat across run lengths, sysmem reads are L2-cached and the direct path opens up; if it collapses, the gather stays.
5. **Whether the 38.9 ms is a floor under capture.** Unmeasurable without attempting capture, because `qwen3_5.py:631-642` and `qwen4_exp.py:1621-1626` switch 48 layers to dual-stream schedules only under capture mode.
6. **Expert routing skew.** Requires live instrumentation; the built-in recorder is broken for top-10. Gates item 10 entirely.
7. **Long-context regime.** One `/start_profile` at 30k context. The entire plan may be optimising 1.5% of the advertised window.

---

## 6. What I would actually do tomorrow

**Morning, no code:** re-profile clean (unknown 1); test BCG's `TypeError` (unknown 2); run the four-restart flag sweep (item 4). By lunch you have a correct baseline and, if the overlap-schedule and chunked-prefill flags behave as the trace predicts, ~2.4 ms of decode and 2–4× of prefill already banked for zero engineering.

**Afternoon, ~150 lines total:** offloader hook guard (item 1), skip-gather-for-GPU-resident-layers (item 3), `arange`/cast memoization and `positions.max()` hoist (item 5). Validate all three with a greedy-diff. Expected combined: ~4–7 ms/token, i.e. **15.5 → ~17–18 tok/s**, with the whole package being bit-identical and individually revertible.

**Day two:** PLE `os.pread` (item 2, +2.4 ms → ~18–19 tok/s) and the int2 config sweep (item 6 stage 1). The config sweep is the highest-variance cheap experiment in the plan — it is the only shot at the 13.7 ms line item that does not cost a week.

**Then decide between the two large projects on measured evidence rather than the dossier's priors:** a batch-1 int2 GEMV kernel (up to 11.6 ms, exact, L) versus CUDA graphs (most of the residual ~18–20 ms host gap, exact, L, VRAM-constrained, with an unmeasured dual-stream schedule change hiding inside it). The dossier assumed graphs were the answer because it believed the 78% idle figure. Corrected to ~40–46% idle, and with 5–7 ms of that gap removable by hand for a day's work, **the MoE kernel is the better bet** — it is the largest single line item, it is additive with graphs rather than subsumed by them, and it is the one place where this machine is demonstrably running 11× below what its own neighbouring kernels achieve.

Everything above is exact. The only approximation worth queuing is the dense-BF16 requantization (item 8, ~12–13%), and it should not start until the fake-quant error survey says which of the five groups can take it.