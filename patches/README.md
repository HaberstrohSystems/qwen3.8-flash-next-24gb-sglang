# SGLang patches

Each script edits the patched SGLang checkout (`$SGLANG`, default `~/quant/sglang`; the `SG = ...`
line at the top of the file) by exact string replacement and supports `apply`, `revert` and `--check`; the module
docstring of every script explains its mechanism and its environment variables. `--check` only reads
the tree. The tree they were measured on is SGLang at commit `73a255206f` ("Introduce Qwen 3.8 Flash
Next") plus the base patch.

## Base layer

`base/sglang-qwen4exp-2bit.patch` is the pre-campaign base patch (the 32k-context state of `docs/WRITEUP.md`)
(604 lines across 15 files): the 2-bit unpack in `fused_moe_kernel_gptq_awq`, the 2-bit `moe_wna16`
loader, the MoE-aware `ExpertStreamer` (`expert_stream.py`), the mmap PLE embedding mode, the offloader
fixes, the `qwen_sparse_attention` allowlist, `packed_modules_mapping` propagation and the `conv_weights`
view fix (`docs/WRITEUP.md` sections 3-5). **It is stale relative to the measured tree**: on 2026-09-03
`git diff --stat` against `73a255206f` shows 28 modified files, 2,954 insertions / 89 deletions, plus six
new modules (`expert_elastic.py`, `expert_gemv.py`, `row_arena.py`, `int8_kv_pool.py`, `int4_kv_pool.py`,
`tiered_kv_pool.py`; `fused_moe_triton_kernels.py.ncontig.orig` is a backup written by `ncontig_gemv.py`,
not source). Whether base patch + the layers below reproduces the tree byte for byte has not been
verified; the layers also depend on each other's wording. **The authoritative artifact is the serving
patch `sglang/qwen4exp-serving-73a255206f.patch`** (34 files, +4,155 / -89; SHA-256
`92f669b2525f9c86190825390fafc2b41a28071c398fcb7fd95716fbce744bb5`), produced from the served tree
on 2026-09-03 and verified in a clean worktree of `73a255206f`: all 34 patched files compare equal to
the served tree (`sglang/PATCH_NOTES.md` section 2). It excludes the `.ncontig.orig` backup but
includes the measurement-only `kv_stats` / `kv_fakeq` hooks and the opt-in `ngram_ple` edits that were
applied in the served tree; `sglang/PATCH_NOTES.md` section 5 gives the exact hunks to drop.

## Accepted layers, in apply order

The accepted list is `assets/phase1_state.json` `patches`. Revert in reverse order.

| # | Item | Script | Target files | Purpose | Measured effect | `--check` on the measured tree |
|---|---|---|---|---|---|---|
| 1 | `hook` | `host_fixes.py apply hook` | `srt/utils/offloader.py` | do not install the offloader forward hook when every offloaded parameter of the module is a streamed expert param | part of S5: 15.4 -> 19.4 tok/s (CAMPAIGN.md:260) | APPLIED |
| 2 | `skipgather` | `host_fixes.py apply skipgather` | `layers/quantization/moe_wna16.py` | skip gather + renumber for fully GPU-resident layers | S5 | MISMATCH (anchor overlaid by `ncontig` edit 9 and `elastic` edit 1 — expected) |
| 3 | `memo` | `host_fixes.py apply memo` | `layers/moe/expert_stream.py` | memoize arange/cast tensors in the streamer | S5 | APPLIED |
| 4 | `rope` | `host_fixes.py apply rope` | `attention/qsa/qsa_indexer.py` | hoist `positions.max().item()`, pre-size the rotary cache once per QSA layer | S5 | APPLIED |
| 5 | `ple` | `host_fixes.py apply ple` | `models/qwen4_exp.py` | parallel `os.pread` PLE row fetch for decode (releases the GIL) | S5 | MISMATCH (memmap lines overlaid by `ple_random` — expected) |
| 6 | `ncontig` | `ncontig_gemv.py apply` | `kernels/ops/moe/fused_moe_triton_kernels.py`, `moe_runner/triton_utils/fused_moe.py`, `moe_wna16.py`, `expert_stream.py`, new `layers/moe/expert_gemv.py` | N-contiguous `[E, K/16, N]` int32 layout, word-load int2 prefill kernel, in-place batch-1 GEMV through int64 address tables | S9: 19.4 -> 21.8 decode, 1,443 -> 2,249 prefill (CAMPAIGN.md:286); numerically equivalent per per-layer A/B (:92-96) | edits 0-8 APPLIED, 9 MISMATCH (overlaid by `elastic`); leaves the `.ncontig.orig` backup |
| 7 | `bcg` | `host_fixes.py apply bcg` | `models/qwen4_exp.py` | wrap the mmap PLE forward as an eager break for breakable CUDA graphs | S6c: 21.8 -> 40.0 (CAMPAIGN.md:291) | APPLIED |
| 8 | `bcg2` | `host_fixes.py apply bcg2` | `model_executor/runner_backend/breakable_cuda_graph_backend.py` | teach the backend's structure helpers the `LogitsProcessorOutput` dataclass | S6c | APPLIED |
| 9 | `placement` | `placement.py apply` | `moe_wna16.py`, `expert_stream.py` | frequency-based, memory-neutral placement v3 (`SGLANG_MOE_PLACEMENT`, `_S=184`) | S10: 40.0 -> 48.4 (CAMPAIGN.md:296) | edits 0, 1, 3, 4 APPLIED; edit 2 clean (superseded by `elastic`) |
| 10 | `elastic` | `elastic.py apply` | `moe_wna16.py`; installs `gemv/row_arena.py`, `gemv/expert_elastic.py` into `srt/layers/moe/` | elastic expert residency in VMM row arenas, live S control file (`SGLANG_MOE_ELASTIC=1`, `_PIN_MB`, `_FILL_MB`, `_RESERVE_ROWS`, `_CTL`) | S13: 56.2 / 2,335 exact (CAMPAIGN.md:319) | all APPLIED, both files installed |
| 11 | `kv_lazy` | `kv_lazy.py apply` | `mem_cache/kv_vmm_backing.py`, `memory_pool.py`, `kv_cache_configurator.py`, `allocator/paged.py`, `allocator/token.py` | lazy VMM KV backing, idle release to a floor, watermark, admission cap (`SGLANG_KV_LAZY=1`, `_TOKENS`, `_SAFETY`, `_FLOOR=4096`, `_MARGIN=2048`, `_HEADROOM_MB=1536`) | S14 (CAMPAIGN.md:330), S15 128k (:336), S18 262,144 VA (:372) | all 12 APPLIED |
| 12 | `ple_random` | `ple_random.py apply` | `models/qwen4_exp.py` | `MADV_RANDOM` / `POSIX_FADV_RANDOM` on the PLE table, `POSIX_FADV_DONTNEED` after bulk gathers | no numerics (CAMPAIGN.md:341, :408) | all APPLIED |
| 13 | `kv_fp8` | `kv_fp8.py apply` | `attention/qsa/sparse_attn.py`, `attention/qwen_sparse_attn_backend.py`, `memory_pool.py` | fp8_e4m3 KV read path for QSA; anchor layer for `kv_int8` | mode at speed parity (CAMPAIGN.md:353); 1-token prompt crash open (:355) | edit 0 clean, 1-3 MISMATCH (overlaid by `kv_int8` — expected per docstring) |
| 14 | `kv_int8` | `kv_int8.py apply` | same + `kv_cache_dtype.py`, `kv_cache_configurator.py`, `pool_configurator.py`, `server_args.py`, new `mem_cache/int8_kv_pool.py` | INT8-G64 KV pool, fused quantize+scatter, dequant in both gather sites (`--kv-cache-dtype int8_g64`) | S17 (CAMPAIGN.md:370); 162k ceiling (:372) | overlaid edits MISMATCH under `kv_int4`; non-overlaid intact |
| 15 | `kv_int4` | `kv_int4.py apply` | same, new `mem_cache/int4_kv_pool.py` | INT4-G32 KV pool, `kv_bits` dispatch (`--kv-cache-dtype int4_g32`) | S19 (CAMPAIGN.md:406-409) | overlaid edits MISMATCH under `kv_tiers`; intact list printed by `kv_tiers.py --check` |
| 16 | `kv_tiers` | `kv_tiers.py apply` | same, new `mem_cache/tiered_kv_pool.py` | INT8 ring of `SGLANG_KV_TIERS_W=8192` slots over the INT4 pool, owner table, device-side tier test (`--kv-cache-dtype int8ring_int4`, the default) | S21 (CAMPAIGN.md:413) | all 12 edits + new file APPLIED |

Layering constraints (from the docstrings): `placement` on top of `ncontig`; `elastic` on top of
`placement`; `kv_fp8 < kv_int8 < kv_int4 < kv_tiers` — each `apply()` refuses unless its prerequisite is
fully applied, and each `revert()` refuses while a later layer is applied. `--check` of an overlaid layer
prints its overlaid edits as MISMATCH and a `P-line` naming the overlay; that is the expected state, not
damage.

## Not in the accepted list

| Item | Script | Status | Notes |
|---|---|---|---|
| `ngram_ple` | `ngram_ple.py` | **opt-in** (currently applied in the measured tree, inert without `--speculative-algorithm NGRAM`) | NGRAM speculation with the PLE: guard drop, linear draft chain, PLE-state commit after verify (also a real bug for MTP topk = 1 under replayssm), startup self-check. Lossless within the decode-vs-prefill floor; accept length 1.11-1.27 -> 22-25 tok/s vs 56 (CAMPAIGN.md:387). Serve flags in the docstring. |
| `kv_stats` | `kv_stats.py` | **measurement-only** (edit 0 applied, edit 1 clean) | per-layer/head K/V statistics at `set_kv_buffer` (`SGLANG_KV_STATS=/path.pt`), the design data for the KV panel |
| `kv_fakeq` | `kv_fakeq.py` | **measurement-only** (applied, env-gated) | fake quantization on the bf16 pool's write path (`SGLANG_KV_FAKEQ=int8_g64|int8_tok|int8_g32|e4m3|int4_g32|...|zero|noise`); must run with `--kv-cache-dtype auto` (ERRATUM, CAMPAIGN.md:394). The smoothing statistics file is read from `SGLANG_KV_STATS_FILE` only; the served tree (and the flat patch) carried the measuring host's default path in that line, so `kv_fakeq.py --check` against the served tree reports edit 1 as MISMATCH |
| `dump`, `moedump` | `host_fixes.py` items | **measurement-only** (clean, not applied) | routing dump (`SGLANG_ROUTE_DUMP`, consumed by `tools/expert_freq.py`) and per-layer MoE I/O dumps for kernel A/Bs |
| `kv_paged_prefix` | `kv_paged_prefix.py` | **rejected** (clean, not applied) | paged prefix-chunk prefill kernel reading the int8/int4/tiered pools in-kernel; correct within 2 row-ulps, but 6.1 ms (tiered) / 3.2 (int8) / 4.2 (int4) vs 1.4 ms materialised per head-layer at prefix 60k (CAMPAIGN.md:439; `docs/KV_PAGED_PREFIX_PLAN.md` "Outcome") |

## Serving assets the accepted layers need

* `assets/moe_configs/` — tuned Triton configs for `dtype=int2_w2a16` / `int2_w2a16_down` at E = 10
  (decode) and E = 512 (prefill), `SGLANG_MOE_CONFIG_DIR` (S0b, +13 % decode, CAMPAIGN.md:252).
* `assets/expert_freq.pt` — routing-mass histogram `[48, 512]`, `SGLANG_MOE_PLACEMENT`.
* a writable control file for `SGLANG_MOE_ELASTIC_CTL` (contents `S 184`; the server writes
  `<file>.status`).

## Rules

* No server starts while a patch workflow is applying or reverting (CAMPAIGN.md:420, :435-436).
* Revert layered patches in reverse; `scripts/phase1.py` does this and aborts on a refusal.
* The model-side change is not a patch: `scripts/requant_int8.py` re-packed 85 dense
  tensors (`linear_attn.out_proj` x36, `self_attn.{q,k,v,o}_proj` x12 each, `lm_head`) to INT8 g128 RTN
  into the served directory (S11b, CAMPAIGN.md:298-300).
