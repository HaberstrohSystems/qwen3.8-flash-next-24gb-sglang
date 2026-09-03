# Upstream PR draft: serving a 176B / 6B-active MoE at 2.572 bpw on one 24 GB GPU

Draft for the SGLang maintainers, written 2026-09-03 against base commit
`73a255206f916366c8d26d4022f82ddfb0ab558d` ("Introduce Qwen 3.8 Flash Next"; the commit is
public on `github.com/sgl-project/sglang`: 95 files, +18,543/-84, nine co-authors -- checked
2026-09-03; the page does not say on which branch it sits). The flat patch is
`qwen4exp-serving-73a255206f.patch` (this directory; 34 files, +4155/-89); `PATCH_NOTES.md` next
to it has the per-file map, the flags and the verification. Every figure below names its source
in this repository (paths relative to its root): `CAMPAIGN.md` = `docs/CAMPAIGN.md` (dated,
append-only log; `:N` is a line number, `HH:MM` a dated entry of 2026-09-02 unless stated),
`state` = `assets/phase1_state.json`, `WRITEUP.md` = `docs/WRITEUP.md`.

Upstream status checked 2026-09-03: `python/sglang/srt/layers/quantization/moe_wna16.py` on `main`
still limits `num_bits` to 4 and 8 (`"num_bits must be 4 or 8"`, `num_bits in [4, 8]`,
`assert layer.quant_config.weight_bits in [4, 8]`) and contains no expert streaming, so none of
the MoE part has landed upstream in another form as far as that file shows. The other files
were not compared against `main`.

## Summary

Qwen3.8-Flash-Next (Qwen4-Exp: 176B total / 6B active, 48 layers = 36 GatedDeltaNet + 12 Qwen
sparse attention (QSA), 512 experts top-10, a 51 GB PLE n-gram embedding table, an MTP head)
quantized with AutoRound to 2-bit experts (2.572 bpw, `WRITEUP.md:59`) runs on one
RTX PRO 4000 Blackwell (24 GB, sm_120) with 32 GB host RAM. The base commit cannot load a 2-bit
MoE at all (`WRITEUP.md:159`). With the patch:

| | first working state (2026-09-01, base patch) | accepted state (2026-09-02/03) | source |
|---|---|---|---|
| decode | 15.5 tok/s | 54-57 tok/s (56.2 / 54.3 / 56.8 / 55.7 / 54.5 at context 101 / 421 / 1701 / 6821 / 10001) | CAMPAIGN.md:20; state `accepted`; `docs/logs/tiers-validate.log:14-18` |
| prefill at 10k | ~1,158 tok/s | 2,271 tok/s | CAMPAIGN.md:20, :413 |
| context | 32,768 | 262,144 tokens of address space; 257,905-token prompt: 171.0 s prefill (1,508 tok/s), decode 51.8; needles 5/5 at 41k and 248k | CAMPAIGN.md:413; `tiers-validate.log:21, :30-34` |
| KV bytes/token (12 QSA layers) | 24 KB (bf16) | 7,308 B at 256k (int8 ring of 8192 slots over int4 rows) | `patches/kv_tiers.py` docstring |
| long-text NLL vs a bf16 KV cache | -- | -0.0001 nats over 8,561 positions (int8 alone +0.001, int4 alone +0.009) | CAMPAIGN.md:413, :406 |
| PCIe traffic per decoded token | 26 GB -> 0.31 GB (base patch) | expert reads through int64 address tables, 51 GB/s from pinned host | WRITEUP.md:11-16; CAMPAIGN.md:196-204 |

All speed numbers are single-request streaming measurements (`tools/bench_speed.py`,
200 streamed tokens; CAMPAIGN.md:306). No standard-benchmark score is published for the 2-bit
model (`docs/HISTORY.md`, "Benchmarks"); the quality evidence is the internal protocol (short
held-out NLL, long-text NLL ladder, logprob oracle, needle retrieval) quoted per PR below.

The work splits into pieces that are general (any offloaded model, any MoE with packed expert
rows, any VMM-backed KV pool) and pieces that are specific to Qwen4-Exp (its QSA gather sites,
its PLE table). The next section is explicit about which is which.

## What is general and what is Qwen4-Exp-specific

| component | files (`python/sglang/srt/...` unless noted) | general? | notes |
|---|---|---|---|
| Offloader: expert-only offload under `SGLANG_MOE_EXPERT_STREAM=1`, `tie_weights=False` for `functional_call`, hard reference to the pinned storage, skip the forward hook when every offloaded parameter is a streamed expert | `utils/offloader.py` (+67/-3) | yes for any model under `--cpu-offload-gb`; the expert predicate is a name suffix list (`w13_qweight`, `w2_qweight`, `w13_scales`, `w2_scales`, `w13_qzeros`, `w2_qzeros`) | WRITEUP.md finding 5 and 9 (`WRITEUP.md:158-168`); `hook` item, S5 (CAMPAIGN.md:260) |
| `GemmaRMSNorm.gemma_weight` re-derived on the parameter's device | `layers/layernorm.py` (+22/-4) | yes, any `GemmaRMSNorm` user under offload | finding 4 |
| `conv_weights` derived on every access instead of a construction-time `.view()` | `models/qwen3_5.py`, `layers/radix_linear_attention.py` | yes for the GDN family; tensors/tuples pass through, so KDA / ShortConv / Lightning are untouched | finding 8, silent NaN under offload (`WRITEUP.md:172-207`) |
| `qwen_sparse_attention` accepted as a layer type; `packed_modules_mapping` propagated to the quant config; `Qwen4ExpForConditionalGeneration` in the `language_model_only` allowlist | `configs/qwen4_exp.py`, `models/qwen4_exp.py`, `server_args.py` | model-specific, trivial | findings 2, 3, 7 |
| Breakable CUDA graph backend understands `LogitsProcessorOutput` in its four structure helpers | `model_executor/runner_backend/breakable_cuda_graph_backend.py` (+35) | yes: capture died with "Unsupported BCG output type" for any graph body that returns the dataclass | `bcg2` item; S6c 21.8 -> 40.0 tok/s (CAMPAIGN.md:291) |
| fp8 KV write path: skip `div_` by a unit scale, saturate to +-448 before the e4m3 cast | `mem_cache/memory_pool.py` (two hunks) | yes: the fp32/bf16 -> e4m3fn cast returns NaN beyond the range | `patches/kv_fp8.py` docstring |
| 2-bit (W2A16) `moe_wna16`: loader (symmetric only), Triton unpack (4 values per byte, zero point 2), `use_int2_w2a16` plumbing, config dtype name `int2_w2a16` | `layers/quantization/moe_wna16.py`, `kernels/ops/moe/fused_moe_triton_kernels.py`, `layers/moe/moe_runner/triton.py`, `.../triton_utils/fused_moe.py`, `.../fused_moe_triton_config.py`, `test/manual/test_triton_moe_wna16.py` | yes for any symmetric 2-bit GPTQ/AutoRound MoE checkpoint; asymmetric `qzeros` raise `NotImplementedError` by design | `WRITEUP.md:356-379` (bit-identical vs torch on the exact `tl` expressions) |
| MoE-aware expert streaming: only the routed experts cross PCIe, one shared staging buffer, ids renumbered onto it | `layers/moe/expert_stream.py` (new, 182 lines) | yes for offloaded `moe_wna16` experts | finding 9: 26 GB -> 0.63 GB per forward (`WRITEUP.md:210-254`) |
| N-contiguous `[E, K/16, N]` int32 expert layout (re-laid once after loading, derived from the tensor dtype so the custom-op schema is untouched), word-load prefill kernel `fused_moe_kernel_gptq_awq_word`, in-place batch-1 GEMV through int64 address tables | `moe_wna16.py`, `fused_moe_triton_kernels.py`, `fused_moe.py`, `layers/moe/expert_gemv.py` (new, 109) | yes for 2-bit `moe_wna16` | 320 GB/s from device, 51 GB/s from pinned host (CAMPAIGN.md:196-204); S9 (CAMPAIGN.md:286) |
| Frequency placement: hottest `S` experts of every layer on the GPU, cold rows in donated pinned slots, memory-neutral | `moe_wna16.py` (`_collect_for_placement`, `_run_placement`), `expert_stream.py` (table-driven gather) | yes, given a routing-mass profile `[L, E]` (`tools/expert_freq.py` builds one from a routing dump) | S10 40.0 -> 48.4 tok/s (CAMPAIGN.md:296); top-171 experts per layer cover 82 % of routing mass (CAMPAIGN.md:206-216) |
| Elastic expert cache: CUDA-VMM row arenas in rank order, 2 MiB tail unmap, addresses fixed for CUDA graphs, live `S` control file | `layers/moe/row_arena.py` (new, 215), `layers/moe/expert_elastic.py` (new, 376) | `row_arena.py` fully general; `expert_elastic.py` general in design, wired to the `moe_wna16` word layout and `expert_gemv.make_tables` | S13 exact, 56.2 tok/s (CAMPAIGN.md:319); the S dial buys +2-3 % decode per ~1.7 GB, its value is VRAM on demand (CAMPAIGN.md:320) |
| Lazy VMM KV backing: address space reserved for `max_total_tokens`, backed in 2048-token steps as pages are handed out, released to a 4096-token floor at pool idle, driver-free headroom watermark, admission cap | `mem_cache/kv_vmm_backing.py`, `memory_pool.py`, `allocator/paged.py`, `allocator/token.py`, `kv_cache_configurator.py` | yes: builds on the existing `KvVmmBufferOwner` (post-capture sizing), which only committed monotonically; the "shrink the expert cache first" step is an import of `ExpertElastic` inside `lazy_ensure` | S14 exact (CAMPAIGN.md:329-330), 128k (:336), watermark (:337), 68,905-token bf16 prompt (:348), admission cap (:350) |
| INT8-G64 / INT4-G32 / tiered KV pools: payload plus fp16 group scales as extra `KvBufferDesc`s on the VMM owner, fused quantize+scatter at write | `mem_cache/int8_kv_pool.py`, `int4_kv_pool.py`, `tiered_kv_pool.py` (new), `pool_configurator.py`, `kv_cache_dtype.py`, `kv_cache_configurator.py`, `server_args.py` | pool classes general; the write kernels live in `qsa/sparse_attn.py` | 12.4 / 6.75 / 7.3 KB per token (docstrings); no attention kernel is touched |
| Dequant-on-gather read path for fp8, int8, int4 and tiered pools (decode/verify compaction into the FA2 scratch; prefix-chunk row gather) | `layers/attention/qsa/sparse_attn.py` (+1082), `layers/attention/qwen_sparse_attn_backend.py` (+113/-20) | **Qwen4-Exp-specific**: QSA's two gather sites are where dequantization is free; other backends would need their own read path | S17 / S19 / S21 (CAMPAIGN.md:370, :406, :413) |
| PLE table served from an mmap'd file (`SGLANG_QWEN4_PLE_MMAP`), parallel `pread` for decode, `MADV_RANDOM` / `POSIX_FADV_RANDOM` / `POSIX_FADV_DONTNEED`, the PLE lookup as the eager break of breakable decode graphs; QSA indexer `positions.max().item()` hoist | `models/qwen4_exp.py` (+203/-12), `layers/attention/qsa/qsa_indexer.py` | **Qwen4-Exp-specific**; the pattern (a huge embedding table served from a file and wrapped as the graph's eager break) may interest other n-gram-table models | `WRITEUP.md:285-294`; `ple`/`bcg` items (S5, S6c); CAMPAIGN.md:341, :408 |
| NGRAM speculation with the PLE: guard drop, forced linear draft chain, PLE state commit after verify | `qwen4_exp.py`, `speculative/ngram_worker.py` (+116), `speculative/spec_utils.py` (+31) | the `spec_utils.py` commit is a bug fix that also affects MTP top-k 1 under `--enable-linear-replayssm-spec` (CAMPAIGN.md:379); `_linearize_chain` is general for any hybrid backend that assumes a chain | correct but no speed win on this model: accept length 1.11-1.27 (CAMPAIGN.md:387); opt-in |

Not proposed upstream, although they are in the flat patch (locations in `PATCH_NOTES.md`
section 5): the measurement hooks `SGLANG_KV_STATS` / `SGLANG_KV_FAKEQ` in `memory_pool.py`,
the `SGLANG_NAN_TRACE` print in `radix_linear_attention.py`, a stray unused `import os as _os`
in `gdn_backend.py` (the file's whole diff), a duplicated `import os as _os` in `qwen3_5.py`,
two hunks in `qwen4_exp.py` that only delete a blank line, and the rejected paged prefix-chunk
kernel, which is not in the tree at all (`patches/kv_paged_prefix.py`: 4-5x slower than
gather + packed kernel, CAMPAIGN.md:439-442).

## Suggested split into reviewable PRs

### PR 1 -- Qwen4-Exp under CPU offload: correctness fixes, breakable-graph support, mmap PLE table

Files: `utils/offloader.py`, `layers/layernorm.py`, `models/qwen3_5.py`,
`layers/radix_linear_attention.py` (without the `SGLANG_NAN_TRACE` print),
`configs/qwen4_exp.py`, `server_args.py` (allowlist entry only), `breakable_cuda_graph_backend.py`,
`mem_cache/memory_pool.py` (the two fp8 unit-scale / saturation hunks only), `qsa/qsa_indexer.py`,
`models/qwen4_exp.py` (mmap embedding, meta-device table creation, pread, madvise, eager-break
wrapper, layer-type entry, `packed_modules_mapping`; without the NGRAM hunks).

Evidence: WRITEUP.md sections 3-4 (nine findings with symptoms, `WRITEUP.md:153-207`); host
fixes S5 15.4 -> 19.4 tok/s (CAMPAIGN.md:260); breakable decode graphs S6c 21.8 -> 40.0 tok/s,
capture 3.97 s / 0.12 GB (CAMPAIGN.md:291 and the dated entry 11:33); PLE readahead fix and
page drop (CAMPAIGN.md:341, :408-409).

Still needed upstream: a unit test for the `conv_weights` property under a simulated
`param.data` swap; a test that the BCG helpers round-trip a `LogitsProcessorOutput`; the mmap
embedding needs a documented file format (`ple.json` carries `file`, `rows` = 320,001,536,
`dim` = 160, `dtype` = `F8_E4M3`, `n_shards`, `rows_per_shard`, `weight_scale`; the table file is
51,200,245,760 bytes) and a server flag instead of `SGLANG_QWEN4_PLE_MMAP`; the `_qsa_ensure_rope`
hoist reads `context_length` from the global server args and should take it from the layer.

### PR 2 -- 2-bit MoE for `moe_wna16`: unpack, expert streaming, N-contiguous layout, in-place GEMV

Files: `kernels/ops/moe/fused_moe_triton_kernels.py`, `moe_runner/triton_utils/fused_moe.py`,
`moe_runner/triton.py`, `moe_runner/triton_utils/fused_moe_triton_config.py`,
`layers/quantization/moe_wna16.py` (loader, streamer hook, layout conversion, GEMV dispatch),
`layers/moe/expert_stream.py`, `layers/moe/expert_gemv.py`, `test/manual/test_triton_moe_wna16.py`;
plus the tuned configs
`E={10,512},N=160,device_name=NVIDIA_RTX_PRO_4000_Blackwell,dtype=int2_w2a16{,_down}.json`
(`assets/moe_configs/configs/triton_3_7_1/`, `SGLANG_MOE_CONFIG_DIR`) relocated into the fused_moe
config tree.

Evidence: kernel expressions bit-identical vs torch and end-to-end vs bf16 dequant (small shapes
bit-identical, large shapes 1e-5 relative, `WRITEUP.md:356-369`); S9 KEPT 19.4 -> 21.8 decode,
1,443 -> 2,249 prefill (CAMPAIGN.md:286) with a per-layer identical-input A/B at 5.1e-5 (M=6) /
1.7e-4 (M=1) relative (dated entry 11:25); tuned int2 configs +13 % decode (S0b,
CAMPAIGN.md:252); GEMV 320 GB/s device / 51 GB/s pinned host (CAMPAIGN.md:196-204); byte-row
table variant rejected at 157 / 14 GB/s (CAMPAIGN.md:218-219).

Still needed upstream: the word-load kernel is a second Triton function
(`fused_moe_kernel_gptq_awq_word`, 294 lines) selected by `B.dtype == torch.int32` -- a
maintainer may prefer a `constexpr` branch of the existing kernel; the layout-by-dtype
convention (int2 + int32 = word layout, int2 + uint8 = byte layout) needs a docstring on
`fused_experts`; the layout conversion re-pins host tensors (`pin_memory()` after the copy),
which doubles the host footprint of a layer transiently; tuned configs exist for one device;
the `.ncontig.orig` backup written by the patch script must not ship (it is not in the flat
patch); the upstream manual test fails at 8 bits on 66 of 96 large shapes before this patch
(`WRITEUP.md:381-385`) and should become a real test with tolerances; asymmetric 2-bit is
refused by design; `SGLANG_MOE_NCONTIG_LAYERS` (a bisection aid) should go.

### PR 3 -- VMM-backed elasticity: expert row arenas and lazy KV backing

Files: `layers/moe/row_arena.py`, `layers/moe/expert_elastic.py`, `moe_wna16.py` (placement
pass, elastic hook, control-file poll in `apply`), `expert_stream.py` (shape-keyed staging,
table-driven gather), `mem_cache/kv_vmm_backing.py` (`uncommit_beyond`, `release_beyond`,
`bytes_per_token`), `memory_pool.py` (`lazy_ensure`, `lazy_release`, lazy allocation in the
classic flow), `allocator/paged.py`, `allocator/token.py`, `kv_cache_configurator.py` (virtual
capacity `SGLANG_KV_LAZY_TOKENS`, admission cap `SGLANG_KV_LAZY_SAFETY`).

Evidence: placement S10 40.0 -> 48.4 tok/s, exact (CAMPAIGN.md:296); elastic S13 56.2 tok/s,
exact (CAMPAIGN.md:319); `row_arena.py` self-test: 2 MiB granularity, shrink returns memory to
the driver (`mem_get_info` confirms), a CUDA graph captured against arena addresses replays after
shrink and regrow (CAMPAIGN.md:304); lazy KV S14 exact, 24 KB/token committed on demand,
release to the 4096-token floor at idle (CAMPAIGN.md:329-330); S15 128k (:336); watermark rule
after the 60k crash (:337); 68,905-token bf16 prompt (:348); admission cap (:350); with the
tiered pool 262,144 tokens of address space with 4096 backed at start
(`docs/logs/tiers-validate.log:3-4`).

Honest limits: on the reference host the expert-cache floor is S = 184 rows per layer because
host RAM (32 GB, 24 GB pinned by 31 offloaded layers) cannot absorb more cold rows, and any
runtime pinned allocation triggers systemd-oomd (CAMPAIGN.md:305, :346-347) -- so `free()`
never pins. The S dial buys only +2-3 % decode per ~1.7 GB (CAMPAIGN.md:320). `kv_lazy`'s idle
release assumes the pool goes fully idle (its docstring reasons with `--max-running-requests 1`,
the published configuration; the concurrent-benchmark restart #23 ran 4, CAMPAIGN.md:454-459).
With 4 concurrent requests and 8 GDN state slots the profiled capacity fell to 260,480 tokens and
the admitted capacity to 200,512 (server log of restart #23, 2026-09-03 00:36:02, not included in
this repository), against 366,976 / 262,144 at the single-request validation
(`docs/logs/tiers-validate.log:3-4`).

Still needed upstream: server args instead of the `SGLANG_MOE_ELASTIC*` / `SGLANG_KV_LAZY*`
environment variables; the control file (`SGLANG_MOE_ELASTIC_CTL`) replaced by an HTTP endpoint;
the expert-cache coupling in `lazy_ensure` made a registered callback rather than an import; a
CI test for `RowArena` (needs a GPU, ~150 MB per its docstring) and for the allocator hooks
with a mocked owner; behaviour with the radix cache enabled (the campaign ran
`--disable-radix-cache`) and under several in-flight requests measured; the scheduler's handling
of `alloc_extend -> None` mid-prefill made graceful instead of relying on the admission cap
(CAMPAIGN.md:349).

### PR 4 -- Quantized KV pools with dequant-on-gather for QSA: fp8 read path, INT8-G64, INT4-G32, tiered

Files: `qsa/sparse_attn.py` (kernels: `_compact_kv_fp8`, `_quant_store_kv_int8`,
`_compact_kv_int8`, `_gather_dequant_rows_int8`, the int4 siblings, `_stamp_ring_owner`,
`_quant_store_kv_tiered`, `_compact_kv_tiered`, `_gather_dequant_rows_tiered` and their
wrappers), `qwen_sparse_attn_backend.py` (`_kv_bits`, `_kv_head_dim`, `_kv_scratch_dtype`,
`_int8_gather_kwargs`, `_kv_tier_kwargs`, the prefix-chunk branch, CPU-fallback guards),
`int8_kv_pool.py`, `int4_kv_pool.py`, `tiered_kv_pool.py`, `memory_pool.py`
(`get_kv_smooth_buffer` / `get_kv_ring_buffer` / `get_kv_ring_owner` on `HybridLinearKVPool`),
`pool_configurator.py` (cell sizes), `kv_cache_dtype.py`, `kv_cache_configurator.py` (pool
class selection), `server_args.py` (`int8_g64`, `int4_g32`, `int8ring_int4`).

Evidence: fp8 mode at speed parity (dated entry 15:12 / CAMPAIGN.md:353); INT8-G64 unit tests
bit-exact (quant+scatter, compact gather, prefix row gather; relative RMS 0.62 %,
CAMPAIGN.md:365); S17: 512-window mean |dlogprob| 0.094 vs bf16 noise 0.099, NLL +0.010
(+-0.008), decode 56.1-57.4, 115,560-token prompt (CAMPAIGN.md:370), ceiling 162,215 tokens
(:372); INT4-G32 S19: all-position NLL +0.0088 / 0.138 vs the bf16 pool (noise +0.0002 /
0.059), 257,905-token prompt, needle 5/5 at 248k (CAMPAIGN.md:406-409); tiered S21: NLL
-0.0001 / 0.074, decode 54-57, prefill 2,271, needles 5/5 at 41k and 248k (CAMPAIGN.md:413;
`docs/logs/tiers-validate.log`). Design grounded in measured K/V statistics (simulated relative
RMS error e4m3 2.66 %, int8 per-token 1.3 %, int8-g64 0.9 %, CAMPAIGN.md:356). A first
all-position series was invalid because the fake-quant hook was bypassed by the int8 pool
subclass; it was caught by a K/V := 0 control and re-measured (ERRATUM CAMPAIGN.md:394, corrected
ladder :406).

Honest limits: the read path exists only for `qwen_sparse_attn_backend` (QSA's compaction and
prefix-chunk gathers); the CPU fallback raises `NotImplementedError` for int8/int4 pools, and any
other GPU backend that meets an int8/int4/tiered pool today would read raw bytes. `fp8_e4m3` mode
has an open crash on 1-token prompts (CAMPAIGN.md:355). INT4 prefill at 10k measured 1,493 tok/s
vs 2,316 for int8 at bring-up (the unpack gather, CAMPAIGN.md:406); the tiered default measured
2,271 (`docs/logs/tiers-validate.log:18`) and 2,301 in the S21 bring-up bench (log not included
in this repository). Smoothing
constants (`sm_k` / `sm_v`) are plumbed but identity; a static-smoothing A/B was mixed and not
adopted (CAMPAIGN.md:444-447). Prefix-chunk prefill still materialises the whole prefix as bf16
per layer and chunk (2 KB/token temporaries); the paged alternative was built and rejected on
timing (CAMPAIGN.md:439-442, `docs/KV_PAGED_PREFIX_PLAN.md` "Outcome").

Still needed upstream: a generic dispatch -- either a `kv_bits`-aware dequant-to-bf16
materialisation every backend can call, or a guard that rejects the new dtypes for backends
without a read path; `kv_bits` / `kv_tiered` formalised on the pool interface (today they are
class attributes read through `getattr`); `SGLANG_KV_TIERS_W` as a server arg; the three 2-line
`SGLANG_KV_STATS` guards in the pool files removed; `TORCH_DTYPE_TO_KV_CACHE_STR` maps
`torch.uint8` to `int4_g32`, which collides with the tiered mode (both store uint8) and should
become an explicit pool-class key; tests relocated (below); documentation of the on-disk layouts
(nibble order, scale index arithmetic, owner-table semantics), which today lives in the patch
docstrings.

### Optional PR 5 -- NGRAM speculation on hybrid PLE models (bug fix + opt-in)

`speculative/spec_utils.py` (PLE n-gram history + short-conv state committed after verify in both
ReplaySSM branches -- they returned before the generic update, so the PLE history never advanced
after a verify; a real bug for MTP top-k 1 too, CAMPAIGN.md:379), `speculative/ngram_worker.py`
(`_linearize_chain`, breadth-1 assertion, `SGLANG_NGRAM_CHECK` / `SGLANG_NGRAM_FORCE_REJECT`
debug checks), `models/qwen4_exp.py` (guard drop, startup self-check). Lossless within the
decode-vs-prefill floor (0 mismatches / 600 tokens; spec-path vs teacher-forced mean |dlogprob|
0.009-0.019 equals the non-speculative baseline, CAMPAIGN.md:386-387) but accept length 1.11-1.27
on prose, reasoning and code, so no speed-up on this model; MTP is infeasible on the reference host
(bf16 MTP experts 4.7 GiB, CAMPAIGN.md:379). Upstream would want the `spec_utils.py` fix
regardless, with a test that the PLE context advances after a verify step (the existing check is
the runtime invariant under `SGLANG_NGRAM_CHECK=1`; the CPU test below covers the linearizer).

## Tests that exist today

All in this repository, none inside the patch. Line counts are `wc -l` of 2026-09-03. GPU tests
import the patched modules and need only a few tens of MB of VRAM (their docstrings); pass
statements are the dated CAMPAIGN entries named in the last column.

| test | lines | covers | PR | pass evidence |
|---|---|---|---|---|
| `tools/test_unpack_2bit.py`, `test_group_layout.py`, `test_vs_bf16.py` | 61, 38, 69 | 2-bit unpack vs uint8 view; group layout; end to end vs bf16 dequant | 2 | `WRITEUP.md:356-369` |
| `test/manual/test_triton_moe_wna16.py` (+34/-6 in the patch) | -- | 2-bit cases (`w2a16`, `w2a16b2`) in the upstream manual test | 2 | none recorded; the 8-bit path of this test fails upstream (`WRITEUP.md:381-385`) |
| `gemv/test_gemv.py`, `test_gemv_ncontig.py`, `test_gemv_tab.py`, `test_tiled_word.py` | 127, 102, 101, 95 | GEMV bit-level agreement and GB/s; bit-exact layout conversion; full-layer decode MoE vs an fp32 reference; A/B through the real `invoke_fused_moe_kernel` | 2 | CAMPAIGN.md:196-204, :222-228; dated entries 09:47 (`test_gemv_tab: correct`), 11:04 (`all IDENTICAL (2e-5)`) |
| `gemv/row_arena.py` self-test (`python3 row_arena.py`) | 215 | 2 MiB granules, shrink returns memory to the driver, graph replay after shrink/regrow, `ArenaOOM` | 3 | CAMPAIGN.md:304 |
| `gemv/test_kv_fp8.py` | 41 | fp8 gather-dequant bit-exact vs `pool.to(fp8).to(bf16)`, e4m3 RMS error, saturation | 4 | none recorded separately (the fp8 mode ran, CAMPAIGN.md:353) |
| `gemv/test_kv_int8.py` | 228 | quant+scatter (int32/int64 loc), scale index, compact gather, prefix row gather with interior gaps (all bit-exact), RMS error < 1.2 %, pool eager + lazy VMM | 4 | CAMPAIGN.md:365, :369 |
| `gemv/test_kv_int4.py` | 352 | pack/unpack, trtllm strided layout, fp16-max clamp, pool, `kv_bits` | 4 | review round CAMPAIGN.md:392 (three findings fixed); S19 server validation :406 |
| `gemv/test_kv_tiers.py` | 604 | ring aliasing, same-launch collisions, stale owner, `SGLANG_KV_TIERS_W` validation, microbench | 4 | CAMPAIGN.md:411 (six review findings fixed); S21 :413 |
| `gemv/test_ngram_chain.py` (CPU) | 412 | `_linearize_chain` against a Python replica of the trie/result C++ under several sibling orders, verify-kernel replica | 5 | dated entry 17:08 (patch + test written); server gate :386-387 |
| `gemv/test_kv_paged_prefix.py` | 629 | the rejected paged kernel (reference only) | -- | CAMPAIGN.md:439-442 |

Server-level protocols (HTTP against a running server, not unit tests; all in `tools/`):
`logprob_diff.py` (teacher-forced oracle, 450 tokens), `nll_eval.py`, `nll_long.py`
(+ `nll_series.sh`, `NLL_LONG_ALL=1`), `needle_test.py`, `longctx_test.py`, `spec_lossless.py`,
`elastic_sweep.py`, `int8_validate.sh` (the `tiers_validate.sh` wrapper used for S21 is not
included), `toolcall_smoke.py`.

## What an upstream version would still need (all PRs)

1. **Configuration surface.** Every feature is switched by environment variables (25 distinct
   `SGLANG_*` names are read by the added lines; the full table is in `PATCH_NOTES.md` section 6).
   Upstream wants server args with validation, and the new `--kv-cache-dtype` values documented.
2. **Generic dispatch** for the quantized pools (PR 4) and a clean interface between the lazy KV
   owner and whoever can give memory back (PR 3).
3. **CI.** The unit tests above need to move under `test/` with GPU markers; none exercises a real
   model end to end. A small synthetic 2-bit MoE (`tools/make_mini_model.py` exists for the
   base patch) would allow an e2e smoke test.
4. **Docs.** Flags, env, the on-disk KV layouts, the placement-profile format
   (`{"mass": [48, 512], "count": [48, 512]}`, `assets/expert_freq.pt`), and the host-memory caveats
   (no runtime pinning, systemd-oomd) -- today spread over patch docstrings,
   `docs/ELASTIC_MEMORY.md` and `CAMPAIGN.md`.
5. **License headers and sign-off.** The patch contains no `Apache`, `SPDX`, `Copyright` or
   `License` line (grep over the whole patch, 2026-09-03); none of the seven new modules carries a
   header. DCO sign-off needed. The repository `LICENSE` (Apache-2.0) covers the patch.
6. **Leftovers removed**: the `~/Downloads/.../kv_stats_fp8run.pt` default path in the fake-quant
   hook, `SGLANG_NAN_TRACE`, the `SGLANG_NGRAM_*` debug switches behind a proper debug flag, the
   unused `import os as _os` in `gdn_backend.py`, the duplicated `import os as _os` in
   `qwen3_5.py`, `import os` next to `import os as _os` in `qwen4_exp.py`, the two blank-line-only
   hunks in `qwen4_exp.py`, and two German comments in `test/manual/test_triton_moe_wna16.py`.
7. **Open bugs disclosed in the PRs**: fp8 1-token crash (CAMPAIGN.md:355); scheduler vs
   `alloc_extend -> None` (CAMPAIGN.md:349); `kv_lazy` single-request assumption; `--page-size 1`
   silently becoming page size 64 under QSA (CAMPAIGN.md:325; the server log reports
   `page_size=64`).
8. **Portability of the tuned configs and the placement profile**: one GPU, one model; the profile
   must be regenerated per model (`tools/expert_freq.py` from a `SGLANG_ROUTE_DUMP` routing dump,
   a `host_fixes.py` item that is not applied in the tree).
9. **Benchmarks**: no standard-benchmark score is published for the 2-bit model
   (`docs/HISTORY.md`, "Benchmarks"). A PR that claims model quality needs one; the evidence
   available today is the internal protocol (short held-out NLL, long-text NLL ladder against a
   bf16 KV cache, logprob oracle, needle retrieval) quoted per PR above.

## Reproduction of the accepted state (for reviewers with the same class of hardware)

Launch line: `scripts/serve.sh`; flags and environment in `PATCH_NOTES.md` section 6 (from
`assets/phase1_state.json`). Environment: torch 2.13.0+cu130, triton 3.7.1, flash-attn 2.8.3
built for sm_120 (`WRITEUP.md:416-418`); the quantization ran with AutoRound 0.14.2. The served
checkpoint (the Hub download) is the sealed 2-bit quant with 85 dense tensors re-packed to INT8
g128 (`scripts/requant_int8.py`, S11b CAMPAIGN.md:298-300), and the 51,200,245,760-byte PLE table
`ple/ple.f8_e4m3.bin` + `ple.json` is a separate file of the same download. Host RAM is the
binding constraint of the reference host (CAMPAIGN.md:305, :347, :454-464).
