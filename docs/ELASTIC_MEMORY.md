# Elastic memory: 256k-token context for a 176B MoE on one 24 GB GPU

> **Status (2026-09-03):** last updated after S21. Corrections: the ladder row "real INT8-G64
> server +0.001 / 0.059" (and the "int8 +0.001" in the tiered paragraph) is invalid per the ERRATUM of
> CAMPAIGN.md:394 — the valid INT8 evidence is the 512-window comparison (mean abs. dlogprob 0.094 vs
> noise 0.099, NLL +0.010, :370); INT4 prefill at 10k is 1,493 vs 2,316 tok/s (:406), the tiered default
> 2,271-2,301; the `empty_cache` rate limit mentioned below was measured afterwards (258k prefill
> 171.0 -> 165.3 s, :448). The "Reproduce" block at the end uses the published launch line
> [`../scripts/serve.sh`](../scripts/serve.sh), which assumes one running request.

Companion to [WRITEUP.md](WRITEUP.md). Everything below runs on the same host as the rest of this
release: one RTX PRO 4000 Blackwell (24 GB), 32 GB host RAM, Qwen3.8-Flash-Next at 2.572 bpw, patched
SGLang. Measured on 2026-09-02; the running log is [`CAMPAIGN.md`](CAMPAIGN.md).

## Result

| | before | after |
|---|---|---|
| decode (10k context) | 56 tok/s | 56 tok/s |
| prefill (10k) | 2340 tok/s | 2316 tok/s |
| context length | 32 768 | 257 905 tokens proven (tiered default), needles 5/5 at 248k |
| KV cache per token | 24 KB (bf16) | 6.8 KB pool + 104 MB int8 ring (tiered default); 12.4 KB (INT8-G64); 6.8 KB (INT4-G32) |
| KV cache at startup | 0.8 GB committed | 0.1 GB committed, rest is address space |
| long-prompt quality | reference | indistinguishable (see below) |

162k prompt (INT8-G64): prefill 95.8 s (1694 tok/s), decode 51.2 tok/s, correct answer, server healthy afterwards.
258k prompt (INT4-G32): prefill 182 s (1415 tok/s), decode 51.9 tok/s, correct answer, server healthy afterwards.
Needle test at 248k tokens (five codes hidden at 10-90 % depth of random prose, INT4-G32): 5/5 retrieved,
prefill 269 s, decode 51.4 tok/s.
Tiered default (int8 ring over int4): 258k prompt prefill 171 s (1508 tok/s), decode 51.8; needles 5/5 at 41k and
at 248k; long-text NLL delta -0.0001 (8561 positions), i.e. int8 quality where attention is dense.

## The three mechanisms

**1. Elastic expert cache (residency is an address).** Every MoE layer keeps its 512 expert rows in a
CUDA VMM arena on the GPU in routing-mass order; an int64 table per (layer, tensor) maps expert id to
the row's address, which is either in the arena or in the row's pinned host slot. The decode GEMV and
the prefill gather read through the table, so moving an expert is a table write plus one row copy, and
CUDA graphs never need recapture (the table pointer is what they captured). Shrinking a layer unmaps
arena chunks (2 MiB granules), so the VRAM really returns to the driver; growing maps them back and
gathers the rows from host memory. Host memory is conserved through a slot pool: a host layer's hot
rows vacate their pinned slots, a GPU layer's cold rows take them. On the reference host the pool is empty at
S = 184 rows per layer, so 184 is the floor without extra pinned memory.
Files: `gemv/row_arena.py`, `gemv/expert_elastic.py`, `patches/elastic.py`.

**2. Lazy KV cache (VMM-backed, pay as you go).** SGLang already had a VMM arena for the KV cache but
committed it monotonically. The lazy patch reserves the KV cache as address space for the whole context
(262 144 tokens here) and backs it in 2048-token steps as the paged allocator hands out pages; when the
pool goes idle the backing beyond a 4096-token floor is unmapped. A watermark rule keeps 1.5 GB free
for prefill working memory: before a commit would cross it, the expert cache gives rows back (down to
the floor), and after the request it regrows at the pool-idle point. Prompts whose KV could not be
backed are refused at admission (capacity = min(requested, 0.77 x profiled)) instead of crashing the
scheduler mid-prefill. File: `patches/kv_lazy.py`.

**3. Own 8-bit KV cache (INT8-G64).** The Qwen sparse attention (QSA) kernels read bf16/fp16 only, so
the stock fp8 KV option cannot be consumed. A dequant-on-read path for fp8_e4m3 was added first
(`patches/kv_fp8.py`); the INT8-G64 scheme was then designed from measured K/V statistics of this
model (collected with `patches/kv_stats.py`, `SGLANG_KV_STATS`, over 19k tokens; the statistics
file is not part of the repository): int8 K and V with one fp16 absmax scale per token, KV head
and 64-channel group, the first group being the rotary dimensions. Quantization happens in one fused
Triton kernel at write time; dequantization happens inside the two existing gather sites (decode
compaction into the FlashAttention scratch, prefix-chunk row gather), so the attention kernels are
untouched. Scales live in the same lazy VMM arena. Simulated relative RMS error on the measured channel
profiles: e4m3 2.7 %, int8 per-token 1.3 %, INT8-G64 0.9 % (0.62 % in the unit test on N(0,3)).
File: `patches/kv_int8.py` (requires `kv_fp8.py` applied first), tests `gemv/test_kv_int8.py`.

## Quality protocol (why short tests are blind)

A prompt shorter than one prefill chunk (1024 tokens) never reads the KV cache: the first chunk
attends the freshly projected bf16 K/V. Short held-out passages therefore report "no change" for any
KV scheme. `tools/nll_long.py` scores the last 512 positions of a 9.6k-token text (their chunks read the
cache) and a 300-token greedy continuation after 3.7k tokens (the decode gather path). The bf16
run-to-run noise floor on this test is mean |dlogprob| 0.099 and NLL +-0.008.

| scheme | mean dlogprob (last 512) | NLL delta |
|---|---|---|
| bf16 vs bf16 (noise) | 0.099 | -0.008 |
| fake-quant int8_g64 | 0.099 | -0.007 |
| fake-quant int8 per-token | 0.100 | -0.006 |
| fake-quant int8_g32 | 0.106 | -0.004 |
| fake-quant fp8_e4m3 | 0.110 | -0.013 |
| real INT8-G64 server | 0.094 | +0.010 |

Short held-out NLL (write path): -0.001 nats/token. Logprob oracle at 10k: mean 0.0019 (noise floor).

With every position from the second chunk on (8561 positions, bf16 pool as reference) the resolution
is about 0.001 nats. This is where the precision ladder becomes visible:

| stored K/V | NLL delta | mean dlogprob (8561 positions) | needle 41k |
|---|---|---|---|
| bf16 vs bf16 (noise) | +0.0002 | 0.059 | 3/5 (answer truncated at 120 tokens) |
| real INT8-G64 server | +0.001 | 0.059 | 5/5 |
| fake-quant int4_g32 | +0.008 | 0.139 | 5/5 |
| real INT4-G32 server | +0.009 | 0.138 | 5/5 at 248k tokens |
| tiered int8 ring (8k) over int4, default | -0.0001 | 0.074 | 5/5 at 41k and 248k |
| fake-quant int2_g16 | +0.300 | 0.62 | |
| K/V := 0 (control) | +3.70 | 3.92 | |

INT8-G64 is free. INT4-G32 costs about 0.009 nats per token (0.5 % of the NLL on this text) and
keeps retrieval intact; int2 is the cliff. A first version of this table was invalid (every scheme
had accidentally run on the int8 pool) and was caught by the K/V := 0 control.

**INT4-G32 mode (256k).** `patches/kv_int4.py` (layered on `kv_int8.py`): nibble-packed K and V
with one fp16 absmax scale per token, head and 32-channel group, 6.8 KB/token including scales; the
profiled capacity becomes 366k tokens, so the full 262 144-token address space is admitted. Prefill is
about 15 % slower than INT8 (the unpack gather), decode unchanged. Serve with `--kv-cache-dtype int4_g32`.
**Tiered mode.** `patches/kv_tiers.py` keeps the last W tokens (default 8192) at
INT8 and everything older at INT4 by dual-writing: every fresh token goes int8 into a ring of W slots
and int4 into the full-context pool; an owner table says whether a ring row still belongs to a slot,
and the gather kernels test that on the device (CUDA-graph safe) and read int8 or int4 accordingly.
No compactor, no unmapping mid-request, 7.3 KB/token at 256k (ring 104 MB). Serve with
`--kv-cache-dtype int8ring_int4` (`SGLANG_KV_TIERS_W` sets the window). Validated (S21): NLL delta -0.0001 over 8561 positions (int8 +0.001, int4 +0.009),
decode 54-57 tok/s, 258k prompt, needles 5/5 at 41k and 248k. It is the default now.

## Speculative decoding (tried, correct, not a win here)

The model's PLE n-gram embedding refused NGRAM speculation. `patches/ngram_ple.py` removes that
guard, forces a linear draft chain (the GDN ReplaySSM fold, the QSA pending ring and the KV move all
assume a chain), and commits the PLE history and short-conv state after verify. A lossless gate
(`tools/spec_lossless.py`: spec-path logprobs vs teacher forcing, top-1/top-2 near-tie rule) shows the
same deviation as the non-speculative server (mean 0.01, 0 mismatches in 600 tokens), so it is
correct. But own-history n-gram drafts accept only 1.1-1.3 tokens per step on prose, reasoning and
code prompts, and QSA caps drafts at 4, so decode does not get faster on this model. It stays an
opt-in for repetitive workloads. The MTP head in the checkpoint is bf16 (4.7 GiB) and does not fit.

## Where the long-context prefill time goes

Per-chunk timing of the 258k run fits `chunk_ms = 615 + 0.49 us x prefix_tokens`. Only about a
quarter of the slowdown at 258k was attributed to the O(prefix) gather; a paged prefix kernel that
attends straight to the int8/int4/tiered pool rows was built and verified (`patches/kv_paged_prefix.py`,
within 2 row-ulps of the packed kernel) but rejected on timing: the gather is only about 0.18 ms per
60k rows and head-layer, whereas dequantizing inside the attention kernel costs it once per query tile,
so the paged kernel is 4-5x slower than gather + packed kernel at every prefix length. The O(prefix)
slope is dominated by the QSA indexer (scoring the compressed prefix for every query), which is part
of the model. A small part were periodic spikes from `empty_cache` calls below the memory watermark,
now rate-limited.

## What the reference host taught

* Host RAM, not VRAM, is the wall. 31 offloaded layers pin 24 GB of the 32 GB; ~2 GB stay free. Any
  pinned allocation at runtime (a few MB) stalls the kernel for tens of seconds and systemd-oomd kills
  the server; a 0.5 GB pinned reserve at startup died the same way. The expert cache therefore never
  pins after load, and its floor is what the host slot pool can absorb.
* The PLE n-gram table (51 GB, mmap) needs `MADV_RANDOM`; with default read-around a long prefill pulls
  gigabytes of page cache per chunk.
* Idle server processes get swapped; requests minutes after startup refault them. Keep them warm
  (`tools/keepalive.sh`) during long measurements.
* Do not exempt the server scope from `systemd-oomd` (`ManagedOOMPreference=omit`); under pressure
  oomd then kills other units of the session (CAMPAIGN.md:460-464). `MemoryMax=30G` is the backstop.
* 256k needs the 4-bit cache on the reference host: 8-bit KV stops at ~162k because the expert floor (host RAM)
  cannot give more VRAM back. Random text (as in needle tests) also churns the PLE page cache: pages are
  dropped after each bulk gather (`patches/ple_random.py`).

## Reproduce

```
# the published configuration: --kv-cache-dtype int8ring_int4 --max-mamba-cache-size 1 --max-running-requests 1,
# SGLANG_MOE_ELASTIC=1 SGLANG_MOE_PLACEMENT=<repo>/assets/expert_freq.pt SGLANG_MOE_PLACEMENT_S=184,
# SGLANG_KV_LAZY=1 SGLANG_KV_LAZY_TOKENS=262144 SGLANG_KV_LAZY_SAFETY=0.77 (all in scripts/serve.sh)
MODEL=~/quant/model scripts/serve.sh         # set SGLANG, VENV, PLE as described in the script
python3 tools/longctx_test.py 222000         # 257,905-token prompt; watches the expert cache shrink and regrow
python3 tools/elastic_sweep.py 184 200 216   # live S sweep without restarts
```
