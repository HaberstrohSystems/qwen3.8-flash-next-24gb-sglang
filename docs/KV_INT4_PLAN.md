# KV-INT4-G32: stage C of the own KV scheme (256k path)

> **Status:** the evidence quoted in the next paragraph (fake INT4-G32 -0.0006 / 0.060, real INT8 +0.0010 /
> 0.059) predates the ERRATUM of CAMPAIGN.md:394 (those runs had executed on the real INT8 pool).
> Corrected figures on the bf16 pool: fake INT4-G32 +0.0078 / 0.139, real INT4-G32 +0.0088 / 0.138,
> noise +0.0002 / 0.059 (CAMPAIGN.md:406). The conclusion stood; the default on top of it is the tiered
> cache ([KV_TIERS_PLAN.md](KV_TIERS_PLAN.md)). Implemented as `patches/kv_int4.py`.

Evidence (CAMPAIGN.md 2026-09-02 18:15): over 8561 cache-reading positions, fake-quantized int4 with
32-channel groups (9.7 % RMS error on the stored values) is indistinguishable from bf16 (NLL delta -0.0006,
mean |dlogprob| 0.060 vs the bf16 noise floor -0.0006 / 0.061). The real INT8-G64 server measured
+0.0010 / 0.059. So a 4-bit main K/V cache is safe for this model, and it halves the KV bytes again:

| | bf16 | INT8-G64 | INT4-G32 |
|---|---|---|---|
| payload per token (12 QSA layers, K+V, 2 heads x 256) | 24 576 B | 12 288 B | 6 144 B |
| scales per token | 0 | 384 B (fp16 x 4 groups) | 768 B (fp16 x 8 groups) |
| total | 24.0 KB | 12.4 KB | 6.75 KB |
| 256k tokens | 6.0 GB | 3.1 GB | 1.7 GB |

## Storage (mirror of patches/kv_int8.py, see KV_INT8_PLAN.md for every anchor)

* `--kv-cache-dtype int4_g32` -> a new pool class `MHATokenToKVPoolInt4(MHATokenToKVPool)` in
  `srt/mem_cache/int4_kv_pool.py` (copy of int8_kv_pool.py with the layout below). Store dtype uint8.
* `k_buffer[l]`, `v_buffer[l]`: uint8 `[rows, 2, 128]` = two 4-bit values per byte (low nibble = even
  channel, high nibble = odd channel), 256 B per head row, `row_bytes` 512.
* `k_scale_buffer[l]`, `v_scale_buffer[l]`: fp16 `[rows, 2, 8]` (absmax/7 per (token, head, 32-channel
  group); group 0-1 = the rotary dims 0..63), `row_bytes` 32, as extra `KvBufferDesc` rows so the lazy VMM
  owner backs them in lockstep (int8 plan section 5).
* Quantization: s = absmax_group / 7 (s = 1 when absmax == 0), q = clamp(rint(x / s), -7, 7) stored as
  q + 8 in the nibble (0..15); dequant (nibble - 8) * s. fp16 scale is the one used for rounding.
* `pool_configurator.py` cell size: `+= n * ((head_dim + v_head_dim) // 32) * 2 * num_layers` scale
  bytes; payload itemsize handled by the pool's `kv_cache_dtype` size hook (int8 plan section 2).

## Kernels (Triton, siblings of the int8 ones in `srt/layers/attention/qsa/sparse_attn.py`)

* `_quant_store_kv_int4[(N, 2)]`: load 256 bf16, group absmax over 32, quantize, pack pairs
  (`(q_even + 8) | ((q_odd + 8) << 4)`), store 128 uint8 + 8 fp16 scales.
* `_compact_kv_int4`: `_compact_kv_int8` with a 128-wide uint8 load, unpack to 256 values
  (`(b & 15) - 8`, `(b >> 4) - 8`, interleave), scale index `(slot*2 + h)*8 + d//32`, bf16 store into the
  FlashAttention scratch (valid columns only).
* `_gather_dequant_rows_int4`: the row gather for prefix-chunk prefill, same unpack; runtime length grid.
* Wrapper branch `if k.dtype == torch.uint8 and pool.kv_bits == 4` BEFORE the int8/fp8 branches (the int8
  pool stores int8, the fp8 pool stores uint8 viewed as float8: keep the dispatch keyed on a pool attribute,
  not on dtype alone).

## Tests

* `gemv/test_kv_int4.py`: pack/unpack round trip bit-exact vs torch reference; quant+scatter into
  random slots; compact gather bit-exact vs torch dequant (3 requests, invalid rows untouched); prefix row
  gather (interior gaps); relative RMS error on N(0,3) (expect ~9.7 %, g32); pool eager + lazy VMM.
* Server: `int8_validate.sh S19_int4kv 140000` (short NLL, all-position long NLL via nll_series-style
  scoring against nll/long_bf16all.json, oracle, bench, longctx). Pass: NLL delta < 0.003 at 8.5k
  positions, decode >= 55.8 tok/s.

## Still needed for 256k after INT4

At the expert-cache floor ~3.0 GB are free. 256k int4 KV = 1.7 GB; the prefix-chunk prefill materializes
the whole prefix as bf16 (2 KB/token = 512 MB at 256k, per call) and the GDN/MoE prefill working memory
is 1.1-1.5 GB -> 3.3-3.7 GB. Two options, in order:
1. paged prefix-chunk kernel: `_sparse_gqa_chunk_prefill` loads K/V rows through the slot table with
   the int4 unpack + scale, no materialized prefix (saves 512 MB and the O(n^2/chunk) traffic);
2. `--chunked-prefill-size 512` (halves the GDN working set; prefill throughput cost to be measured).
The admission cap (SGLANG_KV_LAZY_SAFETY x profiled) and the headroom rule stay as they are.
