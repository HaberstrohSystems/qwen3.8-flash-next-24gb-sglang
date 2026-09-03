# Running a 176B MoE on a single workstation card

> **Status (2026-09-03):** written after the base patch, before the performance campaign. The numbers in
> this document (15.5 tok/s decode, 1,158 tok/s prefill, 32,768-token context, 30 GiB host RAM, the
> 14-file patch size) describe that intermediate state. The final state is 54-57 / 2,271 tok/s at 10k
> with a 262,144-token context ([README](../README.md), CAMPAIGN.md:413); the open items of section 9
> (CUDA graphs vs mmap PLE, tuned MoE configs, naive placement, speculative decoding) were all resolved
> or evaluated afterwards, see [ELASTIC_MEMORY.md](ELASTIC_MEMORY.md) and
> [../patches/README.md](../patches/README.md). The launch configuration of section 7 is superseded by
> `scripts/serve.sh`. The recipe table of section 2 describes the AutoRound run; afterwards the 85
> dense tensors it lists as bf16 (`lm_head`, the QSA `q/k/v/o_proj`, the GDN `out_proj`) were re-packed
> to INT8 g128 by `scripts/requant_int8.py` (CAMPAIGN.md:298-300), so the shipped checkpoint carries
> them at 8 bits; the Hub card's precision table is the shipped state.

Qwen3.8-Flash-Next quantized to 2.57 bpw and served at **15 tok/s with a full 32k context**
on one RTX PRO 4000 (24 GB, sm_120) with 30 GiB of host RAM. Along the way: nine bugs and
gaps in SGLang, eight fixed and one new subsystem built.

Every number below is measured, not estimated.

| | before | after | factor |
|---|---:|---:|---:|
| Decode | 2.24 tok/s | **15.5 tok/s** | 6.9× |
| Prefill | ~500 tok/s | **1,158 tok/s** | 2.3× |
| PCIe per token | 26 GB | **0.31 GB** | 84× |
| Usable context | 13,824 | **32,768** | 2.4× |
| GPU utilization | 84–98 % | 43–58 % | — |

The 98 % before was waiting on the bus, not computing. Utilization *falling* is the good sign here.

---

## 1. The constraint that decides everything

Qwen3.8-Flash-Next is a hybrid MoE preview of the Qwen4 architecture: 36 Gated DeltaNet layers
interleaved with 12 Qwen Sparse Attention layers, 512 experts with top-10 routing,
hyper-connections that widen the residual stream to 10240, and a PLE n-gram table of 320M rows.
125B parameters in the trunk, 51B in the n-gram table, 4B in the MTP head, 6B active per token.

The official recipes list H200, B200, B300, GB300, MI350X and MI355X. Our machine is an
RTX PRO 4000 Blackwell, 24 GB, sm_120, with 30 GiB of RAM and an NVMe. Workstation Blackwell
appears on none of those lists — and as it turns out, that is not an accident.

| Experts at | bpw | experts | + rest | total | fits in ~34 GiB? |
|---|---:|---:|---:|---:|---|
| W2 g128 | 2.125 | 28.2 GiB | 7.0 | 35.2 | barely |
| W3 g128 | 3.125 | 41.5 GiB | 7.0 | 48.5 | no |
| W4 g128 | 4.125 | 54.7 GiB | 7.0 | 61.7 | no |

4-bit would be nearly twice the total memory of the machine. **2-bit is the only option that
fits** — and 2-bit MoE is exactly what neither SGLang nor vLLM can load today. Both cap
`moe_wna16` at 4 and 8 bits.

---

## 2. The quantization recipe

Quantized with AutoRound, mixed by sensitivity. Qwen's own FP8 release leaves everything except
experts and n-gram untouched — that is the best available statement about what is fragile in this
architecture. We could not afford to follow it fully, but followed it as far as the budget allowed.

| Group | Scheme | On disk | Reason |
|---|---|---:|---|
| Experts up/gate | `W2A16 g128` | 20.2 GiB | 96 % of all quantized parameters |
| Experts down | `W2A16 g128` | 10.1 GiB | same group size — mandatory, see finding 3 |
| GDN `in_proj_qkv/z` | `W8A16 g128` | 1.4 GiB | recurrent path, error-prone |
| Shared expert | `W8A16 g128` | 0.2 GiB | runs on every token |
| QSA, hyper-conn., router, `lm_head` | bf16 | 3.6 GiB | Qwen pins these too |
| GDN `out_proj`, `in_proj_a/b` | bf16 | 1.1 GiB | `[48, 2560]` — 48 is not divisible by 32 |

Result: **2.572 bpw, 38.60 GiB**. The PLE table is written out separately as fp8 (47.7 GiB) and
never loaded into memory — it is read from NVMe. The MTP head lands in its own file and is left
out at serving time.

The run took **3 days 8 hours**: 48 blocks, four subset rounds each over the 512 experts with
error compensation in between, `--enable_alg_ext` (AutoRound recommends it for ≤2 bits and lists
W2A16 as validated), 48 calibration samples of 2048 tokens.

### Why each group got the bits it got

The budget forces roughly 2.5 bpw on average. The question is where to spend above that average
and where to go below it. Every choice below has a stated cost, because a recipe without costs is
just an opinion.

**Experts at 2 bit, group size 128.** They are 96 % of the quantized parameters, so nothing else
moves the total. Group size was originally 256 for `gate_up` and 128 for `down` — coarser groups on
the larger tensor, 0.73 GiB cheaper. That had to go: SGLang's `FusedMoE` requires *one* `(bits,
group_size)` for all projections of a layer and raises `Fused MoE layer requires consistent quant
config for all sub-layers` otherwise. Unifying on g128 turned out to cost nothing overall, because
it was paid for by dropping a rule that never worked (below), and it improves 80.5B parameters.

**No per-layer bit gradient.** The first plan gave the first 8 layers' `down_proj` 3 bits, on the
usual argument that early layers are more sensitive. Two things killed it. First, it never took
effect — AutoRound's last-match-wins order let the general 2-bit rule overwrite it, and the disk
proved it (`qweight [4,256]`, `qzeros [1,16]` → pack factor 16 → 2 bit, not 3). Second, `FusedMoE`
uniformity means a bit gradient is only expressible per *whole layer*, not per projection.
Upgrading eight entire layers to 3 bit would have cost 2.2 GiB. We dropped it rather than spend
that on a coarse-grained version of a fine-grained idea.

**Gated DeltaNet split three ways.** `in_proj_qkv` and `in_proj_z` at W8 (1.4 GiB): they are large
and feed a recurrent path where errors accumulate across the sequence rather than washing out.
`out_proj` at bf16 (the "ssm_out is never deep" rule of thumb). `in_proj_a`/`in_proj_b` at bf16 not
by choice but by arithmetic — they are `[48, 2560]`, and 48 is not divisible by 32, so AutoRound
cannot quantize them at all. Declaring them 8-bit anyway is what produced
`No compatible backend found for layer ...linear_attn.in_proj_a` on the first serving attempt:
config said quantized, disk said bf16.

**Shared expert at W8** (0.2 GiB). It runs on *every* token, unlike the routed experts which each
see roughly 2 % of tokens. Cheap insurance on a tensor that is small anyway.

**QSA attention, router, hyper-connections, `lm_head` at bf16** (3.6 GiB together). Qwen's own FP8
release pins all of these, and the router in particular is the one place where a rounding error
does not degrade an output but *redirects* it to a different expert. 12 attention layers out of 48
is a small bill for removing a whole class of risk.

**`lm_head` at bf16** deserves its own line because it is where quality and mechanics agreed.
AutoRound only quantizes it on an explicit `--quant_lm_head`, and with a 248,320-entry vocabulary
every rounding error lands directly in the logits. It also happens to be the layer whose
quantization disables block-wise saving (below). Cost: 0.879 GiB, from 0.305 at W4 g128 to 1.184
at bf16.

**PLE n-gram table stays fp8 and leaves the checkpoint.** 51.2B parameters, untouched at the
precision Qwen shipped. Quantizing it further would have been the single largest saving available
and we did not take it: it is a lookup table, so an error is not averaged over a matmul — it is
the value. It is also the one part that never needs to be resident.

**MTP head skipped entirely** (bf16, own file, 4.85 GiB). Speculative decoding is not on the path
to a working server on this machine; carrying it would have cost VRAM for nothing.

### Calibration: what we cut, and what it cost

| Setting | Chosen | Alternative | Why |
|---|---|---|---|
| `nsamples` | 48 | 128 (AutoRound default) | 128 × 2048 × 10240 × 2 B = 5.0 GiB **per cache**, and AutoRound holds three at the block boundary. That is 15 GiB on a 30 GiB machine, and it thrashed: block 1 took 313 min instead of 84, GPU at 0 %, 133M major faults. 48 samples fit. Cost: 63 % of the calibration pool |
| `seqlen` | 2048 | 1024 | Kept full length rather than more samples — long context exercises the GDN recurrence and the routing realistically. Same memory either way |
| `iters` | 200 | 1000 (`auto-round-best`) | 1000 would have been ~11 days against a 5-day budget. The algorithmic part of "best" was taken instead, see next row |
| `--enable_alg_ext` | on | off | AutoRound recommends it for ≤2 bits and lists W2A16 — our 96 % — as validated. Measured cost on a small model: 398 s vs 392 s, i.e. nothing. Not a placebo either: 816 of 1423 tensors change, and the 607 that don't are exactly the bf16 pass-throughs |
| Subset rounds | 4 | 1 | Experts are parallel additive branches with no intra-block interaction, so tuning them in 4 groups gives sequential error compensation across groups (each round writes back quantized weights). Roughly 2.2× the runtime |

The `nsamples` cut is the one to be honest about. Fewer calibration tokens means each of the 512
experts sees fewer routed tokens: at 48 × 2048 tokens and top-10 of 512, that is ~1,900 tokens per
expert instead of ~5,100. That is a real reduction in signal, and it was forced by host memory, not
chosen for quality.

### Two silent traps in AutoRound

**Rule order is last-match-wins.** AutoRound applies `layer_config` in insertion order and lets
the *last* match win (`compressors/utils.py:506,536`). A recipe written as first-match-wins is
silently inverted. In our case the general 2-bit rule would have overwritten the bf16 pins for
`in_proj_a/b` and `out_proj`. Counter-test: **235 layers** resolved differently than intended,
including the MTP head that was marked SKIP. Emit rules in reverse order, and verify by resolving
every layer both ways.

**One quantized layer outside the transformer blocks disables block-wise saving.** In our case
`lm_head` at 4 bits. The chain is `base.py:1224 → 1425 → 1428 → 1446`:
`has_qlayer_outside_block ∧ need_calib` forces `inplace=False`, which prevents
`is_immediate_packing`, which prevents `is_immediate_saving`. All 48 finished blocks then
accumulate in RAM — about 35 GiB. On a 30 GiB machine that is a guaranteed OOM after ninety
minutes of compute. Putting `lm_head` in bf16 fixes it and is the better choice for quality
anyway (AutoRound only quantizes it on an explicit `--quant_lm_head`).

---

## 3. Nine findings in SGLang

Flash-Next model support lives in an open PR and works on datacenter GPUs. The following only
surface when you run 2-bit, need CPU offload, or sit on sm_120.

| # | Finding | Symptom without the fix |
|---|---|---|
| 1 | `moe_wna16` supports only 4 and 8 bits — Marlin has no 2-bit type, the Triton kernel has no 2-bit unpack | 2-bit MoE cannot load at all |
| 2 | `qwen_sparse_attention` missing from `ALL_DECODER_LAYER_TYPES` | `KeyError`. transformers renames `full_attention` on load, so every checkpoint round-tripped through it carries this name |
| 3 | `packed_modules_mapping` is never propagated to the quant config | Fused names unresolvable. `in_proj_ba` (48+48) falls back to 8 bits → `size_n = 96 is not divisible by tile_n_size = 64` |
| 4 | `gemma_weight` is a non-persistent buffer | Not moved by CPU offload → device mismatch during weight load |
| 5 | `functional_call` without `tie_weights=False` | Fails on `linear_attn.A_log` / `linear_attn.attn.A_log`, which are tied |
| 6 | tilelang compiles with nvcc 13.3 against 13.0 runtime headers | `CUDA compiler and CUDA toolkit headers are incompatible` — QSA kernels won't build |
| 7 | `language_model_only` allowlist omits Qwen4Exp | Vision tower cannot be skipped even though the base class implements it |
| 8 | `conv_weights` captured as a `.view()` at construction time | **Silent corruption.** See below |
| 9 | No MoE-aware offloading | 40× more PCIe traffic than necessary |

---

## 4. Finding 8, the expensive one

The model loaded, ran, produced tokens — and emitted `!!!!!!!!`. That is token 0, the signature
of NaN logits. The checkpoint was provably clean: 75,128 float tensors scanned, zero NaN, zero
Inf, scale magnitudes between 1e-5 and 1.0.

Narrowing it down across 48 layers led into the Gated DeltaNet convolution:

```
# per-layer magnitudes
L0 after GDN: max|x| = 0        L5  after GDN: max|x| = 0
L1 after GDN: max|x| = 0        L9  after GDN: max|x| = 0.016
L2 after GDN: max|x| = 0        L10 after GDN: NaN

# one level deeper, immediately before the convolution
mixed_qkv    = bf16 (10240, 5)  max = 32          <- healthy
conv_weights = bf16 (10240, 4)  max = 9.367e-38   <- the weights are empty
```

On disk `conv1d.weight` had max 0.2236 with no zeros anywhere. The cause is a single line:

```python
# qwen3_5.py, building the GDN module
conv_weights = self.conv1d.weight.view(size(0), size(2))   # a view of NOW
self.attn = RadixLinearAttention(..., conv_weights=conv_weights, ...)
```

The view is taken at construction and held. The normal load path copies into existing storage, so
the view stays valid. But `--cpu-offload-gb` *replaces* `param.data` on every on/offload, and from
then on the held view points at old, uninitialized memory. Those 1e-38 values were never weights.

Fixed by passing the module rather than the view, and deriving the view fresh on each access.
Tensors and tuples pass through unchanged so KDA, ShortConv and Lightning are untouched.

This never shows up on datacenter GPUs because nobody there needs CPU offload. It is the worst
class of bug: no crash, no warning, just a model that talks nonsense.

---

## 5. Finding 9: 26 GB per token

With everything fixed the model ran — at **2.24 tok/s**. For 6B active parameters with part of the
weights resident on the GPU, that makes no sense. The measurement:

```
rxpci  55–61 GB/s sustained    <- PCIe Gen5 x16 saturated
sm     84–98 %                 <- waiting, not computing
40 tokens in 17.8 s            -> ~26 GB per token
```

`--cpu-offload-gb` copies a module's entire `state_dict` to the device on every forward pass.
Correct for a dense model. For a 512-expert MoE with top-10 it transfers 512 experts to use 10:

| Per forward pass | bytes | |
|---|---:|---|
| one expert | 1.31 MB | |
| 10 experts × 48 layers | 0.63 GB | what is needed |
| measured transfer | 26.00 GB | 41.5× too much |

### The streamer

Two changes, both small, both **memory-neutral** — and that is the binding constraint. After
loading, both VRAM and host RAM are full; any additional copy trips the OOM killer. Three
intermediate designs died exactly there before this one worked.

1. **Exclude experts from the bulk transfer.** In `offloader.py`, expert tensors are skipped when
   building `device_state`. They stay in the pinned host memory the offloader already put them in.
   Nothing is moved or copied.
2. **Offload only experts.** Previously whole decoder layers went to the host, including attention,
   GDN and norms — which are needed on *every* token. Those now stay resident on the GPU.

On each forward pass `ExpertStreamer` gathers exactly the selected experts into a single reused
staging buffer (668 MB) and renumbers the top-k ids onto it. The Triton kernel is unchanged — it
still sees one contiguous tensor.

The gather is a Triton kernel reading directly from pinned host memory with GPU-side indices. The
naive route (`uniq.to("cpu")` then one copy per row) forces a device synchronization per layer.
During decode we skip deduplication entirely: with 10 ids, copying a duplicate row is cheaper than
the sync `torch.unique` forces because its output size is data-dependent.

Side effect: with non-expert weights no longer shuttling back and forth, resident usage drops from
19.76 to **18.6 GB**. The freed 3.75 GB goes to the KV cache, which grows from 13,824 to the full
**32,768 tokens**.

---

## 6. Results

| Context | Prefill s | Prefill tok/s | Decode tok/s |
|---:|---:|---:|---:|
| 131 | 0.75 | 174 | 15.2 |
| 547 | 1.11 | 493 | 15.3 |
| 2,211 | 1.96 | 1,129 | 15.5 |
| 8,867 | 7.67 | 1,156 | 15.5 |
| 13,001 | 11.23 | 1,158 | 14.9 |

### Quality at 2 bits

> "No, Anna cannot be 10. Since Ben is older than Clara (12), Ben must be at least 13. Since Anna
> is older than Ben, she must be at least 14. 10 is less than 14, so it is impossible."

Factual answers correct (Berlin 3.7M / Hamburg 1.8M / Munich 1.5M), arithmetic sound (220 km, with
correct unit conversion), code runnable, German and English equally fluent. No waffle, no invented
numbers. Systematic benchmarks are still pending — this is a first impression, not a measurement.

---

## 7. Running it

The quantized weights, the pre-split PLE table and the patch are published — you do not need to
repeat the three-day quantization run or the extraction. What remains on your side is the
environment and the launch configuration, and both have sharp edges worth explaining.

### The PLE table is a separate artifact

SGLang's only offload mode for the n-gram table is `Qwen4ExpPinnedHostEmbedding`, which places all
51 GB into page-locked host memory — impossible below roughly 64 GB of RAM. The patch adds a third
mode that memory-maps the table from disk instead, costing no RAM at all: the page cache keeps hot
rows resident on its own, and a gather reads 16 rows of 160 bytes per token.

That is why the table ships as one contiguous fp8 file plus a small `ple.json`. The JSON carries
`rows`, `dim`, `dtype` and — importantly — `weight_scale`, which does not survive quantization
into the checkpoint and would otherwise be silently lost.

### Two environment details you have to get right

**The JIT compiler.** SGLang compiles kernels for `compute_120a` at runtime. A system CUDA 12.x
cannot target that architecture at all, and the build fails with
`Unsupported gpu architecture 'compute_120a'`. Point `CUDA_HOME` and `PATH` at the toolkit inside
the venv. tilelang additionally compares the compiler version against the runtime headers, so
those two must match — a 13.3 nvcc against 13.0 headers fails with
`CUDA compiler and CUDA toolkit headers are incompatible`.

**FlashAttention 2.** QSA decode prefers classic FA2 and falls back to flash-attn-4, whose CUTLASS
DSL path fails on sm_120 with a layout congruence error. FA2 has to be built from source, but
`FLASH_ATTN_CUDA_ARCHS` lets you build only your own architecture, which turns roughly ninety
minutes into twenty:

```bash
FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=4 \
  pip install --no-build-isolation --no-deps flash-attn==2.8.3.post1
```

### The launch configuration

```bash
SGLANG_MOE_EXPERT_STREAM=1 \
SGLANG_QWEN4_PLE_MMAP=/path/to/ple \
SGLANG_VLM_CACHE_SIZE_MB=0 \
python -m sglang.launch_server \
  --model-path /path/to/quant --port 30000 \
  --tp-size 1 --context-length 32768 \
  --cpu-offload-gb 19 \          # with the patch this offloads experts only
  --no-ple-offload-embedding \
  --disable-cuda-graph \         # see open items
  --language-model-only \        # skips the vision tower entirely
  --max-running-requests 1 --max-mamba-cache-size 10 \
  --max-total-tokens 32768 --chunked-prefill-size 512 \
  --mem-fraction-static 0.95 \
  --page-size 1 --disable-overlap-schedule --disable-radix-cache \
  --weight-loader-drop-cache-after-load
```

Two of those flags are load-bearing and worth understanding rather than copying.

`--cpu-offload-gb 19` reads differently with the patch applied: instead of offloading whole decoder
layers, it now offloads *only* expert weights, so the number is a budget for experts alone. Raise
it and more experts move to the host, which costs nothing per token because only the selected ones
are fetched; lower it and they stay resident. The ceiling is host RAM, not throughput.

`--language-model-only` skips the vision tower entirely — its weights are never loaded and the
tower is never built. For text workloads that is 0.84 GiB of weights plus the multimodal
reservation inside the KV budget, and it was the difference between a KV cache of 13,824 and one of
32,768 tokens.

### On memory pressure

`systemd-oomd` kills on PSI pressure, not on absolute limits, and `user@.service` ships with
`ManagedOOMMemoryPressure=kill`. Setting `MemoryHigh` or a `memory.swap.high` below current usage
*creates* the pressure that gets you killed. Use `MemoryMax` as a backstop and leave `MemoryHigh`
alone. `--weight-loader-drop-cache-after-load` matters here too: without it the page cache from
reading 41 GB of shards is enough to trip it.

---

## 8. How the 2-bit path was verified

The kernel was not reviewed, it was measured — against real AutoRound tensors:

- **Packing order.** Unpacking straight from the `int32` versus unpacking through the `uint8` view
  with byte index and shift → `torch.equal == True`, values 0..3, mean 1.95. That mean confirms a
  symmetric zero point of 2, since GPTQ stores `zero − 1`.
- **Triton expressions.** The exact `tl` expressions from the patch, run as a standalone kernel
  against a PyTorch reference → bit-identical, max difference 0.
- **End to end.** The same weights fed through the same MoE machinery twice — once dequantized as
  bf16, once packed as 2-bit. Routing, activation and reduction are identical, so only
  dequantization differs. Small shapes bit-identical; large shapes 1e-5 relative error, which is
  bf16 accumulation noise. Including the real expert dimensions (e=64, n=640, k=2560, g128).

The 2-bit unpack itself is a generalization of the existing 4-bit path:

```
// 4-bit (existing)              // 2-bit (added)
(offs_k // 2) * stride_bk        (offs_k // 4) * stride_bk
(offs_k % 2) * 4                 (offs_k % 4) * 2
b >> shifter & 0xF               b >> shifter & 0x3
b_zp_num = 8                     b_zp_num = 2        // 2^(bits-1)
b_ptrs += BLOCK_SIZE_K // 2      b_ptrs += BLOCK_SIZE_K // 4
```

**A note for the SGLang maintainers.** The existing `test/manual/test_triton_moe_wna16.py` fails on
large shapes at **8 bits** too — that is the unmodified upstream path, 66 of 96 cases. It explains
why `weight_bits` is pinned to `[8]` there with `[4, 8]` commented out. In its current state the
test cannot validate anything, at any bit width.

---

## 9. Open items

- **CUDA graphs are incompatible with the mmap PLE path.** The 51 GB table lives in non-pinned
  memory and its row indices must reach the CPU in a data-dependent way — both are forbidden during
  graph capture. This costs real throughput: the GPU now sits at 43–58 %, and the remainder is
  kernel launch overhead. A pinned hot-row cache for the PLE would likely unlock it.
- **No tuned MoE configs** exist for `int2_w2a16`; the kernel falls back to defaults. A tuning
  sweep is straightforward and probably worth several tok/s.
- **Asymmetric 2-bit** (`qzeros`) is deliberately not implemented in the loader and raises a clear
  `NotImplementedError`. A silent wrong path would be worse than a refusal. The kernel itself does
  handle the zero-point case.
- **Expert placement is naive.** Which experts stay resident is whatever the offloader happened to
  leave on the GPU. A usage-aware cache — keeping the most frequently routed experts resident —
  should cut the remaining transfer substantially.

---

**Patch size.** 467 inserted lines across 14 files plus one new module, `expert_stream.py`.

**Setup.** RTX PRO 4000 Blackwell 24 GB (sm_120) · 30 GiB RAM · NVMe · torch 2.13.0+cu130 ·
triton 3.7.1 · flash-attn 2.8.3 built for sm_120 · SGLang on the Flash-Next branch ·
AutoRound 0.14.2.

## Addendum: elastic memory

The memory work that followed this write-up (elastic expert residency, lazy VMM KV backing, INT8-G64 KV cache, 162k-token context) is documented in [ELASTIC_MEMORY.md](ELASTIC_MEMORY.md).
