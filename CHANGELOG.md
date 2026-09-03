# Changelog

Release changelog. The dated engineering timeline is [`docs/TIMELINE.md`](docs/TIMELINE.md); what was
tried and rejected, with reasons, is [`docs/HISTORY.md`](docs/HISTORY.md); every number cites
[`docs/CAMPAIGN.md`](docs/CAMPAIGN.md) (`CAMPAIGN.md:N` = line N of the append-only log).

## 1.0.0 — 2026-09-03

First release: Qwen3.8-Flash-Next quantized to 2.572 bits per weight and served on one RTX PRO 4000
Blackwell (24 GB) with 32 GB of host RAM through SGLang `73a255206f` plus the serving patch
`sglang/qwen4exp-serving-73a255206f.patch` (34 files, +4,155 / -89, verified byte for byte against the
served tree, `sglang/PATCH_NOTES.md` section 2). Weights on the Hub:
https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang
(text-only checkpoint, 222,856 tensors, 38,755,352,600 B; PLE table 51,200,245,760 B).

### Added

* **2-bit MoE path** for `moe_wna16` (unpack, loader, tuned `int2_w2a16` Triton configs) and nine
  findings under CPU offload on sm_120 addressed: eight correctness fixes, and whole-layer offloading
  replaced by the MoE-aware `ExpertStreamer` (26 GB -> 0.31 GB of PCIe traffic per token). The PLE
  n-gram table is memory-mapped from NVMe (`docs/WRITEUP.md`; `patches/base/`).
* **N-contiguous expert layout and in-place batch-1 GEMV** through int64 address tables: 320 GB/s from
  HBM, 51 GB/s from pinned host memory (`patches/ncontig_gemv.py`, `gemv/`; CAMPAIGN.md:196-204, :286).
* **Breakable CUDA graphs for decode** with the PLE lookup as the eager break (`patches/host_fixes.py`
  items `bcg`, `bcg2`; CAMPAIGN.md:291).
* **Frequency-based expert placement** from a routing-mass histogram (`assets/expert_freq.pt`,
  `patches/placement.py`; CAMPAIGN.md:296) and the **elastic expert cache** in CUDA VMM row arenas with
  a live S control file (`gemv/row_arena.py`, `gemv/expert_elastic.py`, `patches/elastic.py`;
  CAMPAIGN.md:319-320).
* **Lazy VMM KV cache**: address space for 262,144 tokens, backing in 2,048-token steps, idle release,
  driver-free watermark, admission cap (`patches/kv_lazy.py`; CAMPAIGN.md:330, :337, :350).
* **Quantized KV cache modes for the QSA layers** with dequantization on gather: `fp8_e4m3` read path,
  `int8_g64`, `int4_g32`, and the default tiered `int8ring_int4` (INT8 ring of 8,192 slots over an INT4
  pool, 7,308 B per token at 256k) (`patches/kv_fp8.py` .. `kv_tiers.py`; CAMPAIGN.md:370, :406, :413).
* **PLE page-cache hygiene** (`MADV_RANDOM`, `POSIX_FADV_RANDOM`, page drop after bulk gathers;
  `patches/ple_random.py`; CAMPAIGN.md:341, :408).
* **NGRAM speculation on PLE models**, opt-in (`patches/ngram_ple.py`; CAMPAIGN.md:387).
* Reasoning and tool-call parsers in the launch line (`--reasoning-parser qwen3
  --tool-call-parser qwen3_coder`; CAMPAIGN.md:430-432).
* The launch script `scripts/serve.sh` (published configuration: `--max-running-requests 1
  --max-mamba-cache-size 1`), the measurement tools in `tools/`, the unit tests in `gemv/`, the
  quantization pipeline in `scripts/`.

### Changed (model side)

* 85 dense projection tensors (`linear_attn.out_proj` x36, `self_attn.{q,k,v,o}_proj` x12 each,
  `lm_head`) re-packed from bf16 to INT8 g128 RTN by `scripts/requant_int8.py`; short held-out NLL
  -0.001 nats/token overall, 1.6 GB of VRAM returned (CAMPAIGN.md:298-300).

### Numbers (accepted state, `assets/phase1_state.json`)

* Decode 56.2 / 54.3 / 56.8 / 55.7 / 54.5 tok/s at 101 / 421 / 1,701 / 6,821 / 10,001 tokens of
  context; prefill 2,271 tok/s at 10,001 tokens (`docs/logs/tiers-validate.log`; CAMPAIGN.md:413).
* 257,905-token prompt: prefill 165.3 s (1,560 tok/s), decode 52.3 tok/s (`docs/logs/night.log`;
  CAMPAIGN.md:448). Needles 5/5 at 41,370 and 247,629 tokens.
* Tiered KV cache vs a bf16 KV cache, all-position test: NLL delta -0.0001 nats/token, mean abs.
  dlogprob 0.0743 (noise +0.0002 / 0.059) (CAMPAIGN.md:413, :406).
* Path from the base patch: 15.5 -> 54-57 tok/s decode, 1,158 -> 2,271 tok/s prefill, 32,768 ->
  262,144 tokens of context (README, "The path from 2.24 to 56 tok/s").

### Known limitations

* Host RAM is the binding limit: 31 offloaded layers pin ~24 GB of 32 GB; no runtime pinning; one
  request at a time in the published configuration (CAMPAIGN.md:305, :346-347, :454).
* `fp8_e4m3` mode crashes on a 1-token prompt (CAMPAIGN.md:355).
* The scheduler does not survive `alloc_extend` returning `None` mid-prefill; the admission cap avoids
  it (CAMPAIGN.md:349-350).
* Long prefills are O(prefix) in the model's QSA indexer; the paged prefix-chunk kernel was built and
  rejected on timing (CAMPAIGN.md:439).
* Tuned Triton configs and the placement histogram are specific to the RTX PRO 4000 Blackwell and the
  routing probe's workload.
* No standard-benchmark score is published for the 2-bit model (`docs/HISTORY.md`).
