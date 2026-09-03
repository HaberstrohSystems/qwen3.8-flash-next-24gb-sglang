# Qwen3.8-Flash-Next on one 24 GB GPU

Qwen3.8-Flash-Next is a 176B-parameter mixture-of-experts model (Qwen4-Exp architecture) with 6B active
parameters per token. This repository is the code side of serving it on a single RTX PRO 4000 Blackwell
(24 GB, sm_120) with 32 GB of host RAM: quantized with AutoRound to 2.572 bits per weight (2-bit experts
with group size 128, INT8 dense projections, 16-bit routers and norms) and served through a patched SGLang.
It contains the serving patch, the layered patch scripts it was built from, the kernels, the two serving
assets, the measurement tools, the design documents with their outcomes and the dated engineering log.

**Weights:** [Hugging Face Hub](https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang)
· **Patch:** [`sglang/qwen4exp-serving-73a255206f.patch`](sglang/qwen4exp-serving-73a255206f.patch)
· **License:** Apache-2.0 (code, [`LICENSE`](LICENSE)) / Qwen Community License 1.0 (weights, on the Hub)

## Headline numbers

| Metric | Value |
|---|---|
| Decode at 101 / 421 / 1,701 / 6,821 / 10,001 tokens of context | 56.2 / 54.3 / 56.8 / 55.7 / 54.5 tok/s |
| Prefill, 10,001-token prompt | 2,271 tok/s (4.40 s) |
| Context | 262,144 tokens; a 257,905-token prompt prefills in 165.3 s (1,560 tok/s) and decodes at 52.3 tok/s |
| KV cache, default mode (`int8ring_int4`) | 7,308 B per token at 256k context (6,912 B in the INT4 pool + an INT8 ring of 8,192 slots, 103.8 MB fixed) |
| VRAM at the expert-cache floor | S = 184 experts per layer resident: arena 11.062 GiB (11.88 GB), 84.3 % of routing mass served from VRAM |
| Long-text quality of the default KV mode | NLL delta -0.0001 nats/token, mean abs. dlogprob 0.0743 over 8,561 cache-reading positions vs a bf16 KV cache; bf16-vs-bf16 noise on the same test +0.0002 / 0.0590 |

Sources, row by row: [`docs/logs/tiers-validate.log`](docs/logs/tiers-validate.log) "streaming bench" and
[`assets/phase1_state.json`](assets/phase1_state.json) `accepted.decode_all` (CAMPAIGN.md:413); the same
log; [`docs/logs/night.log`](docs/logs/night.log) "258k spike check" (CAMPAIGN.md:448);
[`patches/kv_tiers.py`](patches/kv_tiers.py) docstring (CAMPAIGN.md:413);
[`docs/logs/elastic.ctl.status`](docs/logs/elastic.ctl.status) (CAMPAIGN.md:320);
[`docs/logs/tiers-validate.log`](docs/logs/tiers-validate.log) (CAMPAIGN.md:396, :413).

`CAMPAIGN.md:N` cites line N of [`docs/CAMPAIGN.md`](docs/CAMPAIGN.md), the append-only engineering log;
line numbers stay valid as the log grows. The *accepted state* is the flag set, environment, patch list and
checkpoint recorded in `assets/phase1_state.json`, on which the published numbers were measured. *S* is the
number of experts per layer resident in VRAM. All measurements were taken on one machine (one RTX PRO 4000
Blackwell, 32 GB host RAM); see [Reproducing the measurements](#reproducing-the-measurements).

**Weights.** The checkpoint (text-only, 222,856 tensors, 38,755,352,600 bytes) and the PLE n-gram table
(`ple/ple.f8_e4m3.bin`, 51,200,245,760 bytes, plus `ple/ple.json`) are published on the Hugging Face Hub
under the Qwen Community License 1.0:
**https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang**.
The model card there describes the quantization recipe and the serving configuration.

## Quick start

Requirements: one 24 GB GPU of compute capability 12.0 (the tuned Triton configs are for the RTX PRO
4000 Blackwell), 32 GB host RAM, an NVMe disk for the PLE table, Python 3.12 (the virtualenv's
`lib/python3.12/site-packages` path is hard-wired in `scripts/serve.sh`), the CUDA 13 toolkit inside
the virtualenv. The serving stack that was measured: torch 2.13.0+cu130, triton 3.7.1, flash-attn
2.8.3.post1 built for sm_120, transformers 5.12.1 (pinned by SGLang at `73a255206f`), SGLang at commit
`73a255206f` with the serving patch.

```bash
# 1. this repository and SGLang at the base commit, with the serving patch applied
mkdir -p ~/quant && cd ~/quant
git clone https://github.com/HaberstrohSystems/qwen3.8-flash-next-24gb-sglang.git
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout 73a255206f916366c8d26d4022f82ddfb0ab558d
git apply --check ../qwen3.8-flash-next-24gb-sglang/sglang/qwen4exp-serving-73a255206f.patch
git apply         ../qwen3.8-flash-next-24gb-sglang/sglang/qwen4exp-serving-73a255206f.patch

# 2. a virtualenv with torch for CUDA 13, SGLang, the CUDA 13 compiler and FlashAttention 2. nvcc and the
#    runtime headers are installed under site-packages/nvidia/cu13/, which scripts/serve.sh puts on PATH and
#    CUDA_HOME (SGLang JIT-compiles kernels for compute_120a with that nvcc).
python3.12 -m venv ~/quant/venv-sglang && . ~/quant/venv-sglang/bin/activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
pip install -e python              # SGLang (the patched checkout is the current directory)
pip install nvidia-cuda-nvcc==13.3.73 nvidia-cuda-runtime==13.3.29   # compiler and headers, same minor version
$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc --version   # release 13.3, V13.3.73
export CUDA_HOME=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=4 pip install --no-build-isolation --no-deps flash-attn==2.8.3.post1

# 3. the weights (about 90 GB: checkpoint shards plus ple/)
hf download HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang --local-dir ~/quant/model

# 4. launch (the published configuration: full 262,144-token context, tiered KV cache, one request at a time)
cd ~/quant/qwen3.8-flash-next-24gb-sglang
SGLANG=~/quant/sglang VENV=~/quant/venv-sglang MODEL=~/quant/model PLE=~/quant/model/ple scripts/serve.sh

# 5. verify
python3 tools/bench_speed.py
```

The nvcc and the CUDA runtime headers inside the venv must be the same minor version (tilelang checks;
`docs/WRITEUP.md` section 7). `nvidia-cuda-nvcc` depends on `nvidia-cuda-runtime` without a version pin,
and neither SGLang nor its tilelang dependency installs the compiler, so step 2 pins both to what the
measured venv carries: `nvidia-cuda-nvcc` 13.3.73 and `nvidia-cuda-runtime` 13.3.29. A system CUDA 12
cannot target `compute_120a`.

[`scripts/serve.sh`](scripts/serve.sh) waits for `/health`, sends one warm-up request and `POST
/freeze_gc`; the comment block at its end explains every flag and environment variable. The server
speaks the OpenAI chat API on `127.0.0.1:30000` with thinking in `reasoning_content` and parsed tool
calls. `tools/bench_speed.py` prints one line per context (tokens, prefill seconds, prefill tok/s,
decode tok/s, decoded tokens); expected on the measured machine: decode 54-57 tok/s at all five
contexts and about 2,270 tok/s prefill at 10,001 tokens (`docs/logs/tiers-validate.log` lines 13-18).

Two flags to know: `--cpu-offload-gb 19` is a budget for expert weights only once the patch is
applied. `--max-running-requests 1 --max-mamba-cache-size 1` serve one request at a time; on a 32 GB
host that is the configuration that keeps memory pressure low (CAMPAIGN.md:454); higher concurrency
needs more host RAM.

Pointers: the patch is in [`sglang/`](sglang/), reproduction of every headline number is described in
[Reproducing the measurements](#reproducing-the-measurements), the license is [`LICENSE`](LICENSE).

## Repository layout

| Path | Contents |
|---|---|
| [`sglang/`](sglang/) | The serving patch against SGLang `73a255206f` ([`qwen4exp-serving-73a255206f.patch`](sglang/qwen4exp-serving-73a255206f.patch)), its per-file notes with verification ([`PATCH_NOTES.md`](sglang/PATCH_NOTES.md)) and the upstream contribution plan ([`UPSTREAM.md`](sglang/UPSTREAM.md)). |
| [`patches/`](patches/) | The layered patch scripts (`apply` / `revert` / `--check`) the served tree was built from, the original base patch, and [`patches/README.md`](patches/README.md) with layering order, the status of every layer and the packaging notes. |
| [`gemv/`](gemv/) | Kernels and unit tests: int2 GEMV, the VMM row arena, the elastic expert cache, KV pool tests, n-gram chain test. |
| [`tools/`](tools/) | Measurement tools: streaming bench, logprob oracle, NLL tests, needle and long-context tests, elastic sweep. |
| [`scripts/`](scripts/) | The quantization pipeline (`01`..`09`, `budget.py`, `pipeline.sh`), the INT8 dense re-pack (`requant_int8.py`), the launch script [`serve.sh`](scripts/serve.sh), the campaign harness (`sweep.sh`, `phase1.py`) and the pre-campaign 32k launch script (`serve-v1-32k.sh`). |
| [`assets/`](assets/) | Tuned `int2_w2a16` Triton configs for the RTX PRO 4000, the routing-mass histogram `expert_freq.pt`, the accepted-state record `phase1_state.json`. |
| [`docs/`](docs/) | [`WRITEUP.md`](docs/WRITEUP.md) (quantization and base patch), [`ELASTIC_MEMORY.md`](docs/ELASTIC_MEMORY.md) (the memory work), the design plans with outcomes, [`HISTORY.md`](docs/HISTORY.md) (what was rejected and why), [`TIMELINE.md`](docs/TIMELINE.md) (the dated timeline), the engineering log [`CAMPAIGN.md`](docs/CAMPAIGN.md), validation logs under [`docs/logs/`](docs/logs/). |
| [`CHANGELOG.md`](CHANGELOG.md) | Release changelog. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to contribute, the unit tests, the validation protocol, the measurement rules. |
| [`CITATION.cff`](CITATION.cff) | Citation metadata. |
| [`LICENSE`](LICENSE) | Apache License 2.0 (code). |

## What the patch does

[`sglang/qwen4exp-serving-73a255206f.patch`](sglang/qwen4exp-serving-73a255206f.patch), the serving patch,
is the full difference between SGLang commit `73a255206f` and the tree that served every number above
(34 files, +4,155 / -89 lines, SHA-256 `92f669b2…744bb5`), produced from the served tree and verified in
a clean worktree: base commit + patch reproduces all 34 files byte for byte
([`sglang/PATCH_NOTES.md`](sglang/PATCH_NOTES.md) section 2). It is the flattened result of the base
patch plus every layer in [`patches/`](patches/); the layered scripts remain useful for peeling
features off one at a time. Terms used below: Gated DeltaNet (GDN) and Qwen sparse attention (QSA) are
the two layer types of the model (36 + 12 of 48 layers); PLE is SGLang's name for the model's 320M-row
n-gram embedding table (`ple.json`, `--no-ple-offload-embedding`); VMM is CUDA virtual memory
management. Mechanisms, in the order they were built:

* **2-bit MoE path** (base patch, [`docs/WRITEUP.md`](docs/WRITEUP.md)). Stock SGLang and vLLM cap
  `moe_wna16` at 4 and 8 bits. The patch adds a 2-bit unpack to the fused MoE Triton kernel and a
  2-bit loader and addresses nine findings that only surface with CPU offload on sm_120: eight
  correctness fixes (the worst one silent: a `conv_weights` view that went stale when the offloader
  replaced `param.data`) and, for the ninth, whole-layer offloading replaced by an `ExpertStreamer`
  that moves only the ten selected experts per layer per token (26 GB -> 0.31 GB of PCIe traffic per
  token). The 51.2 GB fp8 PLE table is memory-mapped from NVMe instead of pinned in host RAM.
* **Read experts where they are** (`patches/ncontig_gemv.py`, `placement.py`). Experts are re-laid
  after loading to an N-contiguous `[E, K/16, N]` int32-word layout; the batch-1 decode GEMV reads each
  expert in place through an int64 address table, from HBM at 320 GB/s or from pinned host memory at
  51 GB/s (PCIe line rate, measured inside the kernel; CAMPAIGN.md:196-204). Residency is an address,
  so placement is a table write: the hottest S = 184 experts of every layer (routing-mass histogram
  `assets/expert_freq.pt`; the top 32 / 64 / 128 / 171 / 256 of 512 experts carry 37 / 53 / 73 / 82 /
  93 % of the mass, CAMPAIGN.md:206-216) live on the GPU, the rest in pinned host slots.
* **Breakable CUDA graphs for decode** (`host_fixes.py` items `bcg`, `bcg2`): the mmap PLE lookup is
  the one eager break; capture 3.97 s / 0.12 GB (CAMPAIGN.md:102, :291).
* **Elastic expert cache** (`gemv/row_arena.py`, `gemv/expert_elastic.py`, `patches/elastic.py`).
  Every (layer, tensor kind) keeps its expert rows in a CUDA VMM arena in routing-mass rank order;
  growing gathers rows from host slots, shrinking copies the tail back and unmaps 2 MiB granules, so
  VRAM really returns to the driver while every address stays fixed and captured graphs stay valid. S
  is a live dial through the `SGLANG_MOE_ELASTIC_CTL` file (`tools/elastic_sweep.py`).
* **Lazy VMM KV cache** (`patches/kv_lazy.py`). The KV cache is reserved as address space for
  262,144 tokens and backed in 2,048-token steps as pages are handed out; at idle the backing beyond a
  4,096-token floor is unmapped. A watermark keeps 1,536 MB driver-free after every commit by shrinking
  the elastic expert cache first; prompts whose KV cannot be backed are refused at admission (capacity =
  min(requested, 0.77 x profiled): 366,976 profiled -> 262,144 admitted, `docs/logs/tiers-validate.log`
  lines 3-4).
* **Quantized KV cache formats for the QSA layers** (`patches/kv_fp8.py` < `kv_int8.py` <
  `kv_int4.py` < `kv_tiers.py`). The sparse-attention kernels read bf16 only, so quantized storage
  gets a dequantizing read path: a fused quantize+scatter Triton write and dequantization inside the
  two existing gather sites (decode compaction into the FlashAttention scratch, prefix-chunk row
  gather). The attention kernels are untouched; scales live in the same lazy VMM arena as the payload.
  The default, the tiered KV cache (`int8ring_int4`), dual-writes every token INT8 into a ring of the
  last 8,192 slots and INT4 into the full-context pool; readers test ownership on the device.

The full mechanism descriptions are in [`docs/ELASTIC_MEMORY.md`](docs/ELASTIC_MEMORY.md) and the
per-layer table in [`patches/README.md`](patches/README.md).

### KV cache modes

| `--kv-cache-dtype` | Bytes/token (12 QSA layers, 2 KV heads x 256) | Longest prompt proven | Quality evidence | Source |
|---|---:|---|---|---|
| `auto` (bf16) | 24 KB | 68,905 tokens proven (48.9 s, 1,408 tok/s, decode 53.3); estimated ceiling 64-80k on the reference host | reference | CAMPAIGN.md:348, :339 |
| `fp8_e4m3` (read path added) | 12 KB | speed parity with bf16 (55.9-57.2 / 2,303-2,339); a 1-token prompt crashes in this mode | 512-window fake-quant e4m3: NLL -0.0133, mean abs. dlogprob 0.1099 (noise 0.099) | CAMPAIGN.md:353, :355, :368 |
| `int8_g64` (INT8, fp16 absmax scale per 64-channel group) | 12.4 KB | 162,215 tokens (95.8 s, 1,694 tok/s, decode 51.2); ~179k refused at admission | 512-window real server: mean abs. dlogprob 0.094 vs noise 0.099, NLL +0.010 (noise +-0.008) | CAMPAIGN.md:370-372 |
| `int4_g32` (nibble-packed, scale per 32-channel group) | 6.75 KB | 257,905 tokens (182.2 s, 1,415 tok/s, decode 51.9); needle 5/5 at 247,629 tokens | all-position NLL +0.0088 / 0.138 vs bf16 pool; prefill 1,493 tok/s at 10k (the unpack gather) vs 2,316 for INT8 | CAMPAIGN.md:406, :407, :409 |
| **`int8ring_int4` (default, the tiered KV cache)** | 7,308 B at 256k + 103.8 MB ring | 257,905 tokens (165.3 s); needles 5/5 at 41k and 248k | all-position NLL -0.0001 / 0.0743 (INT8 quality where attention is dense) | CAMPAIGN.md:413, :448 |

## Reproducing the measurements

All numbers were measured on one RTX PRO 4000 Blackwell (24 GB) with 32 GB host RAM against a server
launched with the flag set and environment of `scripts/serve.sh` (the campaign harness `scripts/sweep.sh`
passed the same set; the concurrency flags were 1 / 1 for every headline log except `night4.log` and
`elastic.ctl.status`, which were recorded at restart #23 with `--max-running-requests 4
--max-mamba-cache-size 8`). The tools talk to `127.0.0.1:30000` and never start a server. Expected
values are those of the accepted state; a different GPU, disk or host will differ.

| Number | Command | Expected (accepted state) | Notes |
|---|---|---|---|
| Decode and prefill at five contexts | `python3 tools/bench_speed.py` | 56.2 / 54.3 / 56.8 / 55.7 / 54.5 tok/s decode; 2,271 tok/s prefill at 10,001 | Self-contained; one streamed generation of 200 tokens per context (CAMPAIGN.md:306). Re-measure after the startup transient of the first minutes (CAMPAIGN.md:336). |
| Needle retrieval at ~41k and ~248k tokens | `python3 tools/needle_test.py 60000` then `python3 tools/needle_test.py 360000` | 5/5 at 41,370 tokens (prefill 41.2 s, decode 52.7); 5/5 at 247,629 tokens (prefill 257.2 s, decode 50.8) | Self-contained; the argument is a target token count: the script builds int(N x 0.55) random words (60000 -> 41,370 tokens, 360000 -> 247,629 tokens); appends to `tools/needle_results.tsv`. Random-word text prefills slower than prose (~920 vs ~1,500 tok/s at 248k, CAMPAIGN.md:408-409). Reference rows: [`docs/logs/needle_results.tsv`](docs/logs/needle_results.tsv). |
| 257,905-token prompt | `python3 tools/longctx_test.py 222000` | prefill 165.3 s (1,560 tok/s), decode 52.3 tok/s | Prints the elastic status before, during and after (the expert cache shrinks to make room for the KV backing and regrows). |
| Long-text NLL vs a bf16 KV cache | `python3 tools/nll_long.py save bf16` on a server started with `--kv-cache-dtype auto`, then `python3 tools/nll_long.py check bf16` on the default mode; `NLL_LONG_ALL=1` for the all-position figure | -0.0001 nats/token, mean abs. dlogprob 0.0743 over 8,561 positions (512-window: within noise +-0.008 / 0.099) | The text is the first 26,000 characters of `docs/CAMPAIGN.md` (9,586 tokens; stable because the log is append-only); override with `NLL_LONG_TEXT=<file>`. Shrink the expert cache to S = 184 first (`tools/elastic_sweep.py 184`) so the input logprobs fit. |
| Short held-out NLL | `python3 tools/nll_eval.py save NAME` / `check NAME` | de 1.512 / en 1.278 / py 0.414 nats/token on the exact 2-bit state; INT8 re-pack -0.001 overall | Three ~700-token passages, blind to the KV cache (CAMPAIGN.md:133-137, :300). |
| Logprob oracle (exactness) | `python3 tools/logprob_diff.py save NAME` on a reference server, `check NAME` on the changed one | same-config noise MAX 0.09 / MEAN 0.002 nats; accept MEAN <= 0.01, MAX <= 0.5 (CAMPAIGN.md:247) | Needs `tools/greedy_diff.py` (prompt set) and `tools/greedy/oa.json` (three 200-token greedy continuations; the oracle scores the first 150 of each, 450 tokens), both included; `tools/logprob/lp2.json` is the reference saved on the accepted INT8-dense state. |
| Expert-cache floor | `cat <ctl>.status` (the file next to `SGLANG_MOE_ELASTIC_CTL`) | S = 184 on all 48 layers, arena 11.062 GiB (11.88 GB), 84.3 % routing mass | Reference: [`docs/logs/elastic.ctl.status`](docs/logs/elastic.ctl.status), recorded at restart #23 (8 GDN state slots). The status file's `_GB` fields are bytes / 2**30, i.e. GiB; its 2.084 GiB free VRAM depends on `--max-mamba-cache-size` (0.16 GB per slot, CAMPAIGN.md:303), so a `serve.sh` server reports a different free figure. Arena size and routing mass are configuration-independent (CAMPAIGN.md:320). |

The validation sequence a KV mode has to pass, and the thresholds, are in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Upstream status

The patch is prepared for SGLang as five reviewable commits (correctness fixes and breakable graphs
under CPU offload; 2-bit `moe_wna16` with expert streaming and the in-place GEMV; VMM-backed
elasticity; quantized KV pools with dequant-on-gather; NGRAM speculation on PLE models). The plan, what
is general and what is Qwen4-Exp-specific, and the hunks that are measurement aids and do not go
upstream are in [`sglang/UPSTREAM.md`](sglang/UPSTREAM.md) and
[`sglang/PATCH_NOTES.md`](sglang/PATCH_NOTES.md) section 5.

## Hardware and software requirements

| Resource | What was used | Notes and sources |
|---|---|---|
| GPU | RTX PRO 4000 Blackwell, 24 GB (23.89 GiB per `nvidia-smi`), sm_120 | The only GPU this was run on. The tuned Triton configs in `assets/moe_configs/` are for this device and Triton 3.7.1 ([`assets/README.md`](assets/README.md)); FlashAttention 2 must be built for sm_120 (`docs/WRITEUP.md` section 7). |
| Host RAM | 32 GB | After loading, 31 offloaded expert layers pin ~24 GB and ~2 GB stay free (CAMPAIGN.md:305); this is the binding limit, see [Limits](#limits). Less than 32 GB has not been tried. |
| Disk | Checkpoint 38,755,352,600 B; PLE table `ple/ple.f8_e4m3.bin` 51,200,245,760 B + `ple/ple.json`; the virtualenv | The PLE table is never loaded into RAM: it is memory-mapped and read as 160-byte rows at random offsets, so the page cache holds the hot rows and the rest streams from disk. |
| NVMe | Required for the PLE table's random 160-byte reads | `tools/nvme_probe.py` measures whether a disk sustains the pattern (O_DIRECT, pessimistic). |
| Software, serving | torch 2.13.0+cu130, triton 3.7.1, flash-attn 2.8.3.post1 built with `FLASH_ATTN_CUDA_ARCHS=120`, transformers 5.12.1 (pinned by SGLang at `73a255206f`), SGLang at commit `73a255206f` + the patch, the cu13 toolkit inside the venv on `PATH` / `CUDA_HOME` (nvcc 13.3.73, cuda-runtime 13.3.29) | `docs/WRITEUP.md` "Setup" and section 7; the venv that ran every measurement |
| Software, quantization | AutoRound 0.14.2 with transformers 5.16.0, torch 2.11.0+cu128, Triton 3.6.0, in its own virtualenv | `scripts/pipeline.sh`; the Hub model card, "Reproduce the quantization" |

The checkpoint is `auto_round:auto_gptq`, bits 2, group_size 128, sym, with 2,342 `extra_config`
entries (its `config.json`); 85 dense projection tensors are re-packed to INT8 g128 (`scripts/requant_int8.py`,
CAMPAIGN.md:298-300); the PLE table is 320,001,536 rows x 160 fp8 (`ple.json`). The AutoRound output
measures 2.572 bits per weight (`docs/WRITEUP.md` section 2).

## The path from 2.24 to 56 tok/s

| Step | Decode tok/s | Prefill tok/s | Source |
|---|---:|---:|---|
| Stock SGLang CPU offload (26 GB of PCIe traffic per token) | 2.24 | ~500 | `docs/WRITEUP.md` section 5 |
| Base patch: 2-bit MoE path, nine findings (eight fixes, the whole-layer offload replaced by the `ExpertStreamer`), mmap PLE | 15.5 | ~1,158 | `docs/WRITEUP.md` section 6; CAMPAIGN.md:20 |
| Campaign baseline (same flags, measured with the old two-request bench) | 13.4 | 1,189 | CAMPAIGN.md:237 |
| Tuned Triton configs for `int2_w2a16` | 15.2 | 1,190 | CAMPAIGN.md:252 |
| `--max-mamba-cache-size 2`, chunked prefill 1024 | 15.4 | 1,662 | CAMPAIGN.md:256, :258 |
| Host-side fixes (offloader hook, gather skip, memo, rope hoist, parallel PLE pread) | 19.4 | 1,443 | CAMPAIGN.md:260 |
| N-contiguous int32-word expert layout + in-place batch-1 GEMV through address tables | 21.8 | 2,249 | CAMPAIGN.md:286 |
| Breakable CUDA graphs for decode (PLE lookup as the eager break) | 40.0 | 2,249 | CAMPAIGN.md:291 |
| Frequency-based expert placement (hottest 184 experts per layer on the GPU) | 48.4 | 2,334 | CAMPAIGN.md:296 |
| INT8 re-pack of 85 dense tensors (approximation, NLL -0.001 overall) | 48.8 | 2,319 | CAMPAIGN.md:300 |
| Bench fix: streamed inter-token measurement replaces the two-request bench (re-base) | 56.0 | 2,362 | CAMPAIGN.md:306 |
| Elastic expert cache (VMM row arenas, live S control) | 56.2 | 2,335 | CAMPAIGN.md:319 |
| Lazy VMM KV backing, then 131,072-token context | 55.4 / 56.9 | 2,323 / 2,340 | CAMPAIGN.md:330, :336 |
| Own INT8-G64 KV cache (162k tokens proven) | 56.1-57.4 | 2,301-2,316 | CAMPAIGN.md:370, :372 |
| Own INT4-G32 KV cache (257,905-token prompt) | 55-57 | 1,493 | CAMPAIGN.md:406, :407 |
| The tiered KV cache (`int8ring_int4`) = the default | 54-57 | 2,271 | CAMPAIGN.md:413 |

Every exact step was gated by a teacher-forced logprob oracle (450 fixed continuation tokens; same-config
noise MAX 0.09 / MEAN 0.002 nats; accept MEAN <= 0.01, MAX <= 0.5 for host-side changes, MEAN <= 0.05
for kernel-class changes after a per-layer identical-input A/B, CAMPAIGN.md:92-96, :247). Approximate
steps were gated by held-out NLL. Steps that did not improve the number were reverted; they are listed
with their reasons in [`docs/HISTORY.md`](docs/HISTORY.md), and the dated sequence is in
[`docs/TIMELINE.md`](docs/TIMELINE.md).

The KV precision ladder that led to the default (all-position test, bf16 pool as reference,
CAMPAIGN.md:406): noise +0.0002 / 0.059; fake INT4-G32 +0.0078 / 0.139 (rerun +0.0059 / 0.137,
CAMPAIGN.md:416); real INT4-G32 +0.0088 / 0.138; fake INT2-G16 +0.3004 / 0.619 (the cliff); K/V := 0
control +3.699 / 3.923 (the metric is sharp). No valid all-position INT8-vs-bf16 figure exists; the INT8
evidence is the 512-window comparison in the KV table (see the ERRATUM note in `docs/HISTORY.md`).

Speculative decoding was evaluated and left opt-in: `patches/ngram_ple.py` makes NGRAM speculation work
with the PLE embedding (linear draft chain, PLE n-gram history committed after verify) and is lossless
within the decode-vs-prefill floor, but own-history n-gram drafts accept only 1.11-1.27 tokens per step,
so decode lands at 22-25 tok/s with eager verify vs 56 without (CAMPAIGN.md:387). The MTP head is bf16
(4.85 GiB, `docs/WRITEUP.md` section 3; its experts alone 4.7 GiB, CAMPAIGN.md:379) and does not fit
next to the expert cache.

## Limits

* **Host RAM is the binding limit.** 31 offloaded layers pin ~24 GB of 32 GB; a runtime
  `cudaHostAlloc` of 4 MB took 22 s and triggered `systemd-oomd`; a 315 MB pinned reserve at startup
  killed the server at load (CAMPAIGN.md:346-347). The elastic expert cache therefore never pins after
  load, its floor is S = 184, and the published configuration serves one request at a time
  (CAMPAIGN.md:454). The server runs in a `systemd-run` scope with `MemoryMax=30G` (27G caused reclaim
  storms at ~55k-token prefills, CAMPAIGN.md:345). Do not exempt the scope from `systemd-oomd`; see
  the host-RAM section of `docs/ELASTIC_MEMORY.md`.
* **`fp8_e4m3` mode crashes on a 1-token prompt** (CAMPAIGN.md:355); the default mode does not.
* **Random text churns the PLE page cache** (fresh 4 KB page per n-gram lookup); `ple_random.py` drops
  pages after bulk gathers, and prefill on random text runs at ~920 tok/s vs ~1,500 on prose at 248k
  (CAMPAIGN.md:408-409).
* **Long prefills are O(prefix) in the model**: the paged prefix-chunk kernel was built, verified and
  rejected on timing (6.1 ms vs 1.4 ms per head-layer at prefix 60k, CAMPAIGN.md:439;
  `docs/KV_PAGED_PREFIX_PLAN.md` "Outcome"); the slope is the QSA indexer.
* **The tuned Triton configs and the placement histogram are specific** to the RTX PRO 4000 Blackwell
  (Triton 3.7.1) and to the routing probe's workload mix; another GPU needs a tuning run, another
  workload may want its own histogram (`assets/README.md`).
* No standard-benchmark score is published for the 2-bit model; the internal quality protocol (NLL
  ladder, logprob oracle, needle retrieval) is what the numbers above rest on (`docs/HISTORY.md`).

Packaging notes on the patch (which environment-gated measurement hooks are in the serving patch, what
has and has not been verified about the layered scripts) are in [`patches/README.md`](patches/README.md).

## License

* The code in this repository (`sglang/`, `patches/`, `gemv/`, `tools/`, `scripts/`, `docs/`) is
  licensed under the Apache License 2.0, see [`LICENSE`](LICENSE).
* The serving patch and the patch scripts contain SGLang diff context and replacement anchors; SGLang
  is Apache-2.0, (c) SGLang contributors.
* The quantization pipeline drives AutoRound (Apache-2.0, (c) Intel Corporation).
* The weights are not in this repository. They are published under the Qwen Community License 1.0 at
  https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang; base
  model Qwen3.8-Flash-Next (c) Qwen Team, Alibaba Group.

## Citation and contact

Maximilian Roland Haberstroh, Haberstroh Systems, 2026 — `info@haberstroh-systems.de`. Citation metadata
is in [`CITATION.cff`](CITATION.cff) (GitHub renders it as "Cite this repository"); the Hub model card
carries the matching BibTeX entry.
