Base branch: qwen4-main-squashed (PR #36497). Stacked on: nothing (part 1 of 5).

# fix(qwen4): CPU-offload correctness, breakable graphs, mmap PLE table

Part 1/5 of the Qwen3.8-Flash-Next 24 GB serving series (RFC issue: *link*). Patch:
`upstream/series-q4head/0001-fix-qwen4-CPU-offload-correctness-breakable-graphs-m.patch`
in the companion repository (9 files, +423 / -33).

## Motivation

Qwen3.8-Flash-Next (Qwen4-Exp: 176B total / 6B active, 48 layers = 36 GatedDeltaNet + 12 Qwen
sparse attention, 512 experts top-10, a 51 GB PLE n-gram embedding table) only fits a 24 GB card
with `--cpu-offload-gb`. On the base commit that path does not work: weight loading fails on a
device mismatch in `GemmaRMSNorm`, `functional_call` rejects the tied `A_log` tensors, the GDN
convolution reads a stale `.view()` of a parameter whose `.data` the offloader has replaced (GDN
output exactly zero, NaN logits), the fused `in_proj_ba` lands at the wrong bit width because
`packed_modules_mapping` never reaches the quant config, and the PLE table is allocated in full
(51.2 GB as fp8, 51,200,245,760 B; 102.4 GB if materialised as bf16, derived as
2 x 51,200,245,760 B) before any offload can act. These are the nine findings of
`docs/WRITEUP.md` section 3 in the companion repository, with finding 8 (the silent NaN) analysed
in section 4.

Reference machine: one RTX PRO 4000 Blackwell (24 GB, sm_120) with 32 GB host RAM, serving a
2.572 bpw AutoRound checkpoint (model card, "Quantization"). Sources below: dated entries of
`docs/CAMPAIGN.md` in https://github.com/HaberstrohSystems/qwen3.8-flash-next-24gb-sglang
("CAMPAIGN <date> <time>"), `docs/DECODE_PERF_PLAN.md` there, and the model card
https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang
("card, <section>").

## Modifications

Correctness under `--cpu-offload-gb` (general, not Qwen4-specific):

- `layers/layernorm.py`: `GemmaRMSNorm` re-derives `gemma_weight` on the device of the parameter
  (`_weight_loader`) and of the input (`_gemma_weight_for`); the non-persistent buffer was left on
  the GPU while the parameter went to the CPU.
- `utils/offloader.py`: `functional_call(..., tie_weights=False)` in both hook variants;
  `_CpuParamOffloader` keeps a hard reference to the pinned host storage. With
  `SGLANG_MOE_EXPERT_STREAM=1` only expert tensors (`w13_qweight`, `w2_qweight`, `w13_scales`,
  `w2_scales`, `w13_qzeros`, `w2_qzeros`) are offloaded, and the forward hook is not installed
  when every offloaded parameter of a module is a streamed expert parameter. The streamer itself
  is part 2.
- `layers/radix_linear_attention.py`, `models/qwen3_5.py`: the GDN module passes its `nn.Conv1d`
  instead of a `.view()` of its weight; `RadixLinearAttention.conv_weights` is a property that
  builds the 2D view from the current parameter on every access. Tensors and tuples pass through
  unchanged, so KDA / ShortConv / Lightning are not affected.
- `model_executor/runner_backend/breakable_cuda_graph_backend.py`: the four output-structure
  helpers understand the `LogitsProcessorOutput` dataclass; capture died with "Unsupported BCG
  output type" for any graph body that returns it.
- `mem_cache/memory_pool.py`: the fp8 write path in both `set_kv_buffer` variants skips the no-op
  `div_` by a unit scale and saturates to +-448 before the e4m3 cast (the cast returns NaN beyond
  the range). Consumed by the fp8 read path of part 4.

Qwen4-Exp specific:

- `models/qwen4_exp.py`: accept the layer type `qwen_sparse_attention` (transformers renames
  `full_attention` on load). The matching `layers_block_type` alias in `configs/qwen4_exp.py` is
  already on `qwen4-main-squashed` since #36772 and is not repeated.
  `Qwen4ExpForConditionalGeneration.__init__` calls `quant_config.update_packed_modules_mapping`
  like `deepseek_v2.py` does for Quark.
- `server_args.py`: `Qwen4ExpForConditionalGeneration` in `LANGUAGE_MODEL_ONLY_ARCHITECTURES`
  (the class already inherits the implementation; only the entry was missing). Saves the vision
  tower (0.84 GiB) and the multimodal reservation in the KV budget for text-only serving.
- `models/qwen4_exp.py`: `Qwen4ExpMmapEmbedding` serves the PLE table from a memory-mapped file
  (`SGLANG_QWEN4_PLE_MMAP=<dir>` with `ple.f8_e4m3.bin` and `ple.json`, which carries `file`,
  `rows`, `dim`, `dtype`, `weight_scale`; the file size `rows x dim x itemsize` is checked). The
  embedding is created on the `meta` device in that mode. Decode fetches rows with parallel
  `os.pread` (16 threads, GIL released; the numpy fancy-index into the memmap held the GIL for
  the whole cold row fetch, measured at 2.56-3.09 ms per token, `docs/DECODE_PERF_PLAN.md`,
  "2. Parallelise the PLE row fetch"), bulk gathers go through the memmap. `MADV_RANDOM` /
  `POSIX_FADV_RANDOM` stop the page-cache read-around per 160-byte row (CAMPAIGN 2026-09-02
  14:12); `POSIX_FADV_DONTNEED` after bulk gathers of >= 512 ids drops the pages again.
  `forward` is wrapped with `eager_on_graph(True)`, so decode runs under
  `--cuda-graph-backend-decode breakable` with the PLE lookup as the one eager break.
- `layers/attention/qsa/qsa_indexer.py`: `_qsa_ensure_rope` hoists the `positions.max().item()`
  device sync (once per QSA layer, 12 per token; `docs/DECODE_PERF_PLAN.md`, "5. Two
  micro-fixes") and pre-sizes the rotary cos/sin cache from `context_length`. On this branch it
  replaces upstream's `_ensure_cos_sin_cache_length(int(positions.max().item()))`
  call in `apply_rope`.

No new server flags; the two switches are `SGLANG_MOE_EXPERT_STREAM` and `SGLANG_QWEN4_PLE_MMAP`
(see the RFC, open question 1). No tests are added in this part.

## Accuracy Tests

Exactness oracle: teacher-forced logprobs on a fixed 10k-token prompt, mean and max |dlogprob|
per token against the previous configuration; noise floor mean 0.002, threshold for host-side
changes mean <= 0.01, max <= 0.5 (card, "Serving fidelity"; CAMPAIGN 2026-09-02 11:25).

- Offload fixes (hook skip, indexer hoist, PLE pread; measured together with the streamer of
  part 2): oracle max 0.078 / mean 0.0019, equivalent (CAMPAIGN 2026-09-02 09:30).
- Breakable decode graphs + `LogitsProcessorOutput` support: max 0.079 / mean 0.0022, equivalent
  (CAMPAIGN 2026-09-02 11:48).
- PLE page-cache fixes involve no numerics; the 248k-token needle test retrieved 5/5 codes after
  the page-drop fix (CAMPAIGN 2026-09-02 20:00).
- Open issue, disclosed: in `fp8_e4m3` mode a 1-token prompt still crashes in the QSA indexer
  prefill; >= 101-token prompts work (CAMPAIGN 2026-09-02 15:26).

Reproduction (server level; the companion repository's tools, run against a server started with
`scripts/serve.sh` there, which sets the flags and environment of the published configuration):

```
python tools/logprob_diff.py --help        # teacher-forced oracle, 450 tokens of a 10k prompt
python tools/needle_test.py --help         # needle retrieval with ignore_eos
```

## Speed Tests and Profiling

Streaming bench (`tools/bench_speed.py`, 200 streamed tokens, single request, ~10k context;
card, "Performance"), measured on `73a255206f` with the flat patch:

- offload fixes: decode 15.4 -> 19.4 tok/s (CAMPAIGN 2026-09-02 09:30; includes the streamer
  items of part 2);
- breakable decode graphs: decode 21.8 -> 40.0 tok/s at prefill 2,249 tok/s, graph capture
  3.97 s / 0.12 GB (CAMPAIGN 2026-09-02 11:33 and 11:48);
- fp8 write path: `fp8_e4m3` mode at parity with bf16, decode 55.9-57.2 / prefill 2,303-2,339
  tok/s (CAMPAIGN 2026-09-02 15:12);
- PLE read-around: before `MADV_RANDOM` / `POSIX_FADV_RANDOM`, ~55k-token prompts drove host
  memory pressure past systemd-oomd (CAMPAIGN 2026-09-02 14:12); the 248k random-text needle run
  survived only after `POSIX_FADV_DONTNEED` (CAMPAIGN 2026-09-02 19:37 and 20:00).

Not re-measured on `78c5024e9d` (see the RFC, open question 3).

Reproduction:

```
git clone https://github.com/sgl-project/sglang.git && cd sglang
git fetch origin qwen4-main-squashed && git checkout 78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385
git am /path/to/upstream/series-q4head/0001-*.patch
pip install -e python
# server and bench: companion repository, scripts/serve.sh and tools/bench_speed.py
```

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit). (black 26.1.0, isort 7.0.0, ruff 0.15.1 `F401,F821,UP037`, codespell and `py_compile` clean on the commit.)
- [ ] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests). (None in this part. Missing: a test for the `conv_weights` property under a simulated `param.data` swap, a round-trip test of the BCG helpers with a `LogitsProcessorOutput`, a test of `Qwen4ExpMmapEmbedding` on a small generated table.)
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations). (Missing: the `ple.json` file format and the two environment variables.)
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed). (Teacher-forced logprob oracle and the streaming bench above; no standard benchmark score.)
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance). (Known deviation: switches are environment variables, not server arguments.)

## Suggested reviewers

- `python/sglang/srt/models`, `python/sglang/srt/layers/attention`: @Fridge003, @ishandhanani,
  @Qiaolin-Yu (NV and model-specific optimizations); @JustinTong0323 as author of #36497.
- `python/sglang/srt/model_executor` (breakable CUDA graph backend): @merrymercy, @hnyls2002,
  @cctry (Scheduler).
- `python/sglang/srt/mem_cache` (fp8 write path): @ispobock, @xiezhq-hermann (KV Cache).
