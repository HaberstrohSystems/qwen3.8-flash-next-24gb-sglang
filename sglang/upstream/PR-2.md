Base branch: qwen4-main-squashed (PR #36497). Stacked on PR-1.

# feat(moe_wna16): 2-bit experts, expert streaming, N-contiguous GEMV

Part 2/5 of the Qwen3.8-Flash-Next 24 GB serving series (RFC issue: https://github.com/sgl-project/sglang/issues/37792). Patch:
`upstream/series-q4head/0002-feat-moe_wna16-2-bit-experts-expert-streaming-N-cont.patch`
in the companion repository (9 files, +1,583 / -33). Nothing in this part is Qwen4-specific.

## Motivation

`moe_wna16` accepts 4 and 8 bits only, so a symmetric 2-bit AutoRound / GPTQ MoE checkpoint
cannot be loaded at all. For Qwen3.8-Flash-Next (512 experts, top-10, `moe_intermediate_size`
640) the 2-bit experts are what makes the model fit next to a 24 GB card: the served checkpoint
is 2.572 bpw (model card, "Quantization"). Two more things are needed once the experts are
offloaded: `--cpu-offload-gb` copies a whole module's `state_dict` to the device on every forward
(measured ~26 GB per token at 2.3 tok/s, while 10 x 48 x 1.31 MB = 0.63 GB of expert rows are
actually needed; `docs/WRITEUP.md` section 5), and the byte-row `[E, N, K/4]` layout reads
128-byte chunks at 640-byte stride, which limits an in-place decode GEMV to 73 GB/s from device
memory and 5 GB/s from pinned host memory (CAMPAIGN 2026-09-02 08:29-08:31).

Sources: dated entries of `docs/CAMPAIGN.md` and `docs/WRITEUP.md` in
https://github.com/HaberstrohSystems/qwen3.8-flash-next-24gb-sglang and the model card
https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang.
Reference machine: one RTX PRO 4000 Blackwell (24 GB, sm_120) with 32 GB host RAM.

## Modifications

- `kernels/ops/moe/fused_moe_triton_kernels.py`: 2-bit unpack in `fused_moe_kernel_gptq_awq` as
  the generalization of the 4-bit branch (`offs_k // 4`, shift `(offs_k % 4) * 2`, mask `0x3`,
  zero point 2). New `fused_moe_kernel_gptq_awq_word`: the same kernel reading the N-contiguous
  int32-word layout (one coalesced 128-byte line per warp load instead of 32 distinct lines). The
  layout is derived from the tensor dtype (int2 + `int32` = word layout, int2 + `uint8` = byte
  layout), so nothing crosses the custom-op schema. Reviewer's choice: keep the second Triton
  function or fold it into a `constexpr` branch of the existing kernel.
- `moe_runner/triton.py`, `triton_utils/fused_moe.py`, `triton_utils/fused_moe_triton_config.py`:
  `use_int2_w2a16` plumbing, shape handling for the word layout, config dtype name `int2_w2a16`
  so a tuned int4 config is never picked up by accident. No int2 config files ship; the tuned
  configs used below (for `E={10,512}, N=160` on the RTX PRO 4000 Blackwell) live in the
  companion repository under `assets/moe_configs` and are selected through `SGLANG_MOE_CONFIG_DIR`.
- `layers/quantization/moe_wna16.py`: 2-bit loader; asymmetric `qzeros` raise
  `NotImplementedError` by design. `process_weights_after_loading` re-lays every 2-bit expert
  tensor once from `[E, N, K/4]` uint8 to `[E, K/16, N]` int32 and scales to `[E, K/128, N]`
  (same bytes; `SGLANG_MOE_NCONTIG=0` keeps the byte layout); pinned host tensors are re-pinned
  after the copy, which transiently doubles one layer's host footprint. Deferred placement pass
  at the last layer (`SGLANG_MOE_PLACEMENT=<expert_freq.pt>` with `{"mass": [L, E]}`,
  `SGLANG_MOE_PLACEMENT_S`, default 184): the hottest S experts of every layer are resident on
  the GPU, cold rows of GPU layers take the pinned slots that host layers' hot rows vacate,
  memory-neutral on both sides. With `SGLANG_MOE_ELASTIC=1` the pass hands the layers to
  `ExpertElastic` (part 3).
- `layers/moe/expert_gemv.py` (new): batch-1 int2 GEMV (`moe_gemv_int2_tab`, used for M <= 16
  in `apply()`, `SGLANG_MOE_GEMV=0` disables) that reads experts in place through int64 address
  tables; an entry may point into device or pinned host memory, the kernel indexes with the
  original expert ids. `to_word_ncontig`, `make_tables`.
- `layers/moe/expert_stream.py` (new, `SGLANG_MOE_EXPERT_STREAM=1`): gathers exactly the routed
  experts of a forward into one shared, shape-keyed staging buffer and renumbers the top-k ids
  onto it; fully GPU-resident layers skip the gather; `arange`/cast tensors are memoized.
- `test/registered/unit/layers/quantization/test_moe_wna16_int2.py` (new, `register_cuda_ci`,
  `est_time=60`, stage `base-b`, runner `1-gpu-small`; synthetic data, no checkpoint): the exact
  2-bit unpack expressions against a torch reference (bit-exact, including the real expert dims
  2560 x 640 g128); `to_word_ncontig` byte -> word round trip (bit-exact and invertible, CPU);
  `fused_moe(use_int2_w2a16=True)` on the byte layout (symmetric and with per-group `qzeros`)
  and on the word layout against a torch MoE reference on dequantized weights;
  `moe_gemv_int2_tab` through `make_tables` addresses against an fp32 reference (w13 and w2
  forms, fp16 and bf16 scales, N not a multiple of the block; a bad block is rejected).
- `test/manual/test_triton_moe_wna16.py`: `w2a16` / `w2a16b2` cases and the 2-bit packing. This
  stays a manual test: its 8-bit path fails the tolerance check with the unmodified upstream
  kernel on the reference machine (129 of 144 cases in the 8-bit / group-128 subset, identical on
  the unpatched tree), so it is not a usable gate.

## Accuracy Tests

- `test_moe_wna16_int2.py`: 7 tests, 34 subtests, pass (packing order, unpack expressions and
  the word-layout round trip bit-exact; `fused_moe` on both layouts and the GEMV within
  atol 1e-3 / rtol 2e-2 of the dequantized references, measured ~4e-3 relative). Also passes on
  the finished 5-commit branch.
- Original A/B of the same kernel path on real shapes (dequantized bf16 vs packed 2-bit through
  the same MoE machinery): bit-identical on small shapes, 1e-5 relative on large ones including
  e=64, n=640, k=2560, g128 (`docs/WRITEUP.md` section 8).
- Word layout and GEMV: A/B through the patched `invoke_fused_moe_kernel` at 64/64/32,
  16/16/128, 16/32/128 identical (2e-5) (CAMPAIGN 2026-09-02 11:04); per-layer dump A/B with
  identical inputs 5.1e-5 (M=6) / 1.7e-4 (M=1) relative, i.e. bf16 output rounding from a
  different fp32 accumulation order (CAMPAIGN 2026-09-02 11:25). End-to-end oracle on the 10k
  prompt max 0.275 / mean 0.0125, inside the kernel-class band (CAMPAIGN 2026-09-02 11:30).
- Placement is exact by construction (rows move, values do not): oracle max 0.057 / mean 0.0013
  (CAMPAIGN 2026-09-02 12:10). Tuned int2 configs: max 0.086 / mean 0.0018 (2026-09-02 09:13).

Reproduction (needs `ninja` on `PATH` (otherwise `FileNotFoundError: ninja`) and an nvcc that
supports the GPU architecture at `CUDA_HOME` (compute_120a on the reference machine; a system
nvcc 12.0 does not, the virtualenv's 13.3 does)):

```
git clone https://github.com/sgl-project/sglang.git && cd sglang
git fetch origin qwen4-main-squashed && git checkout 78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385
git am /path/to/upstream/series-q4head/0001-*.patch /path/to/upstream/series-q4head/0002-*.patch
pip install -e python
export CUDA_HOME=<toolkit whose nvcc supports the GPU architecture>; export PATH="$CUDA_HOME/bin:$PATH"
python -m pytest -v test/registered/unit/layers/quantization/test_moe_wna16_int2.py
```

## Speed Tests and Profiling

Streaming bench, single request, ~10k context (`tools/bench_speed.py`; card, "Performance"),
measured on `73a255206f` with the flat patch:

- PCIe traffic per decoded token 26 GB -> 0.31 GB with expert-only offload and the streamer
  (`docs/WRITEUP.md` section 5; first working path 2.24 -> 15.5 tok/s, sections 4 and 5).
- Tuned int2 configs: decode 13.4 -> 15.2 tok/s (CAMPAIGN 2026-09-02 09:13).
- N-contiguous GEMV micro-benchmark on real layer-5 tensors: 320 GB/s from device memory,
  51 GB/s (PCIe line rate) from pinned host memory, 2.1 ms/token all-device; the byte-shuffled
  table variant reached 157 / 14 GB/s and was rejected (CAMPAIGN 2026-09-02 08:29-08:31, 09:45).
- N-contiguous layout + GEMV end to end: decode 19.4 -> 21.8 tok/s, prefill 1,443 -> 2,249 tok/s
  (CAMPAIGN 2026-09-02 11:30).
- Routing-mass placement: the top 171 of 512 experts per layer cover 82 % of routing mass on a
  2,496-token, 3-domain probe (CAMPAIGN 2026-09-02 09:48-09:53); decode 40.0 -> 48.4 tok/s,
  prefill 2,249 -> 2,334 tok/s (CAMPAIGN 2026-09-02 12:10).

Not re-measured on `78c5024e9d` (see the RFC, open question 3). The placement profile
(`assets/expert_freq.pt`, built by `tools/expert_freq.py` from a routing dump) and the tuned
configs are per model and per GPU.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit). (black 26.1.0, isort 7.0.0, ruff 0.15.1 `F401,F821,UP037`, codespell and `py_compile` clean on the commit.)
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests). (`test/registered/unit/layers/quantization/test_moe_wna16_int2.py`, 7 tests / 34 subtests, `register_cuda_ci`.)
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations). (Missing: the layout-by-dtype convention on `fused_experts`, the placement-profile format, the environment variables.)
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed). (Kernel A/Bs, the logprob oracle, GEMV micro-benchmark and the streaming bench above; no standard benchmark score.)
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance). (Known deviation: switches are environment variables, not server arguments.)

## Suggested reviewers

- `python/sglang/kernels` (Triton MoE kernels): @BBuf (Kernel).
- `python/sglang/srt/layers/moe`, `layers/quantization/moe_wna16.py`: no Merge Oncall area is
  listed for these paths; CODEOWNERS will be requested automatically. @JustinTong0323 as author
  of #36497.
