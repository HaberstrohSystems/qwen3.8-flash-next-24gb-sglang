# Kernels and unit tests

The GEMV tests need the 2-bit checkpoint for their real-tensor tests (the Hub repository:
https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang); the KV
tests need only a GPU and the corresponding patch layer applied (prerequisites per test in
[`../CONTRIBUTING.md`](../CONTRIBUTING.md)).

| File | Role |
|---|---|
| `moe_gemv_int2.py` | First in-place batch-1 int2 GEMV on the loader's `[N, K/4]` layout — exact but 73 GB/s device / 5 GB/s pinned host (CAMPAIGN.md:198-199); kept as the record of the refuted layout. |
| `moe_gemv_int2_tab.py` | The N-contiguous int32-word GEMV through a pointer table: 320 GB/s device, 51 GB/s pinned host (CAMPAIGN.md:200-204). Integrated into SGLang as `srt/layers/moe/expert_gemv.py` by `patches/ncontig_gemv.py`. |
| `row_arena.py` | `RowArena`: CUDA VMM address reservation with prefix backing in 2 MiB granules; shrink is a tail unmap that really returns memory to the driver while every row address stays fixed. Self-test in `__main__` (CAMPAIGN.md:304). Installed by `patches/elastic.py`. |
| `expert_elastic.py` | `ExpertElastic`: rank-ordered residency per (layer, kind), grow/shrink by table write + row copy, host slot pool, control file. Installed by `patches/elastic.py`. |
| `test_gemv.py`, `test_gemv_ncontig.py`, `test_gemv_tab.py` | GEMV correctness vs fp32 and bandwidth on real expert tensors. Their `Q` (the quantized checkpoint directory) defaults to `~/quant/model` (the Hub download) via `os.path.expanduser`; override with the `Q` environment variable. |
| `test_tiled_word.py` | A/B of the word-load int2 branch through the real `invoke_fused_moe_kernel` (`ref` on a pristine tree, `cmp` on the patched tree). Its 14 MB `tiled_ref*.pt` dumps are not in the repository. |
| `test_kv_fp8.py`, `test_kv_int8.py`, `test_kv_int4.py`, `test_kv_tiers.py` | Bit-exact tests of the KV write/gather kernels against torch references. |
| `test_kv_paged_prefix.py` | Test and timing of the rejected paged prefix-chunk kernel. |
| `test_ngram_chain.py` | CPU test of `_linearize_chain` (`patches/ngram_ple.py`). |
