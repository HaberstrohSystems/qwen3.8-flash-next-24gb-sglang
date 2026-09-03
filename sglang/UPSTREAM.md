# Upstream status: the SGLang series for Qwen3.8-Flash-Next on one 24 GB GPU

Status of the contribution to `sgl-project/sglang` as of 2026-09-03. The reviewable form of the
serving patch is the five-commit series under [`upstream/`](upstream/); the flat patch
[`qwen4exp-serving-73a255206f.patch`](qwen4exp-serving-73a255206f.patch) is the verbatim served
diff and stays as the reproduction artifact ([`PATCH_NOTES.md`](PATCH_NOTES.md)).

## Status (2026-09-03)

Posted to `sgl-project/sglang`: RFC issue [#37792](https://github.com/sgl-project/sglang/issues/37792);
pull requests against `qwen4-main-squashed`: [#37793](https://github.com/sgl-project/sglang/pull/37793) (part 1),
[#37794](https://github.com/sgl-project/sglang/pull/37794) (part 5, standalone `spec_utils.py` fix),
[#37796](https://github.com/sgl-project/sglang/pull/37796) (part 2, draft), [#37797](https://github.com/sgl-project/sglang/pull/37797) (part 3, draft),
[#37798](https://github.com/sgl-project/sglang/pull/37798) (part 4, draft). Branches live in the fork
[HaberstrohSystems/sglang](https://github.com/HaberstrohSystems/sglang) (`q4-pr1` .. `q4-pr5`, `q4-pr5-alone`,
`qwen4exp-24gb-serving-q4head`, reference `qwen4exp-24gb-serving`).

## Review target: `qwen4-main-squashed`, not `main`

Qwen3.8-Flash-Next support is not on SGLang `main`. It lives in the open PR #36497 "Introduce
Qwen 3.8 Flash Next" (head branch `sgl-project/sglang:qwen4-main-squashed`, base `main`). Our
base commit `73a255206f` is the first commit of that branch; `main` contains none of the Qwen4
files, so a patch against `main` has nothing to apply to. The series therefore targets
`qwen4-main-squashed` and is rebased onto its head, `78c5024e9d` (2026-08-31), which is five
commits ahead of our base:

| upstream commit | change | touches our files? |
|---|---|---|
| `7c66045d71` (#36649) | fix(qsa): enable trtllm-gen sparse decode on sm_121 | `qwen_sparse_attn_backend.py` |
| `639db13527` (#36772) | fix(qwen4): accept `qwen_sparse_attention` layer type alias | `configs/qwen4_exp.py` (supersedes our hunk) |
| `71946eb488` (#36667) | chore: make qwen4-main-squashed pre-commit clean | `qsa_indexer.py`, `qwen_sparse_attn_backend.py` |
| `99c9362e66` (#36806) | fix(qsa): route exact SM120 to FlashInfer sparse decode | `qwen_sparse_attn_backend.py` |
| `78c5024e9d` (#36845) | fix(qsa): restore SM121 correctness | `qwen_sparse_attn_backend.py` |

PR #36497 itself is open without a `run-ci` label; maintainers report open correctness issues on
SM89 / SM121 and have stated no merge timeline. The series can be reviewed on the branch now and
follows it to `main` when it lands.

## The series

Five commits, author Maximilian Roland Haberstroh, no sign-off (the contribution guide requires
none), each with a body in the PR template's sections and sourced numbers. Two exports:

| directory | base | commits | size |
|---|---|---|---|
| `upstream/series-q4head/` | `78c5024e9d` (head of `qwen4-main-squashed`) | the PRs | 39 files, +8,042 / -94 |
| `upstream/series-base/` | `73a255206f` (the served base) | the same five commits before the rebase | 40 files, +8,082 / -175 |

The difference between the two is the `configs/qwen4_exp.py` alias hunk that #36772 already
carries, plus the conflict resolutions described below. Compared with the flat patch, the series
removes the measurement and debug aids (`SGLANG_KV_STATS`, `SGLANG_KV_FAKEQ` and their default
statistics path, `SGLANG_NAN_TRACE`, `SGLANG_MOE_NCONTIG_LAYERS`, `SGLANG_NGRAM_CHECK`,
`SGLANG_NGRAM_FORCE_REJECT`, unused imports, blank-line-only hunks, two German comments) and adds
seven registered unit tests; the served behaviour is otherwise unchanged.

| part | commit (q4head) | title | files | general | Qwen4-Exp-specific | tests |
|---|---|---|---|---|---|---|
| 1 | `80b055f6fe` | fix(qwen4): CPU-offload correctness, breakable graphs, mmap PLE table | 9, +423/-33 | `GemmaRMSNorm` device fix, offloader `tie_weights=False` and expert-only offload, `conv_weights` property (silent NaN), `LogitsProcessorOutput` in the breakable graph backend, fp8 write saturation | `qwen_sparse_attention` layer type, `packed_modules_mapping`, language-model-only entry, mmap PLE table with `pread` / `madvise` / `fadvise`, PLE lookup as the eager graph break, QSA indexer sync hoist | none |
| 2 | `f43efb9b32` | feat(moe_wna16): 2-bit experts, expert streaming, N-contiguous GEMV | 9, +1,583/-33 | everything: 2-bit unpack and word-load kernel, loader, expert streaming, in-place GEMV through address tables, routing-mass placement | none | `test_moe_wna16_int2.py` (7 tests, 34 subtests) |
| 3 | `5563e85908` | feat(mem_cache): elastic VMM expert row arenas and lazy KV backing | 8, +1,070 | `RowArena`, `ExpertElastic`, lazy KV backing on `KvVmmBufferOwner`, allocator hooks, admission cap | none (wired to the `moe_wna16` layout of part 2) | `test_row_arena.py` (3) |
| 4 | `c73e603910` | feat(qsa): quantized KV pools with dequant-on-gather (fp8/int8/int4) | 14, +4,424/-25 | pool classes `int8_g64`, `int4_g32`, `int8ring_int4` and their write kernels | the read path at QSA's two gather sites | `test_sparse_attn_fp8_gather.py` (2), `test_int8_kv_pool.py` (10), `test_int4_kv_pool.py` (14), `test_tiered_kv_pool.py` (15) |
| 5 | `7ec06ab0ab` | fix(spec): commit PLE state after ReplaySSM verify, NGRAM on Qwen4-Exp | 4, +542/-3 | the `spec_utils.py` bug fix (also MTP top-k 1), `_linearize_chain` | NGRAM guard drop, startup self-check | `test_ngram_linearize_chain.py` (8, CPU) |

PR descriptions: [`upstream/PR-1.md`](upstream/PR-1.md) to [`PR-5.md`](upstream/PR-5.md);
the issue that introduces the series: [`upstream/RFC.md`](upstream/RFC.md).

## What remains for upstream

Open points, all disclosed in the PR texts and asked as questions in the RFC:

1. Configuration surface: every switch is an environment variable (`SGLANG_MOE_EXPERT_STREAM`,
   `SGLANG_QWEN4_PLE_MMAP`, `SGLANG_MOE_NCONTIG`, `SGLANG_MOE_GEMV`, `SGLANG_MOE_PLACEMENT*`,
   `SGLANG_MOE_ELASTIC*`, `SGLANG_KV_LAZY*`, `SGLANG_KV_TIERS_W`) except the three
   `--kv-cache-dtype` values; the control file of the elastic cache should become an endpoint.
2. Generic dispatch for the quantized pools: the read path exists only in the QSA backend, the CPU
   fallback raises, and other GPU backends would read raw bytes. `kv_bits` / `kv_tiered` are
   `getattr` class attributes; `torch.uint8` maps to `int4_g32` in `TORCH_DTYPE_TO_KV_CACHE_STR`.
3. SM120 decode routing on the branch head: since #36806 the QSA backend on exact SM120 runs
   FlashInfer's XQA kernel after the gather instead of FA2. The dequant-on-gather sites are on
   both routes, so no guard was added; the JIT needs an nvcc >= 12.9 at `CUDA_HOME`
   (FlashInfer's bound, `flashinfer/compilation_context.py`), and the published speed numbers
   (measured on `73a255206f` with FA2) have not been re-measured on `78c5024e9d`. A standalone
   probe of the FlashInfer kernel at the backend's shapes (24 query heads, 2 KV heads, head_dim
   256, page 64, topk 2,051 = `indexer_budget` 2048 + `indexer_compress_ratio` 4 - 1) matched a
   torch reference within bf16 precision (max relative error 3.9e-3, 2.5e-3, 2.5e-3, no NaN;
   [`../tools/probe_trtllm_sm120.py`](../tools/probe_trtllm_sm120.py), output
   [`../docs/logs/probe_trtllm_sm120.log`](../docs/logs/probe_trtllm_sm120.log)).
4. Documentation: the new `--kv-cache-dtype` values, the environment variables, the `ple.json`
   format, the placement-profile format and the host-memory caveats.
5. Tests still missing: `conv_weights` under a simulated `param.data` swap, the BCG helpers with
   a `LogitsProcessorOutput`, the mmap embedding on a generated table, allocator hooks with a
   mocked VMM owner, `ExpertElastic`, an end-to-end run on a synthetic 2-bit MoE, and a test that
   the PLE context advances after a verify step.
6. Known open issues in the code: `fp8_e4m3` mode crashes on 1-token prompts; the scheduler does
   not survive `alloc_extend` returning `None` mid-prefill (the admission cap refuses such prompts
   earlier); `lazy_release` assumes the pool goes fully idle (validated at
   `--max-running-requests 1`); the radix cache was disabled throughout the measurements.
7. Portability: the tuned Triton configs and the placement profile are for one GPU and one model.
8. No standard benchmark score is published for the 2-bit checkpoint; the quality evidence is the
   internal protocol (short held-out NLL, long-text NLL ladder against a bf16 KV cache, logprob
   oracle, needle retrieval).

## Related pull requests

- PR #36497 "Introduce Qwen 3.8 Flash Next" (JustinTong0323): the review target.
- Draft PR #36787 "[Qwen 3.8 Flash Next] Add sm120 (RTX PRO 6000 Blackwell) support" (Dev-Jahn,
  124 files): overlaps 11 files of the series (`configs/qwen4_exp.py`, `qsa_indexer.py`,
  `sparse_attn.py`, `qwen_sparse_attn_backend.py`, `kv_cache_configurator.py`, `memory_pool.py`,
  `pool_configurator.py`, `models/qwen3_5.py`, `models/qwen4_exp.py`, `server_args.py`,
  `spec_utils.py`), adds `SGLANG_QSA_DECODE_BACKEND` / `SGLANG_QSA_MQA_BACKEND` /
  `SGLANG_PLE_OFFLOAD_NUMA_INTERLEAVE` and targets 96 GB cards. Cited as related work; not
  merged into the series. Which lands first is one of the RFC's questions.

## How the series was verified

- Flat patch: applies cleanly to `73a255206f` and reproduces all 34 patched files of the served
  tree byte for byte (`PATCH_NOTES.md`, section 2).
- Series on the served base: the five commits reproduce the flat patch minus the removed aids and
  plus the tests (branch-vs-patch diff inspected hunk by hunk).
- Rebase onto `78c5024e9d`: cherry-picks 2, 3 and 5 applied cleanly. Part 1 conflicted in
  `qsa_indexer.py` (two regions: our `_qsa_ensure_rope` helper and its call in `apply_rope`
  against the reformatting of #36667) and part 4 in `qwen_sparse_attn_backend.py` (one region:
  our helper-method block at a blank line #36667 removed). Resolved keeping upstream formatting
  and our logic; verified by diffing each resolved file against the original commit (only the
  upstream hunks of #36649 / #36806 / #36845 remain) and by a branch-vs-branch diff that equals
  the upstream `73a255206f..78c5024e9d` diff. The `configs/qwen4_exp.py` hunk merged as
  identical to #36772 and dropped out of part 1; its commit body says so.
- SM120 routing after the rebase: read at `78c5024e9d` and probed twice. Backend selection
  always instantiates `QwenSparseAttnBackend` for the sparse layers of Qwen4-Exp
  (`--attention-backend triton` only sets the GDN / prefill backends); both decode routes compact
  through `qwen_sparse_kv_extraction_compact_triton` with our scale / ring arguments into a bf16
  scratch, the prefix-chunk gather is not device-routed, and the CPU fallback raises. With nvcc
  12.0 the FlashInfer JIT aborts loudly; with nvcc 13.3 it compiles and matches a torch softmax
  reference (max relative error 2.5e-3 to 3.9e-3, no NaN) at the backend's shapes
  (`../tools/probe_trtllm_sm120.py`, output `../docs/logs/probe_trtllm_sm120.log`).
- Formatting, per commit, run inside the worktree: black 26.1.0, isort 7.0.0, ruff 0.15.1
  (`--select=F401,F821,UP037`), codespell, `py_compile`: clean for all five commits.
- Tests, on an idle GPU with the cleaned worktree on `PYTHONPATH` and the serving toolchain on
  `PATH` (ninja, nvcc 13.3 from the virtualenv's `nvidia/cu13`): each part's tests after its
  cherry-pick, and all seven files on the finished branch: 59 passed, 40 subtests passed,
  0 failed. Without `ninja` on `PATH` the Triton JIT tests fail with `FileNotFoundError: ninja`
  and the real-corpus NGRAM case skips; this is stated in the PR texts.
- Nothing under the served checkout was modified; no `git fetch`, `git push`, `git stash` or
  worktree commands were run.

## Reproduction of the accepted state

Launch line: [`../scripts/serve.sh`](../scripts/serve.sh) (`--max-running-requests 1
--max-mamba-cache-size 1`); flags and environment in `PATCH_NOTES.md`, section 6. Stack: torch
2.13.0+cu130, triton 3.7.1, flash-attn 2.8.3.post1 built for sm_120, transformers 5.12.1, nvcc
13.3 inside the virtualenv, flashinfer 0.6.17. The checkpoint and the PLE table are the Hub
download: https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang.
