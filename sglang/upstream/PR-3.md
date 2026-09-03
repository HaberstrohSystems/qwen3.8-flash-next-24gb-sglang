Base branch: qwen4-main-squashed (PR #36497). Stacked on PR-2.

# feat(mem_cache): elastic VMM expert row arenas and lazy KV backing

Part 3/5 of the Qwen3.8-Flash-Next 24 GB serving series (RFC issue: *link*). Patch:
`upstream/series-q4head/0003-feat-mem_cache-elastic-VMM-expert-row-arenas-and-laz.patch`
in the companion repository (8 files, +1,070). General in design; the expert side is wired to
the `moe_wna16` word layout and address tables of part 2.

## Motivation

On a 24 GB card that also has to hold a 2-bit MoE, VRAM is shared between resident expert rows and
the KV cache, and the split that is right for a short chat is wrong for a 256k-token prompt. Two
facts make a static split unnecessary: expert weights are immutable, so an expert's GPU residency
is a pure cache (eviction = table write, admission = one row copy + table write); and SGLang
already keeps the KV cache in a CUDA VMM arena (`KvVmmBufferOwner`), but commits it monotonically
at startup (CAMPAIGN 2026-09-02 12:40). Host RAM is the wall on the reference machine (RTX PRO
4000 Blackwell 24 GB, 32 GB host RAM): 31 offloaded layers pin 24-26 GB of the 32 GB, so no host
mirror can be added at runtime (CAMPAIGN 2026-09-02 12:49; model card, "Scope and limitations").

Sources: dated entries of `docs/CAMPAIGN.md` and `docs/ELASTIC_MEMORY.md` in
https://github.com/HaberstrohSystems/qwen3.8-flash-next-24gb-sglang and the model card
https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang.

## Modifications

Elastic expert cache (`SGLANG_MOE_ELASTIC=1`, wired to the placement pass of part 2):

- `layers/moe/row_arena.py` (new): `RowArena` reserves virtual address space for `max_rows` once
  (`cuMemAddressReserve`) and backs a rank-ordered prefix with physical chunks on demand
  (`cuMemCreate` + `cuMemMap`, 4 MiB chunks aligned to the 2 MiB device granularity). Shrinking
  is a tail unmap that returns the memory to the driver while every row address stays fixed for
  the process lifetime, so kernels that read rows through an address table, and CUDA graphs that
  captured them, need no recapture. Uses the driver plumbing in `cuda_vmm_utils`.
- `layers/moe/expert_elastic.py` (new): one arena per (layer, tensor kind) in routing-mass rank
  order; the int64 table `addr[e]` points into the arena for rank(e) < S and at the row's pinned
  host slot otherwise. Grow = table-driven host -> arena gather + table rewrite; shrink = D2H copy
  of the tail ranks into pool slots + table rewrite + unmap. Host memory is conserved through a
  slot pool; `free()` never pins at runtime, so S_floor is what the pool can absorb (184 rows per
  layer on the reference machine; `SGLANG_MOE_ELASTIC_PIN_MB` grants a pinned fallback budget,
  `SGLANG_MOE_ELASTIC_FILL_MB` / `_RESERVE_ROWS` steer the startup fill). A control file
  (`SGLANG_MOE_ELASTIC_CTL`: `S <n>`, `fill <MB>`, `free <MB>`, `status`) is polled from the MoE
  apply path outside graph replay and writes a status file.
- `test/registered/unit/layers/moe/test_row_arena.py` (new, `register_cuda_ci`, `est_time=20`,
  stage `base-b`, runner `1-gpu-small`): arena geometry; `ensure_rows` backs a prefix and
  driver-free memory drops by the mapped bytes (within 8 MiB); a Triton gather through an
  address table that mixes arena and pinned-host rows (one pinned host row of 200 KB) reads every
  row; a CUDA graph captured against the arena addresses replays after `shrink_rows` and after
  growing back; `close` returns the memory; `ArenaOOM` is raised for a chunk larger than the
  device. The module's former `__main__` self-test is removed in favour of this test.

Lazy KV backing (`SGLANG_KV_LAZY=1`):

- `mem_cache/kv_vmm_backing.py`: `KvVmmArena.uncommit_beyond`, `KvVmmBufferOwner.release_beyond`,
  `backed_tokens`, `bytes_per_token`.
- `mem_cache/memory_pool.py`: with `SGLANG_KV_LAZY=1` the full-attention pool is allocated
  through the VMM owner in the classic flow as well, with `SGLANG_KV_LAZY_FLOOR` (4096) tokens
  backed at start. `lazy_ensure` commits in `SGLANG_KV_LAZY_MARGIN` (2048) token steps as pages
  are handed out and keeps `SGLANG_KV_LAZY_HEADROOM_MB` (1536) driver-free after every commit:
  below the watermark it shrinks the expert cache first (through `ExpertElastic.free`), and
  `empty_cache()` is only forced when the cache can still shrink, otherwise rate-limited to once
  per 30 s. `lazy_release` unmaps beyond the floor when the pool goes idle and regrows the expert
  cache there.
- `mem_cache/allocator/paged.py`, `allocator/token.py`: `_lazy_hook` before pages are consumed;
  an allocation whose commit fails returns `None` (refused) instead of crashing;
  `_lazy_idle_check` in `_release_page_ids` / `free`.
- `mem_cache/kv_cache_configurator.py`: virtual capacity `SGLANG_KV_LAZY_TOKENS` above the
  profiled value, admitted capacity `min(requested, SGLANG_KV_LAZY_SAFETY x profiled)` (default
  0.85), so a prompt that could not be backed is refused at admission.

Known limits, disclosed: all switches are environment variables (RFC, open question 1); the
`ExpertElastic` import inside `lazy_ensure` should become a registered callback; the control file
should become an HTTP endpoint; `lazy_release` fires when the whole pool is idle (validated at
`--max-running-requests 1`); the scheduler does not survive `alloc_extend` returning `None`
mid-prefill, which is why the admission cap exists (CAMPAIGN 2026-09-02 14:56 and 14:58); the
radix cache was disabled throughout the measurements.

## Accuracy Tests

Both mechanisms move bytes without changing values; the exactness oracle (teacher-forced logprobs
on the 10k prompt; card, "Serving fidelity") confirms it: elastic cache max 0.060 / mean 0.0016
(CAMPAIGN 2026-09-02 13:17); lazy KV max 0.057 / mean 0.0014 (2026-09-02 13:42); 128k
configuration mean 0.0021 (2026-09-02 13:53).

`test_row_arena.py`: 3 tests pass on the idle GPU (2 MiB granularity, driver-free deltas within
8 MiB of the mapped bytes, graph replay valid after shrink and regrow, `ArenaOOM`). Also passes on
the finished 5-commit branch.

Reproduction (needs `ninja` on `PATH` (otherwise `FileNotFoundError: ninja`) and an nvcc that
supports the GPU architecture at `CUDA_HOME` (compute_120a on the reference machine; a system
nvcc 12.0 does not, the virtualenv's 13.3 does) for the Triton gather kernel):

```
git clone https://github.com/sgl-project/sglang.git && cd sglang
git fetch origin qwen4-main-squashed && git checkout 78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385
git am /path/to/upstream/series-q4head/000{1,2,3}-*.patch
pip install -e python
export CUDA_HOME=<toolkit whose nvcc supports the GPU architecture>; export PATH="$CUDA_HOME/bin:$PATH"
python -m pytest -v test/registered/unit/layers/moe/test_row_arena.py
```

## Speed Tests and Profiling

Streaming bench, single request, ~10k context (`tools/bench_speed.py`; card, "Performance"),
measured on `73a255206f` with the flat patch:

- elastic expert cache: decode 56.2 tok/s, prefill 2,335 tok/s (the previous configuration
  measured 56.0 / 2,362 after the bench measurement fix of 2026-09-02 12:58; CAMPAIGN 13:17).
  Live S sweep without restart: S=184 arena 11.06 GB, 1.73 GB free, 84.3 % of routing mass,
  decode 56-57.6; S=200 12.19 GB / 0.62 GB free / 86.7 % / 55.6-57.9: the dial buys +2-3 %
  decode per ~1.7 GB, its value is VRAM on demand for the KV cache (CAMPAIGN 2026-09-02 14:20).
- lazy KV: decode 55.4 / prefill 2,323 (13:42); 288 MB backed at 10k tokens (24 KB/token bf16)
  and release to the 4,096-token floor (144 MB) at idle (14:45); KV committed at startup
  0.8 GB -> 0.1 GB (`docs/ELASTIC_MEMORY.md`, "Result").
- 128k context: decode 55.0-57.5, prefill 2,270-2,340 re-measured on the live server (13:53).
- watermark rule: a 60k-token prompt first crashed because every commit ate the prefill's working
  memory (driver free 0.07 GB; 13:55); with the headroom rule a 68,905-token bf16 prompt ran at
  prefill 48.9 s (1,408 tok/s), decode 53.3 tok/s, expert cache 192 -> 184 during the request and
  regrown afterwards (14:54).
- rate-limited `empty_cache`: 257,905-token prompt (tiered pool of part 4) prefill 171.0 s ->
  165.3 s (1,508 -> 1,560 tok/s), decode 52.3 (CAMPAIGN 2026-09-02 22:19; 2026-09-03 00:55).

Not re-measured on `78c5024e9d` (see the RFC, open question 3).

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit). (black 26.1.0, isort 7.0.0, ruff 0.15.1 `F401,F821,UP037`, codespell and `py_compile` clean on the commit.)
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests). (`test/registered/unit/layers/moe/test_row_arena.py`, 3 tests, `register_cuda_ci`. Missing: allocator-hook tests with a mocked VMM owner, an `ExpertElastic` test.)
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations). (Missing: the environment variables, the control-file protocol, the host-memory caveats.)
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed). (Logprob oracle, streaming bench, live S sweep and long-prompt runs above; no standard benchmark score.)
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance). (Known deviations: environment variables, the control file, the `ExpertElastic` import in `lazy_ensure`.)

## Suggested reviewers

- `python/sglang/srt/mem_cache` (VMM backing, memory pool, allocators, configurator): @ispobock,
  @xiezhq-hermann (KV Cache).
- `python/sglang/srt/layers/moe` (`row_arena.py`, `expert_elastic.py`): no Merge Oncall area is
  listed; CODEOWNERS will be requested automatically. @JustinTong0323 as author of #36497.
