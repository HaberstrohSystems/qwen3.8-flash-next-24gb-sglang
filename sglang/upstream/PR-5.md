Base branch: qwen4-main-squashed (PR #36497). Stacked on PR-4.

# fix(spec): commit PLE state after ReplaySSM verify, NGRAM on Qwen4-Exp

Part 5/5 of the Qwen3.8-Flash-Next 24 GB serving series (RFC issue: *link*). Patch:
`upstream/series-q4head/0005-fix-spec-commit-PLE-state-after-ReplaySSM-verify-NGR.patch`
in the companion repository (4 files, +542 / -3). The `spec_utils.py` change is a bug fix that
stands on its own; the rest is an opt-in. Patch 0005 applies on `qwen4-main-squashed`
(`78c5024e9d`) alone (verified with `git apply --check` against that tree), so this part can be
opened and merged first; it is listed as stacked on part 1 only because both touch
`models/qwen4_exp.py`, and parts 2-4 are not needed for it.

## Motivation

Two things stop speculative decoding on Qwen4-Exp. First, a bug: `commit_mamba_states_after_verify`
returns early in both ReplaySSM branches (fold and ring), before the generic path that rolls the
PLE n-gram history and the PLE short-conv state to the last accepted node
(`hybrid_linear_attn_backend._update_ple_state_after_mtp_verify`). After a verify step the PLE
history therefore silently freezes. This affects MTP with top-k 1 under
`--enable-linear-replayssm-spec` as well, not only NGRAM (CAMPAIGN 2026-09-02 17:08). Second,
`_prepare_ple_batch` refuses NGRAM outright, and even at `--speculative-ngram-max-bfs-breadth 1`
the n-gram corpus returns a star (one chain per anchor, zero-padding nodes as root children),
while the GDN ReplaySSM fold, the QSA pending ring (keyed by position % 4) and the KV move of the
accepted tokens all assume a chain in which row j-1 is row j's parent.

On the reference machine (RTX PRO 4000 Blackwell 24 GB, 32 GB host RAM) MTP is not an option:
the checkpoint's MTP experts are bf16, 4.7 GiB plus a 2.5 GB transient (CAMPAIGN 2026-09-02
17:08; `docs/ELASTIC_MEMORY.md`, "Speculative decoding"), so NGRAM is the only speculation
available there.

Sources: dated entries of `docs/CAMPAIGN.md` and `docs/ELASTIC_MEMORY.md` in
https://github.com/HaberstrohSystems/qwen3.8-flash-next-24gb-sglang and the model card
https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang.

## Modifications

- `speculative/spec_utils.py`: in both ReplaySSM branches of `commit_mamba_states_after_verify`,
  call `attn_backend._update_ple_state_after_mtp_verify(...)` when the backend has it. The
  generic path scatters with physical slot ids, while `get_mamba_indices` returns virtual ids,
  so the state batch indices (and the track indices in the fold branch) go through
  `req_pool.translate_mamba_indices` (identity for the static `HybridReqToTokenPool`, a lookup
  for the unified pool). In the ring branch no track indices are passed, mirroring the conv
  scatter that is skipped there.
- `speculative/ngram_worker.py`: `_linearize_chain` collapses each request's draft tree into its
  longest root chain (ties -> lowest node index, i.e. BFS order), pads with token 0 chained after
  it and sets the mask to `tril`. Padding token 0 after the real drafts is lossless: greedy verify
  accepts a node only if it equals the target argmax at its parent row. The constructor asserts
  `speculative_ngram_max_bfs_breadth == 1`. The function is general for any hybrid backend that
  assumes a chain.
- `models/qwen4_exp.py`: drop the "Qwen4 PLE does not support NGRAM speculation" guard (the
  top-k guard stays). On the first target-verify forward, check that
  `ngram_pool.intermediate_context` and `short_conv_pool.intermediate_conv_state` are allocated
  and raise a `RuntimeError` naming the cause otherwise; without both, the post-verify commit
  scatters nothing and the history freezes silently.
- `test/registered/unit/spec/test_ngram_linearize_chain.py` (new, `register_cpu_ci`,
  `est_time=30`, suite `base-a-test-cpu`): `_linearize_chain` against the trie/result semantics,
  with the corpus output rebuilt by a Python replica of `trie.cpp` / `result.cpp` at breadth 1
  and run under several sibling orders: a single chain is preserved, a star collapses to one
  chain, a partial match keeps the longest chain, padding nodes never displace a real chain,
  branching below the root keeps the longest path, batched inputs stay untouched with
  `batch_get` shapes and dtypes, the verify-side derivations (`reconstruct_indices_from_tree_mask`,
  `verify_tree_greedy` replicas) see a chain with parent(row j) == row j-1 and a prefix
  `accept_index`, and the real JIT `NgramCorpus` fed with a star-producing history is linearized
  and re-checked (skipped when the extension cannot be built, e.g. without `ninja`).

Opt-in; nothing changes unless `--speculative-algorithm NGRAM` is passed. Validated launch:
`--speculative-algorithm NGRAM --speculative-num-draft-tokens 4
--speculative-ngram-min-bfs-breadth 1 --speculative-ngram-max-bfs-breadth 1
--enable-linear-replayssm-spec` with eager decode (`--disable-cuda-graph`); 12 draft tokens did
not fit the mamba intermediate-state reserve on the reference machine (CAMPAIGN 2026-09-02
12:44). Not covered yet: a test that the PLE context advances after a verify step.

## Accuracy Tests

Lossless gate (`docs/ELASTIC_MEMORY.md`, "Speculative decoding"; card, "Serving fidelity"):
speculative-path logprobs versus teacher forcing with a top-1 / near-tie top-2 rule, plus the
prefill oracle.

- `test_ngram_linearize_chain.py`: 8 tests pass on the CPU with the JIT `NgramCorpus` built;
  without `ninja` on `PATH` the real-corpus test skips cleanly (7 pass + 1 skip). Also passes on
  the finished 5-commit branch.
- Prefill oracle mean 0.0015; 0 mismatches in 600 tokens (7 near-ties); spec-path vs
  teacher-forced mean |dlogprob| 0.009-0.019, max 0.25 (CAMPAIGN 2026-09-02 18:25).
- The non-speculative server on the same gate gives mean |dlogprob| 0.019 / 0.009 / 0.011
  (speculative 0.019 / 0.011 / 0.010), max 0.16-0.27 (speculative 0.18-0.25), 0 mismatches,
  4 near-ties: the speculative path is lossless within the decode-vs-prefill floor of about 0.01
  (CAMPAIGN 2026-09-02 18:50).

Reproduction:

```
git clone https://github.com/sgl-project/sglang.git && cd sglang
git fetch origin qwen4-main-squashed && git checkout 78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385
git am /path/to/upstream/series-q4head/0005-*.patch      # applies without parts 1-4
pip install -e python
python -m pytest -v test/registered/unit/spec/test_ngram_linearize_chain.py     # CPU; ninja on PATH for the real-corpus case
# server level (companion repository): tools/spec_lossless.py against a server started with the NGRAM flags above
```

## Speed Tests and Profiling

No speed-up on this model: mean accept length 1.11-1.27 (accept rate 7-9 %) on prose, reasoning
and code prompts, QSA caps drafts at 4, and with eager verify decode is 22-25 tok/s against
56 tok/s for the non-speculative graph path (CAMPAIGN 2026-09-02 18:25 and 18:50;
`docs/ELASTIC_MEMORY.md`, "Speculative decoding"). It is therefore an opt-in for repetitive or
structured workloads; the `spec_utils.py` change is a bug fix regardless of that outcome.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit). (black 26.1.0, isort 7.0.0, ruff 0.15.1 `F401,F821,UP037`, codespell and `py_compile` clean on the commit.)
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests). (`test/registered/unit/spec/test_ngram_linearize_chain.py`, 8 tests, `register_cpu_ci`. Missing: a test that the PLE context advances after a verify step.)
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations). (Missing: a note in the speculative-decoding documentation that NGRAM on Qwen4-Exp requires breadth 1 and `--enable-linear-replayssm-spec`.)
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed). (Lossless gate and the accept-length / throughput measurement above, which shows no speed-up.)
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

## Suggested reviewers

- `python/sglang/srt/speculative`: @hnyls2002, @Qiaolin-Yu (Speculative decoding).
- `python/sglang/srt/models` (`qwen4_exp.py`): @Fridge003, @ishandhanani, @Qiaolin-Yu;
  @JustinTong0323 as author of #36497.
