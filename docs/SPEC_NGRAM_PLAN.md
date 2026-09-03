# NGRAM speculation with PLE: plan (2026-09-02)

> **Status:** the accepted set uses `--cuda-graph-backend-decode breakable`, not `--disable-cuda-graph`
> (the NGRAM serve line in `patches/ngram_ple.py` accounts for it); the 0.003 lossless threshold below
> was replaced by the ~0.01 decode-vs-prefill floor (CAMPAIGN.md:387). Outcome: lossless within that
> floor, accept length 1.11-1.27 tokens per step, 22-25 tok/s vs 56 — opt-in only. Implemented as
> `patches/ngram_ple.py`, gate `tools/spec_lossless.py`.

# Speculative decoding for Qwen4-Exp: verdict, patch list, risks, validation

Base paths: `SRT=$SGLANG/python/sglang/srt`, `PERF=perf`. Every anchor below was re-read in the tree today; where a reader's claim did not survive re-reading it is corrected here.

## 1. Verdict per option

### Option A: NGRAM with PLE support — RECOMMENDED (do this one)

- **Memory**: ~tens of MB, fits in the 2.5 GB headroom with margin. With `--enable-linear-replayssm-spec` there is no `intermediate_ssm` scratch (the 0.16 GB x slots x drafts term that killed S12a); what remains per mamba slot is the GDN ring (`record_len = 4`, rawv/rawk in conv dtype + fp32 g/beta, `memory_pool.py:620-688`), the conv intermediate windows (`:698-706`), the PLE `intermediate_context (slots+1, 4, 2)` int64 (`ple_state_pool.py:170-176`), 4 draft KV rows/step, and the worker's preallocated `(max_bs, 4)` index tensors. Evidence: S12b (replayssm, 12 drafts) got all the way to graph capture and failed only on the QSA width check (CAMPAIGN.md:307-309), so 4 drafts is far inside budget. No checkpoint change, no host RAM.
- **Speed**: structural cap = 4 tokens/step (`num_draft_tokens=4` = root + 3 corpus drafts + bonus; `param.h:30-38`, `result.cpp:18-28`). Every decode step becomes a 4-row eager verify (`ngram_worker.py:354-364`), so the step cost rises: +5-10 % host/launch bookkeeping (eagle_sample, 1 fold launch + 36 conv scatters, KV move, corpus get/put, the Python full-mask loop at `:337-352`) and up to 4x expert rows per layer (per-token top-10 routing, GEMV path `moe_wna16.py:597-603`; ~16 % of routed mass cold at S=184 -> PCIe reads). Per-step ~1.2-1.5x a decode step. Acceptance from an own-history trie (query = last 18 tokens, `trie.cpp:192-202, 236-259`): unmeasured anywhere in the tree; estimate 1.1-1.4 tok/step prose, 1.5-2.3 code/JSON/quoting. **Net expectation: 0.9-1.2x on prose (may be a slowdown), 1.3-1.8x on code/repetitive text; theoretical ceiling ~3x.** Treat it as a workload-dependent opt-in until measured.
- **Why it is cheap to build**: with a *chain* draft, the existing PLE verify path (topk-1 MTP path) is exactly right — the unfold in `compute_ngram_ids` (`qwen4_exp.py:713-716`) and the intermediate windows in `_commit_ple_batch` (`:288-297`) assume row j-1 is row j's parent, which holds for a chain. The reader's proposal to plumb tree parents into the PLE is unnecessary if the worker forces a chain — and forcing a chain is *required anyway* because the GDN fold verify asserts a chain (`gdn_backend.py:750-753`), QSA never reads the tree mask (0 uses of `custom_mask` in `qwen_sparse_attn_backend.py`) and keys its pending ring by `position % 4`, and the int8 KV `move_kv_cache` does not move scale buffers (`memory_pool.py:3070-3092`; `int8_kv_pool.py` overrides only `set_kv_buffer`). Important correction to the ngram reader: `--speculative-ngram-max-bfs-breadth 1` does NOT give a chain. Each anchor re-enters from the root (`trie.cpp:236-260`) and all zero-padding nodes are root children (`result.cpp:34-38`), so a partial match yields a star (root with several depth-1 children, identical positions). That star silently breaks GDN chain metadata, the QSA pending ring (two rows with the same `position % 4`) and the KV move (tgt != src). Patch 2 below linearizes host-side.

### Option B: MTP/NEXTN head — NOT NOW

- **Memory**: the checkpoint's `mtp.*` experts are BF16 (`'^mtp\..*': bits 16`; 3 x 512 x [640,2560] = 4.69 GiB) + ~0.17 GiB dense + a ~2.5 GB transient (draft allocates its own bf16 `embed_tokens` and `lm_head` before `set_embed_and_head` frees them, `qwen3_5_mtp.py:132-137, 153-160`) against ~2.2-2.5 GB free. Does not fit by 3-5 GB. Bf16 MoE cannot ride the elastic cache or the expert streamer (both keyed on `w13_qweight/w2_qweight`, `moe_wna16.py:641`, `expert_elastic.py:35`; GEMV path is `weight_bits == 2` only, `moe_wna16.py:402`), and host RAM has no pinned slots for a 49th 512-expert layer. Even after an int2 requant (~0.67 GB experts, ~1.0 GB steady with graphs/KV delta) the headroom drops to ~1.2 GB — below the 1.1-1.5 GB prefill working set and the 1.18 GB long-context minimum (CAMPAIGN.md:333, :348).
- **Patches it would need on top**: the requant (drop the `^mtp\..*` bits-16 override), `get_embed_and_head` returns `self.lm_head.weight` (`qwen3_vl.py:1581-1582`) but the int8 GPTQ lm_head registers `qweight` only -> AttributeError at `eagle_worker_v2.py:292`; avoid the transient allocation; guard `ExpertElastic.poll(lid=0)` for the draft layer (`moe_wna16.py:593-595`, `expert_elastic.py:339-345`); and the same ReplaySSM PLE-commit bug as Option A (Patch 3).
- **Speed**: same 4-token cap (QSA compress ratio), plus one full draft layer + a 248k-vocab lm_head per draft step; repo's own estimate 1.1-1.35x. Acceptance of a bf16-trained MTP head against an int2 target is unmeasured.
- Verdict: revisit only after host RAM grows or a requantized draft dir exists; NGRAM answers the "is speculation worth it on this box" question first for ~150 lines of patch.

## 2. Ordered minimal patch list for Option A — `PERF/patches/ngram_ple.py` (EDITS style, `--check|apply|revert`)

Launch line for the sweep (server runs `--disable-cuda-graph --disable-overlap-schedule`, sync worker, so everything below runs eagerly):

```
scripts/sweep.sh S12e_ngram_chain --speculative-algorithm NGRAM --speculative-num-draft-tokens 4 \
  --speculative-ngram-min-bfs-breadth 1 --speculative-ngram-max-bfs-breadth 1 \
  --enable-linear-replayssm-spec   (+ the accepted defaults: --kv-cache-dtype int8_g64 --attention-backend triton --max-mamba-cache-size 1 ...)
```
(`_handle_ngram` sets `speculative_eagle_topk := max_bfs_breadth = 1` and `num_steps = 4`, `speculative_hook.py:721-731`; the replayssm topk check at `server_args.py:6399-6407` then passes and the hybrid backend takes its chain path, `hybrid_linear_attn_backend.py:51, 203`.)

**Edit 1 — drop the NGRAM guard** (`SRT/models/qwen4_exp.py:122-124`), old:
```python
    spec_algorithm = forward_batch.spec_algorithm
    if spec_algorithm is not None and spec_algorithm.is_ngram():
        raise NotImplementedError("Qwen4 PLE does not support NGRAM speculation")
```
new: delete the three lines (keep the TBO guard at :120-121 and the topk guard at :125-129; `NgramVerifyInput` has no `topk`, so `getattr(..., 1)` passes — the chain is enforced by Edit 2, not by that guard). Nothing else in the model checks `is_ngram`; the verify row layout (`row_width = spec_info.draft_token_num`, `:148`) and the KV allocation (`end_offset = seq_lens + draft_token_num`, `ngram_worker.py:356-363`) already match.

**Edit 2 — force a linear draft chain in the worker** (`SRT/speculative/ngram_worker.py`), two hunks:

(a) constructor guard (after `self.draft_token_num = ...` in `__init__`, near `:111-120`): `assert server_args.speculative_ngram_max_bfs_breadth == 1, "Qwen4-Exp (GDN replayssm fold + QSA + PLE) needs a chain draft: --speculative-ngram-max-bfs-breadth 1"`.

(b) in `_prepare_draft_tokens` (`:285-298`), old:
```python
        req_drafts, mask = self.ngram_corpus.batch_get(
            req_ids, batch_tokens, total_lens
        )
        total_draft_token_num = len(req_drafts)
```
new:
```python
        req_drafts, mask = self.ngram_corpus.batch_get(
            req_ids, batch_tokens, total_lens
        )
        req_drafts, mask = _linearize_chain(req_drafts, mask, bs, self.draft_token_num)
        total_draft_token_num = len(req_drafts)
```
plus the module-level helper (insert after `_derive_tree_links`, `:40-58`):
```python
def _linearize_chain(req_drafts, mask, bs, D):
    """Collapse each request's draft tree into one root chain.  Even at
    bfs-breadth 1 the corpus fans out at the root (one chain per anchor,
    trie.cpp:236-260) and zero-padding nodes are root children
    (result.cpp:34-38).  GDN replayssm fold, QSA (no tree mask, pending ring
    keyed by position % 4) and the int8 KV move all assume a chain.  Keep the
    deepest root chain, pad with token 0 chained after it; mask := tril."""
    toks = np.asarray(req_drafts).reshape(bs, D).copy()
    tree = np.asarray(mask).reshape(bs, D, D)
    order = np.arange(D)
    for b in range(bs):
        anc = tree[b] & (order < order[:, None])
        parents = np.where(anc.any(-1), (anc * order).argmax(-1), -1)
        parents[0] = -1
        best = [0]
        for root_child in np.flatnonzero(parents == 0):
            chain, cur = [0, int(root_child)], int(root_child)
            while True:
                kids = np.flatnonzero(parents == cur)
                if kids.size == 0:
                    break
                cur = int(kids[0]); chain.append(cur)
            if len(chain) > len(best):
                best = chain
        new = np.zeros(D, dtype=toks.dtype)
        new[: len(best)] = toks[b, best]
        toks[b] = new
    tri = np.tril(np.ones((D, D), dtype=tree.dtype))
    return toks.reshape(-1), np.broadcast_to(tri, (bs, D, D)).reshape(-1).copy()
```
Padding token 0 chained after the real drafts is lossless: greedy verify accepts a node only if it equals the target argmax at its parent row (`eagle_utils.py:726-739`), so an accepted padding token is the greedy token. After this, `reconstruct_indices_from_tree_mask` (`:320-329`) yields `positions = seq_len + arange(4)`, `retrieve_next_token = [1,2,3,-1]`, `retrieve_next_sibling = -1`, and `accept_index` is always a prefix -> `tgt_cache_loc == accept_out_cache_loc` in `move_accept_tokens_to_target_kvcache` (`spec_utils.py:729-753`), which makes the missing int8 scale move a no-op. bs=1 host cost: microseconds.

**Edit 3 — commit PLE side states in the ReplaySSM fold branch** (`SRT/speculative/spec_utils.py:900-910`). This is a real bug independent of NGRAM (it also affects MTP topk=1; the "GDN/PLE/QSA rollback fully plumbed" note in CAMPAIGN.md:184 is wrong under `--enable-linear-replayssm-spec`). Old:
```python
        commit_gdn_replayssm_fold_after_verify(
            spec_state=spec_state,
            state_batch_indices=state_batch_indices,
            accept_lens=accept_lens,
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            null_block_id=-1,
        )
        return
```
new: same call, then before `return`:
```python
        # PLE n-gram history + PLE short-conv state live outside the GDN fold;
        # roll them to the last accepted node like the generic path does
        # (hybrid_linear_attn_backend.py:1284-1289, :1324-1381).
        attn_backend = model_runner.attn_backend
        if hasattr(attn_backend, "_update_ple_state_after_mtp_verify"):
            attn_backend._update_ple_state_after_mtp_verify(
                state_batch_indices,
                last_correct_step_indices,
                batch.mamba_track_indices,
                mamba_steps_to_track,
            )
        return
```
Same insertion in the ring branch before its `return` at `:959` (pass `None` for `mamba_steps_to_track`; that branch discards it as `_` at `:942`). `last_correct_step_indices` is the *node index* of the last accepted row (`_verify_commit_step_indices`, `:784-843`), which is exactly the step index `intermediate_context[slot, j] = [tok(j-1), tok(j)]` was written for (`qwen4_exp.py:288-297`) and the same index the conv scatter uses (`gdn_replayssm_spec_fold.py:252-257`). After the scatter, the next step's root row (= bonus token, `result.cpp:18`) gets the 3-gram `[tok(j-1), tok(j), bonus]`, identical to what non-spec decode would compute.

**Edit 4 — startup self-check (cheap insurance)** (`SRT/models/qwen4_exp.py`, in `_prepare_ple_batch` right after the mode check, only when `mode.is_target_verify()`): assert `pool.ngram_pool.intermediate_context is not None and pool.short_conv_pool.intermediate_conv_state is not None`. Both are allocated only when `speculative_num_draft_tokens` reaches `HybridReqToTokenPool` (`memory_pool.py:1296-1310`, `ple_state_pool.py:170-176`); if either is None, `_update_ple_state_after_mtp_verify` builds an empty `state_pairs` list (`hybrid:1336-1357`) and the PLE history silently never advances — the one failure mode that would pass a startup and fail only the exactness test.

**Edit 5 (optional, debug env `SGLANG_NGRAM_CHECK=1`)** in `ngram_worker.py` after `commit_mamba_states_after_verify` (`:466-472`): `torch.equal(tgt_cache_loc, accept_out_cache_loc)`-style prefix assert on `accept_index`, and compare `req_to_token_pool.ngram_pool.context[slot]` with the last two tokens of `(origin_input_ids + output_ids + accepted[:-1])`. Sync mode, bs=1: a few hundred microseconds; remove for the bench.

Not needed: `NgramVerifyInput` changes, `filter_batch/merge_batch`, the graph-capture dummy (`decode_cuda_graph_runner.py:1518-1530`), `_hash_contexts` (the chain contexts are a contiguous `(N,3)` long tensor, so the fused Triton hash at `qwen4_exp.py:617-632` still applies), and `set_ngram_intermediate_context` capacity (sized by `speculative_num_draft_tokens` = 4).

## 3. Correctness risks and a cheap pre-check for each

| Risk | Where | Pre-check |
|---|---|---|
| **GDN state rollback** (fold replays `accept_lens` steps of the ring into `temporal`; conv scatter to `last_correct_step`) | `gdn_replayssm_spec_fold.py:98-120, 238-257`; fold asserts chain `gdn_backend.py:750-753` | Run the lossless test (section 4, step 3) twice: normally, and with drafts forced to rejection (`SGLANG_NGRAM_FORCE_REJECT=1`: `_linearize_chain` returns all-zero tokens) so every step commits exactly one bonus token — outputs and per-token logprobs must match the non-spec server to the noise floor (mean 0.002). Then `--speculative-num-draft-tokens 2` vs `4`: same outputs. |
| **QSA verify** (no tree mask; pending index-key ring keyed by `position % 4`; paged row "completes at most one group") | `qwen_sparse_attn_backend.py:256-272`; `qsa/metadata.py:249-263` | With Edit 2, positions within a window are `seq_len..seq_len+3` (distinct mod 4). Assert in Edit 5 that `positions` after `reconstruct_indices_from_tree_mask` equal `seq_len + arange(4)`. Correctness of the indexer on verify rows falls out of step 3 (the 12 QSA layers' logits are what the lossless test compares). |
| **PLE window rollback** (n-gram history + short-conv state) | `qwen4_exp.py:288-297, 1162-1192`; `hybrid:1324-1381`; commit only via Edit 3 | Edit 4 (both intermediates non-None). Edit 5 context check. Behavioural: without Edit 3 the history freezes at prefill, so step 3 fails with a large, systematic logprob delta on layer-2-sensitive tokens — run step 3 once with Edit 3 reverted to confirm the test has teeth. |
| **CUDA-graph capture of verify batches** | `decode_cuda_graph_runner.py:311-324, 660-712`; BCG break `qwen4_exp.py:2294-2299` | Not exercised: the accepted config is `--disable-cuda-graph` (sweep.sh:22). If graphs are re-enabled later: the mmap gather is a BCG eager break (`breakable_cuda_graph.py:257-261` re-runs it with captured ids), the NGRAM gate `bs*4 == input_ids.numel()` always holds, and the worker's per-step Python mask loop (`ngram_worker.py:337-352`) stays host-side; pre-check = repeat step 3 with `--cuda-graph-max-bs 1`. |
| **int8 KV move drops scales** | `memory_pool.py:3070-3092`, `int8_kv_pool.py:116-151` | Chain -> `tgt == src`; Edit 5's prefix assert on `accept_index` proves it each step. |
| **Star/multi-anchor trees** | `trie.cpp:236-260`, `result.cpp:34-38` | Edit 2 makes the mask `tril` by construction; assert `mask.reshape(D,D) == tril` in Edit 5. |
| **Prefill path** (the old raise fired on every forward) | `qwen4_exp.py:1782-1790` | `nll_eval.py check int8dense` (prefill/teacher-forced) unchanged. |
| **Overlap scheduling** | `ngram_worker.py:265-278` prev-token splice | Keep `--disable-overlap-schedule` (already in BASE). |

Greedy verify is argmax-compare (`eagle_utils.py:726-739`), so losslessness reduces to "verify-row logits == decode logits", which is what step 3 measures. `speculative_accept_threshold_*` only touch the sampling path.

## 4. Validation protocol (in order; server left running between steps as in `int8_validate.sh`)

1. **References from the accepted non-spec server** (the server is not bitwise reproducible run to run, `logprob_diff.py` docstring): `greedy_diff.py save prespec`; `logprob_diff.py save lp_prespec` (or reuse `lp2`, 10k). Record `bench_speed.py 200` and a timed run of the three `greedy_diff.PROMPTS` (prose/reasoning/code, 200 tokens; use the streamed inter-token rate as bench_speed does).
2. **Bring-up** `S12e_ngram_chain` with Edits 1-5 applied (`ngram_ple.py apply`), keepalive, elastic `S 184`. Check the log for the PLE intermediates assert and that no `NotImplementedError` fires.
3. **Lossless test (the gate)** — new `tools/spec_lossless.py`, three parts, all on the spec server:
   a. `logprob_diff.py check lp_prespec` — teacher-forced prefill path; must stay at the floor (mean ~0.002). Confirms Edit 1 did not disturb prefill.
   b. For each of the 3 prompts: greedy-generate 200 tokens with `return_logprob` (spec path: `compute_spec_logprobs`, `ngram_worker.py:484-490`), then teacher-force the same `prompt+generated` ids with `logprob_start_len=len(prompt)-1, top_logprobs_num=2`. Every generated token must equal the teacher-forced top-1 unless the top-1/top-2 gap is < 0.01 (a near-tie, reported not failed); mean |delta| between the spec-path logprobs and the teacher-forced logprobs <= 0.003, max reported. This checks verify-row logits (12 QSA + 36 GDN + PLE) against the prefill path without depending on run-to-run token reproducibility.
   c. Repeat (b) with `SGLANG_NGRAM_FORCE_REJECT=1` (every step commits 1 token; exercises rollback on every step) and with `--speculative-num-draft-tokens 2`.
   Also `greedy_diff.py check prespec`: report divergence positions; they must look like noise (first divergence typically > 100, as with the bf16 x2 runs), not systematic early divergence.
4. **Speed** — `bench_speed.py 200` for parity with the log, BUT its filler prompt is one sentence repeated (`bench_speed.py:33-36`) and the own-history trie will match it trivially; the number will overstate acceptance. Report the timed greedy prompts (prose/reasoning/code) as the headline, and read the mean accept length from the scheduler decode log lines (`spec accept length`). Acceptance criterion: no prompt class below 56 tok/s (else keep NGRAM as an opt-in for code/structured workloads, not the default); expect code 1.3-1.8x.
5. **Long context** — `longctx_test.py 100000` then `150000` under keepalive with `SGLANG_MOE_ELASTIC_CTL`: correct answer, server alive, elastic status before/after (watch for expert-cache shrink from the extra `2 * alloc_len_per_decode` draft KV reserve under lazy VMM, `memory_pool.py:2366-2392`), VRAM free must not drop below the 1.18 GB seen at 162k.
6. **Width sweep** (only if step 4 is positive): `--speculative-num-draft-tokens 3` and `2` (every step is a verify, so width trades expert traffic against acceptance); pick the best per workload. 4 is the QSA cap; 5 raises at startup and would also push the PLE lookup from the 64-id `pread` fan-out to the memmap path (`qwen4_exp.py:863-870`).
7. Record in CAMPAIGN.md; if step 3 fails, revert with `ngram_ple.py revert` (sweep.sh reverts the flags).

Key files: `SRT/models/qwen4_exp.py`, `SRT/speculative/ngram_worker.py`, `SRT/speculative/spec_utils.py`, `SRT/layers/attention/hybrid_linear_attn_backend.py`, `SRT/layers/attention/linear/gdn_backend.py`, `SRT/layers/attention/qwen_sparse_attn_backend.py`, `SRT/mem_cache/ple_state_pool.py`, `$SGLANG/python/sglang/kernels/jit/csrc/ngram_corpus/{trie.cpp,result.cpp}`, `$SGLANG/python/sglang/kernels/ops/attention/fla/gdn_replayssm_spec_fold.py`, `patches/ple_random.py` (EDITS template), `tools/{greedy_diff.py,logprob_diff.py,longctx_test.py,int8_validate.sh}`, `scripts/sweep.sh`, `tools/bench_speed.py`.

## Reader summaries
[
 {
  "key": "ple",
  "confidence": "high",
  "blockers": [
   "ReplaySSM-spec commit skips PLE: both GDN branches of commit_mamba_states_after_verify return early (speculative/spec_utils.py:882-908 and :913-959) before update_mamba_state_after_mtp_verify (:1026-1041) -> _update_ple_state_after_mtp_verify (layers/attention/hybrid_linear_attn_backend.py:1284) never runs, so the PLE n-gram history and PLE short-conv state are never committed after any verify (affects MTP/EAGLE topk=1 too, not only NGRAM).",
   "NGRAM drafts are a BFS tree, and a chain is not guaranteed even with --speculative-ngram-max-bfs-breadth 1 (multi-anchor root fan-out at kernels/jit/csrc/ngram_corpus/trie.cpp:236-260; zero-padding nodes are children of the root at result.cpp:34-38). The GDN backend takes its chain path whenever topk<=1 (hybrid_linear_attn_backend.py:51, :203, :283) with topk := max_bfs_breadth (speculative_hook.py:721) - a pre-existing hazard for the whole NGRAM+hybrid stack that the PLE fix does not address; out of scope but must be validated separately.",
   "Parent links are not available on the ForwardBatch: NgramVerifyInput carries only retrieve_next_token/sibling/custom_mask (speculative/ngram_info.py:28-32); the proposal needs a new tree_parent field plumbed through worker preallocation (ngram_worker.py:183-229), filter/merge (ngram_info.py:116-140) and the graph-capture dummy (model_executor/runner/decode_cuda_graph_runner.py:1518-1528).",
   "Qwen4ExpMmapEmbedding.gather does a device->host sync plus os.pread (models/qwen4_exp.py:838-849) and is not CUDA-graph capturable; the NGRAM verify graph gate (decode_cuda_graph_runner.py:705-712) would otherwise try to capture it. I could not verify the running server's flags from the tree, so 'verify runs eagerly' is an assumption consistent with the stated launch-bound decode.",
   "I did not find an explicit assert for the QSA '<= 4 rows per verify' bound; only the design comment that paged speculative rows complete at most one compress group (layers/attention/qsa/qsa_indexer.py:356-357). The bound is taken from the task statement.",
   "The mmap lookup cost figures are estimates from the code path, not measurements (no GPU/process work was done per the read-only constraint)."
  ],
  "numbers": [
   "ngram_size=3, ngram_context_len=2 (configs/qwen4_exp.py:27, :108-111)",
   "16 ngram heads per token = (3-1)*8 (models/qwen4_exp.py:442-444); 160 B per PLE row, 2560 B per token (models/qwen4_exp.py:776)",
   "pread fast path threshold: <= 64 ids (models/qwen4_exp.py:843) -> W=4 verify rows = 64 ids exactly; W=5 = 80 ids goes to the memmap path (:851)",
   "NGRAM default speculative_num_draft_tokens=12 and topk := max_bfs_breadth (arg_groups/speculative_hook.py:721-723); must be set explicitly to <= 4 rows for QSA",
   "NGramPool.context shape (mamba_size+1, 2); intermediate_context (spec_state_size+1, speculative_num_draft_tokens, 2) (mem_cache/ple_state_pool.py:164-176)",
   "Estimated D+1 lookup: ~0.1-0.2 ms warm page cache, ~0.4-0.5 ms cold NVMe (16-thread pool, ~100 us/read), vs ~18 ms per decode step at 56 tok/s -> <= ~3% of a step; estimate, not measured"
  ]
 },
 {
  "key": "ngram",
  "confidence": "medium",
  "blockers": [
   "Hard guard: models/qwen4_exp.py:122-124 raises NotImplementedError('Qwen4 PLE does not support NGRAM speculation') in _prepare_ple_batch, hit on every forward (prefill included) once --speculative-algorithm NGRAM is set; must be removed/relaxed before anything runs.",
   "Correctness under --enable-linear-replayssm-spec: PLE short-conv state and ngram context are never rolled forward after verify. commit_mamba_states_after_verify returns from the fold branch (speculative/spec_utils.py:882-910) before attn_backend.update_mamba_state_after_mtp_verify (1036-1043), which is the only caller of _update_ple_state_after_mtp_verify (layers/attention/hybrid_linear_attn_backend.py:1284-1289, 1324-1370). Needs a patch to scatter the PLE intermediates in the fold branch.",
   "Tree vs chain: _handle_ngram forces speculative_eagle_topk = speculative_ngram_max_bfs_breadth (arg_groups/speculative_hook.py:721; default 10) AFTER the ReplaySSM topk validation ran (server_args.py:3691 vs 3738-3740), so startup passes but the GDN fold verify asserts retrieve_parent_token is None (gdn_backend.py:750-753; tree links built at hybrid_linear_attn_backend.py:51, 203-210). QSA also ignores the tree mask (0 uses of custom_mask in qwen_sparse_attn_backend.py; its topk check reads a non-existent NgramVerifyInput.topk, 256-263) and the int8 KV move does not carry scale buffers (memory_pool.py:3067-3092; int8_kv_pool.py overrides only set_kv_buffer). Run with --speculative-ngram-min-bfs-breadth 1 --speculative-ngram-max-bfs-breadth 1 (chain) only.",
   "Width: --speculative-num-draft-tokens must be <= 4 (QSA pending ring, qwen_sparse_attn_backend.py:264-272); that yields only 3 real drafts per step, so a 1+4 window is impossible without changing the QSA ring.",
   "Throughput risk (not a crash): every decode step becomes a 4-row verify (ngram_worker.py:300-354) with per-token MoE routing -> up to 40 expert GEMVs per layer (moe_wna16.py:597-603, expert_gemv.py:78-84) and host reads for non-resident experts (expert_elastic.py:1-16); plus 2x draft KV reservation per step under lazy VMM can trigger expert-cache shrink (memory_pool.py:2366-2392). Net gain on prose may be <= 1.0x; needs measurement.",
   "Acceptance numbers for prose/code are not available in the tree (only the GSM8K >= 1.8 fixture threshold with 16 drafts); the figures above are estimates."
  ],
  "numbers": [
   "num_draft_tokens=4 -> 1 root + 3 drafts per verify row (param.h:37, result.cpp:18-28); max 4 tokens committed/step",
   "QSA compress_ratio=4 caps speculative_num_draft_tokens at 4 (qwen_sparse_attn_backend.py:264-272); '1+4' (=5) raises NotImplementedError",
   "NGRAM defaults: max_trie_depth 18, bfs breadth 1..10, num_draft_tokens 12, topk := max_bfs_breadth (server_args.py:2286-2301; speculative_hook.py:721-727)",
   "Upstream fixture: GSM8K mean accept length >= 1.8 with 16 drafts, breadth 10 (ngram_fixture.py:33-34,44); no prose/code numbers exist in tree",
   "Estimate (unmeasured): 4-wide chain accept ~1.1-1.4 tok/step prose, ~1.5-2.3 code; verify step ~1.2-1.5x decode step due to 40 expert GEMVs/layer (moe_wna16.py:600, expert_gemv.py:78-84) -> net ~0.9-1.2x prose, ~1.3-1.8x code",
   "ReplaySSM ring record_len = num_draft_tokens = 4 per slot per layer, rawv/rawk conv dtype + fp32 g/beta (memory_pool.py:620-688); intermediate_ssm not allocated (713-714)",
   "Post-verify commit: 1 fold launch for all 36 GDN layers + 36 conv scatters (gdn_replayssm_spec_fold.py:238-257)",
   "PLE mmap lookup rows per verify = 4 tokens * 16 heads = 64 = pread path threshold (qwen4_exp.py:863)",
   "KV reserve per decode step = 2 * alloc_len_per_decode (allocation_sizing.py:59)"
  ]
 },
 {
  "key": "mtp",
  "confidence": "high",
  "blockers": [
   "VRAM: MTP experts are BF16 (4.69 GiB) + ~0.17 GiB dense vs ~2.2 GB free; does not fit. Needs a requantization pass (int2 via the same AutoRound recipe; block_name_to_quantize already lists mtp.layers, the '^mtp\\..*' bits-16 override must be dropped) producing a new/merged checkpoint dir.",
   "Host RAM full: bf16 MTP experts cannot use the elastic cache or expert streamer (both keyed on w13_qweight/w2_qweight/scales, moe_wna16.py:641, expert_elastic.py:35; N-contig/GEMV path is weight_bits==2 only, moe_wna16.py:402) and there are no pinned host slots for another 512-expert layer (expert_elastic.py:13-16, 90-110; CAMPAIGN.md:346).",
   "Quantized lm_head sharing crashes: target get_embed_and_head returns self.lm_head.weight (qwen3_vl.py:1581-1582) but the int8 GPTQ lm_head registers qweight only (gptq_linear.py:140 / gptq_marlin.py:147) -> AttributeError at eagle_worker_v2.py:292; set_lm_head_from_target (qwen3_5_mtp.py:163-167) runs only afterwards. Patch required.",
   "Transient ~2.5 GB VRAM spike: draft allocates its own bf16 embed_tokens and bf16 lm_head (qwen3_5_mtp.py:112-137 / qwen4_exp_mtp.py:64-70) before set_embed_and_head deletes them (qwen3_5_mtp.py:153-160). Must be avoided by construction on this GPU.",
   "Even with int2 MTP experts (~0.85 GB total draft) the remaining headroom (~1.2 GB) is below the 1.1-1.5 GB prefill working set and the 1.18 GB long-context minimum; lowering S below the 184/176 host-slot floor is not possible without host RAM.",
   "If MTP experts become int2/MoeWNA16, the draft layer has layer_id 0 and enters _collect_for_placement (moe_wna16.py:443-446) and triggers ExpertElastic.poll(lid=0) from apply() (moe_wna16.py:593-595, expert_elastic.py:344); needs an is_nextn/draft guard.",
   "--cpu-offload-gb 19 (sweep.sh:20) OffloaderV1 is created per ModelRunner (model_runner.py:410-413) with a greedy layer walk (THREADS.md:21); behaviour on the draft runner is unverified and host RAM has no room for anything it offloads.",
   "Speed ceiling: <= 4 draft tokens per verify (QSA compress ratio), verify-side expert union traffic (up to 4x10 experts/layer, 16% cold at S=184), 64 PLE preads + device sync per verify (qwen4_exp.py:860-868), and a 248k-vocab lm_head per draft step -> repo's own estimate 1.1-1.35x, not 2x. Acceptance for this int2 target is unmeasured anywhere in the tree.",
   "NGRAM is not an alternative: qwen4_exp.py:124 raises NotImplementedError for NGRAM (CAMPAIGN.md:374).",
   "Verification exactness under the default accept thresholds (server_args.py:2141-2150) must be confirmed with the oracle/greedy diff before any MTP result is accepted (DECODE_PERF_PLAN.md:172)."
  ],
  "numbers": [
   "MTP head: 1571 tensors, 5,209,501,696 bytes = 4.85 GiB, all in model_extra_tensors.safetensors; one layer (mtp.layers.0)",
   "MTP experts: 3 x 512 x BF16 [640,2560]/[2560,640] = 4800 MiB = 4.69 GiB (bits 16 via extra_config '^mtp\\..*'); non-expert MTP ~168 MiB",
   "Main-model experts int2 g128 (qweight I32 [160,640] for K=2560), ~1.31-1.33 MB per expert; int2 MTP experts would be ~0.67 GB, int4 ~1.3 GB",
   "MTP shared expert int8 GPTQ (~4.7 MiB); attn q_proj 60 MiB, o_proj 30 MiB, k/v 2.5 MiB each, indexer 3.1 MiB; fc_embedding/fc_hidden 12.5 MiB each",
   "Transient at draft build: bf16 embed_tokens 1.27 GB + bf16 lm_head 1.27 GB = ~2.5 GB before set_embed_and_head frees them",
   "Free VRAM steady ~2.2 GB (CAMPAIGN.md:336), 1.18 GB minimum during long prompts (:348), prefill working set 1.1-1.5 GB (:333)",
   "QSA cap: draft tokens <= compress_ratio 4 -> steps <= 3; topk must be 1",
   "Draft KV overhead: +1/12 of 12.4 KB/token int8 KV (~1 KB/token)",
   "Default spec params (3,1,4); Qwen3-Next cookbook uses NEXTN steps 3 topk 1 draft 4",
   "Team's own MTP speedup estimate: 1.1-1.25x today, ~1.35x after host fixes, up to 1.6x with sync-free dedup (CAMPAIGN.md:184-187); ~30% at 2 accepted tokens/step (DECODE_PERF_PLAN.md:153)",
   "Current decode 56 tok/s = 17.9 ms/token; dense/launch floor ~13.4 ms pre-INT8 (~4.5 ms less after), experts ~3.9 ms, host gap ~1.4 ms",
   "lm_head in int8dense dir is GPTQ int8 (qweight I32 [640,248320]); MTP draft lm_head would be bf16 unquantized"
  ]
 }
]