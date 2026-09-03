#!/usr/bin/env python3
"""NGRAM speculation for Qwen4-Exp (PLE + GDN ReplaySSM fold + QSA) -- Edits 1-5 of perf/SPEC_NGRAM_PLAN.md.

  1. qwen4_exp.py   : drop the "PLE does not support NGRAM" guard in _prepare_ple_batch (the topk
                      guard stays; NgramVerifyInput has no topk -> passes).
  2. ngram_worker.py: force a LINEAR draft chain.  Even at --speculative-ngram-max-bfs-breadth 1 the
                      corpus fans out at the root (one chain per anchor, trie.cpp:236-260) and the
                      zero-padding nodes are root children (result.cpp:34-38): a star.  The GDN
                      replayssm fold asserts a chain, QSA never reads the tree mask (pending ring keyed
                      by position % 4) and the int8 KV move does not carry scales -- all three need
                      row j-1 to be row j's parent.  _linearize_chain keeps the longest root chain
                      (ties -> first), pads with token 0 chained after it, mask := tril.  Constructor
                      asserts max_bfs_breadth == 1 (that is what makes speculative_eagle_topk 1, so the
                      hybrid backend takes its chain path).
  3. spec_utils.py  : commit the PLE side states (n-gram history + short-conv state) in both ReplaySSM
                      branches of commit_mamba_states_after_verify -- they returned before the generic
                      update_mamba_state_after_mtp_verify, so the PLE history never advanced after a
                      verify (also affects MTP topk=1 under --enable-linear-replayssm-spec).  Slot ids
                      are translated virtual->physical (translate_mamba_indices) like the generic path.
  4. qwen4_exp.py   : startup self-check -- on the first target-verify forward both PLE intermediates
                      (ngram_pool.intermediate_context, short_conv_pool.intermediate_conv_state) must
                      be allocated, else the commit in (3) silently scatters nothing.
  5. ngram_worker.py: debug checks, env SGLANG_NGRAM_CHECK=1 -- host mask is tril, positions are
                      seq_len + arange(D), accept_index is a row prefix (=> KV move tgt == src),
                      out_cache_loc rows are the contiguous req_to_token slots, and the committed PLE
                      n-gram context equals the last two tokens of origin + output + accepted[:-1].
                      Env SGLANG_NGRAM_FORCE_REJECT=1 zeroes the draft rows (root row kept) so every
                      step commits the bonus token only (unless the target argmax is token 0):
                      exercises the rollback on every step.

Serve with: --speculative-algorithm NGRAM --speculative-num-draft-tokens 4
            --speculative-ngram-min-bfs-breadth 1 --speculative-ngram-max-bfs-breadth 1
            --enable-linear-replayssm-spec --disable-cuda-graph --cuda-graph-backend-decode disabled
            (+ the accepted set).  Graphs: the accepted set (phase1_state.json) DROPS sweep.sh's
            --disable-cuda-graph and ADDS --cuda-graph-backend-decode breakable, and server_args.py
            applies --disable-cuda-graph at the lowest precedence (--cuda-graph-backend-* wins), so
            --disable-cuda-graph alone would leave decode/verify under breakable graphs.  Appending
            --cuda-graph-backend-decode disabled (argparse: last value wins) turns decode graphs off;
            --disable-cuda-graph still turns the prefill BCG default off.  The mmap PLE gather is a
            device->host sync (qwen4_exp.py), so the verify batch must run eagerly.

  python3 ngram_ple.py --check | apply | revert
"""
import os, sys
SG = os.path.join(os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")), "python/sglang")
Q = f"{SG}/srt/models/qwen4_exp.py"
W = f"{SG}/srt/speculative/ngram_worker.py"
S = f"{SG}/srt/speculative/spec_utils.py"

# The helper is one source string so perf/gemv/test_ngram_chain.py tests exactly the inserted code.
LINEARIZE_SRC = '''def _linearize_chain(req_drafts, mask, bs, D, force_reject=False):
    """Collapse each request's draft tree into one root chain (perf/patches/ngram_ple.py).

    Even at bfs-breadth 1 the corpus fans out at the root (one chain per
    anchor, trie.cpp:236-260) and zero-padding nodes are root children
    (result.cpp:34-38).  GDN replayssm fold, QSA (no tree mask, pending ring
    keyed by position % 4) and the int8 KV move all assume a chain: row j-1 is
    row j's parent.  Keep the longest root chain (ties -> lowest node index,
    i.e. BFS order), pad with token 0 chained after it; mask := tril.
    Padding token 0 chained after the real drafts is lossless: greedy verify
    accepts a node only if it equals the target argmax at its parent row.
    force_reject: zero the draft rows (root row kept) so only the bonus token
    is committed each step (rollback exercised every step).
    Inputs/outputs are the flat int arrays of NgramCorpus.batch_get:
    tokens (bs*D,), mask (bs*D*D,) with mask[b, i, j] = j is an ancestor of i.
    """
    toks = np.asarray(req_drafts).reshape(bs, D).copy()
    tree = np.asarray(mask).reshape(bs, D, D)
    order = np.arange(D)
    below = order[None, :] < order[:, None]                      # [i, j]: j < i
    for b in range(bs):
        anc = (tree[b] != 0) & below
        parents = np.where(anc.any(-1), (anc * order).argmax(-1), -1)
        parents[0] = -1
        children = [np.flatnonzero(parents == i) for i in range(D)]

        def longest(i):
            best = [i]
            for k in children[i]:                                # ascending index: first wins ties
                cand = [i] + longest(int(k))
                if len(cand) > len(best):
                    best = cand
            return best

        best = longest(0)
        new = np.zeros(D, dtype=toks.dtype)
        new[: len(best)] = toks[b, best]
        toks[b] = new
    if force_reject:
        toks[:, 1:] = 0
    tri = np.tril(np.ones((D, D), dtype=tree.dtype))
    return toks.reshape(-1), np.broadcast_to(tri, (bs, D, D)).reshape(-1).copy()
'''
_ns = {}
exec("import numpy as np\n" + LINEARIZE_SRC, _ns)
_linearize_chain = _ns["_linearize_chain"]          # importable for the CPU unit test

CHECK_METHOD_SRC = '''    def _ngram_chain_check(self, batch, verify_input, accept_lens, accept_index, predict):
        """SGLANG_NGRAM_CHECK=1 (perf/patches/ngram_ple.py Edit 5): chain invariants after verify.
        Sync mode, bs small: a few hundred microseconds (host syncs)."""
        bs, D = len(batch.reqs), self.draft_token_num
        mask = np.asarray(self._chain_check_mask).reshape(bs, D, D)
        tri = np.tril(np.ones((D, D), dtype=mask.dtype))
        assert (mask == tri).all(), f"ngram chain check: host mask is not tril\\n{mask}"
        seq_lens = batch.seq_lens_cpu
        if seq_lens is None or seq_lens.numel() != bs:
            seq_lens = batch.seq_lens.cpu()
        seq_lens = seq_lens.to(torch.int64)
        pos = verify_input.positions.cpu().view(bs, D)
        exp_pos = seq_lens.view(bs, 1) + torch.arange(D, dtype=torch.int64).view(1, D)
        assert torch.equal(pos, exp_pos), f"ngram chain check: positions {pos.tolist()} != {exp_pos.tolist()}"
        ai = accept_index.cpu().to(torch.int64)
        al = accept_lens.cpu().to(torch.int64)
        out_loc = batch.out_cache_loc.cpu().to(torch.int64).view(bs, D)
        r2t = batch.req_to_token_pool.req_to_token
        rpi = batch.req_pool_indices.cpu().to(torch.int64)
        for b in range(bs):
            n = int(al[b])
            assert 1 <= n <= D, f"ngram chain check: accept_lens[{b}]={n}"
            row = ai[b]
            assert torch.equal(row[:n], b * D + torch.arange(n)) and bool((row[n:] == -1).all()), (
                f"ngram chain check: accept_index[{b}]={row.tolist()} is not a row prefix (n={n})"
            )
            # tgt_cache_loc == accept_out_cache_loc in move_accept_tokens_to_target_kvcache
            # (the int8 KV move carries no scales; a no-op move is required)
            s = int(seq_lens[b])
            slots = r2t[int(rpi[b]), s : s + D].cpu().to(torch.int64)
            assert torch.equal(out_loc[b], slots), (
                f"ngram chain check: out_cache_loc row {b} {out_loc[b].tolist()} != req_to_token {slots.tolist()}"
            )
        ngram_pool = getattr(self.req_to_token_pool, "ngram_pool", None)
        if ngram_pool is not None and ngram_pool.context is not None and ngram_pool.context.shape[1] == 2:
            mslots = self.req_to_token_pool.translate_mamba_indices(
                self.req_to_token_pool.get_mamba_indices(batch.req_pool_indices)
            ).cpu().to(torch.int64)
            ctx = ngram_pool.context[mslots].cpu().tolist()
            pred = predict.cpu().to(torch.int64)
            for b, req in enumerate(batch.reqs):
                n = int(al[b])
                accepted = pred[ai[b, :n]].tolist()
                seq = list(req.origin_input_ids) + list(req.output_ids) + accepted[:-1]
                assert ctx[b] == seq[-2:], (
                    f"ngram chain check: PLE context slot {int(mslots[b])} = {ctx[b]} != {seq[-2:]} "
                    f"(accepted={accepted}, accept_len={n}); PLE history not committed?"
                )
        self._ngram_check_steps = getattr(self, "_ngram_check_steps", 0) + 1
        if self._ngram_check_steps in (1, 10, 100) or self._ngram_check_steps % 1000 == 0:
            logger.info("ngram chain check ok: %d verify steps, last accept_lens=%s", self._ngram_check_steps, al.tolist())

'''

# (path, old, new[, "all"])   "all" = replace every occurrence (identical text at several sites)
EDITS = [
  # ------------------------------------------------------------- Edit 1: drop the NGRAM guard
  (Q, """    spec_algorithm = forward_batch.spec_algorithm
    if spec_algorithm is not None and spec_algorithm.is_ngram():
        raise NotImplementedError("Qwen4 PLE does not support NGRAM speculation")
    if (
        forward_batch.spec_info is not None
""", """    # NGRAM speculation runs on a forced linear draft chain (perf/patches/ngram_ple.py).
    if (
        forward_batch.spec_info is not None
"""),
  # ------------------------------------------------------------- Edit 4: PLE intermediates must exist for verify
  (Q, """    get_req_to_token_pool().ple_window_cache = None
    if mode.is_idle():
        return None
""", """    get_req_to_token_pool().ple_window_cache = None
    if mode.is_idle():
        return None
    if mode.is_target_verify():
        # Both PLE intermediates are allocated only when speculative_num_draft_tokens reaches
        # HybridReqToTokenPool; if either is None the post-verify commit scatters nothing and
        # the PLE history silently freezes (perf/patches/ngram_ple.py Edit 4).
        _pool = get_req_to_token_pool()
        _ngram_pool = getattr(_pool, "ngram_pool", None)
        _conv_pool = getattr(_pool, "short_conv_pool", None)
        if ngram_size is not None and (
            _ngram_pool is None
            or not _ngram_pool.enabled
            or _ngram_pool.intermediate_context is None
        ):
            raise RuntimeError(
                "Qwen4 PLE target verify: ngram_pool.intermediate_context is not allocated "
                "(speculative_num_draft_tokens did not reach HybridReqToTokenPool)"
            )
        if _conv_pool is not None and _conv_pool.enabled and _conv_pool.intermediate_conv_state is None:
            raise RuntimeError(
                "Qwen4 PLE target verify: short_conv_pool.intermediate_conv_state is not allocated "
                "(speculative_num_draft_tokens did not reach HybridReqToTokenPool)"
            )
"""),
  # ------------------------------------------------------------- Edit 2/5: worker imports + env gates
  (W, """import logging
from typing import List, Optional
""", """import logging
import os
from typing import List, Optional
"""),
  (W, """USE_FULL_MASK = True
""", """USE_FULL_MASK = True
# perf/patches/ngram_ple.py: debug checks (Edit 5) and forced rejection (validation step 3c)
_NGRAM_CHECK = os.environ.get("SGLANG_NGRAM_CHECK", "0").lower() in ("1", "true", "yes")
_NGRAM_FORCE_REJECT = os.environ.get("SGLANG_NGRAM_FORCE_REJECT", "0").lower() in ("1", "true", "yes")
"""),
  # ------------------------------------------------------------- Edit 2: module-level helper (after _derive_tree_links)
  (W, """class NGRAMWorker(BaseSpecWorker):
    def alloc_memory_pool(self, **kwargs):
""", LINEARIZE_SRC + """

class NGRAMWorker(BaseSpecWorker):
    def alloc_memory_pool(self, **kwargs):
"""),
  # ------------------------------------------------------------- Edit 2a: constructor guard
  (W, """        self.draft_token_num: int = server_args.speculative_num_draft_tokens
""", """        self.draft_token_num: int = server_args.speculative_num_draft_tokens
        assert server_args.speculative_ngram_max_bfs_breadth == 1, (
            "Qwen4-Exp (GDN replayssm fold + QSA + PLE) needs a chain draft: "
            "--speculative-ngram-max-bfs-breadth 1 (it sets speculative_eagle_topk, "
            f"the hybrid backend's chain switch); got {server_args.speculative_ngram_max_bfs_breadth}"
        )
        if _NGRAM_FORCE_REJECT:
            logger.warning("SGLANG_NGRAM_FORCE_REJECT=1: draft rows zeroed, every step commits the bonus token only")
        if _NGRAM_CHECK:
            logger.warning("SGLANG_NGRAM_CHECK=1: per-step chain / PLE-context checks enabled (bench without it)")
"""),
  # ------------------------------------------------------------- Edit 2b: linearize the corpus tree
  (W, """        req_drafts, mask = self.ngram_corpus.batch_get(
            req_ids, batch_tokens, total_lens
        )
        total_draft_token_num = len(req_drafts)
""", """        req_drafts, mask = self.ngram_corpus.batch_get(
            req_ids, batch_tokens, total_lens
        )
        req_drafts, mask = _linearize_chain(
            req_drafts, mask, bs, self.draft_token_num, force_reject=_NGRAM_FORCE_REJECT
        )
        if _NGRAM_CHECK:
            self._chain_check_mask = mask
        total_draft_token_num = len(req_drafts)
"""),
  # ------------------------------------------------------------- Edit 5: check after the state commit
  (W, """            commit_mamba_states_after_verify(
                self.target_worker,
                batch,
                accept_lens,
                accept_index,
                self.draft_token_num,
            )
            accept_tokens = predict[accept_index].flatten()
""", """            commit_mamba_states_after_verify(
                self.target_worker,
                batch,
                accept_lens,
                accept_index,
                self.draft_token_num,
            )
            if _NGRAM_CHECK:
                self._ngram_chain_check(batch, verify_input, accept_lens, accept_index, predict)
            accept_tokens = predict[accept_index].flatten()
"""),
  (W, """    def _update_ngram_corpus(self, batch: ScheduleBatch):
""", CHECK_METHOD_SRC + """    def _update_ngram_corpus(self, batch: ScheduleBatch):
"""),
  # ------------------------------------------------------------- Edit 3: PLE commit in the fold branch
  (S, """            mamba_steps_to_track=mamba_steps_to_track,
            null_block_id=-1,
        )
        return
""", """            mamba_steps_to_track=mamba_steps_to_track,
            null_block_id=-1,
        )
        # PLE n-gram history + PLE short-conv state live outside the GDN fold;
        # roll them to the last accepted node like the generic path does
        # (hybrid_linear_attn_backend.py _update_ple_state_after_mtp_verify).
        # The generic path scatters with PHYSICAL slot ids
        # (forward_metadata.mamba_cache_indices, translated track indices);
        # get_mamba_indices returns VIRTUAL ids, so translate (identity for the
        # static HybridReqToTokenPool, a lookup for the unified pool).
        # perf/patches/ngram_ple.py Edit 3.
        attn_backend = model_runner.attn_backend
        if hasattr(attn_backend, "_update_ple_state_after_mtp_verify"):
            _ple_track = batch.mamba_track_indices
            if _ple_track is not None:
                _ple_track = req_pool.translate_mamba_indices(_ple_track)
            attn_backend._update_ple_state_after_mtp_verify(
                req_pool.translate_mamba_indices(state_batch_indices),
                last_correct_step_indices,
                _ple_track,
                mamba_steps_to_track,
            )
        return
"""),
  # ------------------------------------------------------------- Edit 3: PLE commit in the ring branch
  (S, """        # skipped here.
        return
""", """        # skipped here.
        # PLE side states (perf/patches/ngram_ple.py Edit 3); the track scatter is
        # skipped like the conv one above (extra_buffer is forbidden with replayssm),
        # so no track indices / steps are passed.  Physical slot ids (see the fold
        # branch): translate the virtual get_mamba_indices() result.
        attn_backend = model_runner.attn_backend
        if hasattr(attn_backend, "_update_ple_state_after_mtp_verify"):
            attn_backend._update_ple_state_after_mtp_verify(
                req_pool.translate_mamba_indices(state_batch_indices),
                last_correct_step_indices,
                None,
                None,
            )
        return
"""),
]


def _mode(e):
    return e[3] if len(e) > 3 else "one"


def state():
    out = []
    for e in EDITS:
        p, a, b = e[0], e[1], e[2]
        t = open(p, encoding="utf-8").read()
        out.append((p, a in t, b in t))
    return out


def check():
    for i, (p, pr, ap) in enumerate(state()):
        print(f"  {i} {'APPLIED' if ap else ('clean' if pr else 'MISMATCH'):<8} {_mode(EDITS[i]):<4} "
              f"{os.path.relpath(p, SG)}: {EDITS[i][1].strip().splitlines()[0][:60]}")


def apply():
    st = state()
    if not all(pr or ap for _, pr, ap in st):
        print("  [!] mismatch"); check(); return
    for e, (_, pr, ap) in zip(EDITS, st):
        p, a, b = e[0], e[1], e[2]
        if not ap:
            t = open(p, encoding="utf-8").read()
            open(p, "w", encoding="utf-8").write(t.replace(a, b, -1 if _mode(e) == "all" else 1))
    print("  applied (NGRAM chain + PLE commit; serve with --speculative-algorithm NGRAM "
          "--speculative-num-draft-tokens 4 --speculative-ngram-min-bfs-breadth 1 "
          "--speculative-ngram-max-bfs-breadth 1 --enable-linear-replayssm-spec "
          "--disable-cuda-graph --cuda-graph-backend-decode disabled)")


def revert():
    for e, (_, pr, ap) in zip(EDITS, state()):
        p, a, b = e[0], e[1], e[2]
        if ap:
            t = open(p, encoding="utf-8").read()
            open(p, "w", encoding="utf-8").write(t.replace(b, a, -1 if _mode(e) == "all" else 1))
    print("  reverted")


if __name__ == "__main__":
    {"--check": check, "apply": apply, "revert": revert}[sys.argv[1] if len(sys.argv) > 1 else "--check"]()
