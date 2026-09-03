"""CPU unit test for _linearize_chain (perf/patches/ngram_ple.py, Edit 2 of SPEC_NGRAM_PLAN.md).

Inputs are built in the layout NgramCorpus.batch_get returns: flat int64 tokens (bs*D,) and a flat
int64 ancestor mask (bs*D*D,), produced by a Python replica of trie.cpp buildRecency (bfs breadth 1:
every anchor re-enters from the root and inserts one chain, shared prefixes merge, node budget D-1)
followed by result.cpp fillResult (BFS order, zero-padding nodes as ROOT children, mask row i = copy
of the parent row + the diagonal).  One thing the replica cannot pin down: Node.next is a
std::unordered_map (result.h), so the BFS order AMONG SIBLINGS -- and therefore which chain is
"first" on a length tie -- is hash order in the real corpus (observed: history [7,8,20,5,8,30],
query [7,8] returns [8, 30, 20, 0] although 20 was inserted first).  The replica takes a
child_order hook and every star/partial case is run under several sibling orders; the assertions
only require the result to be ONE of the depth-maximal root chains, never a specific sibling.
The function under test is imported from the patch module, so the tested source is byte-for-byte
the text the patch inserts into ngram_worker.py.

Checks
  1. a single chain is preserved (tokens and tril mask unchanged);
  2. a star (several anchors -> several depth-1 root children) collapses to ONE depth-1 child, padded,
     under every sibling order; the unlinearized star has duplicate positions;
  3. a partial match (one 2-chain + one 1-chain) keeps the longest chain under every sibling order;
  4. padding nodes (root children with token 0) never displace a real chain; no-match -> [root,0,0,0];
  5. branching below the root (D=6) keeps the longest path, not the first child;
  6. batched (bs=3) inputs: per-request results, output dtypes/shapes match batch_get, inputs untouched;
  7. force_reject: draft rows zeroed, root row kept, mask tril;
  8. what the verify code derives from the mask: a replica of sgl_kernel reconstruct_indices_from_tree_mask
     (ngram_utils.cu) gives positions seq_len..seq_len+D-1, retrieve_next_token [1..D-1,-1],
     retrieve_next_sibling all -1, parent(row j) == row j-1; the real host-side _derive_tree_links from
     ngram_worker.py agrees; the UNlinearized star has duplicate positions (why the patch is needed);
  9. a replica of verify_tree_greedy (kernels/aot/csrc/cpu/spec.cpp) on the chain: accept_index is always
     a row prefix b*D + arange(n), -1 after; accept_lens in [1, D]; a padding-0 node is accepted only when
     the target argmax at its parent row is 0;
 10. (optional, SGLANG_NGRAM_TEST_REAL_CORPUS=1) the real JIT NgramCorpus at breadth 1 / D=4 fed with a
     history that PROVABLY creates a multi-anchor star (the raw mask of request 0 is asserted to be
     non-tril); its batch_get output is linearized and re-checked with 8/9 and against the set of
     depth-maximal root chains of the raw tree.

  ~/quant/venv-sglang/bin/python3 perf/gemv/test_ngram_chain.py
"""
import os
import sys
import time
from collections import deque

import numpy as np

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(H, "patches"))
from ngram_ple import _linearize_chain  # noqa: E402  (exec'd from the patch's LINEARIZE_SRC)


# ----------------------------------------------------------------------------- corpus replicas
def fill_result(last_token, D, tree, child_order=None):
    """result.cpp:12-51 fillResult.  tree[node] = {token: child_node}; BFS from root 0; the root row
    holds last_token; zero padding to D with prevs = 0 (root children); mask[i] = mask[parent][:parent+1]
    then the diagonal.  child_order(items) -> items models the std::unordered_map iteration order of
    Node.next (result.h): unspecified in the real corpus, insertion order by default here."""
    child_order = child_order or (lambda items: items)
    tokens, prevs = [last_token], [-1]
    queue = deque((tok, nxt, 0) for tok, nxt in child_order(list(tree[0].items())))
    while queue:
        tok, nxt, prev = queue.popleft()
        tokens.append(tok)
        prevs.append(prev)
        for t, n in child_order(list(tree[nxt].items())):
            queue.append((t, n, len(tokens) - 1))
    while len(tokens) < D:
        tokens.append(0)
        prevs.append(0)
    n = len(tokens)
    mask = np.zeros((n, n), dtype=np.int64)
    mask[0, 0] = 1
    for i in range(n):
        if prevs[i] != -1:
            mask[i, : prevs[i] + 1] = mask[prevs[i], : prevs[i] + 1]
        mask[i, i] = 1
    return np.array(tokens, dtype=np.int64), mask


# sibling orders a std::unordered_map could produce (insertion, reversed, by token, by a hash-like key)
SIBLING_ORDERS = [
    lambda items: items,
    lambda items: list(reversed(items)),
    lambda items: items[1:] + items[:1],                                   # rotate: the middle child first
    lambda items: sorted(items, key=lambda kv: kv[0]),
    lambda items: sorted(items, key=lambda kv: (kv[0] * 2654435761) % 97),
]


def build_recency(last_token, D, anchor_paths, child_order=None):
    """trie.cpp:236-260 buildRecency at bfs breadth 1: each anchor (deepest first) re-enters from the
    root and inserts its chain (an existing token at the parent is reused); node ids 1..D-1."""
    n_real = D - 1
    tree = [dict() for _ in range(D)]
    cursor = 1
    for path in anchor_paths:
        parent = 0
        for tok in path:
            if cursor > n_real:
                break
            if tok in tree[parent]:
                parent = tree[parent][tok]
            else:
                tree[parent][tok] = cursor
                parent = cursor
                cursor += 1
    return fill_result(last_token, D, tree, child_order)


def maximal_root_chains(toks, mask, D):
    """All root->leaf token paths of maximal depth in one request's (tokens, mask) tree (padding nodes
    included: they are root children with token 0).  A correct _linearize_chain result must start with
    one of them (after the root) and be zero-padded after it."""
    toks = np.asarray(toks).reshape(D)
    m = np.asarray(mask).reshape(D, D)
    parents = [-1] + [max(j for j in range(i) if m[i, j]) for i in range(1, D)]
    children = [[i for i in range(D) if parents[i] == p] for p in range(D)]

    def paths(i):
        if not children[i]:
            return [[]]
        return [[c] + rest for c in children[i] for rest in paths(c)]

    all_paths = paths(0)
    depth = max(len(p) for p in all_paths)
    return depth, {tuple(int(toks[k]) for k in p) for p in all_paths if len(p) == depth}


def assert_linearized(toks, mask, D, expect_depth=None):
    """_linearize_chain(one request) is a depth-maximal root chain + zero padding, mask tril."""
    depth, chains = maximal_root_chains(toks, mask, D)
    if expect_depth is not None:
        assert depth == expect_depth, (depth, chains)
    t, m = _linearize_chain(*batch_get_like([(toks, mask)]), 1, D)
    assert (m.reshape(D, D) == tril(D)).all(), m
    assert t[0] == toks[0], (t, toks)
    assert tuple(t[1 : 1 + depth].tolist()) in chains, (t.tolist(), chains)
    assert (t[1 + depth :] == 0).all(), t
    return t, m, chains


def batch_get_like(results):
    """Flatten per-request (tokens, mask) exactly like NgramCorpus.batch_get: int64 (bs*D,), (bs*D*D,)."""
    toks = np.concatenate([t for t, _ in results]).astype(np.int64)
    mask = np.concatenate([m.reshape(-1) for _, m in results]).astype(np.int64)
    return toks, mask


# ----------------------------------------------------------------------------- verify-side replicas
def reconstruct_ref(mask, seq_lens, bs, D):
    """Replica of sgl_kernel reconstructIndicesFromTreeMask (ngram_utils.cu:16-81)."""
    tree = mask.reshape(bs, D, D).astype(bool)
    positions = np.empty(bs * D, dtype=np.int64)
    ri = np.empty((bs, D), dtype=np.int64)
    nt = np.full((bs, D), -1, dtype=np.int64)
    ns = np.full((bs, D), -1, dtype=np.int64)
    parents = np.full((bs, D), -1, dtype=np.int64)
    for b in range(bs):
        for t in range(D):
            depth, parent = 0, -1
            for i in range(t - 1, -1, -1):
                if tree[b, t, i]:
                    depth += 1
                    if parent == -1:
                        parent = i
            ri[b, t] = b * D + t
            positions[b * D + t] = depth + seq_lens[b]
            parents[b, t] = parent
            for i in range(t + 1, D):
                if tree[b, i, t]:
                    nt[b, t] = i
                    break
            if parent != -1:
                for i in range(t + 1, D):
                    if tree[b, i, parent] and not tree[b, i, parent + 1 : i].any():
                        ns[b, t] = i
                        break
    return positions, ri, nt, ns, parents


def verify_greedy_ref(candidates, ri, nt, ns, target_predict, bs, D):
    """Replica of verify_tree_greedy_kernel_impl (kernels/aot/csrc/cpu/spec.cpp:47-92); accept_index
    has num_spec_step = D columns (NgramVerifyInput.max_tree_depth == draft_token_num)."""
    predicts = np.zeros(bs * D, dtype=np.int64)
    accept_index = np.full((bs, D), -1, dtype=np.int64)
    accept_token_num = np.zeros(bs, dtype=np.int64)
    cand, rif, ntf, nsf, tp = (x.reshape(-1) for x in (candidates, ri, nt, ns, target_predict))
    for bx in range(bs):
        off = bx * D
        last = rif[off]
        accept_index[bx, 0] = last
        n_correct, cur = 0, 0
        for j in range(1, D):
            cur = ntf[off + cur]
            while cur != -1:
                draft_idx, draft_tok = rif[off + cur], cand[off + cur]
                if draft_tok == tp[last]:
                    predicts[last] = tp[last]
                    n_correct += 1
                    accept_index[bx, n_correct] = draft_idx
                    last = draft_idx
                    break
                cur = nsf[off + cur]
            if cur == -1:
                break
        accept_token_num[bx] = n_correct
        predicts[last] = tp[last]
    return predicts, accept_index, accept_token_num + 1  # accept_lens incl. bonus (ngram_worker)


# ----------------------------------------------------------------------------- helpers
def tril(D):
    return np.tril(np.ones((D, D), dtype=np.int64))


def check_chain_derivations(toks, mask, bs, D, derive_tree_links=None, seq_lens=None):
    """Assertions 8/9 on a linearized (toks, mask) pair."""
    if seq_lens is None:
        seq_lens = np.array([100 + 7 * b for b in range(bs)], dtype=np.int64)
    m = mask.reshape(bs, D, D)
    assert (m == tril(D)[None]).all(), f"mask is not tril:\n{m}"
    positions, ri, nt, ns, parents = reconstruct_ref(mask, seq_lens, bs, D)
    exp_pos = (seq_lens[:, None] + np.arange(D)[None, :]).reshape(-1)
    assert (positions == exp_pos).all(), f"positions {positions} != {exp_pos}"
    assert (ri == (np.arange(bs)[:, None] * D + np.arange(D)[None, :])).all()
    exp_nt = np.tile(np.append(np.arange(1, D), -1), (bs, 1))
    assert (nt == exp_nt).all(), f"retrieve_next_token {nt} != {exp_nt}"
    assert (ns == -1).all(), f"retrieve_next_sibling {ns} has a sibling"
    exp_par = np.tile(np.arange(-1, D - 1), (bs, 1))
    assert (parents == exp_par).all(), f"parent(row j) != row j-1: {parents}"
    if derive_tree_links is not None:
        import torch

        h_nt, h_ns = derive_tree_links(mask, bs, D)
        assert torch.equal(h_nt, torch.from_numpy(exp_nt)), f"_derive_tree_links next_token {h_nt}"
        assert torch.equal(h_ns, torch.from_numpy(ns)), f"_derive_tree_links next_sibling {h_ns}"
    # greedy verify: accept 0..D-1 drafts by crafting target_predict per request
    cand = toks.reshape(bs, D)
    for n_acc in range(D):
        tp = np.full((bs, D), -7, dtype=np.int64)          # -7: never equal to a draft
        for b in range(bs):
            tp[b, :n_acc] = cand[b, 1 : n_acc + 1]           # row j predicts draft row j+1
            tp[b, n_acc:] = 999_999 + b                      # bonus token(s)
        predicts, ai, al = verify_greedy_ref(cand, ri, nt, ns, tp, bs, D)
        for b in range(bs):
            n = int(al[b])
            assert n == n_acc + 1, f"accept_lens[{b}]={n} != {n_acc + 1}"
            assert (ai[b, :n] == b * D + np.arange(n)).all() and (ai[b, n:] == -1).all(), f"accept_index {ai[b]}"
            acc = predicts[ai[b, :n]]
            assert (acc[:-1] == cand[b, 1:n]).all() and acc[-1] == 999_999 + b, f"accepted {acc}"
    return positions, ri, nt, ns


def main():
    t0 = time.time()
    D = 4
    root = 1234

    # 1. chain preserved
    toks, mask = build_recency(root, D, [[11, 12, 13]])
    assert toks.tolist() == [root, 11, 12, 13] and (mask == tril(D)).all(), (toks, mask)
    t, m = _linearize_chain(*batch_get_like([(toks, mask)]), 1, D)
    assert t.tolist() == [root, 11, 12, 13] and (m.reshape(D, D) == tril(D)).all(), (t, m)
    print("  1 chain preserved                 ", t.tolist())

    # 2. star: three anchors, each one depth-1 child -> ONE depth-1 child kept, padded with 0.
    #    Which child is "first" is hash order in the real corpus, so run every sibling order.
    toks, mask = build_recency(root, D, [[21], [22], [23]])
    assert toks.tolist() == [root, 21, 22, 23]
    assert mask.tolist() == [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]], mask  # star
    pos_star, _, _, _, _ = reconstruct_ref(mask.reshape(-1), np.array([50]), 1, D)
    assert pos_star.tolist() == [50, 51, 51, 51], pos_star                       # the collision
    seen = []
    for order in SIBLING_ORDERS:
        toks_o, mask_o = build_recency(root, D, [[21], [22], [23]], order)
        assert (mask_o == mask).all() and sorted(toks_o.tolist()) == sorted(toks.tolist())
        t, m, chains = assert_linearized(toks_o, mask_o, D, expect_depth=1)
        assert chains == {(21,), (22,), (23,)}, chains
        assert t.tolist() == [root, toks_o[1], 0, 0], (t, toks_o)      # lowest node index wins the tie
        seen.append(int(t[1]))
    assert set(seen) == {21, 22, 23}, seen                             # the orders really differed
    print("  2 star -> one chain + padding    ", seen, " (star positions were", pos_star.tolist(), ")")

    # 3. partial: root->a->b and root->c  (deepest anchor first, as getExpandableAnchors_ orders them)
    toks, mask = build_recency(root, D, [[31, 32], [33]])
    assert toks.tolist() == [root, 31, 33, 32], toks                         # BFS order: depth-1 nodes first
    assert mask.tolist() == [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 1]], mask
    t, m, chains = assert_linearized(toks, mask, D, expect_depth=2)
    assert chains == {(31, 32)} and t.tolist() == [root, 31, 32, 0], (t, chains)
    # same tree under every sibling order / anchor order: the 2-chain always wins
    for order in SIBLING_ORDERS:
        for anchors in ([[31, 32], [33]], [[33], [31, 32]]):
            toks_o, mask_o = build_recency(root, D, anchors, order)
            t, m, chains = assert_linearized(toks_o, mask_o, D, expect_depth=2)
            assert t.tolist() == [root, 31, 32, 0], (t, toks_o, mask_o)
    print("  3 partial star -> longest chain   ", t.tolist(), " (under", len(SIBLING_ORDERS), "sibling orders x 2 anchor orders)")

    # 4. padding: one real child + two zero pads (root children); no match at all
    toks, mask = build_recency(root, D, [[41]])
    assert toks.tolist() == [root, 41, 0, 0]
    assert mask.tolist() == [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]], mask   # pads = root kids
    t, m = _linearize_chain(*batch_get_like([(toks, mask)]), 1, D)
    assert t.tolist() == [root, 41, 0, 0] and (m.reshape(D, D) == tril(D)).all(), t
    toks, mask = build_recency(root, D, [])
    assert toks.tolist() == [root, 0, 0, 0]
    t, m = _linearize_chain(*batch_get_like([(toks, mask)]), 1, D)
    assert t.tolist() == [root, 0, 0, 0] and (m.reshape(D, D) == tril(D)).all(), t
    # a real 2-chain listed AFTER a padding-like real child keeps the 2-chain
    toks, mask = build_recency(root, D, [[0], [42, 43]])       # token 0 as a real match, then a 2-chain
    t, m = _linearize_chain(*batch_get_like([(toks, mask)]), 1, D)
    assert t.tolist() == [root, 42, 43, 0], t
    print("  4 padding / no-match handled      ", t.tolist())

    # 5. branching below the root (D=6): root->a->{b, c->d}; longest path wins over first child
    D6 = 6
    toks, mask = build_recency(root, D6, [[51, 52], [51, 53, 54]])
    assert toks.tolist() == [root, 51, 52, 53, 54, 0], toks
    t, m = _linearize_chain(*batch_get_like([(toks, mask)]), 1, D6)
    assert t.tolist() == [root, 51, 53, 54, 0, 0], t
    assert (m.reshape(D6, D6) == tril(D6)).all()
    check_chain_derivations(t, m, 1, D6)
    for order in SIBLING_ORDERS:
        toks_o, mask_o = build_recency(root, D6, [[51, 52], [51, 53, 54]], order)
        t, m, chains = assert_linearized(toks_o, mask_o, D6, expect_depth=3)
        assert t.tolist() == [root, 51, 53, 54, 0, 0], (t, toks_o)
    print("  5 branch below root -> longest    ", t.tolist())

    # 6. batch of 3 (chain / star / no-match), dtypes, shapes, inputs untouched
    rs = [build_recency(1, D, [[11, 12, 13]]), build_recency(2, D, [[21], [22], [23]]), build_recency(3, D, [])]
    toks, mask = batch_get_like(rs)
    toks0, mask0 = toks.copy(), mask.copy()
    assert toks.dtype == np.int64 and mask.dtype == np.int64 and toks.shape == (3 * D,) and mask.shape == (3 * D * D,)
    t, m = _linearize_chain(toks, mask, 3, D)
    assert t.dtype == np.int64 and m.dtype == np.int64 and t.shape == (3 * D,) and m.shape == (3 * D * D,), (t.dtype, m.dtype)
    assert t.flags.c_contiguous and m.flags.c_contiguous and t.flags.writeable and m.flags.writeable
    assert t.reshape(3, D).tolist() == [[1, 11, 12, 13], [2, 21, 0, 0], [3, 0, 0, 0]], t
    assert (toks == toks0).all() and (mask == mask0).all(), "inputs mutated"
    print("  6 batched bs=3                    ", t.reshape(3, D).tolist())

    # 7. force_reject
    t_fr, m_fr = _linearize_chain(toks, mask, 3, D, force_reject=True)
    assert t_fr.reshape(3, D).tolist() == [[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]], t_fr
    assert (m_fr.reshape(3, D, D) == tril(D)[None]).all()
    print("  7 force_reject                    ", t_fr.reshape(3, D).tolist())

    # 8/9. derivations the verify code makes from the mask (replicas + the real _derive_tree_links)
    derive = None
    try:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        import warnings

        warnings.filterwarnings("ignore")
        t_imp = time.time()
        from sglang.srt.speculative.ngram_worker import _derive_tree_links as derive

        print(f"  8 imported ngram_worker._derive_tree_links ({time.time() - t_imp:.1f} s)")
    except Exception as e:  # pragma: no cover
        print(f"  8 [!] ngram_worker import failed ({type(e).__name__}: {str(e)[:120]}); replica only")
    positions, ri, nt, ns = check_chain_derivations(t, m, 3, D, derive)
    check_chain_derivations(t_fr, m_fr, 3, D, derive)
    print("  8 positions", positions.tolist(), "next_token", nt.tolist(), "next_sibling", ns.tolist())
    # 9b. padding-0 semantics: accepted only when the parent row's argmax is 0 (then it IS the greedy token)
    cand = t.reshape(3, D)
    tp = np.array([[11, 12, 13, 5], [21, 0, 0, 5], [0, 0, 7, 5]])       # req1: pads accepted, req2: 2 pads
    predicts, ai, al = verify_greedy_ref(cand, ri, nt, ns, tp, 3, D)
    assert al.tolist() == [4, 4, 3], al
    assert ai.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, -1]], ai
    assert predicts[ai[2, :3]].tolist() == [0, 0, 7]
    tp = np.array([[11, 12, 13, 5], [21, 9, 0, 5], [9, 0, 7, 5]])       # a non-zero argmax rejects the pad
    _, ai, al = verify_greedy_ref(cand, ri, nt, ns, tp, 3, D)
    assert al.tolist() == [4, 2, 1] and ai.tolist() == [[0, 1, 2, 3], [4, 5, -1, -1], [8, -1, -1, -1]], (al, ai)
    print("  9 greedy verify: accept_index always a row prefix; pad 0 accepted only as the argmax")

    # 10. optional: the real JIT corpus
    if os.environ.get("SGLANG_NGRAM_TEST_REAL_CORPUS", "0") == "1":
        from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus

        # max_trie_depth=4: the deepest anchor (7,8) of query [7,8] yields the chain 20->5 (the history
        # ends after 5,8,30 so (7,8,20,5) has no deeper child within the trie window) and the depth-1
        # anchor (8) then inserts its most recent child 30 as a SECOND root child -> a star, raw mask
        # not tril.  The chain 20->5 is the unique depth-maximal root chain whichever sibling comes first.
        corpus = NgramCorpus(max_trie_depth=4, min_bfs_breadth=1, max_bfs_breadth=1, draft_token_num=D)
        corpus.batch_put([[7, 8, 20, 5, 8, 30]])
        corpus.synchronize()
        q = [[7, 8], [5, 8], [99, 98]]
        raw_t, raw_m = corpus.batch_get([f"r{i}" for i in range(3)], q, [len(x) for x in q])
        assert raw_t.dtype == np.int64 and raw_m.dtype == np.int64 and raw_t.shape == (3 * D,)
        raw_m3 = raw_m.reshape(3, D, D)
        assert not (raw_m3[0] == tril(D)).all(), f"expected a star for query [7,8], got tril:\n{raw_m3[0]}"
        star_pos, _, _, _, _ = reconstruct_ref(raw_m3[0].reshape(-1), np.array([50]), 1, D)
        assert len(set(star_pos.tolist())) < D, star_pos                # duplicate positions before the fix
        assert sorted(raw_t.reshape(3, D)[0, 1:].tolist()) == [5, 20, 30], raw_t
        depth0, chains0 = maximal_root_chains(raw_t.reshape(3, D)[0], raw_m3[0], D)
        assert depth0 == 2 and chains0 == {(20, 5)}, (depth0, chains0)
        lt, lm = _linearize_chain(raw_t, raw_m, 3, D)
        assert lt.reshape(3, D)[0].tolist() == [8, 20, 5, 0], lt
        check_chain_derivations(lt, lm, 3, D, derive)
        for b in range(3):
            assert_linearized(raw_t.reshape(3, D)[b], raw_m3[b], D)
            chain = [x for x in lt.reshape(3, D)[b, 1:].tolist() if x != 0]
            leaf_paths = corpus.leaf_paths_from_mask(raw_t.reshape(3, D)[b].tolist(), raw_m3[b].tolist())
            leaf_paths = [p[1:] for p in leaf_paths]   # drop the root token
            assert any(p[: len(chain)] == chain for p in leaf_paths) or not chain, (chain, leaf_paths)
        print("  10 real NgramCorpus raw:", raw_t.reshape(3, D).tolist(), "(req 0 is a star, positions",
              star_pos.tolist(), ") -> chain:", lt.reshape(3, D).tolist())
    else:
        print("  10 real NgramCorpus check skipped (SGLANG_NGRAM_TEST_REAL_CORPUS=1 to enable)")

    print(f"\n  ALL PASSED ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
