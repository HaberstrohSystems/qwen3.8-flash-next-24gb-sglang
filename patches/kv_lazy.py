#!/usr/bin/env python3
"""Lazy KV backing: reserve the KV cache as virtual address space, back it on demand.

SGLang already keeps a VMM arena for "post-capture KV sizing" (kv_vmm_backing.KvVmmBufferOwner)
but only commits monotonically up to the final capacity. This patch (SGLANG_KV_LAZY=1):

  * allocates the full-attention KV buffers through that owner in the classic flow too,
    backing only SGLANG_KV_LAZY_FLOOR tokens (default 4096) at start;
  * TokenToKVPoolAllocator.alloc backs the prefix up to the highest slot handed out plus
    SGLANG_KV_LAZY_MARGIN tokens (default 2048), so commits happen in bursts;
  * when the pool becomes completely free the allocator resets to slot order and unmaps the
    backing beyond the floor (physical memory returns to the driver);
  * a failed commit first asks the elastic expert cache (ExpertElastic.inst) to shrink.

With --max-running-requests 1 and no radix cache the live KV prefix is exactly the current
request's length, so this turns the 32k (or 128k) reservation into pay-as-you-go memory.

  python3 kv_lazy.py --check | apply | revert
"""
import os, sys
SG = os.path.join(os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")), "python/sglang")
MP = f"{SG}/srt/mem_cache/memory_pool.py"
TK = f"{SG}/srt/mem_cache/allocator/token.py"
PG = f"{SG}/srt/mem_cache/allocator/paged.py"
VB = f"{SG}/srt/mem_cache/kv_vmm_backing.py"
KC = f"{SG}/srt/mem_cache/kv_cache_configurator.py"

EDITS = [
  # ------------------------------------------------------------- kv_vmm_backing.py
  (VB, """    @property
    def backed_bytes(self) -> int:
        \"\"\"Total physically-backed bytes (sum of scattered per-buffer ranges).\"\"\"
        return self._range_backed
""", """    @property
    def backed_bytes(self) -> int:
        \"\"\"Total physically-backed bytes (sum of scattered per-buffer ranges).\"\"\"
        return self._range_backed

    def uncommit_beyond(self, offset: int, keep_bytes: int) -> int:
        \"\"\"Release every mapping of the buffer at ``offset`` that starts at or beyond
        ``keep_bytes`` (mappings straddling the boundary stay). Returns bytes freed.\"\"\"
        if self._closed:
            return 0
        from sglang.srt.cuda_vmm_utils import _get_cuda_driver, check_drv
        drv = _get_cuda_driver()
        start = self.base + offset
        limit = start + self._committed_by_offset.get(offset, 0)
        cut = start + self._align(int(keep_bytes))
        maps = self._allocation._mappings
        victims = sorted([m for m in maps if start <= m[0] < limit and m[0] >= cut], key=lambda m: -m[0])
        freed = 0
        with torch.cuda.device(self.device_id):
            for m in victims:
                address, size, handle = m
                check_drv(drv.cuMemUnmap(address, size), "cuMemUnmap(uncommit)")
                if handle is not None:
                    check_drv(drv.cuMemRelease(handle), "cuMemRelease(uncommit)")
                maps.remove(m)
                freed += size
        if freed:
            self._committed_by_offset[offset] = self._committed_by_offset.get(offset, 0) - freed
            self._range_backed -= freed
        return freed
"""),
  (VB, """    def ensure_prefix(self, num_tokens: int) -> None:
        \"\"\"Ensure the first ``num_tokens`` slots of every buffer are physically backed.\"\"\"
        self._back_spans(
            [s.desc.prefix_span_bytes(num_tokens, self.page_size) for s in self._specs]
        )
""", """    def ensure_prefix(self, num_tokens: int) -> None:
        \"\"\"Ensure the first ``num_tokens`` slots of every buffer are physically backed.\"\"\"
        self._back_spans(
            [s.desc.prefix_span_bytes(num_tokens, self.page_size) for s in self._specs]
        )
        self.backed_tokens = max(getattr(self, "backed_tokens", 0), int(num_tokens))

    def release_beyond(self, num_tokens: int) -> int:
        \"\"\"Unmap the backing beyond the first ``num_tokens`` slots of every buffer.\"\"\"
        if self._arena is None:
            return 0
        torch.cuda.synchronize()
        freed = 0
        for s in self._specs:
            keep = align_up(s.desc.prefix_span_bytes(num_tokens, self.page_size), self._arena.granularity)
            freed += self._arena.uncommit_beyond(s.offset, keep)
            s.backed_to = min(s.backed_to, self._arena._committed_by_offset.get(s.offset, 0))
        self.backed_tokens = min(getattr(self, "backed_tokens", 0), int(num_tokens))
        return freed

    def bytes_per_token(self) -> int:
        return sum(-(-s.desc.row_bytes // s.desc.tokens_per_row) for s in self._specs)
"""),
  # ------------------------------------------------------------- memory_pool.py
  (MP, """            if self.post_capture_active:
                self._alloc_post_capture_buffers()
            else:
                self._create_buffers_normal()
""", """            if self.post_capture_active:
                self._alloc_post_capture_buffers()
            elif os.environ.get("SGLANG_KV_LAZY") == "1":
                self._alloc_post_capture_buffers()
                self._lazy_floor = min(self.size, int(os.environ.get("SGLANG_KV_LAZY_FLOOR", "4096")))
                self._lazy_margin = int(os.environ.get("SGLANG_KV_LAZY_MARGIN", "2048"))
                self._post_capture_owner.ensure_prefix(self._lazy_floor)
                logger.info("KV lazy backing: %d tokens reserved as VA, %d backed (floor), margin %d, %.1f KB/token",
                            self.size, self._lazy_floor, self._lazy_margin,
                            self._post_capture_owner.bytes_per_token() / 1024)
            else:
                self._create_buffers_normal()
"""),
  (MP, """    def _alloc_post_capture_buffers(self):
        dev = torch.device(self.device)
""", """    def lazy_ensure(self, num_tokens: int) -> None:
        \"\"\"Back KV slots [0, num_tokens + margin). A failed commit first shrinks the elastic
        expert cache, then retries once.\"\"\"
        o = getattr(self, "_post_capture_owner", None)
        if o is None or not hasattr(self, "_lazy_floor"):
            return
        m = self._lazy_margin                                                           # commit in margin-sized steps
        want = min(self.size + self.page_size, -(-(int(num_tokens) + m) // m) * m)     # slot index == size exists
        if want <= o.backed_tokens:
            return
        # watermark: after the commit at least SGLANG_KV_LAZY_HEADROOM_MB must stay driver-free for
        # the prefill's working memory; otherwise the elastic expert cache gives way first.
        headroom = int(os.environ.get("SGLANG_KV_LAZY_HEADROOM_MB", "1536")) << 20
        delta = (want - o.backed_tokens) * o.bytes_per_token()
        if torch.cuda.mem_get_info()[0] - delta < headroom:
            from sglang.srt.layers.moe.expert_elastic import ExpertElastic
            el = ExpertElastic.inst
            # Below the watermark for the whole tail of a long prefill, empty_cache() on every 2048-token
            # commit forces torch to cudaMalloc its working set again each time (periodic ~1 s spikes).
            # Only do the expensive part when the expert cache still has rows to give back, else rate-limit.
            can_shrink = el is not None and any(st.S > el.s_floor() for st in el.layers.values())
            now = __import__("time").time(); last = getattr(self, "_lazy_last_empty", 0.0)
            if can_shrink or now - last > 30.0:
                torch.cuda.empty_cache(); self._lazy_last_empty = now
                if can_shrink and torch.cuda.mem_get_info()[0] - delta < headroom:
                    el.free(((headroom + delta) >> 20) + 64)
                if torch.cuda.mem_get_info()[0] - delta < headroom // 2:
                    raise RuntimeError(f"KV lazy backing: no headroom for {want} tokens "
                                       f"(free {torch.cuda.mem_get_info()[0] >> 20} MB, need {(delta + headroom) >> 20} MB)")
        try:
            try:
                o.ensure_prefix(want)
            except Exception:
                torch.cuda.empty_cache()                 # torch-cached blocks are not driver-free
                o.ensure_prefix(want)
            _free = torch.cuda.mem_get_info()[0]
            logger.info("KV lazy commit: tokens %d -> %d (owner backed %.0f MB, torch reserved %.2f GB, driver free %.2f GB, capturing=%s)",
                        num_tokens, want, o.backed_bytes / 2**20, torch.cuda.memory_reserved() / 2**30, _free / 2**30,
                        torch.cuda.is_current_stream_capturing())
        except Exception as ex:
            from sglang.srt.layers.moe.expert_elastic import ExpertElastic
            el = ExpertElastic.inst
            if el is None:
                raise
            need_mb = ((want - o.backed_tokens) * o.bytes_per_token() >> 20) + 1024  # + prefill working memory
            logger.warning("KV lazy commit failed (%s); shrinking expert cache to free %d MB", ex, need_mb)
            el.free(need_mb)
            o.ensure_prefix(want)

    def lazy_release(self) -> None:
        o = getattr(self, "_post_capture_owner", None)
        if o is None or not hasattr(self, "_lazy_floor"):
            return
        freed = o.release_beyond(self._lazy_floor)
        if freed:
            logger.info("KV lazy backing: released %.0f MB beyond the %d-token floor", freed / 2**20, self._lazy_floor)
        try:                                            # pool idle = no forward in flight: safe point to regrow experts
            from sglang.srt.layers.moe.expert_elastic import ExpertElastic
            el = ExpertElastic.inst
            if el is not None and el.pending_regrow:
                el.regrow()
        except Exception as ex:
            logger.warning("elastic regrow after KV release failed: %s", ex)

    def _alloc_post_capture_buffers(self):
        dev = torch.device(self.device)
"""),
  # ------------------------------------------------------------- kv_cache_configurator.py
  # With lazy backing the KV pool is address space, not memory: let SGLANG_KV_LAZY_TOKENS raise
  # the capacity above the profiled value (physical pages come from the expert cache on demand).
  (KC, """        user_limit = get_schedule().max_total_tokens

        # Apply user-specified upper bound
        if user_limit is not None:
""", """        user_limit = get_schedule().max_total_tokens
        _env = __import__("os").environ
        _lazy_tokens = int(_env.get("SGLANG_KV_LAZY_TOKENS", "0")) if _env.get("SGLANG_KV_LAZY") == "1" else 0
        if _lazy_tokens > 0:
            # The profiled capacity is what the free VRAM at startup could back physically; with the
            # headroom rule (1.5 GB kept free for prefill working memory) the practical limit measured
            # 0.85 x profiled (84k of 106k tokens at bf16). Longer prompts are rejected at admission
            # instead of crashing the scheduler mid-prefill.
            _safety = float(_env.get("SGLANG_KV_LAZY_SAFETY", "0.85"))
            _new = min(_lazy_tokens, int(token_capacity * _safety))
            logging.warning(f"KV lazy backing: token capacity {token_capacity} (profiled) -> {_new} "
                            f"(requested {_lazy_tokens}, safety {_safety})")
            token_capacity = _new

        # Apply user-specified upper bound
        if user_limit is not None:
"""),
  # ------------------------------------------------------------- allocator/paged.py (page_size 64: QSA)
  (PG, """    def alloc(self, need_size: int):
        # page-aligned allocation, returning contiguous indices of pages
""", """    def _lazy_hook(self, pages: torch.Tensor) -> bool:
        \"\"\"Back the KV rows of the pages about to be handed out. False = no physical memory.\"\"\"
        kv = getattr(self._kvcache, "full_kv_pool", self._kvcache)
        ensure = getattr(kv, "lazy_ensure", None)
        if ensure is None or pages.numel() == 0:
            return True
        try:
            ensure((int(pages.max().item()) + 1) * self.page_size)
            return True
        except Exception as ex:
            __import__("logging").getLogger(__name__).warning("KV lazy backing: commit failed, allocation refused (%s)", ex)
            return False

    def _lazy_idle_check(self) -> None:
        if len(self.free_pages) + len(self.release_pages) >= self.num_pages:   # pool idle: slot order + give memory back
            kv = getattr(self._kvcache, "full_kv_pool", self._kvcache)
            release = getattr(kv, "lazy_release", None)
            if release is not None:
                self.clear()
                release()

    def alloc(self, need_size: int):
        # page-aligned allocation, returning contiguous indices of pages
"""),
  (PG, """        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
""", """        out_pages = self.free_pages[:num_pages]
        if not self._lazy_hook(out_pages):
            return None
        self.free_pages = self.free_pages[num_pages:]
"""),
  (PG, """                prefix_lens=prefix_lens_cpu,
            )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices
""", """                prefix_lens=prefix_lens_cpu,
            )
        if num_new_pages > len(self.free_pages):
            return None
        if not self._lazy_hook(self.free_pages[:num_new_pages]):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices
"""),
  (PG, """            decode=True,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices
""", """            decode=True,
        )
        if num_new_pages > len(self.free_pages):
            return None
        if not self._lazy_hook(self.free_pages[:num_new_pages]):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices
"""),
  (PG, """    def _release_page_ids(self, *page_ids: torch.Tensor):
        if self.need_sort:
            self.release_pages = torch.cat((*page_ids, self.release_pages))
        else:
            self.free_pages = torch.cat((*page_ids, self.free_pages))
""", """    def _release_page_ids(self, *page_ids: torch.Tensor):
        if self.need_sort:
            self.release_pages = torch.cat((*page_ids, self.release_pages))
        else:
            self.free_pages = torch.cat((*page_ids, self.free_pages))
        self._lazy_idle_check()
"""),
  # ------------------------------------------------------------- allocator/token.py
  (TK, """        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index
""", """        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        kv = getattr(self._kvcache, "full_kv_pool", self._kvcache)
        ensure = getattr(kv, "lazy_ensure", None)
        if ensure is not None:
            try:
                ensure(int(select_index.max().item()) + 1)
            except Exception as ex:                    # no physical memory: hand the slots back, report "full"
                self.free_pages = torch.cat((select_index, self.free_pages))
                __import__("logging").getLogger(__name__).warning("KV lazy backing: commit failed, allocation refused (%s)", ex)
                return None
        return select_index
"""),
  (TK, """            if self.need_sort:
                self.release_pages = torch.cat((self.release_pages, free_index))
            else:
                self.free_pages = torch.cat((self.free_pages, free_index))
""", """            if self.need_sort:
                self.release_pages = torch.cat((self.release_pages, free_index))
            else:
                self.free_pages = torch.cat((self.free_pages, free_index))
            if self.available_size() >= self.size:            # pool idle: slot order + give memory back
                kv = getattr(self._kvcache, "full_kv_pool", self._kvcache)
                release = getattr(kv, "lazy_release", None)
                if release is not None:
                    self.clear()
                    release()
"""),
]


def state():
    return [(p, a in open(p, encoding="utf-8").read(), b in open(p, encoding="utf-8").read())
            for p, a, b in EDITS]


def check():
    for i, (p, pr, ap) in enumerate(state()):
        print(f"  {i} {'APPLIED' if ap else ('clean' if pr else 'MISMATCH'):<8} "
              f"{os.path.relpath(p, SG)}: {EDITS[i][1].strip().splitlines()[0][:60]}")


def apply():
    st = state()
    if not all(pr or ap for _, pr, ap in st):
        print("  [!] mismatch"); check(); return
    for (p, a, b), (_, pr, ap) in zip(EDITS, st):
        if not ap:
            t = open(p, encoding="utf-8").read()
            open(p, "w", encoding="utf-8").write(t.replace(a, b, 1))
    print("  applied (lazy KV backing; SGLANG_KV_LAZY=1 to activate)")


def cleanup():
    """Remove applied lazy blocks by marker, for a tree patched with an older version of this file."""
    import re
    t = open(MP, encoding="utf-8").read()
    t2 = re.sub(r"    def lazy_ensure\(self, num_tokens: int\) -> None:.*?(?=    def _alloc_post_capture_buffers\(self\):)", "", t, flags=re.S)
    t2 = re.sub(r"            elif os\.environ\.get\(\"SGLANG_KV_LAZY\"\) == \"1\":.*?(?=            else:\n                self\._create_buffers_normal\(\))", "", t2, flags=re.S)
    if t2 != t: open(MP, "w", encoding="utf-8").write(t2); print("  memory_pool.py: lazy blocks removed")
    t = open(TK, encoding="utf-8").read()
    t2 = re.sub(r"        kv = getattr\(self\._kvcache, \"full_kv_pool\", self\._kvcache\)\n        ensure = .*?(?=        return select_index\n)", "", t, flags=re.S)
    t2 = re.sub(r"            if self\.available_size\(\) >= self\.size:            # pool idle.*?release\(\)\n", "", t2, flags=re.S)
    if t2 != t: open(TK, "w", encoding="utf-8").write(t2); print("  token.py: lazy hooks removed")
    t = open(PG, encoding="utf-8").read()
    t2 = re.sub(r"    def _lazy_hook\(self, pages: torch\.Tensor\) -> bool:.*?(?=    def alloc\(self, need_size: int\):)", "", t, flags=re.S)
    t2 = t2.replace("        if not self._lazy_hook(out_pages):\n            return None\n", "")
    t2 = t2.replace("        if not self._lazy_hook(self.free_pages[:num_new_pages]):\n            return None\n", "")
    t2 = t2.replace("        self._lazy_idle_check()\n", "")
    if t2 != t: open(PG, "w", encoding="utf-8").write(t2); print("  paged.py: lazy hooks removed")
    t = open(VB, encoding="utf-8").read()
    t2 = re.sub(r"\n    def uncommit_beyond\(self, offset: int, keep_bytes: int\) -> int:.*?(?=\n    @property\n    def cursor_bytes)", "", t, flags=re.S)
    t2 = re.sub(r"        self\.backed_tokens = max\(getattr\(self, \"backed_tokens\", 0\), int\(num_tokens\)\)\n\n    def release_beyond.*?(?=    def finalize\()", "", t2, flags=re.S)
    if t2 != t: open(VB, "w", encoding="utf-8").write(t2); print("  kv_vmm_backing.py: lazy methods removed")
    t = open(KC, encoding="utf-8").read()
    t2 = re.sub(r"        _env = __import__\(\"os\"\)\.environ\n.*?token_capacity = _lazy_tokens\n", "", t, flags=re.S)
    t2 = re.sub(r"        _lazy_tokens = int\(os\.environ.*?token_capacity = _lazy_tokens\n", "", t2, flags=re.S)
    if t2 != t: open(KC, "w", encoding="utf-8").write(t2); print("  kv_cache_configurator.py: override removed")
    check()


def revert():
    for (p, a, b), (_, pr, ap) in zip(EDITS, state()):
        if ap:
            t = open(p, encoding="utf-8").read()
            open(p, "w", encoding="utf-8").write(t.replace(b, a, 1))
    print("  reverted")


if __name__ == "__main__":
    {"--check": check, "apply": apply, "revert": revert, "cleanup": cleanup}[sys.argv[1] if len(sys.argv) > 1 else "--check"]()
