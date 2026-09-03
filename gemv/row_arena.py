"""RowArena: a GPU cache of fixed-size rows whose physical memory can be handed back.

The arena reserves virtual address space for ``max_rows`` rows once (cuMemAddressReserve) and
backs a PREFIX of it with physical chunks on demand (cuMemCreate + cuMemMap). Rows are kept in
rank order (hottest first), so shrinking the cache is a tail unmap: the memory really returns to
the driver, in chunks, while every row address stays constant for the arena's lifetime. Kernels
that read rows through an address table, and CUDA graphs that captured such kernels, never need
to know.

    a = RowArena(row_bytes, max_rows, device_id)
    a.ensure_rows(n)      rows [0, n) are physically backed (may raise ArenaOOM)
    a.shrink_rows(n)      release every chunk that lies entirely beyond row n
    a.row_addr(i)         base + i * row_bytes (valid as an address even when unbacked)
    a.view(n, tail, dtype) non-owning torch tensor [n, *tail] over the backed prefix
    a.backed_rows         rows guaranteed backed right now

Chunk size is a multiple of the device granularity (2 MiB on current GPUs); with 4 MiB chunks
the cache is resizable in ~20-row steps for a 205 KB expert row.

  python3 row_arena.py   self-test (needs a CUDA device with ~150 MB free)
"""
import torch

from sglang.srt.cuda_vmm_utils import (  # the driver plumbing SGLang already ships
    _get_cuda_driver, align_up, check_drv, get_device_granularity,
    make_device_allocation_prop, make_rw_access_desc, tensor_from_pointer,
)


class ArenaOOM(RuntimeError):
    """cuMemCreate failed: the device has no free physical memory for another chunk."""


class RowArena:
    def __init__(self, row_bytes: int, max_rows: int, device_id: int = 0,
                 chunk_bytes: int = 4 << 20, name: str = ""):
        drv = _get_cuda_driver()
        self.row_bytes, self.max_rows, self.device_id, self.name = int(row_bytes), int(max_rows), int(device_id), name
        self.gran = get_device_granularity(self.device_id)
        self.chunk = align_up(chunk_bytes, self.gran)
        self.prop = make_device_allocation_prop(self.device_id, handle_types=None)
        self.access = [make_rw_access_desc(self.device_id)]
        self.va_bytes = align_up(self.row_bytes * self.max_rows, self.chunk)
        self.base = int(check_drv(drv.cuMemAddressReserve(self.va_bytes, 0, 0, 0), "cuMemAddressReserve"))
        self._handles = []            # chunk i backs [i*chunk, (i+1)*chunk)
        self._closed = False

    # ---------------------------------------------------------------- geometry
    @property
    def backed_bytes(self) -> int:
        return len(self._handles) * self.chunk

    @property
    def backed_rows(self) -> int:
        return min(self.max_rows, self.backed_bytes // self.row_bytes)

    def row_addr(self, i: int) -> int:
        return self.base + i * self.row_bytes

    def chunks_for_rows(self, n: int) -> int:
        return -(-(min(int(n), self.max_rows) * self.row_bytes) // self.chunk) if n > 0 else 0

    def bytes_to_reach(self, n: int) -> int:
        """Bytes cuMemCreate would need to make rows [0, n) backed (chunk-granular)."""
        return max(0, self.chunks_for_rows(n) - len(self._handles)) * self.chunk

    def rows_for_bytes(self, nbytes: int) -> int:
        return min(self.max_rows, int(nbytes) // self.row_bytes)

    # ---------------------------------------------------------------- resize
    def ensure_rows(self, n: int) -> int:
        """Back rows [0, n). Returns the number of bytes newly mapped."""
        n = max(0, min(int(n), self.max_rows))
        want_chunks = -(-(n * self.row_bytes) // self.chunk) if n else 0
        drv = _get_cuda_driver(); added = 0
        with torch.cuda.device(self.device_id):
            while len(self._handles) < want_chunks:
                off = len(self._handles) * self.chunk
                err, handle = drv.cuMemCreate(self.chunk, self.prop, 0)
                if err != drv.CUresult.CUDA_SUCCESS:
                    raise ArenaOOM(f"{self.name}: cuMemCreate({self.chunk >> 20} MiB) -> {err}")
                try:
                    check_drv(drv.cuMemMap(self.base + off, self.chunk, 0, handle, 0), "cuMemMap")
                    check_drv(drv.cuMemSetAccess(self.base + off, self.chunk, self.access, 1), "cuMemSetAccess")
                except BaseException:
                    drv.cuMemRelease(handle); raise
                self._handles.append(handle); added += self.chunk
        return added

    def shrink_rows(self, n: int) -> int:
        """Keep rows [0, n) backed, release every chunk entirely beyond. Returns bytes freed.
        The caller must make sure no kernel still reads the released rows (sync first)."""
        n = max(0, min(int(n), self.max_rows))
        keep_chunks = -(-(n * self.row_bytes) // self.chunk) if n else 0
        drv = _get_cuda_driver(); freed = 0
        while len(self._handles) > keep_chunks:
            off = (len(self._handles) - 1) * self.chunk
            handle = self._handles.pop()
            check_drv(drv.cuMemUnmap(self.base + off, self.chunk), "cuMemUnmap")
            check_drv(drv.cuMemRelease(handle), "cuMemRelease")
            freed += self.chunk
        return freed

    # ---------------------------------------------------------------- views
    def view(self, n_rows: int, tail=(), dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        """Non-owning tensor [n_rows, *tail] over the backed prefix. Re-create after a shrink."""
        assert n_rows <= self.backed_rows, f"{self.name}: view of {n_rows} rows, only {self.backed_rows} backed"
        return tensor_from_pointer(self.base, n_rows * self.row_bytes, shape=(n_rows,) + tuple(tail),
                                   dtype=dtype, device_id=self.device_id)

    def close(self):
        if self._closed:
            return
        self._closed = True
        torch.cuda.synchronize()
        self.shrink_rows(0)
        check_drv(_get_cuda_driver().cuMemAddressFree(self.base, self.va_bytes), "cuMemAddressFree")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------- self-test
def _selftest():
    import triton, triton.language as tl

    @triton.jit
    def _gather(tab_ptr, idx_ptr, out_ptr, row_bytes, BLOCK: tl.constexpr):
        r = tl.program_id(0); e = tl.load(idx_ptr + r)
        src = tl.load(tab_ptr + e).to(tl.pointer_type(tl.uint8))
        for off in range(0, row_bytes, BLOCK):
            o = off + tl.arange(0, BLOCK); m = o < row_bytes
            tl.store(out_ptr + r * row_bytes + o, tl.load(src + o, mask=m, other=0), mask=m)

    row_bytes, E, S = 204800, 512, 184                      # a w13 int2 expert row
    free0 = torch.cuda.mem_get_info()[0]
    a = RowArena(row_bytes, E, 0, name="w13")
    added = a.ensure_rows(S)
    free1 = torch.cuda.mem_get_info()[0]
    print(f"  granularity {a.gran >> 20} MiB, chunk {a.chunk >> 20} MiB, VA {a.va_bytes >> 20} MiB")
    print(f"  ensure_rows({S}): mapped {added >> 20} MiB, backed_rows {a.backed_rows}, free -{(free0 - free1) >> 20} MiB")
    assert a.backed_rows >= S and abs((free0 - free1) - added) < (8 << 20)

    v = a.view(S, (row_bytes,))
    for i in range(S):
        v[i].fill_(i % 251)
    host = torch.full((E - S, row_bytes), 7, dtype=torch.uint8).pin_memory()   # "cold" rows on the host
    tab = torch.empty(E, dtype=torch.int64)
    for e in range(E):
        tab[e] = a.row_addr(e) if e < S else host.data_ptr() + (e - S) * row_bytes
    tab = tab.cuda()
    idx = torch.tensor([0, 5, 183, 184, 511, 63, 64], dtype=torch.int64).cuda()
    out = torch.empty(len(idx), row_bytes, dtype=torch.uint8, device="cuda")

    def run():
        _gather[(len(idx),)](tab, idx, out, row_bytes, BLOCK=4096)
    run(); torch.cuda.synchronize()
    exp = [i % 251 if i < S else 7 for i in idx.tolist()]
    got = out[:, 0].tolist() + out[:, -1].tolist()
    assert got == exp + exp, (got, exp)
    print("  table gather through GPU + host addresses: ok")

    # CUDA graph captured against the arena addresses
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        run()
    torch.cuda.synchronize()
    with torch.cuda.graph(g, stream=s):
        run()
    torch.cuda.synchronize()

    # shrink to 64 rows: memory returns, addresses stay
    torch.cuda.synchronize()
    free_a = torch.cuda.mem_get_info()[0]
    freed = a.shrink_rows(64)
    free_b = torch.cuda.mem_get_info()[0]
    print(f"  shrink_rows(64): freed {freed >> 20} MiB, backed_rows {a.backed_rows}, free +{(free_b - free_a) >> 20} MiB")
    assert abs((free_b - free_a) - freed) < (8 << 20)
    # repoint evicted rows to the host, replay the graph
    tab_cpu = tab.cpu()
    for e in range(64, S):
        tab_cpu[e] = host.data_ptr() + 0 * row_bytes       # any valid host row
    tab.copy_(tab_cpu); g.replay(); torch.cuda.synchronize()
    exp2 = [i % 251 if i < 64 else 7 for i in idx.tolist()]
    assert out[:, 0].tolist() == exp2, (out[:, 0].tolist(), exp2)
    print("  graph replay after shrink (evicted rows repointed to host): ok")

    # grow back, refill, replay
    a.ensure_rows(S)
    v = a.view(S, (row_bytes,))
    for i in range(64, S):
        v[i].fill_(i % 251)
    for e in range(64, S):
        tab_cpu[e] = a.row_addr(e)
    tab.copy_(tab_cpu); g.replay(); torch.cuda.synchronize()
    assert out[:, 0].tolist() == exp, out[:, 0].tolist()
    assert v[:64, 0].tolist() == [i % 251 for i in range(64)]          # survivors intact
    print("  grow back + graph replay: ok (survivor rows intact)")

    # OOM path
    big = RowArena(64 << 20, 4096, 0, name="oom-probe")               # 256 GiB of VA
    try:
        big.ensure_rows(4096); print("  (no OOM raised; device had 256 GiB?)")
    except ArenaOOM as ex:
        print(f"  OOM path: {type(ex).__name__} raised as expected")
    big.close(); a.close()
    print(f"  closed: free {(torch.cuda.mem_get_info()[0] - free0) >> 20:+} MiB vs start")


if __name__ == "__main__":
    _selftest()
