# The serving patch

The serving patch against SGLang `73a255206f`, produced from the served tree on 2026-09-03 and
verified in a clean worktree (`PATCH_NOTES.md` section 2). The weights it serves are on the Hub:
https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang

| File | What it is |
|---|---|
| `qwen4exp-serving-73a255206f.patch` | The complete difference between SGLang commit `73a255206f916366c8d26d4022f82ddfb0ab558d` ("Introduce Qwen 3.8 Flash Next") and the tree that served the accepted state. 34 files, +4,155 / -89 lines, 5,439 lines, 251,796 bytes, SHA-256 `92f669b2525f9c86190825390fafc2b41a28071c398fcb7fd95716fbce744bb5`. Plain `git diff` output: `git apply` or `patch -p1`. Verified: base commit + patch reproduces all 34 patched files of the served tree byte for byte (`PATCH_NOTES.md` section 2). |
| `PATCH_NOTES.md` | Per-file map of the patch grouped by feature, the hunks an upstreamer should drop (measurement-only `kv_stats` / `kv_fakeq`, opt-in `ngram_ple`, debug switches), the server flags and environment, the validation evidence per feature, known issues. |
| `UPSTREAM.md` | The upstream contribution plan for the SGLang maintainers: what is general and what is Qwen4-Exp-specific, and the split into reviewable commits (correctness fixes and breakable graphs under CPU offload; 2-bit `moe_wna16` with expert streaming and the in-place GEMV; VMM-backed elasticity; quantized KV pools; NGRAM speculation on PLE models). |

Apply:

```
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout 73a255206f916366c8d26d4022f82ddfb0ab558d
git apply --check /path/to/qwen4exp-serving-73a255206f.patch && git apply /path/to/qwen4exp-serving-73a255206f.patch
```

Relationship to [`../patches/`](../patches/): the scripts there are the exact-string edit layers that
were applied one after another to the served tree; the serving patch is their flattened result plus the
base 2-bit patch. Layer order (only needed to peel layers off with the scripts): base < host_fixes items <
ncontig_gemv < placement < elastic < kv_lazy < ple_random < kv_fp8 < kv_int8 < kv_int4 < kv_tiers.

The published launch line for the patched tree is [`../scripts/serve.sh`](../scripts/serve.sh)
(`--max-running-requests 1 --max-mamba-cache-size 1`); `PATCH_NOTES.md` section 6 lists its flags
and environment. Every headline log in `../docs/logs/` was recorded with that flag set, except
`night4.log` and `elastic.ctl.status`, which were recorded on the concurrent-benchmark restart #23
(`--max-running-requests 4 --max-mamba-cache-size 8`, not the published configuration).
