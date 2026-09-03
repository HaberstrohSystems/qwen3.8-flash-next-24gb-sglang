# The SGLang changes

Two forms of the same change to SGLang: the flat serving patch, which is the verbatim diff of
the tree that served the published numbers, and the five-commit series under `upstream/`, which
is the reviewable form for `sgl-project/sglang`. The weights they serve are on the Hub:
https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang

| File | What it is |
|---|---|
| `qwen4exp-serving-73a255206f.patch` | The complete difference between SGLang commit `73a255206f916366c8d26d4022f82ddfb0ab558d` ("Introduce Qwen 3.8 Flash Next", the first commit of the branch `qwen4-main-squashed` of PR #36497) and the tree that served the accepted state. 34 files, +4,155 / -89 lines, 5,439 lines, 251,796 bytes, SHA-256 `92f669b2525f9c86190825390fafc2b41a28071c398fcb7fd95716fbce744bb5`. Plain `git diff` output: `git apply` or `patch -p1`. Base commit + patch reproduces all 34 patched files of the served tree byte for byte (`PATCH_NOTES.md` section 2). Use this to reproduce the measurements. |
| `PATCH_NOTES.md` | Per-file map of the flat patch grouped by feature, the measurement-only and debug hunks that the series drops (`kv_stats` / `kv_fakeq`, `SGLANG_NAN_TRACE`, NGRAM debug switches), the server flags and environment, the validation evidence per feature, known issues. |
| `UPSTREAM.md` | Status of the upstream contribution: review target and why it is not `main`, the five parts and what each contains, what remains, related PRs, how the series was verified. |
| `upstream/series-q4head/0001..0005` | The series rebased onto the head of `qwen4-main-squashed` (`78c5024e9d`): 39 files, +8,042 / -94. `git am` format. This is what the PRs contain. |
| `upstream/series-base/0001..0005` | The same five commits on the served base `73a255206f`: 40 files, +8,082 / -175. Reference for the rebase. |
| `upstream/RFC.md` | The issue text that introduces the series to the maintainers (feature-request template). |
| `upstream/PR-1.md` .. `PR-5.md` | The PR descriptions (PR template: Motivation, Modifications, Accuracy Tests, Speed Tests and Profiling, Checklist, suggested reviewers, reproduction commands). |

Apply the flat patch:

```
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout 73a255206f916366c8d26d4022f82ddfb0ab558d
git apply --check /path/to/qwen4exp-serving-73a255206f.patch && git apply /path/to/qwen4exp-serving-73a255206f.patch
```

Apply the series instead (review form; drops the measurement hooks, adds the registered tests):

```
git fetch origin qwen4-main-squashed && git checkout 78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385
git am /path/to/upstream/series-q4head/000*.patch
```

Relationship to [`../patches/`](../patches/): the scripts there are the exact-string edit layers
that were applied one after another to the served tree; the flat patch is their flattened result
plus the base 2-bit patch. Layer order (only needed to peel layers off with the scripts): base <
host_fixes items < ncontig_gemv < placement < elastic < kv_lazy < ple_random < kv_fp8 < kv_int8 <
kv_int4 < kv_tiers.

The published launch line for the patched tree is [`../scripts/serve.sh`](../scripts/serve.sh)
(`--max-running-requests 1 --max-mamba-cache-size 1`); `PATCH_NOTES.md` section 6 lists its flags
and environment. Every headline log in `../docs/logs/` was recorded with that flag set, except
`night4.log` and `elastic.ctl.status`, which were recorded on the concurrent-benchmark restart #23
(`--max-running-requests 4 --max-mamba-cache-size 8`, not the published configuration).
