# Scripts

| File | Role |
|---|---|
| `serve.sh` | Launches the published configuration (accepted flag set and environment of `assets/phase1_state.json`, single request): the `systemd-run` scope, the health wait, the warm-up request and `POST /freeze_gc`. Set `MODEL`, `PLE`, `SGLANG`, `VENV`. The comment block at the end explains every flag and variable. |
| `serve-v1-32k.sh` | The launch script of the pre-campaign 32k state (`docs/WRITEUP.md` section 7): 32k context, no CUDA graphs, one request, `MemoryMax=27G`. Historical; its flag comments are still the best explanation of the base flags. |
| `sweep.sh` | The flag-sweep harness the campaign restarted the server with: base flags, `DROP=` removal, `EXTRA_ENV`, `MemoryMax=30G`, health wait, warm-up, `freeze_gc`, bench. Paths are `$HOME/quant/...` defaults (`Q` = the checkpoint directory); override before use. |
| `phase1.py` | The campaign driver: steps with drop/add flags, patches and env; restart through `sweep.sh`; oracle check; keep-or-revert; `--restart` / `--bringup STEP` / `--status`. State in `phase1_state.json`. |
| `pipeline.sh`, `01_inspect_model.py` .. `09_seal_output.py`, `budget.py` | The quantization pipeline (`docs/WRITEUP.md`). |
| `requant_int8.py` | Step 10: the INT8 g128 RTN re-pack of the 85 dense tensors (`linear_attn.out_proj` x36, `self_attn.{q,k,v,o}_proj` x12 each, `lm_head`) from the sealed 2-bit quant into a **new** directory — the served checkpoint (S11b, CAMPAIGN.md:298-300). `--selftest` for the pack/unpack round trip, `--dry-run` to list. `SEALED` (the sealed AutoRound output) and `SRC` (the bf16 source) default to `~/quant/...`; override through the environment variables of the same names. |

`phase1.py` and `sweep.sh` are the campaign harness as run; they expect the working layout in which
`patches/`, `tools/logprob_diff.py` and `phase1_state.json` sat beside them (see the note in
`phase1.py`'s docstring). `serve.sh` is the launch line to use.
