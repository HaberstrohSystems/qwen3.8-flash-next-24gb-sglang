# Serving assets

| File | Used by | Notes |
|---|---|---|
| `moe_configs/configs/triton_3_7_1/E={10,512},N=160,device_name=NVIDIA_RTX_PRO_4000_Blackwell,dtype=int2_w2a16{,_down}.json` | `SGLANG_MOE_CONFIG_DIR` (base patch config lookup) | Tuned Triton configs for the 2-bit MoE kernel: E = 10 covers decode (the routed experts of one token), E = 512 prefill. Step S0b: 13.4 -> 15.2 tok/s (CAMPAIGN.md:252). Device- and Triton-version-specific; re-tune for another GPU. |
| `expert_freq.pt` | `SGLANG_MOE_PLACEMENT` (`patches/placement.py`, `patches/elastic.py`) | `{"mass": FloatTensor[48, 512], "count": LongTensor[48, 512]}`, routing mass per (layer, expert) from a 2,496-token routing probe over three domains (CAMPAIGN.md:206-216). Built by `tools/expert_freq.py` from a `SGLANG_ROUTE_DUMP` directory. Workload-dependent: a different workload mix may want its own histogram. |
| `phase1_state.json` | `scripts/phase1.py`; documentation | Kept as the record of the accepted state (only the home directory is abbreviated to `~`) (S21 flag set plus the parsers, single request): flags to drop from `sweep.sh`'s base set, flags to add (`--max-mamba-cache-size 1`, `--max-running-requests 1`), the accepted patch list, the environment string, the accepted bench numbers. Its `add` list and `env` string still contain the measuring host's paths (model directory, `assets/` copy, control file); `scripts/serve.sh` is the same configuration with the paths as variables. `accepted.decode_all` and `accepted.prefill_10k` are the headline numbers. |

The control file for the elastic cache (`SGLANG_MOE_ELASTIC_CTL`) is not an asset: it is any writable
file containing `S 184`; the server writes `<file>.status` next to it.
