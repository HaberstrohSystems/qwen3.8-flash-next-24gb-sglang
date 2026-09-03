#!/usr/bin/env bash
# Launches the published serving configuration: the accepted flag set and environment of the
# campaign (assets/phase1_state.json; CAMPAIGN.md:413, :454-459) with single-request concurrency.
# Includes the systemd-run scope, the health wait, one warm-up request and POST /freeze_gc.
#
# Set these for your machine.
SGLANG="${SGLANG:-$HOME/quant/sglang}"                        # patched SGLang checkout (73a255206f + patch)
VENV="${VENV:-$HOME/quant/venv-sglang}"                       # its virtualenv (torch 2.13.0+cu130, triton 3.7.1)
MODEL="${MODEL:?set MODEL to the checkpoint directory (the Hub download)}"
PLE="${PLE:-$MODEL/ple}"                                      # directory with ple.f8_e4m3.bin + ple.json
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="${ASSETS:-$REPO/assets}"                              # moe_configs/, expert_freq.pt
CTL="${CTL:-$REPO/elastic.ctl}"                               # writable elastic control file (git-ignored)
SRVLOG="${SRVLOG:-$HOME/quant/logs/server.log}"

CU="$VENV/lib/python3.12/site-packages/nvidia/cu13"           # cu13 toolkit inside the venv (sm_120 JIT)
[ -f "$CTL" ] || echo "S 184" > "$CTL"
mkdir -p "$(dirname "$SRVLOG")"

cd "$SGLANG" || exit 1

# MemoryMax=30G: 27G caused cgroup reclaim storms at ~55k-token prefills (CAMPAIGN.md:345).
# Do not exempt this scope from systemd-oomd (docs/ELASTIC_MEMORY.md, host-RAM section; CAMPAIGN.md:460-464).
setsid systemd-run --user --scope --unit="sglang-$(date +%s)" -p MemoryMax=30G \
  env PATH="$CU/bin:$VENV/bin:$PATH" CUDA_HOME="$CU" \
      SGLANG_QWEN4_PLE_MMAP="$PLE" \
      SGLANG_VLM_CACHE_SIZE_MB=0 \
      SGLANG_MOE_EXPERT_STREAM=1 \
      SGLANG_MOE_CONFIG_DIR="$ASSETS/moe_configs" \
      SGLANG_MOE_PLACEMENT="$ASSETS/expert_freq.pt" \
      SGLANG_MOE_PLACEMENT_S=184 \
      SGLANG_MOE_ELASTIC=1 \
      SGLANG_MOE_ELASTIC_PIN_MB=512 \
      SGLANG_MOE_ELASTIC_CTL="$CTL" \
      SGLANG_KV_LAZY=1 \
      SGLANG_MOE_ELASTIC_FILL_MB=2048 \
      SGLANG_MOE_ELASTIC_RESERVE_ROWS=0 \
      SGLANG_KV_LAZY_TOKENS=262144 \
      SGLANG_KV_LAZY_SAFETY=0.77 \
      SGLANG_KV_TIERS_W=8192 \
  "$VENV/bin/python3" -m sglang.launch_server \
    --host 127.0.0.1 --port 30000 --tp-size 1 --cpu-offload-gb 19 --no-ple-offload-embedding \
    --mem-fraction-static 0.95 --language-model-only --page-size 1 --disable-overlap-schedule \
    --disable-radix-cache --weight-loader-drop-cache-after-load \
    --chunked-prefill-size 1024 --max-prefill-tokens 32768 --cuda-graph-backend-decode breakable \
    --model-path "$MODEL" --max-total-tokens 262144 --context-length 262144 \
    --kv-cache-dtype int8ring_int4 --attention-backend triton --max-mamba-cache-size 1 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --max-running-requests 1 \
  > "$SRVLOG" 2>&1 < /dev/null &

T0=$(date +%s)
until curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1; do
  sleep 10
  if [ $(( $(date +%s) - T0 )) -gt 900 ]; then echo "TIMEOUT (see $SRVLOG)"; exit 1; fi
done
echo "up after $(( $(date +%s) - T0 )) s"
sleep 5
# Warm once, then freeze the Python heap of scheduler + detokenizer (gc.freeze; CAMPAIGN.md:343).
curl -s -m 120 http://127.0.0.1:30000/generate -H 'Content-Type: application/json' \
  -d '{"text":"Warmup.","sampling_params":{"max_new_tokens":4,"temperature":0}}' >/dev/null
curl -s -m 30 -X POST http://127.0.0.1:30000/freeze_gc >/dev/null && echo "gc frozen"

# What the non-obvious flags and variables do (sources in parentheses):
#   --cpu-offload-gb 19            expert weights only (base patch); budget for experts alone.
#   --no-ple-offload-embedding     the mmap PLE mode replaces SGLang's pinned-host PLE offload.
#   --cuda-graph-backend-decode breakable   decode under breakable CUDA graphs, PLE lookup as the eager
#                                  break (host_fixes items bcg/bcg2; 21.8 -> 40.0 tok/s, CAMPAIGN.md:291).
#   --chunked-prefill-size 1024 --max-prefill-tokens 32768   S3 (CAMPAIGN.md:258).
#   --max-total-tokens/--context-length 262144   address space only; backed on demand (kv_lazy).
#   --kv-cache-dtype int8ring_int4   tiered KV, INT8 ring of SGLANG_KV_TIERS_W slots over INT4 (S21).
#   --attention-backend triton     drops the FlashInfer workspace (+0.42 GB, CAMPAIGN.md:353).
#   --max-mamba-cache-size 1 --max-running-requests 1   one request at a time; on a 32 GB host this
#                                  keeps memory pressure low (CAMPAIGN.md:454). Higher concurrency
#                                  needs more host RAM (each concurrent request multiplies the host
#                                  expert-row traffic).
#   --reasoning-parser qwen3 --tool-call-parser qwen3_coder   thinking out of content; the chat
#                                  template's <function=..><parameter=..> tool-call format (CAMPAIGN.md:430-432).
#   SGLANG_MOE_PLACEMENT(_S)       frequency placement, S = 184 hottest experts per layer (S10).
#   SGLANG_MOE_ELASTIC*            elastic residency: 512 MB pinned fallback budget, 2048 MB startup fill
#                                  reserve, no pinned reserve rows (CAMPAIGN.md:347), control file.
#   SGLANG_KV_LAZY*                lazy VMM backing; virtual capacity 262144; admission cap 0.77 x profiled
#                                  (defaults not set here: FLOOR 4096, MARGIN 2048, HEADROOM_MB 1536).
