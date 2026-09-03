#!/usr/bin/env bash
# Restart the server with one flag variant, wait for health, benchmark, record.
#
#   scripts/sweep.sh <name> [extra sglang flags...]
#
# Base flags are the accepted state; a variant ADDS flags. To REMOVE a base flag,
# set DROP="--flag-a --flag-b" in the environment (exact tokens, no values).
# Results append to sweep.log at the repository root. The sealed model is never touched.
#
# Campaign harness as run. It expects the layout it ran in: this script beside phase1.py, patches/,
# tools/logprob_diff.py and phase1_state.json in one directory; here H is the repository root, so
# tools/bench_speed.py resolves, and the ~/quant/... defaults (model, venv, checkout) are those of the
# measuring host. Override them before use; scripts/serve.sh is the published launch line.
set -u
NAME="$1"; shift
H="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$H/sweep.log"
SRVLOG="$HOME/quant/logs/server-$NAME.log"
VENV="$HOME/quant/venv-sglang"
CU="$VENV/lib/python3.12/site-packages/nvidia/cu13"
Q="${Q:-$HOME/quant/model}"                                   # the 2-bit checkpoint directory
DROP="${DROP:-}"

BASE=(--model-path "$Q" --host 127.0.0.1 --port 30000
      --tp-size 1 --context-length 32768 --cpu-offload-gb 19
      --no-ple-offload-embedding --disable-cuda-graph
      --max-running-requests 1 --max-mamba-cache-size 10
      --max-total-tokens 32768 --chunked-prefill-size 512
      --mem-fraction-static 0.95 --language-model-only
      --page-size 1 --disable-overlap-schedule --disable-radix-cache
      --weight-loader-drop-cache-after-load)

# apply DROP: remove listed flags (and a following value if the next token is not a flag)
ARGS=()
i=0
while [ $i -lt ${#BASE[@]} ]; do
  tok="${BASE[$i]}"
  if [[ " $DROP " == *" $tok "* ]]; then
    i=$((i+1))
    if [ $i -lt ${#BASE[@]} ] && [[ "${BASE[$i]}" != --* ]]; then i=$((i+1)); fi
    continue
  fi
  ARGS+=("$tok"); i=$((i+1))
done
ARGS+=("$@")

# stop the running server by PID. Bracketed patterns cannot match this script's
# own command line (the literal text "schedule[r]" does not match the regex).
for p in $(pgrep -f "sglang::schedule[r]"; pgrep -f "sglang.launch_serve[r]"); do
  kill "$p" 2>/dev/null
done
sleep 6
nvidia-smi --query-gpu=memory.used --format=csv,noheader | xargs echo "  VRAM after stop:"

echo "=== $NAME  $(date -Is) ===" | tee -a "$LOG"
echo "  flags: ${ARGS[*]}" >> "$LOG"
[ -n "$DROP" ] && echo "  dropped: $DROP" >> "$LOG"

cd "$HOME/quant/sglang" || exit 1
setsid systemd-run --user --scope --unit="sglang-$NAME-$(date +%s)" -p MemoryMax=30G \
  env PATH="$CU/bin:$VENV/bin:$PATH" CUDA_HOME="$CU" \
      SGLANG_QWEN4_PLE_MMAP="$HOME/quant/ple" SGLANG_VLM_CACHE_SIZE_MB=0 \
      SGLANG_MOE_EXPERT_STREAM=1 ${EXTRA_ENV:-} \
  "$VENV/bin/python3" -m sglang.launch_server "${ARGS[@]}" \
  > "$SRVLOG" 2>&1 < /dev/null &

T0=$(date +%s)
until curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1; do
  sleep 10
  if ! pgrep -f "sglang::schedule[r]" >/dev/null && [ $(( $(date +%s) - T0 )) -gt 60 ]; then
    echo "  FAILED to start" | tee -a "$LOG"
    grep -aiE "Error|Traceback" "$SRVLOG" | tail -4 | tee -a "$LOG"
    exit 1
  fi
  if [ $(( $(date +%s) - T0 )) -gt 900 ]; then echo "  TIMEOUT" | tee -a "$LOG"; exit 1; fi
done
echo "  up after $(( $(date +%s) - T0 )) s" | tee -a "$LOG"
sleep 5
# warm once, then freeze the Python heap of scheduler + detokenizer (gc.freeze): a gen-2 collection
# over a partly swapped heap froze the box during 55k-token prefills (CAMPAIGN 2026-09-02 14:30).
curl -s -m 120 http://127.0.0.1:30000/generate -H 'Content-Type: application/json' \
  -d '{"text":"Warmup.","sampling_params":{"max_new_tokens":4,"temperature":0}}' >/dev/null
curl -s -m 30 -X POST http://127.0.0.1:30000/freeze_gc >/dev/null && echo "  gc frozen" | tee -a "$LOG"
"$HOME/quant/venv/bin/python3" "$H/tools/bench_speed.py" 2>&1 | tail -7 | tee -a "$LOG"
echo >> "$LOG"
