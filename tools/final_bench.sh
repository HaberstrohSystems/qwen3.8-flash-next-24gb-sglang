#!/bin/bash
# GPQA Diamond through sglang.test.run_eval (chat API, model-card thinking sampling: temperature 1.0,
# top_p 0.95, top_k 20): the local server, or an OpenRouter model as the reference (the campaign used
# qwen/qwen3.8-flash, CAMPAIGN.md:437). No score of the 2-bit model is published (docs/HISTORY.md).
# Usage: final_bench.sh local|openrouter [num_examples] [openrouter model]   (key: OPENROUTER_API_KEY in the environment)
cd "$(dirname "$0")"; PY=~/quant/venv-sglang/bin/python3; MODE=$1; N=${2:-}
OUT=bench_final; mkdir -p $OUT
common=(--eval-name gpqa --max-tokens 16384 --temperature 1.0 --top-p 0.95 --top-k 20 --thinking-mode qwen-3)
[ -n "$N" ] && common+=(--num-examples "$N")
if [ "$MODE" = local ]; then
  $PY -m sglang.test.run_eval --port 30000 "${common[@]}" --num-threads ${THREADS:-8} 2>&1 | tee $OUT/gpqa_local.log
else
  [ -z "$OPENROUTER_API_KEY" ] && { echo "OPENROUTER_API_KEY missing"; exit 1; }
  MODEL=${3:-qwen/qwen3.8-27b}; TAG=$(echo "$MODEL" | tr '/' '_')
  OPENAI_API_KEY=$OPENROUTER_API_KEY $PY -m sglang.test.run_eval --base-url https://openrouter.ai/api --model "$MODEL" "${common[@]}" --num-threads 16 2>&1 | tee "$OUT/gpqa_${TAG}_openrouter.log"
fi
