#!/bin/bash
# Validation protocol for a KV mode (plan section 7). Usage: int8_validate.sh STEP_NAME [longctx_tokens]
# bring-up (accepted set + step flags/patches), keepalive, cache to S=184 for the 1 GB logit chunks,
# short NLL, long-text NLL vs bf16, streaming bench, long-context test. Server is left running.
BENCH=${BENCH:-$(cd "$(dirname "$0")" && pwd)/bench_speed.py}   # streaming bench: the one next to this script unless BENCH is set
cd "$(dirname "$0")"; PY=~/quant/venv/bin/python3; STEP=$1; LC=${2:-100000}
echo "=== $STEP bring-up ==="; $PY phase1.py --bringup "$STEP" > "logs/phase1-$STEP-validate.log" 2>&1
curl -sf -m 5 http://127.0.0.1:30000/health >/dev/null || { echo "  server not up"; grep -a "Error" ~/quant/logs/server-$STEP.log | grep -av Pyspy | tail -3 | cut -c1-200; exit 1; }
grep -a "KV lazy backing: [0-9]* tokens\|token capacity" ~/quant/logs/server-$STEP.log | tail -2 | cut -c1-160
./keepalive.sh 3000 & KA=$!
echo "=== short NLL (single chunk, write path) ==="; $PY nll_eval.py check int8dense 2>&1 | tail -2
$PY -c "import elastic_sweep as e; e.command('S 184')" >/dev/null 2>&1
echo "=== long-text NLL vs bf16 (cache read paths) ==="; $PY nll_long.py check bf16 2>&1 | tail -2
echo "=== logprob oracle lp2 (10k) ==="; $PY logprob_diff.py check lp2 2>&1 | grep LOGPROB
kill $KA 2>/dev/null
$PY -c "import elastic_sweep as e; e.command('fill 2048')" >/dev/null 2>&1
echo "=== streaming bench ==="; $PY "$BENCH" 200 2>&1 | tail -5
echo "=== longctx $LC ==="; ./keepalive.sh 1200 & KA=$!; SGLANG_MOE_ELASTIC_CTL=$PWD/elastic.ctl timeout 1500 $PY longctx_test.py $LC 2>&1 | grep -v "^  during" | tail -4 | cut -c1-200; kill $KA 2>/dev/null
curl -sf -m 5 http://127.0.0.1:30000/health >/dev/null && echo "server alive" || echo "server DEAD"
