#!/bin/bash
# All-position long-text NLL series. Usage: nll_series.sh REFNAME STEP...
#   STEP = "accepted" (phase1 --restart) or a phase1 step name (--bringup). The first "accepted:save"
#   entry saves the reference REFNAME; every other entry checks against it. NLL_LONG_ALL=1 -> all
#   positions >= 1024 are scored (1 GB logit chunks: the expert cache is shrunk to S=184 first).
cd "$(dirname "$0")"; PY=~/quant/venv/bin/python3; REF=$1; shift
export NLL_LONG_ALL=1
for item in "$@"; do
  step=${item%%:*}; mode=${item#*:}; [ "$mode" = "$item" ] && mode=check
  echo "=== $step ($mode) ==="
  if [ "$step" = accepted ]; then $PY -c "import json;s=json.load(open('phase1_state.json'));s['server_matches_accepted']=False;json.dump(s,open('phase1_state.json','w'),indent=2)"; $PY phase1.py --restart > "logs/phase1-nll-$step.log" 2>&1
  else $PY phase1.py --bringup "$step" > "logs/phase1-nll-$step.log" 2>&1; fi
  curl -sf -m 5 http://127.0.0.1:30000/health >/dev/null || { echo "  server not up"; continue; }
  ./keepalive.sh 2400 & KA=$!
  $PY -c "import elastic_sweep as e; e.command('S 184')" >/dev/null 2>&1
  out=$(NLL_LONG_TAG=$step $PY nll_long.py $mode $REF 2>&1 | tail -3); echo "$out"
  kill $KA 2>/dev/null
  echo "- $(date '+%Y-%m-%d %H:%M') nll_long ALL-positions [$step $mode vs $REF]: $(echo "$out" | tr '\n' ' ' | cut -c1-360)" >> CAMPAIGN.md
done
