#!/bin/bash
# DeepSWE 1.1 driver: one `pier run` per task so images can be deleted after each task (disk is tight).
# Usage: deepswe_run.sh local|openrouter N [PARALLEL]     tasks = first N of $TASKLIST (default deepswe/tasklist_seed0.txt)
# Env: MODEL_CLASS (local default litellm = tool calling, alt litellm_textbased), OR_MODEL (default qwen/qwen3.8-flash)
set -u
SIDE=$1; N=$2; PAR=${3:-1}
H=$(cd "$(dirname "$0")" && pwd); TASKS=${TASKS:-$HOME/quant/harness/deep-swe/tasks}
TASKLIST=${TASKLIST:-$H/deepswe/tasklist_seed0.txt}   # random.Random(0) shuffle of the DeepSWE 1.1 task names (CAMPAIGN.md:427)
JOBS=$H/bench_final/deepswe/$SIDE; mkdir -p "$JOBS/logs"
SAMPLING='{"temperature":1.0,"top_p":0.95,"drop_params":true}'
if [ "$SIDE" = local ]; then
  MODEL=openai/qwen38-flash-next-2bit
  AGENT=(--ae MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=30 --ak "model_class=\"${MODEL_CLASS:-litellm}\"" --ae OPENAI_BASE_URL=http://172.17.0.1:30001/v1 --ae OPENAI_API_KEY=EMPTY
         --ak "model_kwargs=${SAMPLING%\}},\"extra_body\":{\"top_k\":20}}")
  export OPENAI_API_KEY=EMPTY OPENAI_BASE_URL=http://172.17.0.1:30001/v1
else
  MODEL=openrouter/${OR_MODEL:-qwen/qwen3.8-flash}
  AGENT=(--ae MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=30 --ak "model_class=\"${MODEL_CLASS:-openrouter}\"" --ak "model_kwargs=${SAMPLING%\}},\"top_k\":20}")
  : "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY in the environment}"
fi
run_one() {
  t=$1
  if ls "$JOBS/$t"/*/verifier/reward.json >/dev/null 2>&1 || ls "$JOBS/$t"/*/*/verifier/reward.json >/dev/null 2>&1; then echo "skip $t (done)"; return; fi
  rm -rf "$JOBS/$t"
  img=$(grep -o 'docker_image = "[^"]*"' "$TASKS/$t/task.toml" | cut -d'"' -f2)
  echo "[$(date +%H:%M)] start $t ($img)"
  pier run -p "$TASKS/$t" --agent mini-swe-agent --model "$MODEL" -o "$JOBS" --job-name "$t" -n 1 -y --delete --override-memory-mb 4096 "${AGENT[@]}" > "$JOBS/logs/$t.log" 2>&1
  rc=$?
  rw=$(find "$JOBS/$t" -name reward.json | head -1); r=$( [ -n "$rw" ] && python3 -c "import json,sys; print(json.load(open('$rw')))" 2>/dev/null | cut -c1-120)
  echo "[$(date +%H:%M)] end $t rc=$rc reward=${r:-none}"
  # drop this task's images: pier's built images are named <task[:32]>__<id>-*, the base image carries the ext_id
  ext=$(basename "${img#*:}" -v1.1)
  docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep -E "^${t:0:32}__|$ext" | awk '{print $2}' | sort -u | xargs -r docker rmi -f >/dev/null 2>&1
  docker image prune -f >/dev/null 2>&1
}
export -f run_one; export JOBS TASKS MODEL; export AGENT_STR="${AGENT[*]}"
head -n "$N" "$TASKLIST" | while read -r t; do
  run_one "$t" &
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 20; done
done; wait
echo "##### deepswe $SIDE done"
