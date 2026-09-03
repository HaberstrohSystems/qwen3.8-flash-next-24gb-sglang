#!/bin/bash
# Keep the tokenizer/detokenizer/scheduler processes warm (their idle pages get swapped out on the reference host);
# one tiny request every 5 s. Usage: keepalive.sh SECONDS
for i in $(seq 1 $(( ${1:-3600} / 5 ))); do
  curl -s -m 900 http://127.0.0.1:30000/generate -H 'Content-Type: application/json' -d '{"text":"Ping.","sampling_params":{"max_new_tokens":1,"temperature":0}}' >/dev/null 2>&1
  sleep 5
done
