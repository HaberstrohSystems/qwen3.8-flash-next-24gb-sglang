#!/usr/bin/env bash
# Launch the patched SGLang server.
#
# Set these four for your machine.
SGLANG="${SGLANG:-$HOME/quant/sglang}"          # patched SGLang checkout
VENV="${VENV:-$HOME/quant/venv-sglang}"         # its virtualenv
MODEL="${MODEL:?set MODEL to the quantized checkpoint directory}"
PLE="${PLE:-$HOME/quant/ple}"                   # extracted n-gram table

# CUDA runtime shipped inside the venv. Needed because tilelang compiles
# against the headers it finds, and a mismatch with the installed driver
# surfaces as a kernel compile failure much later.
CU="$VENV/lib/python3.12/site-packages/nvidia/cu13"

# Stop any previous instance first.
for p in $(pgrep -f "sglang.launch_server"); do kill "$p" 2>/dev/null; done
sleep 5

cd "$SGLANG" || exit 1

# The cgroup limit is a safety net, not a tuning knob: it makes an OOM hit
# only this job instead of taking the desktop session with it.
exec systemd-run --user --scope --unit=sglang-$(date +%s) -p MemoryMax=27G \
  env PATH="$CU/bin:$VENV/bin:$PATH" CUDA_HOME="$CU" \
      SGLANG_QWEN4_PLE_MMAP="$PLE" \
      SGLANG_MOE_EXPERT_STREAM=1 \
      SGLANG_VLM_CACHE_SIZE_MB=0 \
  "$VENV/bin/python3" -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port 30000 \
    --tp-size 1 --context-length 32768 \
    --cpu-offload-gb 19 \
    --no-ple-offload-embedding \
    --disable-cuda-graph \
    --language-model-only \
    --max-running-requests 1 \
    --max-mamba-cache-size 10 \
    --max-total-tokens 32768 \
    --chunked-prefill-size 512 \
    --mem-fraction-static 0.95 \
    --page-size 1 \
    --disable-overlap-schedule \
    --disable-radix-cache \
    --weight-loader-drop-cache-after-load

# ---------------------------------------------------------------------------
# What each of the less obvious flags is doing here:
#
#   SGLANG_MOE_EXPERT_STREAM=1   enables the MoE-aware streamer. Without it,
#       --cpu-offload-gb copies every expert of every layer to the GPU on every
#       forward pass: 26 GB per token instead of 0.31 GB.
#
#   SGLANG_QWEN4_PLE_MMAP        path to the n-gram table written by
#       03_split_ple.py. SGLang's built-in offload mode puts all 51 GB in
#       page-locked host memory, which is impossible below ~64 GB of RAM.
#
#   --cpu-offload-gb 19          WITH THE PATCH APPLIED this offloads expert
#       weights only, so it is a budget for experts alone. Raising it moves
#       more experts to the host at no per-token cost, because only the
#       selected ones are ever fetched.
#
#   --no-ple-offload-embedding   turns off the built-in PLE offload, which the
#       mmap path replaces.
#
#   --disable-cuda-graph         the mmap PLE is incompatible with graph
#       capture. This costs throughput and is the first thing to revisit.
#
#   --language-model-only        the vision tower is present in the checkpoint
#       but was never exercised. Drop this flag at your own risk.
#
#   --max-mamba-cache-size 10    36 of 48 layers are Gated DeltaNet with a
#       constant per-sequence state. Ten slots is enough for one request and
#       leaves the VRAM for weights.
#
#   --mem-fraction-static 0.95   only safe with no desktop session on the GPU.
#       With a desktop attached, use 0.90 and expect ~1.3 GiB less.
