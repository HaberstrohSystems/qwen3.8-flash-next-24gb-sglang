#!/usr/bin/env bash
# Full chain: prepare -> quantize (with subset tuning) -> audit -> seal.
#
# RULES, each of which exists because it was broken once:
#   * The output carries a timestamp and is NEVER overwritten.
#   * The output is NEVER deleted, not even on a rerun.
#   * Only intermediates are deleted, and only once their successor is verified.
#
# Set these three for your machine.
S="$(cd "$(dirname "${BASH_SOURCE[0]}")/scripts" && pwd)"
PY="${PY:-$HOME/quant/venv/bin/python}"
SRC="${SRC:-$HOME/quant/src-flashnext}"      # the downloaded FP8 checkpoint
BF16="${BF16:-$HOME/quant/src-bf16}"         # stacked bf16 intermediate (~240 GB)
PLE="${PLE:-$HOME/quant/ple}"                # extracted n-gram table (~48 GB)
OUTDIR="${OUTDIR:-$HOME/quant/out}"

L="$HOME/quant/logs"; MAIN="$L/pipeline.log"
STAMP=$(date +%Y%m%d-%H%M)
OUT="$OUTDIR/qwen38-fn-$STAMP"

# Do NOT name this variable "GROUPS". That is a bash builtin (the list of the
# user's group IDs); assignments to it are ignored and $GROUPS returns the
# primary group ID instead - here 1000. A run started that way tunes 1000
# rounds per block rather than 4, and nothing warns you.
#   bash -c 'GROUPS=4; echo $GROUPS'   ->   1000
# 512 experts / 4 rounds = 128 experts per round.
SUBSET_N=4

mkdir -p "$L" "$OUTDIR"
step(){ echo "" >>"$MAIN"; echo "════════ $1  $(date -Is) ════════" >>"$MAIN"; }
die(){ echo "!!!! ABORT: $1 (rc=$2) $(date -Is)" >>"$MAIN"; exit "$2"; }
free_gb(){ df --output=avail -BG / | tail -1 | tr -dc '0-9'; }

echo "===== pipeline · output: $OUT =====" >> "$MAIN"

step "1/4 prepare source (stack + bf16)"
if [ ! -f "$BF16/model.safetensors.index.json" ]; then
  [ "$(free_gb)" -lt 150 ] && die "not enough disk for preparation ($(free_gb) GB)" 1
  $PY -u "$S/02_prepare_source.py" "$SRC" -o "$BF16" >> "$L/prepare.log" 2>&1 \
    || die "prepare" $?
else
  echo "  already present" >> "$MAIN"
fi
du -sh "$BF16" >> "$MAIN" 2>&1; echo "  free: $(free_gb) GB" >> "$MAIN"

step "2/4 quantize (subset tuning, $SUBSET_N rounds per block)"
[ "$(free_gb)" -lt 60 ] && die "not enough disk for the output ($(free_gb) GB)" 1
#
# --alg-ext: SignRoundV2. AutoRound recommends it explicitly for bits<=2 and
# lists W2A16 as validated - the experts here are 96% of the quantized
# parameters and are exactly W2A16. Measured on a miniature model: 398 s vs
# 392 s (inside the noise), peak RAM unchanged, 816 of 1423 tensors change
# (so it is not a placebo), structure identical.
#
# --nsamples 48 instead of 128. This is a memory decision, not a quality one.
# At 128 the anonymous working set was 35 GiB on a 30 GiB machine, of which
# ~18.5 GiB was calibration cache (128 x 2048 x 10240 x 2 B = 5.0 GiB each,
# and AutoRound holds three at a block boundary). The kernel then evicted the
# HOT caches to keep the page cache of the mmap-ed 240 GB source: 133M major
# faults, GPU at 0%, one block took 313 minutes instead of 84. Extrapolated:
# 10 days instead of 2.8. With 48 samples the cache is ~6.9 GiB and the
# anonymous set ~23.4 GiB, which fits. Sequence length stays at 2048 - what
# is cut is the number of samples, not the context each one sees.
#
# Do NOT try to fix this with memory.swap.high. A value below what is already
# swapped out invites systemd-oomd to kill the run, and oomd acts on pressure,
# not on absolute limits.
$PY -u "$S/05_quantize.py" --model "$BF16" --ple "$PLE" -o "$OUT" \
    --policy quality --ram-reserve 10 --subset-groups "$SUBSET_N" --alg-ext \
    --nsamples 48 \
    >> "$L/quant-$STAMP.log" 2>&1 || die "quantize" $?

# AutoRound may create a subdirectory
REAL=$(find "$OUT" -name "model.safetensors.index.json" -o -name "model.safetensors" \
       | head -1 | xargs -r dirname)
REAL="${REAL:-$OUT}"
echo "  output: $REAL" >> "$MAIN"

step "3/4 size audit"
$PY -u "$S/08_verify_quant.py" "$REAL" >> "$MAIN" 2>&1 \
  || echo "  (audit reports a deviation, see above)" >> "$MAIN"

step "4/4 seal"
$PY -u "$S/09_seal_output.py" "$REAL" --seal  >> "$MAIN" 2>&1 || die "seal" $?
$PY -u "$S/09_seal_output.py" "$REAL" --check >> "$MAIN" 2>&1 || die "checksums" $?

echo "" >> "$MAIN"
echo "════════ DONE $(date -Is) ════════" >> "$MAIN"
echo "  sealed quantization in: $REAL" >> "$MAIN"
echo "  free: $(free_gb) GB" >> "$MAIN"
