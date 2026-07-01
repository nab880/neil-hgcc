#!/bin/sh
# The memory wall: which DP/TP/PP/ZeRO configs FIT a model on a GPU budget, and
# how ZeRO trades memory for the communication of study (A)/(E). Pure Python
# (capacity.py) -- no simulator needed, so it reaches 175B which is too big to
# simulate cheaply. Knobs: MODEL, N (GPU budget), DEV (capacity GB).
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
CAP="python3 $HERE/capacity.py"
MODEL="${MODEL:-70b}"
N="${N:-16}"
DEV="${DEV:-80}"

echo "# memory wall: $MODEL on N=$N x A100-${DEV}GB"
echo
echo "## feasibility frontier (ZeRO-1): which factorizations fit"
$CAP --model "$MODEL" --gpus "$N" --zero 1 --gpu-mem "$DEV"
echo
echo "## ZeRO trades memory for comms (pure DP=$N, the throughput-optimal layout)"
printf "  %-8s %10s\n" "stage" "per-rank"
for z in 0 1 2 3; do
  t=$($CAP --model "$MODEL" --dp "$N" --tp 1 --pp 1 --zero "$z" --gpu-mem "$DEV" \
        | sed -n 's/.*TOTAL *\([0-9.]*\) GB.*/\1/p')
  v=$($CAP --model "$MODEL" --dp "$N" --tp 1 --pp 1 --zero "$z" --gpu-mem "$DEV" \
        | grep -o 'FITS\|OOM' | tail -1)
  printf "  ZeRO-%-3s %8s GB  %s\n" "$z" "$t" "$v"
done
echo
echo "# done"
