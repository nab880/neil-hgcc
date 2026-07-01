#!/bin/sh
# WS2-2d: does the optimal DP x PP x TP factorization SHIFT once the fabric
# actually contends? For a fixed GPU budget N, sweep every factorization and find
# the fastest, on a single crossbar (no contention -- the control) and on the
# fat-tree (real contention), for both all-reduce algorithms.
#
# If the fat-tree optimum differs from the crossbar optimum, contention -- not the
# per-algorithm cost in isolation -- is choosing the parallelization, which is the
# headline an analytical model cannot produce.
#
# Requires sst + the installed mercury_3d demo on PATH. Slow; run in background.
# Knobs: N (GPU budget), LAYERS (timing skeleton), MICROBATCH, BW.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
APP="$HERE/parallel3d.py"
N="${N:-16}"
export LAYERS="${LAYERS:-2}"
export MICROBATCH="${MICROBATCH:-8}"
BW="${BW:-50GB/s}"

run() { # TOPO ALG TP PP -> simulated step ms (or ABORT)
  out=$(TOPO="$1" SUMI_ALLREDUCE_ALG="$2" TP_SIZE="$3" PP_SIZE="$4" NRANKS="$N" \
        LINK_BW="$BW" sst "$APP" 2>&1) || true
  if echo "$out" | grep -qiE "abort|couldn't get a flow|fatal"; then echo "ABORT"; return; fi
  echo "$out" | grep -i "simulated time" | grep -o "[0-9.]* ms" | grep -o "[0-9.]*" | head -1
}

# factorizations: TP*PP divides N, DP = N/(TP*PP)
facs=""
tp=1
while [ "$tp" -le "$N" ]; do
  if [ $((N % tp)) -eq 0 ]; then
    pp=1
    while [ $((tp * pp)) -le "$N" ]; do
      if [ $(( (N / tp) % pp )) -eq 0 ]; then facs="$facs $tp:$pp"; fi
      pp=$((pp * 2))
    done
  fi
  tp=$((tp * 2))
done

sweep() { # TOPO ALG  -> prints table, returns best "DPxPPxTP=ms"
  topo="$1"; alg="$2"
  best_ms=""; best_cfg=""
  printf "  %-14s %12s\n" "DPxPPxTP" "step_ms"
  for f in $facs; do
    tp=${f%:*}; pp=${f#*:}; dp=$((N / (tp * pp)))
    ms=$(run "$topo" "$alg" "$tp" "$pp")
    printf "  %-14s %12s\n" "${dp}x${pp}x${tp}" "$ms"
    case "$ms" in ABORT|"") continue;; esac
    if [ -z "$best_ms" ] || awk -v a="$ms" -v b="$best_ms" 'BEGIN{exit !(a<b)}'; then
      best_ms="$ms"; best_cfg="${dp}x${pp}x${tp}"
    fi
  done
  echo "  -> optimum: $best_cfg  ($best_ms ms)"
  echo "$topo $alg OPT $best_cfg $best_ms" >> "$HERE/.sweep_opt.$$"
}

echo "# 3D factorization sweep: N=$N LAYERS=$LAYERS MICROBATCH=$MICROBATCH BW=$BW"
echo ""
rm -f "$HERE/.sweep_opt.$$"
for cfg in "single recdouble" "fattree recdouble" "fattree ring"; do
  set -- $cfg
  echo "=== topology=$1  algorithm=$2 ==="
  sweep "$1" "$2"
  echo ""
done

echo "=== OPTIMUM per configuration (does it shift?) ==="
awk '{printf "  %-9s %-10s -> %-12s %s ms\n", $1, $2, $4, $5}' "$HERE/.sweep_opt.$$"
rm -f "$HERE/.sweep_opt.$$"
