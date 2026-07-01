#!/bin/sh
# WS3 validation: compare the packet-level simulator against the analytical
# (no-contention) closed-form baseline (analytic_baseline.py), for the DP
# transformer step, across both all-reduce algorithms.
#
# The story the table tells:
#   * compute floor: analytic == sim (the model is the same; sanity).
#   * single crossbar (minimal contention): sim ~= analytic on comms too.
#   * fat-tree at scale: sim > analytic -- the gap is network contention, which
#     the analytical model structurally cannot capture and the simulator is for.
#
# Requires sst + the installed mercury_llm_train demo on PATH. Slow (many sims);
# run it in the background. Knobs: NSET (rank counts), BW (link bandwidth).
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
TRAIN="$HERE/../mercury_llm_train/train.py"
NSET="${NSET:-8 16}"
BW="${BW:-50GB/s}"
LAYERS="${LAYERS:-8}"
export LAYERS

# A100 model params shared by sim (set in train.py) and analytic (passed below).
FP32=1.95e13 ; TENSOR=3.12e14 ; MEMBW=2000e9 ; SM=108 ; LAT=100e-9
bw_bps() { echo "$1" | sed 's/GB\/s//' | awk '{printf "%g", $1*1e9}'; }

sim_step() { # TOPO ALG NRANKS -> simulated step ms (overlap off so comms is exposed)
  TOPO="$1" SUMI_ALLREDUCE_ALG="$2" NRANKS="$3" LINK_BW="$BW" LLM_OVERLAP=0 \
    sst "$TRAIN" 2>&1 | grep -i "simulated time" | grep -o "[0-9.]* ms" | grep -o "[0-9.]*"
}

analytic_step() { # ALG NRANKS -> analytic step ms (no contention, overlap off)
  python3 "$HERE/analytic_baseline.py" --alg "$1" --nranks "$2" --layers "$LAYERS" \
    --fp32-peak "$FP32" --tensor-peak "$TENSOR" --mem-bw "$MEMBW" \
    --link-bw "$(bw_bps "$BW")" --link-lat "$LAT" --sm-count "$SM" --overlap 0 \
    | sed -n 's/^step_ms=//p'
}

pct() { awk -v s="$1" -v a="$2" 'BEGIN{ if(a>0) printf "%+.1f%%", 100*(s-a)/a; else printf "n/a" }'; }

printf "%-12s %-4s %12s %12s %8s %12s %8s\n" \
  alg N analytic_ms xbar_ms d_xbar fattree_ms d_ftree
echo "--------------------------------------------------------------------------------"
for alg in recdouble ring; do
  for n in $NSET; do
    an=$(analytic_step "$alg" "$n")
    xb=$(sim_step single "$alg" "$n")
    ft=$(sim_step fattree "$alg" "$n")
    printf "%-12s %-4s %12s %12s %8s %12s %8s\n" \
      "$alg" "$n" "$an" "$xb" "$(pct "$xb" "$an")" "$ft" "$(pct "$ft" "$an")"
  done
done
echo ""
echo "d_xbar  = sim(single crossbar) vs analytic  (expect small -> model validated)"
echo "d_ftree = sim(fat-tree)        vs analytic  (expect positive -> contention)"
