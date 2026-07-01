#!/bin/sh
# MoE all-to-all study: how much does routing imbalance (SKEW) cost, and does the
# packet fabric amplify it? An MoE layer's dispatch/combine is a pair of
# all-to-alls; under skewed routing the hot expert's rank is BOTH the compute
# straggler and a fabric fan-in hotspot. SKEW in [0,1]: 0 = uniform routing,
# 1 = every token to expert 0.
#
# We sweep SKEW on a single crossbar (no contention -- isolates the compute
# straggler) and on the fat-tree (real packet contention -- adds the fan-in
# hotspot), so the gap between them is the share of the slowdown the fabric owns,
# which an analytical all-to-all cost cannot produce. A second sweep scales N at
# fixed skew.
#
# Requires sst + the installed mercury_moe demo on PATH. Knobs: N (experts),
# LAYERS, BW.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
# The shared SST modules (platform_file_hg_test, scale_topo) live one level up in
# examples/, and SST puts the cwd on sys.path -- so run from there.
EXAMPLES="$(dirname "$HERE")"
APP="$HERE/moe.py"
N="${N:-8}"
export LAYERS="${LAYERS:-4}"
BW="${BW:-100GB/s}"

run() { # TOPO SKEW N -> simulated step ms (or ABORT)
  out=$(cd "$EXAMPLES" && TOPO="$1" SKEW="$2" NRANKS="$3" LAYERS="$LAYERS" LINK_BW="$BW" \
        sst "$APP" 2>&1) || true
  if echo "$out" | grep -qiE "abort|couldn't get a flow|fatal"; then echo "ABORT"; return; fi
  # SST prints "simulated time: X s|ms|us"; normalise everything to ms.
  echo "$out" | sed -n 's/.*simulated time: *//p' | head -1 | awk \
    '{v=$1; u=$2; if(u=="s")v*=1000; else if(u=="us")v/=1000; else if(u=="ns")v/=1e6;
      printf "%.3f", v}'
}

echo "# MoE all-to-all sweep: N=$N LAYERS=$LAYERS BW=$BW"
echo ""

echo "## 1) routing skew, crossbar (no contention) vs fat-tree (contention)"
echo "##    crossbar isolates the compute straggler; the fat-tree adds the"
echo "##    all-to-all fan-in hotspot. gap = fabric's share of the imbalance cost."
printf "  %-6s %14s %14s %10s\n" "SKEW" "crossbar_ms" "fattree_ms" "fab_x"
for skew in 0.0 0.2 0.4 0.6 0.8 1.0; do
  xbar=$(run single "$skew" "$N")
  ftree=$(run fattree "$skew" "$N")
  ratio="-"
  case "$xbar" in ''|ABORT) xbar="$xbar";; *)
    case "$ftree" in ''|ABORT) :;; *)
      ratio=$(awk -v f="$ftree" -v x="$xbar" 'BEGIN{if(x>0)printf "%.2f", f/x; else print "-"}');;
    esac;;
  esac
  printf "  %-6s %14s %14s %10s\n" "$skew" "$xbar" "$ftree" "$ratio"
done

echo ""
echo "## 2) scale at fixed skew=0.6, fat-tree (does the hotspot worsen with N?)"
printf "  %-6s %14s\n" "N" "fattree_ms"
n=4
while [ "$n" -le 32 ]; do
  printf "  %-6s %14s\n" "$n" "$(run fattree 0.6 "$n")"
  n=$((n * 2))
done

echo ""
echo "# done"
