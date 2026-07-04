#!/bin/sh
# Build the 3D-parallel (DP x PP x TP) transformer training step with hg++,
# install it where SST's element loader finds it, and run the three sweeps from
# the README:
#   1) the 3D split  -- step time vs every DPxPPxTP factorization of a fixed GPU
#                       budget (the headline: model-parallel vs replication)
#   2) DP all-reduce -- step time vs fabric for pure DP (the gradient cost the
#                       split trades against; bucketed like real DDP)
#   3) fabric shift  -- how the best split moves between slow (cross-node) and
#                       fast (NVLink) fabric
#
# The global batch (MICROBATCH) is held fixed across the sweep and split across
# the DP replicas, so every factorization does equal work -- the textbook
# fixed-global-batch comparison. Requires hg++ and sst on PATH. Knobs are env
# vars read by the skeleton (TP_SIZE, PP_SIZE, MICROBATCH) and by parallel3d.py
# (NRANKS, LINK_BW), so the sweep never recompiles.
set -eu

HGXX="${HGXX:-hg++}"
SST_BIN="${SST_BIN:-sst}"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$TEST_DIR"

EXT="${SST_ELEMENTS_EXT:-}"
if [ -z "$EXT" ]; then
  PREFIX="$(dirname "$(dirname "$(command -v "$SST_BIN")")")"
  EXT="$PREFIX/lib/sst-elements-library/ext"
fi

echo "# building mercury_3d with hg++"
"$HGXX" -c parallel3d.cu -o parallel3d.o
"$HGXX" parallel3d.o -o libmercury_3d.so
cp libmercury_3d.so "$EXT/"

# step <NRANKS> <TP_SIZE> <PP_SIZE> <MICROBATCH> <LINK_BW> -> simulated step time
step() {
  NRANKS="$1" TP_SIZE="$2" PP_SIZE="$3" MICROBATCH="$4" LINK_BW="$5" \
    "$SST_BIN" "$TEST_DIR/parallel3d.py" 2>/dev/null \
    | sed -n 's/.*simulated time: *//p' | head -1
}

echo
echo "## 1) the 3D split  (fixed 8-GPU budget, global batch M=8, 150GB/s)"
echo "##    DP = 8/(TP*PP); model-parallel shards both compute and the gradient,"
echo "##    pure DP replicates the model -> the biggest gradient all-reduce."
printf "   %-14s %-12s\n" "DPxPPxTP" "step_time"
# enumerate the factorizations TP*PP <= 8 (DP fills the rest), TP/PP powers of two
for tp in 1 2 4 8; do
  for pp in 1 2 4 8; do
    [ $((tp * pp)) -le 8 ] || continue
    dp=$((8 / (tp * pp)))
    printf "   DP%d PP%d TP%-5d %-12s\n" "$dp" "$pp" "$tp" \
      "$(step 8 "$tp" "$pp" 8 150GB/s)"
  done
done

echo
echo "## 2) DP gradient all-reduce vs fabric  (pure DP=8, PP=1, TP=1, M=8)"
echo "##    the un-sharded gradient is bucketed like DDP; bandwidth-bound."
printf "   %-10s %-14s\n" "LINK_BW" "step_time"
for bw in 12GB/s 50GB/s 150GB/s 600GB/s; do
  printf "   %-10s %-14s\n" "$bw" "$(step 8 1 1 8 "$bw")"
done

echo
echo "## 3) the split moves with the fabric  (M=8); slow fabric punishes comms"
printf "   %-14s %-14s %-14s\n" "DPxPPxTP" "12GB/s" "300GB/s"
printf "   DP1 PP8 TP1    %-14s %-14s\n" "$(step 8 1 8 8 12GB/s)" "$(step 8 1 8 8 300GB/s)"
printf "   DP1 PP1 TP8    %-14s %-14s\n" "$(step 8 8 1 8 12GB/s)" "$(step 8 8 1 8 300GB/s)"
printf "   DP8 PP1 TP1    %-14s %-14s\n" "$(step 8 1 1 8 12GB/s)" "$(step 8 1 1 8 300GB/s)"

echo
echo "# done"
