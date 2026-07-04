#!/bin/sh
# Build the data-parallel transformer training step with hg++, install it where
# SST's element loader finds it, and run the three sweeps from the README:
#   1) fabric bandwidth  -- the comms-bound -> compute-bound crossover
#   2) the levers        -- GPUDirect and compute/comms overlap at NVLink speed
#   3) data-parallel scaling -- step time vs rank count
#
# Requires hg++ and sst on PATH (e.g. source the sst-hgcc module env). Knobs are
# environment variables read by the skeleton (GPUDIRECT, LLM_OVERLAP) and by
# train.py (NRANKS, LINK_BW), so the sweep never recompiles.
set -eu

HGXX="${HGXX:-hg++}"
SST_BIN="${SST_BIN:-sst}"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$TEST_DIR"

# Locate the installed element ext dir (where libmercury_llm_train.so must land).
EXT="${SST_ELEMENTS_EXT:-}"
if [ -z "$EXT" ]; then
  PREFIX="$(dirname "$(dirname "$(command -v "$SST_BIN")")")"
  EXT="$PREFIX/lib/sst-elements-library/ext"
fi

echo "# building mercury_llm_train with hg++"
"$HGXX" -c train.cu -o train.o
"$HGXX" train.o -o libmercury_llm_train.so
cp libmercury_llm_train.so "$EXT/"

step() { # NRANKS GPUDIRECT LLM_OVERLAP LINK_BW -> simulated step time
  NRANKS="$1" GPUDIRECT="$2" LLM_OVERLAP="$3" LINK_BW="$4" \
    "$SST_BIN" "$TEST_DIR/train.py" 2>/dev/null \
    | sed -n 's/.*simulated time: *//p' | head -1
}

echo
echo "## 1) fabric sweep  (4 ranks, GPUDirect on, overlap on): comms -> compute crossover"
printf "   %-10s %-14s\n" "LINK_BW" "step_time"
for bw in 12GB/s 50GB/s 150GB/s 600GB/s; do
  printf "   %-10s %-14s\n" "$bw" "$(step 4 true 1 "$bw")"
done

echo
echo "## 2) levers  (4 ranks, 150GB/s NVLink-class fabric)"
printf "   %-12s %-9s %-14s\n" "gpu_direct" "overlap" "step_time"
for gd in true false; do for ov in 1 0; do
  printf "   %-12s %-9s %-14s\n" "$gd" "$ov" "$(step 4 "$gd" "$ov" 150GB/s)"
done; done

echo
echo "## 3) data-parallel scaling  (GPUDirect on, overlap on, 150GB/s)"
printf "   %-8s %-14s\n" "ranks" "step_time"
for n in 1 2 4 8 16; do
  printf "   %-8s %-14s\n" "$n" "$(step "$n" true 1 150GB/s)"
done

echo
echo "# done"
