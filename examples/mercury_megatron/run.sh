#!/bin/sh
# Build the Megatron-style tensor+pipeline parallel transformer with hg++, install
# it where SST's element loader finds it, and run the three sweeps from the README:
#   1) pipeline bubble    -- step time vs microbatch count (pure PP)
#   2) tensor all-reduce  -- step time vs fabric bandwidth (pure TP, critical path)
#   3) TP/PP split        -- step time vs the TPxPP layout for a fixed GPU budget
#
# Requires hg++ and sst on PATH. Knobs are environment variables read by the
# skeleton (TP_SIZE, MICROBATCH) and by megatron.py (NRANKS, LINK_BW), so the
# sweep never recompiles.
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

echo "# building mercury_megatron with hg++"
"$HGXX" -c megatron.cu -o megatron.o
"$HGXX" megatron.o -o libmercury_megatron.so
cp libmercury_megatron.so "$EXT/"

step() { # NRANKS TP_SIZE MICROBATCH LINK_BW -> simulated step time
  NRANKS="$1" TP_SIZE="$2" MICROBATCH="$3" LINK_BW="$4" \
    "$SST_BIN" "$TEST_DIR/megatron.py" 2>/dev/null \
    | sed -n 's/.*simulated time: *//p' | head -1
}

echo
echo "## 1) pipeline bubble  (pure PP: TP=1, 4 stages, 150GB/s): bubble = (P-1)/(M+P-1)"
printf "   %-6s %-14s\n" "M" "step_time"
for m in 1 2 4 8 16; do
  printf "   %-6s %-14s\n" "$m" "$(step 4 1 "$m" 150GB/s)"
done

echo
echo "## 2) tensor all-reduce vs fabric  (pure TP=4, PP=1, M=4): on the critical path"
printf "   %-10s %-14s\n" "LINK_BW" "step_time"
for bw in 12GB/s 50GB/s 150GB/s 600GB/s; do
  printf "   %-10s %-14s\n" "$bw" "$(step 4 4 4 "$bw")"
done

echo
echo "## 3) TP/PP split for a fixed 8-GPU budget (150GB/s); the optimum moves with M"
printf "   %-12s %-14s %-14s\n" "TPxPP" "M=1" "M=8"
for tp in 1 2 4 8; do
  pp=$((8 / tp))
  printf "   TP=%d PP=%-5d %-14s %-14s\n" "$tp" "$pp" \
    "$(step 8 "$tp" 1 150GB/s)" "$(step 8 "$tp" 8 150GB/s)"
done

echo
echo "# done"
