#!/bin/sh
# Build the CUDA halo skeleton with hg++, install it where SST's element loader
# finds it, and run the gpu_direct sweep. Requires hg++ and sst on PATH (or set
# HGXX / SST_BIN). See README.md for the full walk-through.
set -eu

SST_BIN="${SST_BIN:-sst}"
HGXX="${HGXX:-hg++}"
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

EXT="$(sst-config --prefix)/lib/sst-elements-library/ext"

"$HGXX" -c halo.cu -o halo.o
"$HGXX" halo.o -o libmercury_cuda_halo.so
cp libmercury_cuda_halo.so "$EXT/"

echo "== mercury_cuda_halo: GPUDirect sweep (NRANKS=${NRANKS:-4}) =="
for gd in false true; do
  printf "gpu_direct=%-5s -> " "$gd"
  GPUDIRECT="$gd" "$SST_BIN" halo.py 2>&1 \
    | grep -oE "simulated time: [0-9.]+ [a-z]+" | head -1
done
