#!/bin/sh
# FSDP / ZeRO-3 study: the time cost of the memory savings study (G) priced. Builds
# fsdp.cu, then sweeps FSDP (prefetch on/off) against the plain-DP baseline
# (../mercury_llm_train) at equal config, across fabric bandwidth. The questions:
# does prefetch hide the per-layer all-gather (FSDP -> DP), and what is FSDP's ~1.5x
# communication volume worth. Requires hg++ and sst on PATH. Run from anywhere.
set -eu

HGXX="${HGXX:-hg++}"
SST_BIN="${SST_BIN:-sst}"
HERE="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES="$(dirname "$HERE")"
N="${N:-8}"
LAYERS="${LAYERS:-8}"

EXT="${SST_ELEMENTS_EXT:-}"
if [ -z "$EXT" ]; then
  PREFIX="$(dirname "$(dirname "$(command -v "$SST_BIN")")")"
  EXT="$PREFIX/lib/sst-elements-library/ext"
fi

echo "# building mercury_fsdp"
"$HGXX" -c "$HERE/fsdp.cu" -o "$HERE/fsdp.o" >/dev/null 2>&1
"$HGXX" "$HERE/fsdp.o" -o "$HERE/libmercury_fsdp.so" >/dev/null 2>&1
cp "$HERE/libmercury_fsdp.so" "$EXT/"

ms() { sed -n 's/.*simulated time: *//p' | head -1 \
       | awk '{v=$1;u=$2;if(u=="s")v*=1000;else if(u=="us")v/=1000;printf "%.1f",v}'; }

# fsdp <prefetch> <bw> ; dp <bw> -- both run from examples/ (shared modules there).
fsdp() {
  (cd "$EXAMPLES" && NRANKS="$N" LAYERS="$LAYERS" FSDP_PREFETCH="$1" LINK_BW="$2" GPU_MEM=80GB \
     "$SST_BIN" mercury_fsdp/fsdp.py 2>/dev/null) | ms
}
dp() {
  (cd "$EXAMPLES" && NRANKS="$N" LAYERS="$LAYERS" LLM_OVERLAP=1 LINK_BW="$1" \
     "$SST_BIN" mercury_llm_train/train.py 2>/dev/null) | ms
}

echo
echo "## step time (ms): plain DP vs FSDP, by fabric bandwidth (N=$N, $LAYERS layers)"
printf "  %-8s %10s %12s %12s\n" "LINK_BW" "DP" "FSDP(pf)" "FSDP(no-pf)"
for bw in 12GB/s 50GB/s 100GB/s 300GB/s; do
  printf "  %-8s %10s %12s %12s\n" "$bw" "$(dp "$bw")" "$(fsdp 1 "$bw")" "$(fsdp 0 "$bw")"
done

echo
echo "## per-rank footprint (FSDP ZeRO-3 shards; cross-checks capacity.py --zero 3)"
(cd "$EXAMPLES" && NRANKS="$N" LAYERS="$LAYERS" GPU_MEM=80GB "$SST_BIN" mercury_fsdp/fsdp.py 2>/dev/null) \
  | sed -n 's/.*\(mem_footprint=[0-9]*\).*/  \1 bytes/p' | head -1

echo
echo "# done"
