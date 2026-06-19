#!/bin/sh
# Context-parallelism (ring attention) study. Three angles: (1) the HIDING CROSSOVER --
# sweep context length SEQ at fixed ring length and bandwidth with overlap on/off, and
# find where overlap stops helping (attention compute, quadratic in SEQ, overtakes the
# K/V transfer, linear in SEQ, so the ring is fully hidden); (2) FABRIC SENSITIVITY --
# at fixed long SEQ sweep LINK_BW to show the exposed regime is bandwidth-bound and the
# hidden regime is flat; (3) SCALE -- ring length N at fixed SEQ (more ranks = smaller
# blocks = more steps but less compute/step). Requires hg++ and sst on PATH.
set -eu

HGXX="${HGXX:-hg++}"
SST_BIN="${SST_BIN:-sst}"
HERE="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES="$(dirname "$HERE")"

EXT="${SST_ELEMENTS_EXT:-}"
if [ -z "$EXT" ]; then
  PREFIX="$(dirname "$(dirname "$(command -v "$SST_BIN")")")"
  EXT="$PREFIX/lib/sst-elements-library/ext"
fi

echo "# building mercury_context"
"$HGXX" -c "$HERE/ring_attn.cu" -o "$HERE/ring_attn.o" >/dev/null 2>&1
"$HGXX" "$HERE/ring_attn.o" -o "$HERE/libmercury_context.so" >/dev/null 2>&1
cp "$HERE/libmercury_context.so" "$EXT/"

ms() { sed -n 's/.*simulated time: *//p' | head -1 \
       | awk '{v=$1;u=$2;if(u=="s")v*=1000;else if(u=="us")v/=1000;printf "%.1f",v}'; }
run() { (cd "$EXAMPLES" && env "$@" "$SST_BIN" mercury_context/ring_attn.py 2>/dev/null) | ms; }

echo
echo "## 1) the hiding crossover (N=8 ring, 100GB/s, 2 layers): step time vs SEQ,"
echo "##    overlap on/off. At short context overlap helps (K/V ring exposed); past a"
echo "##    crossover SEQ the two converge (attention compute hides the ring)."
printf "  %-8s %14s %14s %8s\n" "SEQ" "overlap" "no-overlap" "gain"
for seq in 8192 16384 32768 65536 131072; do
  on=$(run  NRANKS=8 SEQ="$seq" LAYERS=2 LINK_BW=100GB/s CP_OVERLAP=1)
  off=$(run NRANKS=8 SEQ="$seq" LAYERS=2 LINK_BW=100GB/s CP_OVERLAP=0)
  g=$(awk -v a="$on" -v b="$off" 'BEGIN{if(a>0)printf "%.2f",b/a; else print "-"}')
  printf "  %-8s %14s %14s %8s\n" "$seq" "$on" "$off" "$g"
done

echo
echo "## 2) fabric sensitivity (N=8, 2 layers, overlap on): SEQ in the exposed regime"
echo "##    (16k) vs the hidden regime (128k), sweeping LINK_BW. Exposed scales with"
echo "##    bandwidth (transfer > compute); hidden is flat (compute hides the ring)."
printf "  %-8s %14s %14s\n" "LINK_BW" "SEQ=16k" "SEQ=128k"
for bw in 25GB/s 100GB/s 400GB/s; do
  lo=$(run NRANKS=8 SEQ=16384  LAYERS=2 LINK_BW="$bw" CP_OVERLAP=1)
  hi=$(run NRANKS=8 SEQ=131072 LAYERS=2 LINK_BW="$bw" CP_OVERLAP=1)
  printf "  %-8s %14s %14s\n" "$bw" "$lo" "$hi"
done

echo
echo "## 3) scale the ring (SEQ=65536, 100GB/s, 2 layers, overlap on): N ranks ="
echo "##    SEQ/N tokens/rank. More ranks = more steps but quadratically less"
echo "##    compute per step -- the ring-length vs block-size tradeoff."
printf "  %-8s %14s\n" "NRANKS" "step"
for n in 4 8 16; do
  t=$(run NRANKS="$n" SEQ=65536 LAYERS=2 LINK_BW=100GB/s CP_OVERLAP=1)
  printf "  %-8s %14s\n" "$n" "$t"
done

echo
echo "# done"
