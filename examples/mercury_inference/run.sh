#!/bin/sh
# Disaggregated inference study. Three angles: (1) decode is latency-bound -- its
# small TP all-reduce barely moves with bandwidth but strongly prefers recursive-
# halving over ring (the inverse of training study E); (2) the bulk KV-cache transfer
# from prefill to decode contends with decode traffic on the fabric; (3) the
# prefill:decode split. Requires hg++ and sst on PATH. Knobs: STEPS, PROMPT, BW.
set -eu

HGXX="${HGXX:-hg++}"
SST_BIN="${SST_BIN:-sst}"
HERE="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES="$(dirname "$HERE")"
STEPS="${STEPS:-16}"
PROMPT="${PROMPT:-2048}"
BW="${BW:-100GB/s}"

EXT="${SST_ELEMENTS_EXT:-}"
if [ -z "$EXT" ]; then
  PREFIX="$(dirname "$(dirname "$(command -v "$SST_BIN")")")"
  EXT="$PREFIX/lib/sst-elements-library/ext"
fi

echo "# building mercury_inference"
"$HGXX" -c "$HERE/infer.cu" -o "$HERE/infer.o" >/dev/null 2>&1
"$HGXX" "$HERE/infer.o" -o "$HERE/libmercury_inference.so" >/dev/null 2>&1
cp "$HERE/libmercury_inference.so" "$EXT/"

ms() { sed -n 's/.*simulated time: *//p' | head -1 \
       | awk '{v=$1;u=$2;if(u=="s")v*=1000;else if(u=="us")v/=1000;printf "%.1f",v}'; }
run() { (cd "$EXAMPLES" && env "$@" "$SST_BIN" mercury_inference/infer.py 2>/dev/null) | ms; }

echo
echo "## 1) decode is latency-bound (pure decode, TP=8, $STEPS steps): flat in"
echo "##    bandwidth, but recursive-halving beats ring -- the inverse of study E."
printf "  %-8s %14s %14s\n" "LINK_BW" "rec-halving" "ring"
for bw in 12GB/s 100GB/s 300GB/s; do
  rh=$(run NRANKS=8 PREFILL_RANKS=0 DECODE_STEPS="$STEPS" REQUESTS=1 LINK_BW="$bw" SUMI_ALLREDUCE_ALG=recdouble)
  rg=$(run NRANKS=8 PREFILL_RANKS=0 DECODE_STEPS="$STEPS" REQUESTS=1 LINK_BW="$bw" SUMI_ALLREDUCE_ALG=ring)
  printf "  %-8s %14s %14s\n" "$bw" "$rh" "$rg"
done

echo
echo "## 2) the prefill->decode handoff hides under a long enough decode (fat-tree,"
echo "##    12GB/s, prompt=4096, TP=4): decode-only vs disaggregated, by decode length."
echo "##    The handoff (prefill compute + bulk KV transfer) is a fixed cost; long"
echo "##    generations amortize it (disagg -> decode-only), short ones are handoff-bound."
printf "  %-12s %14s %14s %8s\n" "DECODE_STEPS" "decode-only" "disagg" "x"
for st in 2 8 32; do
  c=$(run TOPO=fattree NRANKS=4 PREFILL_RANKS=0 DECODE_STEPS="$st" REQUESTS=2 PROMPT_LEN=4096 LINK_BW=12GB/s)
  g=$(run TOPO=fattree NRANKS=6 PREFILL_RANKS=2 DECODE_STEPS="$st" REQUESTS=2 PROMPT_LEN=4096 LINK_BW=12GB/s)
  x=$(awk -v a="$g" -v b="$c" 'BEGIN{if(b>0)printf "%.2f",a/b; else print "-"}')
  printf "  %-12s %14s %14s %8s\n" "$st" "$c" "$g" "$x"
done

echo
echo "# done"
