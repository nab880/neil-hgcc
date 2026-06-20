#!/bin/sh
# Co-design Stage 1b: the inference (decode) building blocks. For each per-pool hardware
# point and serving config (TP x batch x context), gate with capacity.py --infer (weights
# + KV <= that pool's HBM capacity), then run mercury_inference in pure-decode mode and
# record the per-token decode latency. Capacity is THE inference axis: a bigger GPU_MEM
# lets a config drop to a lower feasible TP, and lower TP is the latency win (TP1~4ms vs
# TP8~38ms). One replica per row (pool size enters only in pareto.py, as a multiplier),
# so this sweep is small. Requires hg++ and sst on PATH. Output: infer.csv.
#
# Network bandwidth is recorded but ~flat on decode (study I) -- it is here only so the
# CSV is self-describing and the optional long-context-prefill extension can reuse it.
set -eu

HGXX="${HGXX:-hg++}"
SST_BIN="${SST_BIN:-sst}"
HERE="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES="$(dirname "$HERE")"
CAP="$EXAMPLES/memory_model/capacity.py"
OUT="${OUT:-$HERE/infer.csv}"

H=4096; L=64
DECODE_STEPS="${DECODE_STEPS:-8}"            # per-token step is uniform; latency = t_sim/steps
# NETS must mirror sweep_train.sh so the two pools share hardware points -- otherwise no
# uniform-hardware fleet is constructible and the differentiated-vs-uniform test is moot.
# (Decode is flat in network, so these rows differ only in the pool's labeled net + cost.)
if [ "${FULL:-0}" = "1" ]; then
  CAPS="${CAPS:-80 160 320}"; NETS="${NETS:-50 150 600}"; SIM_LAYERS="${SIM_LAYERS:-64}"
else
  CAPS="${CAPS:-80 320}"; NETS="${NETS:-50 600}"; SIM_LAYERS="${SIM_LAYERS:-8}"
fi
BATCHES="${BATCHES:-8 32}"; CTXS="${CTXS:-2048 16384}"

EXT="${SST_ELEMENTS_EXT:-}"
if [ -z "$EXT" ]; then
  PREFIX="$(dirname "$(dirname "$(command -v "$SST_BIN")")")"
  EXT="$PREFIX/lib/sst-elements-library/ext"
fi

echo "# building mercury_inference" >&2
"$HGXX" -c "$EXAMPLES/mercury_inference/infer.cu" -o "$EXAMPLES/mercury_inference/infer.o" >/dev/null 2>&1
"$HGXX" "$EXAMPLES/mercury_inference/infer.o" -o "$EXAMPLES/mercury_inference/libmercury_inference.so" >/dev/null 2>&1
cp "$EXAMPLES/mercury_inference/libmercury_inference.so" "$EXT/"

ms() { sed -n 's/.*simulated time: *//p' | head -1 \
       | awk '{v=$1;u=$2;if(u=="s")v*=1000;else if(u=="us")v/=1000;printf "%.3f",v}'; }

echo "cap_gb,net_gbps,TP,batch,ctx,t_decode_ms,latency_ms,per_replica_tput" > "$OUT"
for cap in $CAPS; do
 for net in $NETS; do
  for TP in 1 2 4 8; do
   for B in $BATCHES; do
    for ctx in $CTXS; do
      # Gate: weights + KV cache for one TP group must fit this pool's capacity.
      python3 "$CAP" --infer --H $H --L $L --tp "$TP" --batch "$B" --seq "$ctx" \
        --gpu-mem "$cap" | tail -1 | grep -q FITS || continue
      log=$(cd "$EXAMPLES" && env NRANKS="$TP" PREFILL_RANKS=0 DECODE_STEPS="$DECODE_STEPS" \
            REQUESTS=1 BATCH="$B" PROMPT_LEN="$ctx" LAYERS="$SIM_LAYERS" \
            LINK_BW="${net}GB/s" GPU_MEM="${cap}GB" SUMI_ALLREDUCE_ALG=recdouble \
            "$SST_BIN" mercury_inference/infer.py 2>/dev/null)
      tsim=$(printf '%s\n' "$log" | ms)
      [ -n "$tsim" ] || { echo "  WARN: no time cap=$cap TP=$TP B=$B ctx=$ctx" >&2; continue; }
      td=$(awk -v t="$tsim" -v s="$DECODE_STEPS" 'BEGIN{printf "%.4f", t/s}')
      tput=$(awk -v b="$B" -v td="$td" 'BEGIN{printf "%.2f", b/(td/1000)}')   # tokens/s, one replica
      echo "$cap,$net,$TP,$B,$ctx,$td,$td,$tput" >> "$OUT"
      echo "  cap=$cap net=$net TP$TP B$B ctx$ctx -> ${td}ms/token tput=$tput tok/s" >&2
    done
   done
  done
 done
done
echo "# wrote $OUT ($(($(wc -l < "$OUT") - 1)) rows)" >&2
