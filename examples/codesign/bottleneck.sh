#!/bin/sh
# Harvest the bottleneck signal for the co-design's frontier configs: re-run each pool's
# binding layout across LINK_BW and capture BOTH the step time (simulated) and the GPU-busy
# time (total_gpu_time). compute_fraction = gpu/sim is the per-config bottleneck indicator
# -- ~1.0 = compute-bound (the fabric is hidden), <1 = communication-exposed. Cheap: only
# the per-split best training layout and the binding decode config per capacity tier, not
# the full grid. Output: bottleneck.csv. Requires hg++ and sst on PATH; reuses the L=64
# train/infer CSVs to pick the binding layouts.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES="$(dirname "$HERE")"
SST_BIN="${SST_BIN:-sst}"
export PATH="/Users/neil/Documents/GitHub/clean-mv2/install/bin:$PATH"
H=4096; L=64
NETS="${NETS:-50 150 600}"
OUT="$HERE/bottleneck.csv"

ms()   { sed -n 's/.*simulated time: *//p' | head -1 \
         | awk '{v=$1;u=$2;if(u=="s")v*=1000;else if(u=="us")v/=1000;printf "%.3f",v}'; }
gpums(){ sed -n 's/.*total_gpu_time=\([0-9.eE+-]*\).*/\1/p' | head -1 | awk '{printf "%.3f",$1*1000}'; }

echo "pool,label,net_gbps,sim_ms,gpu_ms,compute_frac" > "$OUT"

# --- training: the max-throughput layout per split (the frontier's training block) ---
for Nt in 8 16 32 64; do
  # best (TP,PP,zero,alg) for this split from the L=64 grid
  read TP PP ZERO ALG <<EOF
$(awk -F, -v nt=$Nt '$1==nt{if($10>m){m=$10;tp=$4;pp=$5;z=$7;a=$8}} END{print tp,pp,z,a}' "$HERE/train_L64_full.csv")
EOF
  for net in $NETS; do
    log=$(cd "$EXAMPLES" && env NRANKS=$Nt TP_SIZE=$TP PP_SIZE=$PP MICROBATCH=8 LAYERS=$L \
          ZERO=$ZERO LINK_BW="${net}GB/s" GPU_MEM=320GB SUMI_ALLREDUCE_ALG=$ALG \
          "$SST_BIN" mercury_3d/parallel3d.py 2>/dev/null)
    s=$(printf '%s\n' "$log" | ms); g=$(printf '%s\n' "$log" | gpums)
    cf=$(awk -v g="$g" -v s="$s" 'BEGIN{if(s>0)printf "%.3f",g/s; else print "NA"}')
    echo "train,Nt$Nt(TP${TP}xPP${PP}xDP$((Nt/(TP*PP)))),$net,$s,$g,$cf" >> "$OUT"
    echo "  train Nt=$Nt net=$net -> step ${s}ms gpu ${g}ms cf=$cf" >&2
  done
done

# --- decode: the binding (min feasible TP) config per capacity tier, ctx=16384 b=8 ---
for cap in 80 160 320; do
  for tp in 1 2 4 8; do
    python3 "$EXAMPLES/memory_model/capacity.py" --infer --H $H --L $L --tp $tp --batch 8 \
      --seq 16384 --gpu-mem $cap | tail -1 | grep -q FITS && { TP=$tp; break; }
  done
  for net in $NETS; do
    log=$(cd "$EXAMPLES" && env NRANKS=$TP PREFILL_RANKS=0 DECODE_STEPS=8 REQUESTS=1 BATCH=8 \
          PROMPT_LEN=16384 LAYERS=$L LINK_BW="${net}GB/s" GPU_MEM="${cap}GB" \
          SUMI_ALLREDUCE_ALG=recdouble "$SST_BIN" mercury_inference/infer.py 2>/dev/null)
    s=$(printf '%s\n' "$log" | ms); g=$(printf '%s\n' "$log" | gpums)
    # per-token latency = step/DECODE_STEPS; compute_frac uses the same ratio (gpu also /steps)
    lat=$(awk -v s="$s" 'BEGIN{printf "%.3f",s/8}')
    cf=$(awk -v g="$g" -v s="$s" 'BEGIN{if(s>0)printf "%.3f",g/s; else print "NA"}')
    echo "decode,cap${cap}GB(TP$TP),$net,$lat,$(awk -v g="$g" 'BEGIN{printf "%.3f",g/8}'),$cf" >> "$OUT"
    echo "  decode cap=$cap net=$net -> lat ${lat}ms cf=$cf" >&2
  done
done
echo "# wrote $OUT ($(($(wc -l < "$OUT")-1)) rows)" >&2
