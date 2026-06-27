#!/bin/sh
# Co-design Stage 1a: the training building blocks. For each fleet split N_t and each
# per-pool hardware point (HBM capacity x network bandwidth), enumerate the feasible
# TPxPPxDP layouts, derive the minimum ZeRO stage that FITS from capacity.py (NOT swept
# independently -- higher stages cost comms, study H), run mercury_3d, and emit one CSV
# row per layout actually run. HBM *bandwidth* is not an axis (decode/train both flat in
# it -- see CODESIGN-EXPERIMENT.md); the inference pool's axis is capacity, handled in
# sweep_infer.sh. Requires hg++ and sst on PATH. Output: train.csv.
#
# Feasibility always uses the real model depth (L=64); the sim may use a reduced
# SIM_LAYERS timing skeleton (proportional cost, same comm pattern) for a fast first
# pass. FULL=1 sets the paper grid (SIM_LAYERS=64, all splits).
set -eu

HGXX="${HGXX:-hg++}"
SST_BIN="${SST_BIN:-sst}"
HERE="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES="$(dirname "$HERE")"
CAP="$EXAMPLES/memory_model/capacity.py"
OUT="${OUT:-$HERE/train.csv}"

H=4096; L=64                         # the 13B-class target; L=64 is the feasibility depth
if [ "${FULL:-0}" = "1" ]; then
  SPLITS="${SPLITS:-8 16 32 64}"; CAPS="${CAPS:-80 160 320}"; NETS="${NETS:-50 150 600}"
  SIM_LAYERS="${SIM_LAYERS:-64}"; MICRO="${MICRO:-8}"
else                                 # fast smoke grid
  SPLITS="${SPLITS:-16 64}"; CAPS="${CAPS:-80 320}"; NETS="${NETS:-50 600}"
  SIM_LAYERS="${SIM_LAYERS:-8}"; MICRO="${MICRO:-8}"
fi

EXT="${SST_ELEMENTS_EXT:-}"
if [ -z "$EXT" ]; then
  PREFIX="$(dirname "$(dirname "$(command -v "$SST_BIN")")")"
  EXT="$PREFIX/lib/sst-elements-library/ext"
fi

echo "# building mercury_3d" >&2
"$HGXX" -c "$EXAMPLES/mercury_3d/parallel3d.cu" -o "$EXAMPLES/mercury_3d/parallel3d.o" >/dev/null 2>&1
"$HGXX" "$EXAMPLES/mercury_3d/parallel3d.o" -o "$EXAMPLES/mercury_3d/libmercury_3d.so" >/dev/null 2>&1
cp "$EXAMPLES/mercury_3d/libmercury_3d.so" "$EXT/"

# Simulated time -> ms (normalize SST's s/ms/us unit).
ms() { sed -n 's/.*simulated time: *//p' | head -1 \
       | awk '{v=$1;u=$2;if(u=="s")v*=1000;else if(u=="us")v/=1000;printf "%.3f",v}'; }

# fit_zero DP TP PP CAP -> the min ZeRO in {0,1,2,3} whose footprint FITS, else "none".
fit_zero() {
  for z in 0 1 2 3; do
    if python3 "$CAP" --H $H --L $L --dp "$1" --tp "$2" --pp "$3" --zero "$z" \
         --gpu-mem "$4" | tail -1 | grep -q FITS; then echo "$z"; return; fi
  done
  echo none
}

# t_step is capacity-independent (ZeRO stage changes the footprint, not the gradient
# all-reduce volume -- verified: cap80/z1 and cap320/z0 give identical times). So the sim
# runs ONCE per (Nt,net,layout,alg) and the capacity axis is replayed from capacity.py's
# per-cap min-ZeRO gate -- a |CAPS|x cut in sim runs, the dominant cost at LAYERS=64.
CAP_MAX=$(echo $CAPS | tr ' ' '\n' | sort -n | tail -1)

# Row-level resume: keep an existing CSV and skip any (Nt,net,TP,PP,alg) combo already in
# it, so a sweep killed mid-split (e.g. machine sleep) resumes exactly where it stopped --
# even the 9h Nt=64 split banks incrementally. Delete the CSV to force a fresh sweep.
[ -f "$OUT" ] || echo "Nt,cap_gb,net_gbps,TP,PP,DP,zero,alg,t_step_ms,train_tput" > "$OUT"
row_done() {   # Nt net TP PP alg -> 0 if a matching row is already present
  awk -F, -v a="$1" -v b="$2" -v c="$3" -v d="$4" -v e="$5" \
      '$1==a && $3==b && $4==c && $5==d && $8==e {found=1} END{exit !found}' "$OUT"
}
for Nt in $SPLITS; do
 for net in $NETS; do
  for TP in 1 2 4 8; do
   [ "$TP" -le "$Nt" ] || continue            # TP capped at 8 (single NVLink domain) and at Nt
   for PP in 1 2 4 8; do
    [ $((TP * PP)) -le "$Nt" ] || continue
    [ $((Nt % (TP * PP))) -eq 0 ] || continue
    [ $((L % PP)) -eq 0 ] || continue          # whole layers per stage
    DP=$((Nt / (TP * PP)))
    # Skip layouts that OOM even at the largest capacity (infeasible everywhere).
    [ "$(fit_zero "$DP" "$TP" "$PP" "$CAP_MAX")" != "none" ] || continue
    # Algorithm only changes the DP gradient all-reduce; moot when DP=1.
    algs="ring recdouble"; [ "$DP" -gt 1 ] || algs="ring"
    for alg in $algs; do
      row_done "$Nt" "$net" "$TP" "$PP" "$alg" && continue   # resume: already computed
      # One timing run (ZERO=0; timing is ZeRO-independent), replayed across capacities.
      log=$(cd "$EXAMPLES" && env NRANKS="$Nt" TP_SIZE="$TP" PP_SIZE="$PP" MICROBATCH="$MICRO" \
            LAYERS="$SIM_LAYERS" ZERO=0 LINK_BW="${net}GB/s" GPU_MEM="${CAP_MAX}GB" \
            SUMI_ALLREDUCE_ALG="$alg" "$SST_BIN" mercury_3d/parallel3d.py 2>/dev/null)
      t=$(printf '%s\n' "$log" | ms)
      [ -n "$t" ] || { echo "  WARN: no time Nt=$Nt net=$net TP=$TP PP=$PP $alg" >&2; continue; }
      tput=$(awk -v dp="$DP" -v t="$t" 'BEGIN{printf "%.4f", dp/(t/1000)}')
      for cap in $CAPS; do
        zero=$(fit_zero "$DP" "$TP" "$PP" "$cap")
        [ "$zero" != "none" ] || continue      # infeasible at this capacity
        echo "$Nt,$cap,$net,$TP,$PP,$DP,$zero,$alg,$t,$tput" >> "$OUT"
      done
      echo "  Nt=$Nt net=$net TP$TP PP$PP DP$DP $alg -> ${t}ms tput=$tput (x$(echo $CAPS|wc -w) caps)" >&2
    done
   done
  done
 done
done
echo "# wrote $OUT ($(($(wc -l < "$OUT") - 1)) rows)" >&2
