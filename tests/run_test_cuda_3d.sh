#!/bin/sh
# Smoke test for the 3D-parallel demo (examples/mercury_3d): a DP x PP x TP
# transformer training step where data, pipeline, and tensor parallelism all
# coexist. Runs the actual example in a few configs and asserts the three
# strategies behave -- a full 3-way (2x2x2) split completes, model-parallel
# pipeline beats critical-path tensor parallelism at equal work, and pure data
# parallelism's bucketed gradient all-reduce responds to fabric bandwidth.
# Small, fast invariants (not the full README sweep).
set -eu

SST_BIN="${SST_BIN:-sst}"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
P3D_PY="$TEST_DIR/../examples/mercury_3d/parallel3d.py"
cd "$TEST_DIR"

sim_seconds() {
  grep "simulated time:" "$1" \
    | sed -n 's/.*simulated time: \([0-9.eE+-]*\) \([a-z]*\).*/\1 \2/p' | head -1 \
    | awk '{ u=$2;
        m=(u=="s")?1:(u=="ms")?1e-3:(u=="us")?1e-6:(u=="ns")?1e-9:(u=="ps")?1e-12:1;
        printf "%.12g", $1*m }'
}

# run <log> <NRANKS> <TP_SIZE> <PP_SIZE> <MICROBATCH> <LINK_BW>
run() {
  NRANKS="$2" TP_SIZE="$3" PP_SIZE="$4" MICROBATCH="$5" LINK_BW="$6" \
    "$SST_BIN" "$P3D_PY" > "$1" 2>&1
  grep -q "mercury_3d:.*done" "$1" \
    || { echo "FAIL: step did not complete ($1)"; cat "$1"; exit 1; }
}

run out_3d_full.log  8 2 2 8 150GB/s     # 2x2x2: all three comms coexist
run out_3d_pp.log    4 1 4 4 150GB/s     # pure PP (model-parallel pipeline)
run out_3d_tp.log    4 4 1 4 150GB/s     # pure TP (critical-path all-reduce)
run out_3d_dp_slow.log 4 1 1 4 12GB/s    # pure DP, slow fabric  -> big gradient
run out_3d_dp_fast.log 4 1 1 4 300GB/s   # pure DP, fast fabric

pp=$(sim_seconds out_3d_pp.log)
tp=$(sim_seconds out_3d_tp.log)
dp_slow=$(sim_seconds out_3d_dp_slow.log)
dp_fast=$(sim_seconds out_3d_dp_fast.log)
test -n "$pp" && test -n "$tp" && test -n "$dp_slow" && test -n "$dp_fast"

awk -v pp="$pp" -v tp="$tp" -v slow="$dp_slow" -v fast="$dp_fast" 'BEGIN{
  ok = 1
  # (1) At equal work, model-parallel pipeline beats critical-path tensor
  #     parallelism: PP overlaps via microbatching, TP all-reduce cannot hide.
  if (!(pp < tp)) { printf "FAIL: PP (%g s) not < TP (%g s)\n", pp, tp; ok = 0 }
  # (2) Pure-DP gradient all-reduce is bandwidth-bound -> responds to fabric.
  if (!(slow > 1.5 * fast)) { printf "FAIL: DP fabric: 12GB/s (%g s) not > 1.5x 300GB/s (%g s)\n", slow, fast; ok = 0 }
  if (ok) {
    printf "PASS: 3d smoke -- 2x2x2 coexists; PP %g s < TP %g s (%.2fx); DP fabric %g->%g s (%.1fx)\n",
           pp, tp, tp/pp, slow, fast, slow/fast
  } else { exit 1 }
}'
