#!/bin/sh
# Smoke test for the tensor+pipeline parallel demo (examples/mercury_megatron):
# a Megatron-style TPxPP training step. Runs the actual example at 4 ranks in a
# few configs and asserts the two parallelism strategies behave -- the step
# completes, the pipeline bubble amortizes as microbatches grow (PP), and the
# tensor all-reduce responds to fabric bandwidth on the critical path (TP).
# Small, fast invariants (not the full README sweep).
set -eu

SST_BIN="${SST_BIN:-sst}"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
MEGA_PY="$TEST_DIR/../examples/mercury_megatron/megatron.py"
cd "$TEST_DIR"

sim_seconds() {
  grep "simulated time:" "$1" \
    | sed -n 's/.*simulated time: \([0-9.eE+-]*\) \([a-z]*\).*/\1 \2/p' | head -1 \
    | awk '{ u=$2;
        m=(u=="s")?1:(u=="ms")?1e-3:(u=="us")?1e-6:(u=="ns")?1e-9:(u=="ps")?1e-12:1;
        printf "%.12g", $1*m }'
}

# run <log> <TP_SIZE> <MICROBATCH> <LINK_BW>  (4 ranks)
run() {
  NRANKS=4 TP_SIZE="$2" MICROBATCH="$3" LINK_BW="$4" \
    "$SST_BIN" "$MEGA_PY" > "$1" 2>&1
  grep -q "mercury_megatron:.*done" "$1" \
    || { echo "FAIL: step did not complete ($1)"; cat "$1"; exit 1; }
}

run out_mega_pp_m1.log  1 1 150GB/s    # pure PP (4 stages), 1 microbatch  -> full bubble
run out_mega_pp_m4.log  1 4 150GB/s    # pure PP, 4 microbatches           -> bubble amortized
run out_mega_tp_slow.log 4 4 12GB/s    # pure TP, slow fabric              -> critical-path comms
run out_mega_tp_fast.log 4 4 300GB/s   # pure TP, fast fabric

pp_m1=$(sim_seconds out_mega_pp_m1.log)
pp_m4=$(sim_seconds out_mega_pp_m4.log)
tp_slow=$(sim_seconds out_mega_tp_slow.log)
tp_fast=$(sim_seconds out_mega_tp_fast.log)
test -n "$pp_m1" && test -n "$pp_m4" && test -n "$tp_slow" && test -n "$tp_fast"

awk -v m1="$pp_m1" -v m4="$pp_m4" -v slow="$tp_slow" -v fast="$tp_fast" 'BEGIN{
  ok = 1
  # (1) PP bubble amortizes: 4 microbatches cost less than 4x one (the M=1 run is
  #     mostly bubble), i.e. per-microbatch time drops as M grows.
  if (!(m4 < 4 * m1)) { printf "FAIL: bubble: M=4 (%g s) not < 4x M=1 (%g s)\n", m4, m1; ok = 0 }
  # (2) TP all-reduce is on the critical path -> responds strongly to the fabric.
  if (!(slow > 1.5 * fast)) { printf "FAIL: TP fabric: 12GB/s (%g s) not > 1.5x 300GB/s (%g s)\n", slow, fast; ok = 0 }
  if (ok) {
    printf "PASS: megatron smoke -- PP bubble amortizes (M=1 %g s, M=4 %g s = %.2fx not 4x), TP fabric %g->%g s (%.1fx)\n",
           m1, m4, m4/m1, slow, fast, slow/fast
  } else { exit 1 }
}'
