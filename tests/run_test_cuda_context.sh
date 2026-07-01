#!/bin/sh
# Smoke test for the context parallelism / ring attention demo
# (examples/mercury_context): K/V blocks rotate around a ring while attention
# computes locally. Asserts the step completes and that ring overlap (CP_OVERLAP)
# reduces time on a slow fabric where the K/V transfer is otherwise exposed.
set -eu

SST_BIN="${SST_BIN:-sst}"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
CTX_PY="$TEST_DIR/../examples/mercury_context/ring_attn.py"
cd "$TEST_DIR"

sim_seconds() {
  grep "simulated time:" "$1" \
    | sed -n 's/.*simulated time: \([0-9.eE+-]*\) \([a-z]*\).*/\1 \2/p' | head -1 \
    | awk '{ u=$2;
        m=(u=="s")?1:(u=="ms")?1e-3:(u=="us")?1e-6:(u=="ns")?1e-9:(u=="ps")?1e-12:1;
        printf "%.12g", $1*m }'
}

# run <log> <NRANKS> <CP_OVERLAP> <LAYERS> <LINK_BW>
run() {
  NRANKS="$2" CP_OVERLAP="$3" LAYERS="$4" LINK_BW="$5" \
    "$SST_BIN" "$CTX_PY" > "$1" 2>&1
  grep -q "mercury_context:.*done" "$1" \
    || { echo "FAIL: step did not complete ($1)"; cat "$1"; exit 1; }
}

run out_ctx_overlap.log    4 1 2 12GB/s   # slow fabric, overlap on
run out_ctx_nooverlap.log  4 0 2 12GB/s   # slow fabric, overlap off

t_ov=$(sim_seconds out_ctx_overlap.log)
t_nov=$(sim_seconds out_ctx_nooverlap.log)
test -n "$t_ov" && test -n "$t_nov"

awk -v ov="$t_ov" -v nov="$t_nov" 'BEGIN{
  ok = 1
  # Overlap hides K/V transfer behind attention compute on a slow fabric.
  if (!(ov < nov)) {
    printf "FAIL: overlap (%g s) not < no-overlap (%g s)\n", ov, nov; ok = 0
  }
  if (ok) {
    printf "PASS: context smoke -- overlap %g s < no-overlap %g s (%.2fx speedup)\n",
           ov, nov, nov/ov
  } else { exit 1 }
}'
