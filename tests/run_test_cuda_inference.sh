#!/bin/sh
# Smoke test for the disaggregated inference demo (examples/mercury_inference):
# prefill workers + decode pool with KV-cache transfer. Asserts the step completes
# in disaggregated mode, and that decode is latency-bound: more decode steps cost
# proportionally more time (linear scaling with step count).
set -eu

SST_BIN="${SST_BIN:-sst}"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
INFER_PY="$TEST_DIR/../examples/mercury_inference/infer.py"
cd "$TEST_DIR"

sim_seconds() {
  grep "simulated time:" "$1" \
    | sed -n 's/.*simulated time: \([0-9.eE+-]*\) \([a-z]*\).*/\1 \2/p' | head -1 \
    | awk '{ u=$2;
        m=(u=="s")?1:(u=="ms")?1e-3:(u=="us")?1e-6:(u=="ns")?1e-9:(u=="ps")?1e-12:1;
        printf "%.12g", $1*m }'
}

# run <log> <NRANKS> <PREFILL_RANKS> <DECODE_STEPS> <LINK_BW>
run() {
  NRANKS="$2" PREFILL_RANKS="$3" DECODE_STEPS="$4" LINK_BW="$5" \
    "$SST_BIN" "$INFER_PY" > "$1" 2>&1
  grep -q "mercury_inference:.*done" "$1" \
    || { echo "FAIL: step did not complete ($1)"; cat "$1"; exit 1; }
}

run out_infer_s32.log  4 1 32  100GB/s   # disaggregated, 32 decode steps
run out_infer_s64.log  4 1 64  100GB/s   # same config, 64 decode steps

t_s32=$(sim_seconds out_infer_s32.log)
t_s64=$(sim_seconds out_infer_s64.log)
test -n "$t_s32" && test -n "$t_s64"

awk -v s32="$t_s32" -v s64="$t_s64" 'BEGIN{
  ok = 1
  # Decode is latency-bound: doubling steps should roughly double time.
  ratio = s64 / s32
  if (!(ratio > 1.5 && ratio < 3.0)) {
    printf "FAIL: 2x steps ratio %.2f not in [1.5, 3.0] (s32=%g s, s64=%g s)\n",
           ratio, s32, s64; ok = 0
  }
  if (ok) {
    printf "PASS: inference smoke -- 32 steps %g s, 64 steps %g s (%.2fx, decode latency-bound)\n",
           s32, s64, ratio
  } else { exit 1 }
}'
