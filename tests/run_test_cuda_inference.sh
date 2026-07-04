#!/bin/sh
# Smoke test for the disaggregated inference demo (examples/mercury_inference):
# prefill workers + decode pool with KV-cache transfer. Asserts the step completes
# in disaggregated mode, and that decode is latency-bound: doubling the number of
# decode steps doubles the simulated time.
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

run out_infer_s16.log  4 1 16  100GB/s   # disaggregated, 16 decode steps
run out_infer_s32.log  4 1 32  100GB/s   # same config, 32 decode steps

t_s16=$(sim_seconds out_infer_s16.log)
t_s32=$(sim_seconds out_infer_s32.log)
test -n "$t_s16" && test -n "$t_s32"

awk -v s16="$t_s16" -v s32="$t_s32" 'BEGIN{
  ok = 1
  # Decode is latency-bound: doubling steps should roughly double time (1.8-2.2x).
  ratio = s32 / s16
  if (!(ratio > 1.8 && ratio < 2.2)) {
    printf "FAIL: 2x steps ratio %.2f not in [1.8, 2.2] (s16=%g s, s32=%g s)\n",
           ratio, s16, s32; ok = 0
  }
  if (ok) {
    printf "PASS: inference smoke -- 16 steps %g s, 32 steps %g s (%.2fx, decode latency-bound)\n",
           s16, s32, ratio
  } else { exit 1 }
}'
