#!/bin/sh
# Smoke test for the AI-at-scale demo (examples/mercury_llm_train): a data-parallel
# transformer training step. Runs the actual example at 2 ranks in four configs and
# asserts the pipeline's defining behaviors hold -- the step completes, the gradient
# all-reduce responds to fabric bandwidth, compute/comms overlap helps, and GPUDirect
# helps once the network is fast. Small, fast invariants (not the full README sweep).
set -eu

SST_BIN="${SST_BIN:-sst}"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_PY="$TEST_DIR/../examples/mercury_llm_train/train.py"
cd "$TEST_DIR"

# Simulated time (seconds) from an SST run log, normalizing the unit.
sim_seconds() {
  grep "simulated time:" "$1" \
    | sed -n 's/.*simulated time: \([0-9.eE+-]*\) \([a-z]*\).*/\1 \2/p' | head -1 \
    | awk '{ u=$2;
        m=(u=="s")?1:(u=="ms")?1e-3:(u=="us")?1e-6:(u=="ns")?1e-9:(u=="ps")?1e-12:1;
        printf "%.12g", $1*m }'
}

# run <log> <gpu_direct> <overlap> <link_bw>
run() {
  NRANKS=2 GPUDIRECT="$2" LLM_OVERLAP="$3" LINK_BW="$4" \
    "$SST_BIN" "$TRAIN_PY" > "$1" 2>&1
  grep -q "mercury_llm_train:.*layers done" "$1" \
    || { echo "FAIL: step did not complete ($1)"; cat "$1"; exit 1; }
}

run out_llm_slow_ov.log   true  1 12GB/s     # comms-bound, overlapped
run out_llm_slow_noov.log true  0 12GB/s     # comms-bound, no overlap
run out_llm_fast_gd.log   true  1 300GB/s    # compute-bound, GPUDirect
run out_llm_fast_nogd.log false 1 300GB/s    # compute-bound, PCIe staging

slow_ov=$(sim_seconds out_llm_slow_ov.log)
slow_noov=$(sim_seconds out_llm_slow_noov.log)
fast_gd=$(sim_seconds out_llm_fast_gd.log)
fast_nogd=$(sim_seconds out_llm_fast_nogd.log)
test -n "$slow_ov" && test -n "$slow_noov" && test -n "$fast_gd" && test -n "$fast_nogd"

awk -v slow="$slow_ov" -v fast="$fast_gd" -v noov="$slow_noov" -v nogd="$fast_nogd" 'BEGIN{
  ok = 1
  # (1) gradient all-reduce responds to fabric bandwidth (slow >> fast)
  if (!(slow > 1.5 * fast)) { printf "FAIL: fabric: 12GB/s (%g s) not > 1.5x 300GB/s (%g s)\n", slow, fast; ok = 0 }
  # (2) compute/comms overlap helps (no-overlap slower than overlap)
  if (!(noov > slow))       { printf "FAIL: overlap: no-overlap (%g s) not slower than overlap (%g s)\n", noov, slow; ok = 0 }
  # (3) GPUDirect helps once the fabric is fast (PCIe staging exposed without it)
  if (!(nogd > fast))       { printf "FAIL: gpu_direct: off (%g s) not slower than on (%g s)\n", nogd, fast; ok = 0 }
  if (ok) {
    printf "PASS: llm_train smoke -- fabric %g->%g s (%.1fx), overlap saves %g s, GPUDirect saves %g s\n",
           slow, fast, slow/fast, noov-slow, nogd-fast
  } else { exit 1 }
}'
