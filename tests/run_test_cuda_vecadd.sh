#!/bin/sh
# Run the CUDA vecadd skeleton under SST and assert the model is live: the app
# finishes with a nonzero modeled GPU time, the time responds to a platform
# bandwidth knob (roofline), and a calibration-table entry overrides it. The GPU
# library is built/installed by sst-elements (mercury/libraries/gpu).
set -eu

SST_BIN="${SST_BIN:-sst}"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$TEST_DIR"

gpu_time() { # total modeled GPU seconds from a run log
  grep "\[gpu\] rank summary:" "$1" \
    | sed -n 's/.*total_gpu_time=\([0-9.eE+-]*\) s.*/\1/p' | head -1
}

# 1) Baseline: the app runs to completion with nonzero modeled GPU time.
"$SST_BIN" "$TEST_DIR/test_cuda_vecadd.py" 2>&1 | tee out_cuda.log
grep -q "test_cuda_vecadd: done" out_cuda.log
base=$(gpu_time out_cuda.log)
test -n "$base"
case "$base" in 0|0.|0.0|0.0e+00|0e+00|.0) echo "FAIL: total_gpu_time is zero"; exit 1 ;; esac

# 2) Roofline responds to bandwidth: a lower gpu_mem_bandwidth makes the
#    memory-bound kernel term (and the total) larger.
GPU_MEM_BANDWIDTH=2GB/s "$SST_BIN" "$TEST_DIR/test_cuda_vecadd.py" > out_bw_hi.log 2>&1
GPU_MEM_BANDWIDTH=1GB/s "$SST_BIN" "$TEST_DIR/test_cuda_vecadd.py" > out_bw_lo.log 2>&1
hi=$(gpu_time out_bw_hi.log); lo=$(gpu_time out_bw_lo.log)
awk -v hi="$hi" -v lo="$lo" 'BEGIN{
  if (!(lo > hi)) { printf "FAIL: 1GB/s (%g s) not slower than 2GB/s (%g s)\n", lo, hi; exit 1 }
  printf "PASS: roofline responds to bandwidth (2GB/s=%g s < 1GB/s=%g s)\n", hi, lo }'

# 3) Calibration overrides the roofline: the table pins vecAdd to 0.25 s, far
#    above the roofline baseline.
GPU_KERNEL_TIMES="$TEST_DIR/cuda_calibration.json" \
  "$SST_BIN" "$TEST_DIR/test_cuda_vecadd.py" > out_cal.log 2>&1
cal=$(gpu_time out_cal.log)
awk -v cal="$cal" -v base="$base" 'BEGIN{
  if (!(cal > base*10)) { printf "FAIL: calibrated (%g s) did not override roofline (%g s)\n", cal, base; exit 1 }
  printf "PASS: calibration overrides roofline (calibrated=%g s vs roofline=%g s)\n", cal, base }'
