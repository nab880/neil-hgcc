#!/bin/sh
# Run the CUDA vecadd skeleton under SST and assert it simulated: the app
# finished and the GpuLibrary stub reported a nonzero total GPU time. The
# library is built/installed by sst-elements (mercury/libraries/gpu).
set -eu

SST_BIN="${SST_BIN:-sst}"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$TEST_DIR"
"$SST_BIN" "$TEST_DIR/test_cuda_vecadd.py" 2>&1 | tee out_cuda.log

# The app ran to completion ...
grep -q "test_cuda_vecadd: done" out_cuda.log

# ... and the GPU stub charged a nonzero, finite amount of modeled time
# (the summary line is "[gpu] rank summary: ... total_gpu_time=T s").
grep -q "\[gpu\] rank summary:" out_cuda.log
total=$(grep "\[gpu\] rank summary:" out_cuda.log | sed -n 's/.*total_gpu_time=\([0-9.eE+-]*\) s.*/\1/p' | head -1)
test -n "$total"
# Reject zero / all-zero (e.g. 0, 0.0, 0e+00).
case "$total" in
  0|0.|0.0|0.0e+00|0e+00|.0) echo "FAIL: total_gpu_time is zero ($total)"; exit 1 ;;
esac
echo "PASS: total_gpu_time=$total s"
