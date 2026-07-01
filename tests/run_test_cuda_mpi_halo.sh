#!/bin/sh
# Acceptance test (CUDA_PLAN §1): a device-buffer halo exchange under SST. Run
# the gpu_direct sweep and assert (a) every rank finishes both runs, and (b)
# turning gpu_direct on shortens the time-to-solution -- the PCIe staging of the
# device send/recv buffers is the cost it removes. NRANKS (2-8) is swept too.
set -eu

SST_BIN="${SST_BIN:-sst}"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
NRANKS="${NRANKS:-2}"
cd "$TEST_DIR"

# Simulated time (seconds) from an SST run log, normalizing the unit.
sim_seconds() {
  grep "simulated time:" "$1" \
    | sed -n 's/.*simulated time: \([0-9.eE+-]*\) \([a-z]*\).*/\1 \2/p' | head -1 \
    | awk '{ u=$2;
        m=(u=="s")?1:(u=="ms")?1e-3:(u=="us")?1e-6:(u=="ns")?1e-9:(u=="ps")?1e-12:1;
        printf "%.12g", $1*m }'
}

NRANKS="$NRANKS" GPUDIRECT=false "$SST_BIN" "$TEST_DIR/test_cuda_mpi_halo.py" > out_halo_off.log 2>&1
NRANKS="$NRANKS" GPUDIRECT=true  "$SST_BIN" "$TEST_DIR/test_cuda_mpi_halo.py" > out_halo_on.log 2>&1
cat out_halo_off.log

# (a) every rank finished in both runs
done_off=$(grep -c "test_cuda_mpi_halo: rank .* done" out_halo_off.log || true)
done_on=$(grep -c "test_cuda_mpi_halo: rank .* done" out_halo_on.log || true)
test "$done_off" -eq "$NRANKS" || { echo "FAIL: gpu_direct=off, $done_off/$NRANKS ranks done"; exit 1; }
test "$done_on"  -eq "$NRANKS" || { echo "FAIL: gpu_direct=on, $done_on/$NRANKS ranks done"; exit 1; }

# (b) gpu_direct on is faster (no PCIe staging on the critical path)
off=$(sim_seconds out_halo_off.log)
on=$(sim_seconds out_halo_on.log)
test -n "$off" && test -n "$on"
awk -v off="$off" -v on="$on" 'BEGIN{
  if (off <= on) { printf "FAIL: gpu_direct off (%g s) not slower than on (%g s)\n", off, on; exit 1 }
  printf "PASS: %d ranks; time-to-solution off=%g s > on=%g s (GPUDirect saves %g s)\n",
         '"$NRANKS"', off, on, off-on
}'
