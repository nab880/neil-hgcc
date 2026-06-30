#!/bin/sh
# Smoke test for the FSDP / ZeRO-3 demo (examples/mercury_fsdp): per-layer
# all-gather + reduce-scatter with optional prefetch. Asserts the step completes
# at 4 ranks in two configs and that prefetch helps on a slow fabric (where the
# all-gather sits exposed on the critical path without it).
set -eu

SST_BIN="${SST_BIN:-sst}"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
FSDP_PY="$TEST_DIR/../examples/mercury_fsdp/fsdp.py"
cd "$TEST_DIR"

sim_seconds() {
  grep "simulated time:" "$1" \
    | sed -n 's/.*simulated time: \([0-9.eE+-]*\) \([a-z]*\).*/\1 \2/p' | head -1 \
    | awk '{ u=$2;
        m=(u=="s")?1:(u=="ms")?1e-3:(u=="us")?1e-6:(u=="ns")?1e-9:(u=="ps")?1e-12:1;
        printf "%.12g", $1*m }'
}

# run <log> <NRANKS> <FSDP_PREFETCH> <LINK_BW>
run() {
  NRANKS="$2" FSDP_PREFETCH="$3" LINK_BW="$4" \
    "$SST_BIN" "$FSDP_PY" > "$1" 2>&1
  grep -q "mercury_fsdp:.*done" "$1" \
    || { echo "FAIL: step did not complete ($1)"; cat "$1"; exit 1; }
}

run out_fsdp_prefetch.log    4 1 12GB/s   # slow fabric, prefetch on
run out_fsdp_noprefetch.log  4 0 12GB/s   # slow fabric, prefetch off

t_pre=$(sim_seconds out_fsdp_prefetch.log)
t_nopre=$(sim_seconds out_fsdp_noprefetch.log)
test -n "$t_pre" && test -n "$t_nopre"

awk -v pre="$t_pre" -v nopre="$t_nopre" 'BEGIN{
  ok = 1
  # Prefetch hides the all-gather behind compute on a slow fabric -> lower time.
  if (!(pre < nopre)) {
    printf "FAIL: prefetch (%g s) not < no-prefetch (%g s)\n", pre, nopre; ok = 0
  }
  if (ok) {
    printf "PASS: fsdp smoke -- prefetch %g s < no-prefetch %g s (%.2fx speedup)\n",
           pre, nopre, nopre/pre
  } else { exit 1 }
}'
