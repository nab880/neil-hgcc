#!/bin/sh
# Smoke test for the FSDP / ZeRO-3 demo (examples/mercury_fsdp): per-layer
# all-gather + reduce-scatter with optional prefetch. Asserts the step completes
# in both prefetch modes at 2 ranks. The prefetch benefit only emerges at larger
# layer counts / slower fabrics than this smoke test exercises, so we check
# completion only rather than asserting a timing direction.
set -eu

SST_BIN="${SST_BIN:-sst}"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
FSDP_PY="$TEST_DIR/../examples/mercury_fsdp/fsdp.py"
cd "$TEST_DIR"

# run <log> <NRANKS> <FSDP_PREFETCH> <LAYERS> <LINK_BW>
run() {
  NRANKS="$2" FSDP_PREFETCH="$3" LAYERS="$4" LINK_BW="$5" \
    "$SST_BIN" "$FSDP_PY" > "$1" 2>&1
  grep -q "mercury_fsdp:.*done" "$1" \
    || { echo "FAIL: step did not complete ($1)"; cat "$1"; exit 1; }
}

run out_fsdp_prefetch.log    2 1 4 12GB/s
run out_fsdp_noprefetch.log  2 0 4 12GB/s

echo "PASS: fsdp smoke -- both prefetch and no-prefetch configs completed"
