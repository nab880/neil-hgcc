#!/bin/sh
set -eu

SST_BIN="${SST_BIN:-sst}"
EXAMPLE_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$EXAMPLE_DIR"
"$SST_BIN" "$EXAMPLE_DIR/compute_app.py" 2>&1 | tee out.log
grep -q "compute acc=" out.log
grep -q "Simulation is complete" out.log
# The compute pragma must charge the loop's work (about 0.48 us at 2.1 GHz):
# without it the run takes about 1.32 us.
t=$(sed -n 's/.*simulated time: \([0-9.]*\) us.*/\1/p' out.log)
[ -n "$t" ] || { echo "run.sh: no simulated time in us in out.log" >&2; exit 1; }
awk -v t="$t" 'BEGIN { exit !(t > 1.5 && t < 3) }' ||
  { echo "run.sh: expected 1.5 to 3 us, got $t us" >&2; exit 1; }
