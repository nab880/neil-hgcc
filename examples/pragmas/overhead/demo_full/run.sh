#!/bin/sh
set -eu

SST_BIN="${SST_BIN:-sst}"
EXAMPLE_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$EXAMPLE_DIR"
"$SST_BIN" "$EXAMPLE_DIR/overhead_app.py" 2>&1 | tee out.log
grep -q "paid launch overhead" out.log
grep -q "Simulation is complete" out.log
# The app1.launch_overhead of 10us must be charged once: about 11.3 us in
# all, against about 1.3 us without it and 21.3 us if charged twice.
t=$(sed -n 's/.*simulated time: \([0-9.]*\) us.*/\1/p' out.log)
[ -n "$t" ] || { echo "run.sh: no simulated time in us in out.log" >&2; exit 1; }
awk -v t="$t" 'BEGIN { exit !(t >= 10 && t < 20) }' ||
  { echo "run.sh: expected 10 to 20 us, got $t us" >&2; exit 1; }
