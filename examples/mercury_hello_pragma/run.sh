#!/bin/sh
set -eu

SST_BIN="${SST_BIN:-sst}"
EXAMPLE_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$EXAMPLE_DIR"
"$SST_BIN" "$EXAMPLE_DIR/hello_pragma.py" 2>&1 | tee out.log
grep -q "Hello from rank" out.log
grep -q "Simulation is complete" out.log
