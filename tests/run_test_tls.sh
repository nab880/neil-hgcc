#!/bin/sh
set -eu

SST_BIN="${SST_BIN:-sst}"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$TEST_DIR"
"$SST_BIN" "$TEST_DIR/test_tls.py" 2>&1 | tee out.log
grep -q "my_global: 1" out.log
