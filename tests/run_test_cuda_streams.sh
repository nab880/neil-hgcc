#!/bin/sh
# Two kernels on independent streams overlap; on one stream they serialize.
# Assert the app finished and the serialized phase took clearly longer than the
# overlapped one (the busy-until stream model gives real overlap).
set -eu

SST_BIN="${SST_BIN:-sst}"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$TEST_DIR"
"$SST_BIN" "$TEST_DIR/test_cuda_streams.py" 2>&1 | tee out_streams.log

grep -q "test_cuda_streams: done" out_streams.log

line=$(grep "overlap_ms=" out_streams.log | head -1)
overlap=$(echo "$line" | sed -n 's/.*overlap_ms=\([0-9.eE+-]*\).*/\1/p')
serial=$(echo "$line" | sed -n 's/.*serial_ms=\([0-9.eE+-]*\).*/\1/p')
test -n "$overlap" && test -n "$serial"

# serial should be ~2x overlap; require a clear separation (> 1.5x) and both > 0.
awk -v o="$overlap" -v s="$serial" 'BEGIN{
  if (o <= 0 || s <= 0) { print "FAIL: non-positive times"; exit 1 }
  if (s <= o*1.5)       { printf "FAIL: serial %g not > 1.5x overlap %g\n", s, o; exit 1 }
  printf "PASS: overlap=%g ms, serial=%g ms (%.2fx)\n", o, s, s/o
}'
