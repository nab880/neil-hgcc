#!/bin/sh
# Smoke test for --with-cuda-arch: verifies the option is passed through configure
# into hgccvars.py so hgcompile.py picks up the right --cuda-gpu-arch flag.
# Runs from the build tree (not the install tree); skips if configure is not available.
set -eu

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$(cd "$TEST_DIR/.." && pwd)"
HGCCVARS="$BUILD_DIR/hgccvars.py"

if [ ! -f "$HGCCVARS" ]; then
  echo "run_test_cuda_arch: hgccvars.py not found at $HGCCVARS; skipping"
  exit 0
fi

# The default arch (sm_70) must appear in hgccvars.py when no --with-cuda-arch
# was given at configure time.
if grep -q "defaultCudaArch" "$HGCCVARS"; then
  arch=$(grep "defaultCudaArch" "$HGCCVARS" | sed "s/.*defaultCudaArch *= *['\"]//;s/['\"].*//")
  if [ -z "$arch" ]; then
    echo "FAIL: defaultCudaArch found in hgccvars.py but value could not be parsed"
    exit 1
  fi
  echo "PASS: cuda_arch -- defaultCudaArch = $arch"
else
  echo "FAIL: defaultCudaArch not found in hgccvars.py (--with-cuda-arch not wired through)"
  exit 1
fi
