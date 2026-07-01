#!/bin/bash
# Standalone smoke check for the P2 Track 1 CUDA replacement headers (off the
# CI path; tests/lit-tests/cuda is the gating version). Drives a kernel-free .cu
# through the real compiler, exercising the shipped replacement <cuda_runtime.h>
# and its runtime-API -> sst_hg_cuda ABI lowering. The cuda/ replacements
# auto-add for any .cu, so no --replacements flag is needed.
#
# NOTE: this only verifies the preprocess + ssthg_clang result (the ABI lowering
# in the sst.pp intermediate). The host -c does not yet complete for a
# skeleton.h-laden CUDA TU -- CUDA-mode preprocessing bakes in clang's
# cuda_wrappers/ libc++ shims that the host -c's -x c++ mode rejects, a Track 2
# integration concern. Link + run-under-SST additionally awaits the Mercury GPU
# library (Phase 2 Track 3).
set -e

SST_HG_DELETE_TEMP_SOURCES=0 hg++ -c test_cuda_headers.cu -o test_cuda_headers.o || true

pp=sst.pp.test_cuda_headers.cu
if [ -f "$pp" ] && grep -q sst_hg_cuda_ "$pp"; then
  echo "OK: replacement <cuda_runtime.h> lowered the runtime API to the"
  echo "    sst_hg_cuda ABI (symbols in $pp):"
  grep -oE "sst_hg_cuda_[a-z_]+" "$pp" | sort -u | sed 's/^/      /'
else
  echo "FAIL: ssthg_clang intermediate missing or no ABI lowering" >&2
  exit 1
fi
