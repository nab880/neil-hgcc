// P2 Track 1: each replacement sub-header parses self-contained (no umbrella ordering).
// The compile-to-object step is expected to fail (no skeleton.h / SST runtime symbols),
// so we check only that the preprocessing pass produced output and logged no parse errors.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -ferror-limit=100 -o %t.d/out.o > %t.d/log 2>&1 || true
// RUN: ls %t.d/sst.pp.headers-parse.cu
// RUN: %FileCheck %s --check-prefix=NOWARN --input-file=%t.d/log
// RUN: %FileCheck %s --check-prefix=NOERR  --input-file=%t.d/log

#include <driver_types.h>
#include <vector_types.h>
#include <device_launch_parameters.h>
#include <hg_cuda_shims.h>

// Types from the headers above are usable together.
dim3 g_block(128);
cudaMemcpyKind g_kind = cudaMemcpyHostToDevice;
cudaStream_t g_stream = 0;

int uses_them() {
  return (int)g_block.x + (int)g_kind + (g_stream != 0);
}

// NOWARN-NOT: Replacement header
// NOERR-NOT: error:
