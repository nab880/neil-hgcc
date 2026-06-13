// P2 Track 1: each replacement header is self-contained -- including the
// sub-headers directly (not via the cuda_runtime.h umbrella) must parse through
// preprocess + ssthg_clang, so none silently relies on umbrella include
// ordering. The cuda/ replacement dir is auto-added on the include path for any
// .cu. We assert the ssthg_clang intermediate is produced (parse succeeded) and
// no replacement header was missing. (Full host -c is a Track 2 concern; see
// headers-positive.cu.)

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -ferror-limit=100 -o %t.d/out.o > %t.d/log 2>&1 || true
// RUN: ls %t.d/sst.pp.headers-parse.cu
// RUN: %FileCheck %s --check-prefix=NOWARN --input-file=%t.d/log

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
