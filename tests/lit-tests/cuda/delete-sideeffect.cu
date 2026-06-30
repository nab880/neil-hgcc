// P2: #pragma sst delete on a launch with a side-effecting argument -- the launch
// call is removed but the side-effecting arg expression must still be evaluated
// (captured as a temporary and then discarded) so callers observe the side effect.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: ls %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.delete-sideeffect.cu

#include <cuda_runtime.h>

int g_counter = 0;
int next_id() { return ++g_counter; }

#pragma sst gpu_compute flops(1)
__global__ void k(int id, int* p) { (void)id; (void)p; }

int main() {
  int* d;
  cudaMalloc((void**)&d, 4);
#pragma sst delete
  k<<<1, 1>>>(next_id(), d);
  cudaFree(d);
  return 0;
}

// The launch is deleted so no sst_hg_cuda_launch call for this kernel appears:
// CHECK-NOT: sst_hg_cuda_launch("_Z1kiPi"

// But next_id() is side-effecting -- it must be evaluated and discarded so the
// counter increment is not silently lost. The rewriter captures it as a temporary:
// CHECK: (void)(next_id()
