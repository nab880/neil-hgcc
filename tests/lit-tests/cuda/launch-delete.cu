// P2: #pragma sst delete removes a launch entirely (no sst_hg_cuda_launch emitted).

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: ls %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.launch-delete.cu

#include <cuda_runtime.h>

#pragma sst gpu_compute flops(1)
__global__ void k(int* p) { *p = 0; }

int main() {
  int* d;
  cudaMalloc((void**)&d, 4);
#pragma sst delete
  k<<<1, 1>>>(d);
  cudaFree(d);
  return 0;
}

// The launch was deleted, so no launch call survives (only the ABI declaration
// from the header mentions sst_hg_cuda_launch as a prototype):
// CHECK-NOT: sst_hg_cuda_launch("_Z1kPi"
// CHECK-NOT: __sst_g_0
