// P2: cudaGetLastError and cudaPeekAtLastError stubs compile and link.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: ls %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.error-query.cu

#include <cuda_runtime.h>

#pragma sst gpu_compute flops(1)
__global__ void k(int* p) { *p = 0; }

int main() {
  int* d;
  cudaMalloc((void**)&d, 4);
  k<<<1, 1>>>(d);
  // Both error-query stubs must be reachable (many real programs call one or both).
  cudaError_t e1 = cudaGetLastError();
  cudaError_t e2 = cudaPeekAtLastError();
  (void)e1; (void)e2;
  cudaFree(d);
  return 0;
}

// Both calls survive into the preprocessed output:
// CHECK: cudaGetLastError
// CHECK: cudaPeekAtLastError
