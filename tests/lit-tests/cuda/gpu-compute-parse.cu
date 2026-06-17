// P2: #pragma sst gpu_compute parses and is consumed (deleteOnUse).

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: ls %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.gpu-compute-parse.cu

#include <cuda_runtime.h>

#pragma sst gpu_compute flops(2) intops(4) read(8) write(8)
__global__ void k(int* p) { *p = 0; }

int main() {
  int* d;
  cudaMalloc((void**)&d, 4);
  k<<<1, 1>>>(d);
  cudaFree(d);
  return 0;
}

// The pragma text is consumed, not passed through:
// CHECK-NOT: pragma sst gpu_compute
