// P2: __host__ __device__ function bodies are NOT stripped (only __global__ bodies are).

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: ls %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.host-device-exempt.cu

#include <cuda_runtime.h>

// This function is callable from both host and device; its body must survive
// the rewrite unchanged so the host caller can link against it.
__host__ __device__ float clamp01(float x) {
  return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}

#pragma sst gpu_compute flops(4) intops(0) read(4) write(4)
__global__ void scale(float* data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = clamp01(data[i]);
}

int main() {
  float* d;
  cudaMalloc((void**)&d, 16);
  scale<<<1, 16>>>(d, 16);
  cudaDeviceSynchronize();
  cudaFree(d);
  return 0;
}

// The qualifier macros pretty-print as their expanded __attribute__ spelling.

// __host__ __device__ body is left intact (ternary expression survives):
// CHECK: __attribute__((host)) __attribute__((device)) float clamp01(float x) {
// CHECK: return x < 0.0f

// __global__ body is stripped to {}:
// CHECK: __attribute__((global)) void scale(float *data, int n) {}
