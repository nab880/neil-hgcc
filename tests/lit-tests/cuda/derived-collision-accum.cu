// P4: kernel parameters named after the cost accumulators (intops/readBytes/
// writeBytes) must still compile. The derived accumulation declares its own
// accumulators of those names in a nested block that shadows -- rather than
// redeclares -- the like-named parameter bindings. Regression pin for the
// shadow-block invariant beyond derived-param-collision.cu's gridDim/flops case.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: ls %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.derived-collision-accum.cu

#include <cuda_runtime.h>

// Straight-line body so a cost is actually derived (exercises the lambda path).
__global__ void k(float* out, int intops, int readBytes, int writeBytes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = intops + readBytes + writeBytes;
}

int main() {
  const int n = 1 << 16;
  float* d;
  cudaMalloc((void**)&d, n*sizeof(float));
  k<<<(n+255)/256, 256>>>(d, 1, 2, 3);
  cudaDeviceSynchronize();
  cudaFree(d);
  return 0;
}

// The cost lambda is emitted and the params are bound; if the accumulators were
// declared in the same scope as the params (not a nested shadowing block) the
// generated code would redeclare and host compile would fail (out.o absent).
// CHECK: __sst_cost_0 = [&] {
// CHECK: auto intops = __sst_arg0_1
// CHECK: auto readBytes = __sst_arg0_2
// CHECK: auto writeBytes = __sst_arg0_3
// CHECK: sst_hg_cuda_launch("_Z1kPfiii"
