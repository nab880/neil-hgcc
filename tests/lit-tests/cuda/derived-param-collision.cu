// P4: kernel params whose names collide with the launch-rewrite's synthetic
// locals (gridDim/blockDim/threadIdx/blockIdx) or the cost accumulators
// (flops/intops/readBytes/writeBytes) must not produce redeclaration errors in
// the generated launch code. The cost is confined to a lambda; a param that
// shadows a builtin suppresses that builtin's synthetic decl.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: ls %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.derived-param-collision.cu

#include <cuda_runtime.h>

// "gridDim" shadows a synthetic builtin; "flops" collides with a cost
// accumulator. Straight-line body so cost is actually derived.
__global__ void collide(float* out, int gridDim, float flops, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = flops + gridDim + n;
}

int main() {
  const int n = 1 << 16;
  float* d;
  cudaMalloc((void**)&d, n*sizeof(float));
  collide<<<(n+255)/256, 256>>>(d, 4, 1.5f, n);
  cudaDeviceSynchronize();
  cudaFree(d);
  return 0;
}

// The cost lambda is emitted; the "gridDim" builtin is suppressed (param wins),
// and the accumulators are hoisted through unique names so nothing redeclares:
// CHECK: __sst_cost_0 = [&] {
// CHECK-NOT: dim3 gridDim = __sst_g_0
// CHECK: auto gridDim = __sst_arg0_1
// CHECK: auto flops = __sst_arg0_2
// CHECK: sst_hg_cuda_launch("_Z7collidePfifi"
