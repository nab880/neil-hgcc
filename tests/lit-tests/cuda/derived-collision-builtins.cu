// P4: kernel parameters named after the launch-rewrite's synthetic builtins
// (gridDim/blockDim/threadIdx/blockIdx) must still compile. When a parameter
// shadows a builtin, the rewrite suppresses that builtin's synthetic dim3 decl
// and lets the parameter binding supply the name. Exercises the
// paramNames.count(...) suppression branch for all four builtins at once.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: ls %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.derived-collision-builtins.cu

#include <cuda_runtime.h>

// Straight-line body so a cost is derived (the lambda path is what binds these).
__global__ void k(int* out, int gridDim, int blockDim,
                  int threadIdx, int blockIdx) {
  out[0] = gridDim + blockDim + threadIdx + blockIdx;
}

int main() {
  int* d;
  cudaMalloc((void**)&d, sizeof(int));
  k<<<1, 1>>>(d, 1, 2, 3, 4);
  cudaDeviceSynchronize();
  cudaFree(d);
  return 0;
}

// Every synthetic builtin decl is suppressed because a param shadows it; the
// param bindings supply the names instead:
// CHECK-NOT: dim3 gridDim = __sst_g
// CHECK-NOT: dim3 blockDim = __sst_b
// CHECK-NOT: dim3 threadIdx(
// CHECK-NOT: dim3 blockIdx(
// CHECK: auto gridDim = __sst_arg0_1
// CHECK: sst_hg_cuda_launch("_Z1kPiiiii"
