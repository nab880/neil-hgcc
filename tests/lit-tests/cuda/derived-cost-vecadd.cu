// P4: per-thread F/I/R/W derived from kernel body; bounds-check guard assumed taken.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: ls %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.derived-cost-vecadd.cu

#include <cuda_runtime.h>

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 16;
  float* da; float* db; float* dc;
  cudaMalloc((void**)&da, n*sizeof(float));
  cudaMalloc((void**)&db, n*sizeof(float));
  cudaMalloc((void**)&dc, n*sizeof(float));
  vecAdd<<<(n+255)/256, 256>>>(da, db, dc, n);
  cudaDeviceSynchronize();
  cudaFree(da); cudaFree(db); cudaFree(dc);
  return 0;
}

// The builtins are bound inside the cost lambda, the params aliased, and the
// derived accumulation counts 2 loads + 1 store from the guarded body:
// CHECK: __sst_cost_0 = [&] {
// CHECK: dim3 gridDim = __sst_g_0;
// CHECK: dim3 threadIdx(0, 0, 0);
// CHECK: readBytes += tripCount0 * 8;
// CHECK: writeBytes += tripCount0 * 4;
// CHECK: sst_hg_cuda_launch("_Z6vecAddPKfS0_Pfi"
// CHECK: __sst_cost_0.f, __sst_cost_0.i
// CHECK: __sst_cost_0.r, __sst_cost_0.w);
