// P4: loop kernels fall back to zero work + warning (straight-line derivation only).

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: ls %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.derived-loop-fallback.cu
// RUN: %FileCheck %s --check-prefix=WARN --input-file=%t.d/log

#include <cuda_runtime.h>

__global__ void scale(float* a, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    a[i] *= 2.0f;
  }
}

int main() {
  const int n = 1 << 20;
  float* da;
  cudaMalloc((void**)&da, n*sizeof(float));
  scale<<<256, 256>>>(da, n);
  cudaDeviceSynchronize();
  cudaFree(da);
  return 0;
}

// No derivation (no bound builtins, no accumulation); the launch is zeroed:
// CHECK-NOT: dim3 gridDim =
// CHECK: sst_hg_cuda_launch("_Z5scalePfi"
// CHECK: 0, 0, 0, 0);

// WARN: cost could not be derived
