// P2: launch rewrite runs in ENCAPSULATE mode (SST_HG_SKELETONIZE=0), not only skeletonize.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: env SST_HG_SKELETONIZE=0 %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: ls %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.launch-encapsulate.cu

#include <cuda_runtime.h>

#pragma sst gpu_compute flops(2) intops(1) read(8) write(4)
__global__ void vecAdd(const float* a, const float* b, float* c, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 16;
  const unsigned bytes = n * sizeof(float);
  float* da; float* db; float* dc;
  cudaMalloc((void**)&da, bytes);
  cudaMalloc((void**)&db, bytes);
  cudaMalloc((void**)&dc, bytes);
  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  vecAdd<<<grid, block>>>(da, db, dc, n);
  cudaDeviceSynchronize();
  cudaFree(da); cudaFree(db); cudaFree(dc);
  return 0;
}

// Even in ENCAPSULATE mode, the body is stripped and the launch is lowered with
// the pragma's costs (f=2,i=1,r=8,w=4) -- the pragma is active in every mode too.
// CHECK: int n) {}
// CHECK: dim3 __sst_g_0 = (grid);
// CHECK: sst_hg_cuda_launch("_Z6vecAddPKfS0_Pfi", __sst_g_0.x
// CHECK: 2, 1, 8, 4);
