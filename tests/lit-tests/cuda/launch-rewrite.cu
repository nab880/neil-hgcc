// P2 golden: <<<>>> + gpu_compute lowers to sst_hg_cuda_launch; kernel body stripped.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: ls %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.launch-rewrite.cu

#include <cuda_runtime.h>

#pragma sst gpu_compute flops(1) intops(3) read(8) write(4)
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

// Kernel body stripped to {} (no device statements survive):
// CHECK: void vecAdd(
// CHECK: int n) {}

// Launch lowered to the ABI with dim3 temps and the pragma's costs (f=1,i=3,r=8,w=4):
// CHECK: dim3 __sst_g_0 = (grid);
// CHECK: dim3 __sst_b_0 = (block);
// CHECK: sst_hg_cuda_launch("_Z6vecAddPKfS0_Pfi", __sst_g_0.x
// CHECK: 1, 3, 8, 4);
// CHECK-NOT: __device_stub__vecAdd
