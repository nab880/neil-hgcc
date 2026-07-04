// End-to-end vecadd demo: hg++ launch rewrite + GpuLibrary modeled time.

#define ssthg_app_name test_cuda_vecadd
#include <skeleton.h>

#include <cuda_runtime.h>
#include <cstdio>

#pragma sst gpu_compute flops(2) intops(1) read(8) write(4)
__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main(int /*argc*/, char** /*argv*/) {
  const int n = 1024;
  const size_t bytes = n * sizeof(float);

  float ha[n];
  float hb[n];
  float hc[n];

  float *da = nullptr, *db = nullptr, *dc = nullptr;
  cudaMalloc((void**)&da, bytes);
  cudaMalloc((void**)&db, bytes);
  cudaMalloc((void**)&dc, bytes);

  cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  vecAdd<<<blocks, threads>>>(da, db, dc, n);

  cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);

  std::printf("test_cuda_vecadd: done\n");
  return 0;
}
