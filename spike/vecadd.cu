/* Phase-0 vecadd spike: kernel, launch, and runtime surface for the rewriter. */
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void vecAdd(const float* a, const float* b, float* c, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main(int argc, char** argv)
{
  const int n = 1 << 16;
  const size_t bytes = n * sizeof(float);

  float* ha = (float*)malloc(bytes);
  float* hb = (float*)malloc(bytes);
  float* hc = (float*)malloc(bytes);
  for (int i = 0; i < n; ++i) {
    ha[i] = (float)i;
    hb[i] = (float)(n - i);
  }

  float* da;
  float* db;
  float* dc;
  cudaMalloc((void**)&da, bytes);
  cudaMalloc((void**)&db, bytes);
  cudaMalloc((void**)&dc, bytes);

  cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  vecAdd<<<grid, block>>>(da, db, dc, n);
  cudaDeviceSynchronize();

  cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);

  printf("vecadd done n=%d c[0]=%f\n", n, hc[0]);
  free(ha);
  free(hb);
  free(hc);
  return 0;
}
