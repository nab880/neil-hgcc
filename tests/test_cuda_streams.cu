// Two streams overlap; cudaEvents measure overlap vs serial execution.

#define ssthg_app_name test_cuda_streams
#include <skeleton.h>

#include <cuda_runtime.h>
#include <cstdio>

// Memory-bound kernels so kernel time dominates launch overhead.
#pragma sst gpu_compute read(64) write(64)
__global__ void kA(float* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; a[i] += 1.0f; }
#pragma sst gpu_compute read(64) write(64)
__global__ void kB(float* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; a[i] += 2.0f; }

int main(int /*argc*/, char** /*argv*/) {
  const int n = 1 << 20, t = 256, b = (n + t - 1) / t;
  float *da, *db;
  cudaMalloc((void**)&da, n * sizeof(float));
  cudaMalloc((void**)&db, n * sizeof(float));
  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);
  cudaEvent_t a0, a1, b0, b1;
  cudaEventCreate(&a0); cudaEventCreate(&a1);
  cudaEventCreate(&b0); cudaEventCreate(&b1);

  cudaEventRecord(a0, 0);
  kA<<<b, t, 0, s1>>>(da);
  kB<<<b, t, 0, s2>>>(db);
  cudaEventRecord(a1, 0);
  cudaDeviceSynchronize();

  cudaEventRecord(b0, 0);
  kA<<<b, t, 0, s1>>>(da);
  kB<<<b, t, 0, s1>>>(db);
  cudaEventRecord(b1, 0);
  cudaDeviceSynchronize();

  float overlap_ms = 0.0f, serial_ms = 0.0f;
  cudaEventElapsedTime(&overlap_ms, a0, a1);
  cudaEventElapsedTime(&serial_ms, b0, b1);
  std::printf("test_cuda_streams: overlap_ms=%f serial_ms=%f\n", overlap_ms, serial_ms);
  std::printf("test_cuda_streams: done\n");

  cudaFree(da);
  cudaFree(db);
  return 0;
}
