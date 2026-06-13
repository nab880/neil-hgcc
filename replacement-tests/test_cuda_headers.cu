// Standalone smoke source for the P2 Track 1 CUDA replacement headers. A
// kernel-free TU (no <<<, so it compiles through host -c -- launch rewriting
// is Track 2) that exercises the shipped replacement <cuda_runtime.h> and its
// runtime-API -> sst_hg_cuda ABI lowering. Link + run-under-SST awaits the
// Mercury GPU library (Phase 2 Track 3).
#include <cuda_runtime.h>
#include <skeleton.h>
#define ssthg_app_name test_cuda_headers

int main(int argc, char* argv[]) {
  const int n = 1024;
  const unsigned bytes = n * sizeof(float);

  float* d = 0;
  cudaMalloc((void**)&d, bytes);

  float* h = new float[n];
  cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);

  cudaStream_t s;
  cudaStreamCreate(&s);
  cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s);
  cudaStreamSynchronize(s);
  cudaStreamDestroy(s);

  cudaDeviceSynchronize();
  cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);
  cudaFree(d);
  delete[] h;
  return 0;
}
