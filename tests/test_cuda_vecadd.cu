// Phase 2 end-to-end demo: a textbook CUDA vecadd compiled with hg++ and run
// under SST/Mercury. The kernel never executes -- the rewriter lowers the
// <<<>>> launch to sst_hg_cuda_launch and the GpuLibrary stub charges modeled
// time -- so this exercises the whole chain: CUDA driver (P1), launch rewrite +
// gpu_compute pragma (P2 Track 2), and the elements stub library (P2 Track 3).

#define ssthg_app_name test_cuda_vecadd
#include <skeleton.h>

#include <cuda_runtime.h>
#include <cstdio>

// Per-thread costs are pinned by the pragma; the stub ignores them and charges
// a fixed gpu_kernel_time, but the rewriter still threads them into the launch.
#pragma sst gpu_compute flops(2) intops(1) read(8) write(4)
__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main(int /*argc*/, char** /*argv*/) {
  const int n = 1024;
  const size_t bytes = n * sizeof(float);

  // Stack host buffers: the "model don't execute" memcpy ignores the host
  // contents (it only charges modeled time), and skeletonize mode rewrites
  // heap malloc/free, so we avoid the heap here.
  float ha[n];
  float hc[n];

  float *da = nullptr, *db = nullptr, *dc = nullptr;
  cudaMalloc((void**)&da, bytes);
  cudaMalloc((void**)&db, bytes);
  cudaMalloc((void**)&dc, bytes);

  cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, ha, bytes, cudaMemcpyHostToDevice);

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
