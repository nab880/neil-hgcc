// P2 Track 1+2 deliverable: the shipped replacement <cuda_runtime.h> resolves
// for any .cu (auto-added, no -I needed) and lowers the CUDA runtime API to the
// sst_hg_cuda ABI. As of Track 2 the host -c runs in CUDA host-only mode and the
// rewriter skips device code, so this kernel-free TU now compiles ALL THE WAY to
// an object. We assert the object is produced and the sst.pp carries the ABI
// calls with no missing-replacement warning.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: ls %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.headers-positive.cu
// RUN: %FileCheck %s --check-prefix=NOWARN --input-file=%t.d/log

#include <cuda_runtime.h>

int main() {
  float* d = 0;
  float h[4] = {0, 0, 0, 0};
  cudaMalloc((void**)&d, sizeof(h));
  cudaMemcpy(d, h, sizeof(h), cudaMemcpyHostToDevice);

  cudaStream_t s;
  cudaStreamCreate(&s);
  cudaMemcpyAsync(d, h, sizeof(h), cudaMemcpyHostToDevice, s);
  cudaStreamSynchronize(s);
  cudaStreamDestroy(s);

  cudaDeviceSynchronize();
  cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
  cudaFree(d);
  return 0;
}

// The inline wrappers' ABI calls reach the rewriter intermediate:
// CHECK-DAG: sst_hg_cuda_malloc
// CHECK-DAG: sst_hg_cuda_memcpy
// CHECK-DAG: sst_hg_cuda_stream_create
// CHECK-DAG: sst_hg_cuda_device_sync

// The replacement headers resolved (no missing-replacement warning):
// NOWARN-NOT: Replacement header
