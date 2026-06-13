// P2 Track 1 deliverable test: the shipped replacement <cuda_runtime.h>
// resolves for any .cu (auto-added, no -I needed) and lowers the CUDA runtime
// API to the sst_hg_cuda ABI via its inline wrappers. Proven by inspecting the
// ssthg_clang intermediate sst.pp.*: it carries the ABI calls, and no
// "Replacement header ... not found" warning is emitted.
//
// NOTE: this kernel-free TU does NOT yet compile all the way through host -c.
// CUDA-mode preprocessing bakes in clang's cuda_wrappers/ libc++ shims (pulled
// via the force-included skeleton.h -> <list>), which do not recompile under
// the host -c's -x c++ mode. That is a Track 2 integration concern (host
// compile must move to -x cuda --cuda-host-only, and the skeletonization
// replacement machinery collides with cuda_wrappers/new), independent of the
// header lowering this test covers. So the compile is allowed to fail (|| true)
// and we assert only the preprocess + ssthg_clang result.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -ferror-limit=100 -o %t.d/out.o > %t.d/log 2>&1 || true
// RUN: ls %t.d/sst.pp.headers-positive.cu
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
