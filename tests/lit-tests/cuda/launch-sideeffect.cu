// P2 Track 2: the model does not execute the kernel, but a launch argument with
// side effects must still be evaluated. The rewriter binds each side-effecting
// arg with (void)(arg) inside the launch statement-expression. A plain DeclRef
// arg is pure and is not bound.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: ls %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.launch-sideeffect.cu

#include <cuda_runtime.h>

int g_counter = 0;
int next_id() { return ++g_counter; }

__global__ void k(int id, int* p) { (void)id; (void)p; }

int main() {
  int* d;
  cudaMalloc((void**)&d, 4);
  // next_id() has a side effect and must be preserved; d is a pure DeclRef.
  k<<<1, 1>>>(next_id(), d);
  cudaFree(d);
  return 0;
}

// CHECK: (void)(next_id());
// CHECK: sst_hg_cuda_launch("_Z1kiPi"
