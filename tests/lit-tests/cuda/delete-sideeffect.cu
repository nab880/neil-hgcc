// P2: #pragma sst delete on a <<<>>> launch removes the whole statement. Unlike
// the automatic no-cost lowering path (see launch-sideeffect.cu), an explicit
// user delete is wholesale: replace(stmt, "") drops the launch AND its argument
// expressions, so any side effect in the args is intentionally discarded.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: ls %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.delete-sideeffect.cu

#include <cuda_runtime.h>

int g_counter = 0;
int next_id() { return ++g_counter; }

#pragma sst gpu_compute flops(1)
__global__ void k(int id, int* p) { (void)id; (void)p; }

int main() {
  int* d;
  cudaMalloc((void**)&d, 4);
#pragma sst delete
  k<<<1, 1>>>(next_id(), d);
  cudaFree(d);
  return 0;
}

// The launch is deleted: no lowering call for this kernel appears...
// CHECK-NOT: sst_hg_cuda_launch("_Z1kiPi"
// ...and the deleted statement's args go with it: none of the capture-temp or
// void-cast forms the lowering paths would have emitted for next_id() appear.
// CHECK-NOT: __sst_arg{{[0-9]+}}_{{[0-9]+}} = (next_id()
// CHECK-NOT: (void)(next_id()
