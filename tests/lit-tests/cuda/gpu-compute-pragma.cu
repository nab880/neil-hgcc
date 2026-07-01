// P2/P4: #pragma sst gpu_compute overrides derived per-thread costs at decl or launch.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: ls %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.gpu-compute-pragma.cu

#include <cuda_runtime.h>

// costs on the kernel decl (override the derivable body cost)
#pragma sst gpu_compute flops(10) intops(20) read(30) write(40)
__global__ void decl_costed(int* p) { *p = 0; }

// costs supplied at the launch statement
__global__ void stmt_costed(int* p) { *p = 0; }

// no pragma -> cost derived from the body, not zero
__global__ void derived(int* p) { *p = 0; }

int main() {
  int* d;
  cudaMalloc((void**)&d, 4);

  decl_costed<<<1, 1>>>(d);

#pragma sst gpu_compute flops(5) intops(6) read(7) write(8)
  stmt_costed<<<1, 1>>>(d);

  derived<<<1, 1>>>(d);

  cudaFree(d);
  return 0;
}

// The pragma costs reach the launch verbatim (precedence over derivation):
// CHECK: sst_hg_cuda_launch("_Z11decl_costedPi"
// CHECK: 10, 20, 30, 40);
// CHECK: sst_hg_cuda_launch("_Z11stmt_costedPi"
// CHECK: 5, 6, 7, 8);

// The un-annotated straight-line kernel derives a cost and passes the
// accumulated values from the cost lambda (not 0,0,0,0):
// CHECK: sst_hg_cuda_launch("_Z7derivedPi"
// CHECK: __sst_cost_2.f, __sst_cost_2.i
// CHECK: __sst_cost_2.r
// CHECK: __sst_cost_2.w);
