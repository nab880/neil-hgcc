// P2: __device__/__constant__ globals are not privatized like host file-scope globals.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: ls %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.device-global-exempt.cu

#include <cuda_runtime.h>

__device__ int dev_counter;
__constant__ float dev_coef;
int host_global = 7;

#pragma sst gpu_compute flops(1)
__global__ void k() { dev_counter = (int)dev_coef; }

int main() {
  k<<<1, 1>>>();
  return host_global;
}

// The host global is privatized (offset lookup):
// CHECK: __offset_host_global

// The device/constant globals are left alone (no offset/standin generated):
// CHECK-NOT: __offset_dev_counter
// CHECK-NOT: __offset_dev_coef
// CHECK-NOT: sst_hg_dev_counter
