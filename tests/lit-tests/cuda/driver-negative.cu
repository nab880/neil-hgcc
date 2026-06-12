// P1 driver plumbing negative test: a .cu must flow through preprocess and
// ssthg_clang (which passes CUDA through untouched until the P2 rewriter
// lands) and then fail at exactly the host -c step, which compiles -x c++,
// where the surviving <<< is a parse error. The sst.pp intermediate existing
// proves the stages before host -c succeeded. -I spike/stubs satisfies
// <cuda_runtime.h> until the P2 replacement headers ship.
//
// -ferror-limit=100 (forwarded by the driver to the host -c): the
// preprocessed TU embeds __clang_cuda_builtin_vars.h from the LLVM resource
// dir, whose __declspec(property) members are themselves C++ parse errors;
// at the default limit of 20 clang stops before reaching the <<< line.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -I %S/../../../spike/stubs -ferror-limit=100 -o %t.d/out.o > %t.d/log 2>&1 || true
// RUN: %FileCheck %s --input-file=%t.d/log
// RUN: ls %t.d/sst.pp.driver-negative.cu

#include <cuda_runtime.h>

__global__ void kern(int* p) { *p = 1; }

int main() {
  int* p = 0;
  kern<<<1, 1>>>(p);
  cudaDeviceSynchronize();
  return 0;
}

// CHECK: error: expected expression
