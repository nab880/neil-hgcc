// P2: unresolved dependent-template launches are a hard error naming gpu_compute.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: not %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: %FileCheck %s --input-file=%t.d/log

#include <cuda_runtime.h>

template <class K>
void launch_it(K kern, int* p) {
  kern<<<1, 1>>>(p);
}

int main() { return 0; }

// CHECK: error: could not resolve the kernel of a <<<>>> launch
// CHECK-SAME: gpu_compute
