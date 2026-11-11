// `#pragma sst malloc` rewrites a malloc-backed init so the buffer becomes 0.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.malloc.cc

#include <stdlib.h>

void sst_hg_lit_malloc() {
#pragma sst malloc
  int* buf = (int*)malloc(64 * sizeof(int));
  (void)buf;
}

// CHECK-LABEL: sst_hg_lit_malloc
// CHECK: buf = 0
