// Core: in C, skeletonize mode sends free(p) and (free)(p) to Mercury's
// sst_hg_free; putting "sst_hg_" in front of (free) reads sst_hg_(free). A
// local function pointer named free is not the C library's free and is left
// alone. The compile must succeed: sst_hg_free is declared by Mercury's
// skeleton.h, which hgcc includes in every file.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.free-call.c

#include <stdlib.h>

void sst_hg_lit_my_free(void* p);

void sst_hg_lit_free_call(void* a, void* b, void* c) {
  free(a);
  (free)(b);
  void (*free)(void*) = sst_hg_lit_my_free;
  free(c);
}

// CHECK-LABEL: sst_hg_lit_free_call
// CHECK: {{^ *}}sst_hg_free(a);
// CHECK-NEXT: {{^ *}}sst_hg_free(b);
// CHECK-NEXT: {{^ *}}void (*free)(void *) = sst_hg_lit_my_free;
// CHECK-NEXT: {{^ *}}free(c);
