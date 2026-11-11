// #pragma sst keep_if <cond> -- wraps next stmt in `if (cond) { ... } else if (0)`.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.keep_if.cc

extern int sst_hg_kif_cond;

int sst_hg_lit_keep_if() {
  int x = 0;
#pragma sst keep_if sst_hg_kif_cond
  x = 100;
  return x;
}

// CHECK-LABEL: sst_hg_lit_keep_if
// CHECK: if (sst_hg_kif_cond)
// CHECK: x = 100
// CHECK: else if (0)
