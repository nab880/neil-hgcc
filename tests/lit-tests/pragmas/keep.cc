// #pragma sst keep -- guarded statement survives src2src untouched.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.keep.cc

int sst_hg_keep_g = 9;

int sst_hg_lit_keep() {
#pragma sst keep
  return sst_hg_keep_g;
}

// CHECK-LABEL: sst_hg_lit_keep
// CHECK:       return sst_hg_keep_g
