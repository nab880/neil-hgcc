// Smoke: `#pragma sst null_type` elides the variable and its uses in SKELETONIZE mode.

// Known to fail: the rewrite deletes the variable's later use too and leaves
// "(void);", which does not compile.
// XFAIL: *

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.null_type.cc

void sst_hg_lit_null_type() {
#pragma sst null_type
  int* sst_hg_lit_nt = nullptr;
  (void)sst_hg_lit_nt;
}

// CHECK-LABEL: sst_hg_lit_null_type
// CHECK-NOT: int* sst_hg_lit_nt
