// `#pragma sst delete` drops the following statement.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.delete.cc

void sst_hg_lit_delete() {
  int sst_hg_keep = 1;
#pragma sst delete
  int sst_hg_drop = 42;
  (void)sst_hg_keep;
}

// CHECK-LABEL: sst_hg_lit_delete
// CHECK-NOT: sst_hg_drop = 42
// CHECK: sst_hg_keep
