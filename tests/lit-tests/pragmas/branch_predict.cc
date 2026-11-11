// #pragma sst branch_predict <prob> -- annotates an if-stmt with a branch probability (metadata only).

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.branch_predict.cc

extern int sst_hg_lit_cond();

int sst_hg_lit_branch_predict() {
  int x = 0;
#pragma sst branch_predict 0.9
  if (sst_hg_lit_cond()) {
    x = 1;
  }
  return x;
}

// CHECK-LABEL: sst_hg_lit_branch_predict
// CHECK:       if (
