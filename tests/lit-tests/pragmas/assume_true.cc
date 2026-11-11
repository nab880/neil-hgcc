// `#pragma sst assume_true` rewrites the next `if` condition to `true`.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.assume_true.cc

extern int sst_hg_lit_runtime();
extern void sst_hg_lit_taken();

void sst_hg_lit_assume_true() {
#pragma sst assume_true
  if (sst_hg_lit_runtime()) {
    sst_hg_lit_taken();
  }
}

// CHECK-LABEL: sst_hg_lit_assume_true
// CHECK: if (true)
