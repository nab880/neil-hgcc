// Smoke: `#pragma sst assume_false` only rewrites in SKELETONIZE mode; ensure hgcc accepts it.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.assume_false.cc

extern int sst_hg_lit_cond();

int sst_hg_lit_assume_false() {
  int x = 0;
#pragma sst assume_false
  if (sst_hg_lit_cond()) {
    x = 1;
  }
  return x;
}

// CHECK: sst_hg_lit_assume_false
