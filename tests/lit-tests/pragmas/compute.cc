// Smoke: `#pragma sst compute` only rewrites in SKELETONIZE mode; ensure hgcc accepts it.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.compute.cc

void sst_hg_lit_compute() {
  int acc = 0;
#pragma sst compute
  for (int i = 0; i < 1000; ++i) {
    acc += i;
  }
  (void)acc;
}

// CHECK: sst_hg_lit_compute
