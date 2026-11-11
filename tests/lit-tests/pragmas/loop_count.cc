// Smoke: `#pragma sst loop_count` only rewrites in SKELETONIZE mode; ensure hgcc accepts it.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.loop_count.cc

extern int sst_hg_lit_expensive_bound();

void sst_hg_lit_loop_count() {
  int acc = 0;
#pragma sst loop_count 8
  for (int i = 0; i < sst_hg_lit_expensive_bound(); ++i) {
    acc += i;
  }
  (void)acc;
}

// CHECK: sst_hg_lit_loop_count
