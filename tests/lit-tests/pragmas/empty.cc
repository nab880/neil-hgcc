// `#pragma sst empty` replaces the function body with `{}`.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.empty.cc

#pragma sst empty
void sst_hg_lit_empty() {
  int expensive = 0;
  for (int i = 0; i < 1000; ++i) expensive += i;
  (void)expensive;
}

// CHECK-LABEL: sst_hg_lit_empty
// CHECK: {}
