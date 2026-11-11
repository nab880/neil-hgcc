// `#pragma sst instead` replaces the next statement with the provided code.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.instead.cc

extern void sst_hg_lit_shim();

void sst_hg_lit_instead() {
  int heavy = 0;
#pragma sst instead sst_hg_lit_shim()
  for (int i = 0; i < 1000; ++i) heavy += i;
  (void)heavy;
}

// CHECK-LABEL: sst_hg_lit_instead
// CHECK-NOT: heavy += i
// CHECK: sst_hg_lit_shim()
// CHECK-NOT: heavy += i
