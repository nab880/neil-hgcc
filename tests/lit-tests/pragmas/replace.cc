// `#pragma sst replace` swaps a call to the named function with the replacement text.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.replace.cc

extern int sst_hg_lit_orig();
extern void sst_hg_lit_sink_int(int v);

void sst_hg_lit_replace() {
#pragma sst replace sst_hg_lit_orig 123
  int x = sst_hg_lit_orig();
  sst_hg_lit_sink_int(x);
}

// CHECK-LABEL: sst_hg_lit_replace
// CHECK: x = 123
