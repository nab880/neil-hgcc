// #pragma sst call <tokens> -- inserts the verbatim call before the next stmt.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.call.cc

extern void sst_hg_lit_injected();

void sst_hg_lit_call() {
  int x = 0;
#pragma sst call sst_hg_lit_injected()
  x = 1;
  (void)x;
}

// CHECK: sst_hg_lit_injected()
// CHECK: x = 1
