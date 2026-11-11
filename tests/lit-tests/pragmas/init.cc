// `#pragma sst init` rewrites the initializer.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.init.cc

extern void sst_hg_lit_sink_int(int v);

void sst_hg_lit_init() {
#pragma sst init 77
  int v = 3;
  sst_hg_lit_sink_int(v);
}

// CHECK-LABEL: sst_hg_lit_init
// CHECK: int v = 77
