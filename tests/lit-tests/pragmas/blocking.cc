// #pragma sst blocking api(<name>) -- inserts `sst_hg_blocking_call(cond,timeout,"<name>")`.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.blocking.cc

void sst_hg_lit_blocking() {
  int x = 0;
#pragma sst blocking api(sst_hg_lit_api)
  x = 1;
  (void)x;
}

// CHECK-LABEL: sst_hg_lit_blocking
// CHECK: sst_hg_blocking_call({{.*}}"sst_hg_lit_api"
