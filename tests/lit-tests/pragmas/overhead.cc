// `#pragma sst overhead <param>` inserts `sst_hg_advance_time("<param>");`
// before the next statement. The compile must succeed: Mercury's skeleton.h
// declares the hook and libhg defines it.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.overhead.cc

void sst_hg_lit_overhead() {
  int x = 0;
#pragma sst overhead sst_hg_lit_ov
  x = 1;
  (void)x;
}

// CHECK-LABEL: sst_hg_lit_overhead
// CHECK-NOT: sst_hg_advance_time
// CHECK: sst_hg_advance_time("sst_hg_lit_ov")
// CHECK-NOT: sst_hg_advance_time
// CHECK: x = 1
// CHECK-NOT: sst_hg_advance_time
