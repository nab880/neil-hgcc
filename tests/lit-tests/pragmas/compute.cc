// `#pragma sst compute` replaces the loop with a modeled instruction count
// and a call to Mercury's compute hook. The rewrite happens only in
// skeletonize mode, which is hgcc's default; this test does not set
// SST_HG_SKELETONIZE, so a driver that fails to select the mode is caught.
// The compile must succeed: the hook is declared by Mercury's skeleton.h.

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

// CHECK-LABEL: sst_hg_lit_compute
// CHECK-NOT: acc += i
// CHECK: tripCount0
// CHECK: sst_hg_compute_detailed({{ *}}flops,{{ *}}intops,{{ *}}readBytes{{ *}})
// CHECK-NOT: acc += i
