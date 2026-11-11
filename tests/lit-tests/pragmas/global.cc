// #pragma sst global <name> -- marks <name> as a global in a dependent scope (metadata only).

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.global.cc

int sst_hg_lit_gv = 3;

template <class T>
int sst_hg_lit_dependent(T) {
#pragma sst global sst_hg_lit_gv
  return sst_hg_lit_gv;
}

int sst_hg_lit_instantiator() {
  return sst_hg_lit_dependent<int>(0);
}

// CHECK-LABEL: sst_hg_lit_dependent
// CHECK:       sst_hg_lit_gv
