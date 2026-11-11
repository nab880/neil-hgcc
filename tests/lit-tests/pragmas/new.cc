// `#pragma sst new` collapses a C++ `new` into `nullptr`.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.new.cc

struct sst_hg_lit_obj { int v; };

void sst_hg_lit_new() {
#pragma sst new
  sst_hg_lit_obj* p = new sst_hg_lit_obj();
  (void)p;
}

// CHECK-LABEL: sst_hg_lit_new
// CHECK: sst_hg_lit_obj {{.*}}= nullptr
