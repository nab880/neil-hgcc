// `#pragma sst null_ptr` rewrites uses of the marked pointer to `nullptr`.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.null_ptr.cc

void sst_hg_lit_null_ptr() {
#pragma sst null_ptr replace(nullptr)
  int* sst_hg_lit_np = nullptr;
  int* sst_hg_lit_other = sst_hg_lit_np;
  (void)sst_hg_lit_other;
}

// CHECK-LABEL: sst_hg_lit_null_ptr
// CHECK: sst_hg_lit_other = nullptr
