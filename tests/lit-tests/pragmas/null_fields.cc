// #pragma sst null_fields <names> -- marks the listed struct fields as null.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.null_fields.cc

#pragma sst null_fields ptr
struct sst_hg_lit_nf {
  int keep;
  int* ptr;
};

// CHECK: struct sst_hg_lit_nf
// CHECK: int keep
// CHECK: ptr
