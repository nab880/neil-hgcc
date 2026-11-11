// #pragma sst nonnull_fields <names> -- keeps only the listed fields; others are skeletonized.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.nonnull_fields.cc

#pragma sst nonnull_fields keep
struct sst_hg_lit_nnf {
  int keep;
  int* drop;
};

// CHECK: struct sst_hg_lit_nnf
// CHECK: int keep
