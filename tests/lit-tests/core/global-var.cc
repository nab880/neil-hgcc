// Core src2src: file-scope globals become offset lookups via get_sst_hg_global_data().

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.global-var.cc

int sst_hg_lit_global = 5;

int read_global() {
  return sst_hg_lit_global;
}

// CHECK-DAG: get_sst_hg_global_data
// CHECK-DAG: __offset_sst_hg_lit_global
