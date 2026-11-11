// Core src2src: thread_local vars become offset lookups via get_sst_hg_tls_data().

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.tls-var.cc

thread_local int sst_hg_lit_tls = 7;

int read_tls() {
  return sst_hg_lit_tls;
}

// CHECK-DAG: get_sst_hg_tls_data
// CHECK-DAG: __offset_sst_hg_lit_tls
