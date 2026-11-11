// `#pragma sst return` replaces the function body with a literal return.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.return.cc

#pragma sst return 42
int sst_hg_lit_return() {
  int sum = 0;
  for (int i = 0; i < 1000; ++i) sum += i;
  return sum;
}

// CHECK-LABEL: sst_hg_lit_return
// CHECK: return 42
