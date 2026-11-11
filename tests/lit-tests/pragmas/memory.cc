// #pragma sst memory <spec> -- attaches a memory-intensity override (metadata only).

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.memory.cc

void sst_hg_lit_memory() {
  int acc = 0;
#pragma sst memory 1024
  for (int i = 0; i < 16; ++i) {
    acc += i;
  }
  (void)acc;
}

// CHECK-LABEL: sst_hg_lit_memory
// CHECK:       for
