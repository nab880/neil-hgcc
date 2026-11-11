// #pragma sst advance_time <unit> <amount> -- inserts ssthg_{compute,msleep,usleep,nanosleep}.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.advance_time.cc

void sst_hg_lit_advance_time() {
  int x = 0;
#pragma sst advance_time usec 5
  x = 1;
  (void)x;
}

// CHECK-LABEL: sst_hg_lit_advance_time
// CHECK: ssthg_usleep(5)
