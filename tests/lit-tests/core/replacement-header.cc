// Core: baseline `--replacements=pthread.h` swaps the system header and compiles.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc --replacements pthread.h -DSST_HG_USE_MERCURY_PTHREAD -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.replacement-header.cc

#include <pthread.h>

void touches_pthread() {
  pthread_t id;
  (void)id;
}

// CHECK: pthread_t
