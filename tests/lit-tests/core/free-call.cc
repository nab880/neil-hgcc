// Core: in skeletonize mode a call to the C library's free goes to Mercury's
// sst_hg_free, whether written free, ::free, std::free or (free). The whole
// callee is replaced, since putting "sst_hg_" in front of std::free reads
// sst_hg_std::free. A free in the user's own namespace is not the C library's
// and is left alone. Text a statement pragma inserts in front of the call is
// kept, and nothing after the callee is lost.
// The compile must succeed: sst_hg_free is declared by Mercury's skeleton.h,
// which hgcc includes in every file.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.free-call.cc

#include <cstdlib>

namespace pool {
void free(void* p);
}

void sst_hg_lit_log_free(void* p);

void sst_hg_lit_free_call(void* a, void* b, void* c, void* d, void* e) {
  free(a);
  ::free(b);
  std::free(c);
  (free)(d);
  pool::free(e);
}

void sst_hg_lit_free_after_pragma(void* a, void* b) {
#pragma sst call sst_hg_lit_log_free(a)
  std::free(a);
  free(b);
}

// CHECK-LABEL: sst_hg_lit_free_call
// CHECK: {{^ *}}sst_hg_free(a);
// CHECK-NEXT: {{^ *}}sst_hg_free(b);
// CHECK-NEXT: {{^ *}}sst_hg_free(c);
// CHECK-NEXT: {{^ *}}sst_hg_free(d);
// CHECK-NEXT: {{^ *}}pool::free(e);
// CHECK-LABEL: sst_hg_lit_free_after_pragma
// CHECK: {{^ *}}sst_hg_lit_log_free(a);
// CHECK-NEXT: {{^ *}}sst_hg_free(a);
// CHECK-NEXT: {{^ *}}sst_hg_free(b);
