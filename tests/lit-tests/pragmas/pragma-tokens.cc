// Pragma arguments with operators, brackets and keywords: the token printer
// used to stop the build on anything but a short list of tokens, such as the
// `=` and braces of the README's `#pragma sst instead {y = 0;}`. They are
// written as spelled, with the source's spacing. The compile must succeed.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.pragma-tokens.cc

int sst_hg_lit_expensive();
void sst_hg_lit_log(int v);

int sst_hg_lit_tokens_instead() {
  int y = 1;
#pragma sst instead {y = 0;}
  y = sst_hg_lit_expensive();
  return y;
}

#pragma sst return n + 1
int sst_hg_lit_tokens_return(int n) {
  return sst_hg_lit_expensive() * n;
}

void sst_hg_lit_tokens_call(int a) {
  int x = 0;
#pragma sst call sst_hg_lit_log(a + 1)
  x = a;
  (void)x;
}

void sst_hg_lit_tokens_init(int n) {
#pragma sst init (long int)n * 2
  long v = sst_hg_lit_expensive();
  sst_hg_lit_log((int)v);
}

// CHECK-LABEL: sst_hg_lit_tokens_instead
// CHECK: y = 0;
// CHECK-NOT: sst_hg_lit_expensive
// CHECK: return y;
// CHECK-LABEL: sst_hg_lit_tokens_return
// CHECK: return n + 1;
// CHECK-LABEL: sst_hg_lit_tokens_call
// CHECK: sst_hg_lit_log(a + 1);
// CHECK-NEXT: x = a;
// CHECK-LABEL: sst_hg_lit_tokens_init
// CHECK: long v = (long int)n * 2;
