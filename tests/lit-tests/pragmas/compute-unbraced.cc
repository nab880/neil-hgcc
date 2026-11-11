// `#pragma sst compute` on a loop that is the unbraced branch of an if, as in
// Mantevo HPCCG's ddot, and the same through `#pragma omp parallel for`, on
// an unbraced do body, and on the if itself. The replaced statement ends
// before its semicolon; if the rewrite leaves that semicolon behind, the
// result reads "{ ... }; else" (or "; while") and does not compile. The
// compile must succeed. The hook is declared here so the test does not
// depend on skeleton.h declaring it; hgcc still force-includes skeleton.h.
// The RUN line asks for skeletonize mode: on hg++'s default path a plain
// OpenMP region is left as written.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: env SST_HG_SKELETONIZE=1 %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.compute-unbraced.cc

#include <stdint.h>
extern "C" void sst_hg_compute_detailed(uint64_t flops, uint64_t intops, uint64_t bytes);

double sst_hg_lit_compute_unbraced(const double* x, const double* y, int n) {
  double sum = 0.0;
  if (x == y)
#pragma sst compute
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
  else
#pragma sst compute
    for (int i = 0; i < n; i++) sum += x[i] * y[i];
  return sum;
}

double sst_hg_lit_omp_unbraced(const double* x, const double* y, int n) {
  double sum = 0.0;
  if (x == y)
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
  else
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) sum += x[i] * y[i];
  return sum;
}

double sst_hg_lit_do_unbraced(const double* x, int n, int reps) {
  double sum = 0.0;
  do
#pragma sst compute
    for (int i = 0; i < n; i++) sum += x[i];
  while (--reps > 0);
  return sum;
}

double sst_hg_lit_if_compute(double a, double b, bool c) {
  double x = 0.0, y = 0.0;
#pragma sst compute
  if (c) x = a * b; else y = a + b;
  return x + y;
}

// CHECK-LABEL: sst_hg_lit_compute_unbraced
// CHECK-NOT: sum +=
// CHECK: sst_hg_compute_detailed(
// CHECK-NOT: };
// CHECK: else
// CHECK-NOT: sum +=
// CHECK: sst_hg_compute_detailed(
// CHECK-NOT: sum +=
// CHECK: return sum
// CHECK-LABEL: sst_hg_lit_omp_unbraced
// CHECK-NOT: sum +=
// CHECK: sst_hg_compute_detailed(
// CHECK-NOT: };
// CHECK: else
// CHECK-NOT: sum +=
// CHECK: sst_hg_compute_detailed(
// CHECK-NOT: sum +=
// CHECK: return sum
// CHECK-LABEL: sst_hg_lit_do_unbraced
// CHECK: {{^ *}}do{{$}}
// CHECK-NOT: sum +=
// CHECK: sst_hg_compute_detailed(
// CHECK-NOT: };
// CHECK: while
// CHECK-LABEL: sst_hg_lit_if_compute
// CHECK-NOT: a * b
// CHECK: sst_hg_compute_detailed(
// CHECK-NOT: a + b
// CHECK-NOT: };
// CHECK: return x + y
