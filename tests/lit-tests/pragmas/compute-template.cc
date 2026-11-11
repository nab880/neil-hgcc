// `#pragma sst compute` on a loop in a template's own body: when the loop
// uses a template-dependent type, the compute visitor cannot count its work
// (it used to crash), so the loop is left as written with a warning and runs
// natively in each instantiation. A template loop that uses no dependent
// type, such as one bounded by a non-type parameter, is still modelled.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o 2>&1 | %FileCheck %s --check-prefix=WARN
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.compute-template.cc

template <class T>
T sst_hg_lit_tmpl_sum(const T* x, int n) {
  T sum = 0;
#pragma sst compute
  for (int i = 0; i < n; ++i) {
    sum += x[i] * x[i];
  }
  return sum;
}

template <class V>
struct sst_hg_lit_tmpl_holder {
  V v;
  double total(int n) const {
    double s = 0.0;
#pragma sst compute
    for (int i = 0; i < n; ++i) {
      s += v.data[i];
    }
    return s;
  }
};

template <int N>
double sst_hg_lit_tmpl_fixed(const double* x) {
  double s = 0.0;
#pragma sst compute
  for (int i = 0; i < N; ++i) {
    s += x[i];
  }
  return s;
}

struct sst_hg_lit_tmpl_vec {
  double data[4];
};

double sst_hg_lit_tmpl_use(const double* x) {
  sst_hg_lit_tmpl_holder<sst_hg_lit_tmpl_vec> h{};
  return sst_hg_lit_tmpl_sum(x, 4) + h.total(4) + sst_hg_lit_tmpl_fixed<4>(x);
}

// WARN: warning: #pragma sst compute: the statement uses a template-dependent type
// WARN: warning: #pragma sst compute: the statement uses a template-dependent type
// WARN-NOT: warning: #pragma sst compute

// CHECK-LABEL: sst_hg_lit_tmpl_sum
// CHECK: sum += x[i] * x[i];
// CHECK-LABEL: struct sst_hg_lit_tmpl_holder {
// CHECK: s += v.data[i];
// CHECK-LABEL: sst_hg_lit_tmpl_fixed
// CHECK-NOT: s += x[i];
// CHECK: sst_hg_compute_detailed(
