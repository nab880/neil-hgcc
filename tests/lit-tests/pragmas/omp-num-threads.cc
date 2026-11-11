// In skeletonize mode `#pragma omp parallel` becomes a call to
// sst_hg_compute_detailed, or, with a num_threads clause, to
// sst_hg_compute_detailed_nthr whose last argument is the clause's expression.
// The whole clause expression must be kept: with only its first token,
// num_threads(nt + 1) charged nt threads and num_threads(omp_get_max_threads())
// emitted the bare function name, which does not compile. A num_threads inside
// another clause is a variable, not the clause, and an OpenMP 6.0 list
// num_threads(4, 2) gives the outermost team's count first. The compile must
// succeed; a malformed clause stops the build.
// hgcc consumes the pragma itself, so no -fopenmp is needed.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: env SST_HG_SKELETONIZE=1 %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.omp-num-threads.cc
// RUN: not env SST_HG_SKELETONIZE=1 %hgcc -DSST_HG_LIT_NO_PAREN -c %s -o %t.d/bad1.o 2>&1 | %FileCheck %s --check-prefix=NOPAREN
// RUN: not env SST_HG_SKELETONIZE=1 %hgcc -DSST_HG_LIT_EMPTY -c %s -o %t.d/bad2.o 2>&1 | %FileCheck %s --check-prefix=EMPTY

extern "C" int omp_get_max_threads(void);
int sst_hg_lit_pick(int a, int b);

void sst_hg_lit_omp_num_threads(double* a, int n, int nt, int num_threads) {
#pragma omp parallel for num_threads(32)
  for (int i = 0; i < n; ++i) {
    a[i] = 1.0;
  }
#pragma omp parallel for num_threads(nt + 1)
  for (int i = 0; i < n; ++i) {
    a[i] = 2.0;
  }
#pragma omp parallel for num_threads(omp_get_max_threads())
  for (int i = 0; i < n; ++i) {
    a[i] = 3.0;
  }
#pragma omp parallel for num_threads(strict: nt * 2)
  for (int i = 0; i < n; ++i) {
    a[i] = 4.0;
  }
#pragma omp parallel for firstprivate(num_threads) num_threads(4)
  for (int i = 0; i < n; ++i) {
    a[i] = 5.0;
  }
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    a[i] = 6.0;
  }
#pragma omp parallel for shared(num_threads)
  for (int i = 0; i < n; ++i) {
    a[i] = 7.0;
  }
#pragma omp parallel for num_threads(nt, 2)
  for (int i = 0; i < n; ++i) {
    a[i] = 8.0;
  }
#pragma omp parallel for num_threads(sst_hg_lit_pick(nt, 2))
  for (int i = 0; i < n; ++i) {
    a[i] = 11.0;
  }
#pragma omp parallel for num_threads([nt, n] { return nt + n; }(), 2)
  for (int i = 0; i < n; ++i) {
    a[i] = 12.0;
  }
#pragma omp parallel for num_threads(strict: nt + 3, 2)
  for (int i = 0; i < n; ++i) {
    a[i] = 13.0;
  }
#ifdef SST_HG_LIT_NO_PAREN
#pragma omp parallel for num_threads
  for (int i = 0; i < n; ++i) {
    a[i] = 9.0;
  }
#endif
#ifdef SST_HG_LIT_EMPTY
#pragma omp parallel for num_threads(strict:)
  for (int i = 0; i < n; ++i) {
    a[i] = 10.0;
  }
#endif
}

// hgcc clang-formats the kept sources when it can, which may wrap the third
// call before its last argument, so that check is split in two.
// CHECK-LABEL: sst_hg_lit_omp_num_threads
// CHECK: sst_hg_compute_detailed_nthr({{.*}}readBytes,{{ *}}32);
// CHECK: sst_hg_compute_detailed_nthr({{.*}}readBytes,{{ *}}nt + 1);
// CHECK: sst_hg_compute_detailed_nthr({{.*}}readBytes,
// CHECK: omp_get_max_threads());
// CHECK: sst_hg_compute_detailed_nthr({{.*}}readBytes,{{ *}}nt * 2);
// CHECK: sst_hg_compute_detailed_nthr({{.*}}readBytes,{{ *}}4);
// CHECK: sst_hg_compute_detailed({{.*}}readBytes);
// CHECK: sst_hg_compute_detailed({{.*}}readBytes);
// CHECK: sst_hg_compute_detailed_nthr({{.*}}readBytes,{{ *}}nt);
// CHECK: sst_hg_compute_detailed_nthr({{.*}}readBytes,
// CHECK: sst_hg_lit_pick(nt, 2));
// CHECK: sst_hg_compute_detailed_nthr({{.*}}readBytes,
// CHECK: [nt, n]
// CHECK: return nt + n;
// CHECK: }());
// CHECK: sst_hg_compute_detailed_nthr({{.*}}readBytes,{{ *}}nt + 3);
// NOPAREN: {{.*}}:{{[0-9]+}}:{{[0-9]+}}: error: num_threads clause must be followed by '('
// EMPTY: {{.*}}:{{[0-9]+}}:{{[0-9]+}}: error: num_threads clause needs an expression and a closing ')'
