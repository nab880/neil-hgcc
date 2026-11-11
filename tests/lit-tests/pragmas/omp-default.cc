// On hg++'s default path skeletonize mode applies the `#pragma sst` rewrites
// the user wrote, but a plain `#pragma omp parallel` region runs as written:
// modelling it would drop the work it does, such as a reduction whose result
// the caller uses or an MPI call in an `omp master` block. Only the pragma is
// dropped, as in encapsulate mode, and its clauses are not parsed. When the
// user asks for skeletonize mode, by variable or flag, the region is modelled.

// RUN: rm -rf %t.d && mkdir -p %t.d/default %t.d/asked %t.d/flag %t.d/bad
// RUN: %hgcc -c %s -o %t.d/default/out.o
// RUN: %FileCheck --check-prefix=DEFAULT %s --input-file=%t.d/default/sst.pp.omp-default.cc
// RUN: env SST_HG_SKELETONIZE=1 %hgcc -c %s -o %t.d/asked/out.o
// RUN: %FileCheck --check-prefix=ASKED %s --input-file=%t.d/asked/sst.pp.omp-default.cc
// RUN: %hgcc --skeletonize= -c %s -o %t.d/flag/out.o
// RUN: %FileCheck --check-prefix=ASKED %s --input-file=%t.d/flag/sst.pp.omp-default.cc
// RUN: %hgcc -DSST_HG_LIT_BAD_CLAUSE -c %s -o %t.d/bad/out.o
// RUN: not env SST_HG_SKELETONIZE=1 %hgcc -DSST_HG_LIT_BAD_CLAUSE -c %s -o %t.d/bad/asked.o 2>&1 | %FileCheck --check-prefix=BAD %s

#include <mpi.h>

double sst_hg_lit_omp_default(const double* x, int n) {
  double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < n; ++i) {
    sum += x[i] * x[i];
  }
#pragma sst compute
  for (int i = 0; i < n; ++i) {
    sum += x[i];
  }
  return sum;
}

void sst_hg_lit_omp_master_send(const double* buf, int n) {
#pragma omp parallel
  {
#pragma omp master
    MPI_Send(buf, n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
  }
}

#pragma omp parallel
int sst_hg_lit_omp_on_function(int n) {
  return n + 1;
}

#ifdef SST_HG_LIT_BAD_CLAUSE
void sst_hg_lit_omp_bad_clause(double* a, int n) {
#pragma omp parallel for num_threads
  for (int i = 0; i < n; ++i) {
    a[i] = 0.0;
  }
}
#endif

// DEFAULT-LABEL: sst_hg_lit_omp_default
// DEFAULT-NOT: #pragma omp
// DEFAULT: sum += x[i] * x[i];
// DEFAULT-NOT: sum += x[i];
// DEFAULT: sst_hg_compute_detailed(
// DEFAULT: return sum;
// DEFAULT-LABEL: sst_hg_lit_omp_master_send
// DEFAULT: mask_mpi_send(
// DEFAULT-LABEL: sst_hg_lit_omp_on_function
// DEFAULT-NOT: sst_hg_compute_detailed
// DEFAULT: return n + 1;
// ASKED-LABEL: sst_hg_lit_omp_default
// ASKED-NOT: sum +=
// ASKED: sst_hg_compute_detailed(
// ASKED-NOT: sum +=
// ASKED: sst_hg_compute_detailed(
// ASKED-NOT: sum +=
// ASKED: return sum;
// ASKED-LABEL: sst_hg_lit_omp_master_send
// ASKED-NOT: mask_mpi_send(
// ASKED: sst_hg_compute_detailed(
// BAD: error: num_threads clause must be followed by '('
