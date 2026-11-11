// Core: `#include <mpi.h>` resolves to the mask-mpi replacement header by
// default, so an unmodified MPI source compiles and its MPI_* calls reach the
// mask_mpi_* entry points.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.mpi-header.cc

#include <mpi.h>

int sst_hg_lit_mpi_header(int argc, char** argv) {
  int rank = -1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Finalize();
  return rank;
}

// CHECK-LABEL: sst_hg_lit_mpi_header
// CHECK: mask_mpi_init
// CHECK: mask_mpi_comm_rank
// CHECK: mask_mpi_finalize
