// Core: `#include <mpi.h>` takes Mercury's virtual mpi.h ahead of a real
// mpi.h in one of the user's -I directories, with a warning naming the one
// skipped; --disable-mpi takes the user's header instead, without a warning,
// and so does a link-only run, which includes no header.

// RUN: rm -rf %t.d && mkdir -p %t.d/real %t.d/virtual %t.d/disabled
// RUN: printf '#define SST_HG_LIT_REAL_MPI 1\nint MPI_Init(int*, char***);\nint MPI_Finalize(void);\n' > %t.d/real/mpi.h
// RUN: %hgcc -I %t.d/real -c %s -o %t.d/virtual/out.o 2>&1 | %FileCheck --check-prefix=WARN %s
// RUN: %FileCheck --check-prefix=VIRTUAL %s --input-file=%t.d/virtual/sst.pp.mpi-header-user-dir.cc
// RUN: %hgcc --disable-mpi -I %t.d/real -c %s -o %t.d/disabled/out.o 2>&1 | %FileCheck --allow-empty --check-prefix=QUIET %s
// RUN: %FileCheck --check-prefix=REAL %s --input-file=%t.d/disabled/sst.pp.mpi-header-user-dir.cc
// RUN: %hgcc -I %t.d/real %t.d/virtual/out.o -o %t.d/libuserdir.so 2>&1 | %FileCheck --allow-empty --check-prefix=QUIET %s

#include <mpi.h>

int sst_hg_lit_mpi_user_dir(int argc, char** argv) {
#ifdef SST_HG_LIT_REAL_MPI
  int real = 1;
#else
  int real = 0;
#endif
  MPI_Init(&argc, &argv);
  MPI_Finalize();
  return real;
}

// WARN: hgcc: warning: <mpi.h> resolves to Mercury's virtual MPI ahead of {{.*}}real/mpi.h
// VIRTUAL-LABEL: sst_hg_lit_mpi_user_dir
// VIRTUAL: int real = 0;
// VIRTUAL: mask_mpi_init
// QUIET-NOT: warning
// REAL-LABEL: sst_hg_lit_mpi_user_dir
// REAL: int real = 1;
// REAL-NOT: mask_mpi_init
