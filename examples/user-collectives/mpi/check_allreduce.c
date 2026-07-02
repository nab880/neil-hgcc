/*
 * Rank-count-flexible MPI_Allreduce correctness app for the Tier 2 validation
 * matrix. Unlike tests/mv2/test_mv2_allreduce.c (which hard-codes size==8), this
 * accepts any number of ranks and uses a multi-element buffer so it exercises a
 * reduce-scatter's per-rank chunking (including the remainder when nelems is not
 * a multiple of the rank count).
 *
 * It calls MPI_Allreduce, so the *engine* algorithm selected via
 * SUMI_ALLREDUCE_ALG / app1.allreduce_alg is what actually runs -- select an
 * algorithm and this app must still print PASS.
 *
 * Every rank contributes (rank+1) into every element; the expected value in each
 * element is n(n+1)/2.
 */
#define ssthg_app_name check_allreduce
#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NELEMS
#define NELEMS 1024   /* override at compile time: EXTRA_CFLAGS="-DNELEMS=256" */
#endif

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = -1, size = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int* in = malloc(sizeof(int) * NELEMS);
  int* out = malloc(sizeof(int) * NELEMS);
  for (int i = 0; i < NELEMS; ++i) { in[i] = rank + 1; out[i] = -1; }

  int rc = MPI_Allreduce(in, out, NELEMS, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "check_allreduce: rank %d MPI_Allreduce rc=%d\n", rank, rc);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  int expected = size * (size + 1) / 2;
  for (int i = 0; i < NELEMS; ++i) {
    if (out[i] != expected) {
      fprintf(stderr, "check_allreduce: rank %d elem %d got %d expected %d\n",
              rank, i, out[i], expected);
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
  }

  if (rank == 0) {
    printf("PASS: check_allreduce (%d ranks, %d elems, SUM=%d)\n",
           size, NELEMS, expected);
  }

  free(in); free(out);
  MPI_Finalize();
  return 0;
}
