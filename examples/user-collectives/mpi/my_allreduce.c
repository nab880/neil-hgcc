/*
 * Template: implement your own all-reduce algorithm in plain MPI point-to-point.
 *
 * This is Tier 1 of the user-collectives tool: you express a collective purely
 * in MPI_Send/MPI_Recv, and it rides the real Mercury/Iris point-to-point path,
 * so the reported time is a full packet-level measurement on your chosen
 * topology -- no simulator edits, no libsumi rebuild.
 *
 * How to use:
 *   1. Edit the region marked  >>> YOUR ALGORITHM HERE <<<  below.
 *   2. ./build.sh                         (compiles lib<APP>.so)
 *   3. APP=my_allreduce NRANKS=8 sst run.py
 *   4. Read the reported simulated time; bump NRANKS / TOPO to sweep.
 *
 * The app self-checks: every rank contributes (rank+1) and must end with the
 * known sum n(n+1)/2. A wrong algorithm aborts instead of silently reporting a
 * meaningless time. Mirrors tests/mv2/test_mv2_allreduce.c.
 */
#define ssthg_app_name my_allreduce
#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

/* Reduce one int element: acc += val. Swap in your own op if you like. */
static void reduce_sum(int* acc, const int* val) { *acc += *val; }

/*
 * >>> YOUR ALGORITHM HERE <<<
 *
 * Compute the all-reduce (MPI_SUM) of the single int `in` across MPI_COMM_WORLD
 * and write the result to *out, using only point-to-point messages.
 *
 * Worked example: a naive ring. Each original contribution travels once around
 * the ring: `trav` is the value passing through this rank (its own first, then
 * whatever it just received), and `acc` sums each contribution as it arrives.
 * After N-1 steps every rank has added all N contributions exactly once.
 * Replace this body with your scheme (recursive doubling, hierarchical, tree,
 * ...). Keep the send/recv paired to avoid deadlock (this ordering -- even
 * ranks send-then-recv, odd ranks recv-then-send -- is deadlock-free;
 * MPI_Sendrecv would be simpler but this shows the shape).
 */
static void my_allreduce(int in, int* out) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int right = (rank + 1) % size;
  int left  = (rank - 1 + size) % size;

  int acc = in;      /* running sum of contributions seen */
  int trav = in;     /* value forwarded this step (received value next step) */
  int recv = 0;
  for (int step = 0; step < size - 1; ++step) {
    if (rank % 2 == 0) {
      MPI_Send(&trav, 1, MPI_INT, right, 0, MPI_COMM_WORLD);
      MPI_Recv(&recv, 1, MPI_INT, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      MPI_Recv(&recv, 1, MPI_INT, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&trav, 1, MPI_INT, right, 0, MPI_COMM_WORLD);
    }
    reduce_sum(&acc, &recv);  /* add the arriving contribution */
    trav = recv;              /* forward it onward next step */
  }
  *out = acc;
}
/* >>> END OF YOUR ALGORITHM <<< */

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = -1, size = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int in = rank + 1;
  int out = -1;
  my_allreduce(in, &out);

  int expected = size * (size + 1) / 2;   /* sum of 1..size */
  if (out != expected) {
    fprintf(stderr, "my_allreduce: rank %d got %d expected %d\n",
            rank, out, expected);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  if (rank == 0) {
    printf("PASS: my_allreduce (%d ranks, SUM=%d)\n", size, out);
  }

  MPI_Finalize();
  return 0;
}
