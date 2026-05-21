#define ssthg_app_name compute_demo
#include <skeleton.h>
#include <mask_mpi.h>
#include <iostream>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int acc = 0;
#pragma sst compute
  for (int i = 0; i < 1000; ++i) {
    acc += i;
  }

  std::cerr << "Rank " << rank << " compute acc=" << acc << std::endl;

  MPI_Finalize();
  return 0;
}
