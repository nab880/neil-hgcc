#define ssthg_app_name blocking_demo
#include <skeleton.h>
#include <mask_mpi.h>
#include <iostream>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int ready = 1;
#pragma sst blocking api(wait_for_data) condition(ready) timeout(100)
  std::cerr << "Rank " << rank << " passed blocking call" << std::endl;

  MPI_Finalize();
  return 0;
}
