#define ssthg_app_name overhead_demo
#include <skeleton.h>
#include <mask_mpi.h>
#include <iostream>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Charges the app parameter `app1.launch_overhead` (set in overhead_app.py)
  // as busy time before the statement runs.
#pragma sst overhead launch_overhead
  std::cerr << "Rank " << rank << " paid launch overhead" << std::endl;

  MPI_Finalize();
  return 0;
}
