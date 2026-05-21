#define ssthg_app_name mercury_hello_pragma
#include <skeleton.h>
#include <mask_mpi.h>
#include <iostream>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#pragma sst advance_time usec 10
  std::cerr << "Hello from rank " << rank << std::endl;

  MPI_Finalize();
  return 0;
}
