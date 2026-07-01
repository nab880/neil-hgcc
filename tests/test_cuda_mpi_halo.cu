// CUDA halo exchange with stencil/comms overlap; gpu_direct toggles PCIe staging.

#define ssthg_app_name test_cuda_mpi_halo
#include <skeleton.h>

#include <mask_mpi.h>
#include <cuda_runtime.h>
#include <cstdio>

#pragma sst gpu_compute read(8) write(8)
__global__ void stencil(float* u) { int i = blockIdx.x*blockDim.x + threadIdx.x; u[i] *= 1.01f; }

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int n = 1 << 18, t = 256, b = (n + t - 1) / t;
  const int halo = 1024;
  float *u, *sendbuf, *recvbuf;
  cudaMalloc((void**)&u, n * sizeof(float));
  cudaMalloc((void**)&sendbuf, halo * sizeof(float));
  cudaMalloc((void**)&recvbuf, halo * sizeof(float));
  cudaStream_t s;
  cudaStreamCreate(&s);

  int left  = (rank - 1 + size) % size;
  int right = (rank + 1) % size;

  const int nsteps = 10;
  for (int step = 0; step < nsteps; ++step) {
    stencil<<<b, t, 0, s>>>(u);
    MPI_Sendrecv(sendbuf, halo, MPI_FLOAT, right, 0,
                 recvbuf, halo, MPI_FLOAT, left,  0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    cudaDeviceSynchronize();
  }

  cudaFree(u);
  cudaFree(sendbuf);
  cudaFree(recvbuf);
  std::printf("test_cuda_mpi_halo: rank %d done\n", rank);
  MPI_Finalize();
  return 0;
}
