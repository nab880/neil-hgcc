// Example: a GPU-accelerated halo-exchange stencil, the CUDA_PLAN §1 use case.
// Compiled with hg++ and run under SST/Mercury, it answers questions like "does
// compute/comms overlap survive at scale?" and "is GPUDirect worth it?" without
// any real GPU. The kernel never executes; its time is modeled (roofline or a
// calibration table), the halo MPI rides the simulated Merlin network, and the
// gpu_direct knob toggles PCIe staging of the device buffers.
//
// See README.md for the build + calibrate-once + gpu_direct-sweep walk-through.

#define ssthg_app_name mercury_cuda_halo
#include <skeleton.h>

#include <mask_mpi.h>
#include <cuda_runtime.h>
#include <cstdio>

// Per-thread costs feed the roofline; a calibration entry for this kernel's
// mangled name overrides them (see gpu_calibration.json).
#pragma sst gpu_compute flops(8) read(16) write(8)
__global__ void stencil(float* u, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > 0 && i < n - 1) u[i] = 0.25f * (u[i - 1] + 2.0f * u[i] + u[i + 1]);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int n = 1 << 20, t = 256, b = (n + t - 1) / t;
  const int halo = 4096;
  float *u, *sendbuf, *recvbuf;
  cudaMalloc((void**)&u, n * sizeof(float));
  cudaMalloc((void**)&sendbuf, halo * sizeof(float));
  cudaMalloc((void**)&recvbuf, halo * sizeof(float));
  cudaStream_t compute, comms;
  cudaStreamCreate(&compute);
  cudaStreamCreate(&comms);

  int left  = (rank - 1 + size) % size;
  int right = (rank + 1) % size;

  for (int step = 0; step < 20; ++step) {
    // Interior compute on its own stream -- overlaps the halo exchange.
    stencil<<<b, t, 0, compute>>>(u, n);
    // Pack + exchange the halo straight from device memory (GPUDirect-style).
    cudaMemcpyAsync(sendbuf, u, halo * sizeof(float), cudaMemcpyDeviceToDevice, comms);
    cudaStreamSynchronize(comms);
    MPI_Sendrecv(sendbuf, halo, MPI_FLOAT, right, 0,
                 recvbuf, halo, MPI_FLOAT, left,  0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    cudaMemcpyAsync(u, recvbuf, halo * sizeof(float), cudaMemcpyDeviceToDevice, comms);
    cudaDeviceSynchronize();
  }

  cudaFree(u);
  cudaFree(sendbuf);
  cudaFree(recvbuf);
  std::printf("mercury_cuda_halo: rank %d done\n", rank);
  MPI_Finalize();
  return 0;
}
