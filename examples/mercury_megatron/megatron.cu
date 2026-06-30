// Example: Megatron-style tensor + pipeline parallel transformer training under SST/Mercury.

#define ssthg_app_name mercury_megatron
#include <skeleton.h>

#include <mask_mpi.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

#define H 4096
#define T 2048
#define L 8
#define FF (4 * H)
#define ACT (T * H)

#pragma sst gpu_compute flops(8192) intops(0) read(16) write(4)
__global__ void gemm(const float* a, const float* w, float* y, int n) {}

__global__ void act_norm(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * 0.5f + 1.0f;
}

static inline int grid_for(long threads, int block) { return (int)((threads + block - 1) / block); }

static void tp_layer(float* d, cudaStream_t s, MPI_Comm tp, int tp_size,
                     int passes) {
  const int blk = 256;
  for (int p = 0; p < passes; ++p) {
    gemm    <<<grid_for(T * 3 * H / tp_size, blk), blk, 0, s>>>(d, d, d, T * 3 * H / tp_size);
    gemm    <<<grid_for(T * H / tp_size,     blk), blk, 0, s>>>(d, d, d, T * H / tp_size);
  }
  act_norm<<<grid_for(T * H, blk), blk, 0, s>>>(d, d, T * H);
  cudaStreamSynchronize(s);
  if (tp_size > 1)
    MPI_Allreduce(NULL, NULL, ACT, MPI_FLOAT, MPI_SUM, tp);
  for (int p = 0; p < passes; ++p) {
    gemm    <<<grid_for(T * FF / tp_size, blk), blk, 0, s>>>(d, d, d, T * FF / tp_size);
    gemm    <<<grid_for(T * H / tp_size,  blk), blk, 0, s>>>(d, d, d, T * H / tp_size);
  }
  act_norm<<<grid_for(T * H, blk), blk, 0, s>>>(d, d, T * H);
  cudaStreamSynchronize(s);
  if (tp_size > 1)
    MPI_Allreduce(NULL, NULL, ACT, MPI_FLOAT, MPI_SUM, tp);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const char* tps = getenv("TP_SIZE");
  int tp_size = tps ? atoi(tps) : 1;
  if (tp_size < 1) tp_size = 1;
  int pp_size = size / tp_size;
  if (pp_size < 1) { pp_size = 1; tp_size = size; }
  const int tp_rank = rank % tp_size;
  const int pp_rank = rank / tp_size;
  const int M = getenv("MICROBATCH") ? atoi(getenv("MICROBATCH")) : 4;
  const int layers_here = (L + pp_size - 1) / pp_size;

  MPI_Comm tp_comm;
  MPI_Comm_split(MPI_COMM_WORLD, pp_rank, tp_rank, &tp_comm);
  const int prev = (pp_rank - 1) * tp_size + tp_rank;
  const int next = (pp_rank + 1) * tp_size + tp_rank;

  float* d = nullptr;
  cudaMalloc((void**)&d, (size_t)T * H * sizeof(float));
  cudaStream_t s;
  cudaStreamCreate(&s);

  for (int m = 0; m < M; ++m) {
    if (pp_rank > 0)
      MPI_Recv(NULL, ACT, MPI_FLOAT, prev, m, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int l = 0; l < layers_here; ++l) tp_layer(d, s, tp_comm, tp_size, 1);
    if (pp_rank < pp_size - 1)
      MPI_Send(NULL, ACT, MPI_FLOAT, next, m, MPI_COMM_WORLD);
  }
  for (int m = 0; m < M; ++m) {
    if (pp_rank < pp_size - 1)
      MPI_Recv(NULL, ACT, MPI_FLOAT, next, M + m, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int l = 0; l < layers_here; ++l) tp_layer(d, s, tp_comm, tp_size, 2);
    if (pp_rank > 0)
      MPI_Send(NULL, ACT, MPI_FLOAT, prev, M + m, MPI_COMM_WORLD);
  }

  cudaStreamDestroy(s);
  cudaFree(d);
  MPI_Comm_free(&tp_comm);
  if (rank == 0)
    std::printf("mercury_megatron: TP=%d x PP=%d, %d microbatches, %d layers/stage done\n",
                tp_size, pp_size, M, layers_here);
  MPI_Finalize();
  return 0;
}
