// Example: expert-parallel MoE training with dispatch/combine all-to-all traffic; SKEW sweeps routing imbalance.

#define ssthg_app_name mercury_moe
#include <skeleton.h>

#include <mask_mpi.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>

#define H 4096
#define T 2048
#define FF (4 * H)
#define TOPK 2

#pragma sst gpu_compute flops(8192) intops(0) read(16) write(4)
__global__ void attn(const float* x, const float* w, float* y, int n) {}

#pragma sst gpu_compute flops(8192) intops(0) read(16) write(4)
__global__ void expert_up(const float* x, const float* w, float* y, int n) {}

#pragma sst gpu_compute flops(32768) intops(0) read(16) write(4)
__global__ void expert_down(const float* x, const float* w, float* y, int n) {}

static inline int grid_for(long threads, int block) { return (int)((threads + block - 1) / block); }

static void attention(float* d, cudaStream_t s, int passes) {
  const int blk = 256;
  for (int p = 0; p < passes; ++p) {
    attn<<<grid_for(T * 3 * H, blk), blk, 0, s>>>(d, d, d, T * 3 * H);
    attn<<<grid_for(T * T,     blk), blk, 0, s>>>(d, d, d, T * T);
    attn<<<grid_for(T * H,     blk), blk, 0, s>>>(d, d, d, T * H);
    attn<<<grid_for(T * H,     blk), blk, 0, s>>>(d, d, d, T * H);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // SKEW in [0,1]: 0 = uniform routing, 1 = every token to expert 0.
  int layers = getenv("LAYERS") ? atoi(getenv("LAYERS")) : 4;
  if (layers < 1) layers = 1; if (layers > 64) layers = 64;
  double skew = getenv("SKEW") ? atof(getenv("SKEW")) : 0.0;
  if (skew < 0.0) skew = 0.0; if (skew > 1.0) skew = 1.0;

  float* d = nullptr;
  cudaMalloc((void**)&d, (size_t)T * H * sizeof(float));
  cudaStream_t s;
  cudaStreamCreate(&s);

  // Per-destination token counts; skew concentrates traffic on expert 0.
  const long total_slots = (long)T * TOPK;
  int* sendcounts = (int*)malloc(sizeof(int) * size);
  int* sdispls    = (int*)malloc(sizeof(int) * size);
  int* recvcounts = (int*)malloc(sizeof(int) * size);
  int* rdispls    = (int*)malloc(sizeof(int) * size);
  const double uniform = (1.0 - skew) * (double)total_slots / size;
  long num_to0 = (long)(skew * total_slots + uniform);
  long num_toj = (long)uniform;
  for (int j = 0; j < size; ++j)
    sendcounts[j] = (int)((j == 0 ? num_to0 : num_toj) * H);
  int per_src_to_me = (int)((rank == 0 ? num_to0 : num_toj) * H);
  for (int j = 0; j < size; ++j) recvcounts[j] = per_src_to_me;
  sdispls[0] = rdispls[0] = 0;
  for (int j = 1; j < size; ++j) {
    sdispls[j] = sdispls[j-1] + sendcounts[j-1];
    rdispls[j] = rdispls[j-1] + recvcounts[j-1];
  }
  // Hot expert receives size x more tokens -> compute straggler.
  long recv_tokens = (long)(rank == 0 ? num_to0 : num_toj) * size;
  if (recv_tokens < 1) recv_tokens = 1;

  auto moe_layer = [&](int passes) {
    attention(d, s, passes);
    cudaStreamSynchronize(s);
    if (size > 1)
      MPI_Alltoallv(NULL, sendcounts, sdispls, MPI_FLOAT,
                    NULL, recvcounts, rdispls, MPI_FLOAT, MPI_COMM_WORLD);
    // Expert FFN on received tokens (routing-skewed load). Threads = tokens*FF / tokens*H;
    // block counts use long arithmetic because a hot expert can exceed 32 bits.
    const int blk = 256;
    for (int p = 0; p < passes; ++p) {
      long up_threads   = recv_tokens * (long)FF;
      long down_threads = recv_tokens * (long)H;
      expert_up  <<<(int)((up_threads   + blk - 1) / blk), blk, 0, s>>>(d, d, d, (int)recv_tokens);
      expert_down<<<(int)((down_threads + blk - 1) / blk), blk, 0, s>>>(d, d, d, (int)recv_tokens);
    }
    cudaStreamSynchronize(s);
    if (size > 1)
      MPI_Alltoallv(NULL, recvcounts, rdispls, MPI_FLOAT,
                    NULL, sendcounts, sdispls, MPI_FLOAT, MPI_COMM_WORLD);
  };

  for (int l = 0; l < layers; ++l) moe_layer(1);
  for (int l = 0; l < layers; ++l) moe_layer(2);

  cudaStreamSynchronize(s);
  cudaStreamDestroy(s);
  cudaFree(d);
  free(sendcounts); free(sdispls); free(recvcounts); free(rdispls);
  if (rank == 0)
    std::printf("mercury_moe: %d experts (1/rank), top-%d, skew=%.2f, %d layers,"
                " hot/cold recv tokens = %ld/%ld done\n",
                size, TOPK, skew, layers, (long)num_to0 * size, (long)num_toj * size);
  MPI_Finalize();
  return 0;
}
