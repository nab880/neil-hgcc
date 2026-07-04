// Example: fully-sharded data parallelism (FSDP / ZeRO-3) under SST/Mercury --
// modeled, never executed. Where plain DP (../mercury_llm_train) replicates the
// model and pays ONE post-backward gradient all-reduce, FSDP shards the parameters,
// gradients, and optimizer across the data-parallel group and pays, PER LAYER, a
// parameter all-gather on the forward AND backward critical path plus a gradient
// reduce-scatter. That is ~1.5x DP's communication volume (3 vs 2 of the (N-1)/N
// half-collectives), and the all-gather sits on the critical path -- so FSDP's whole
// viability rests on whether prefetching the next layer's gather hides it behind the
// current layer's compute. FSDP_PREFETCH is that knob; it pairs with the study (G)
// memory model -- this is the time cost of the ZeRO-3 footprint that makes DP fit.
//
// Modeled, never executed: kernels carry pinned flop counts; the all-gather and
// reduce-scatter are NULL-buffer collectives timed from their per-rank shard counts;
// the sharded persistent state is allocated as device cookies (no real memory) so the
// per-rank mem_footprint cross-checks examples/memory_model/capacity.py --zero 3.

#define ssthg_app_name mercury_fsdp
#include <skeleton.h>

#include <mask_mpi.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

#define H 4096
#define T 2048
#define L 8
#define FF (4 * H)
#define GRAD_ELEMS (12 * H * H)     // full params (and grads) per layer

#pragma sst gpu_compute flops(8192) intops(0) read(16) write(4)
__global__ void qkv_proj(const float* x, const float* W, float* y, int n) {}
#pragma sst gpu_compute flops(8192) intops(0) read(16) write(4)
__global__ void attn_scores(const float* q, const float* k, float* s, int n) {}
#pragma sst gpu_compute flops(4096) intops(0) read(16) write(4)
__global__ void attn_av(const float* s, const float* v, float* o, int n) {}
#pragma sst gpu_compute flops(8192) intops(0) read(16) write(4)
__global__ void attn_out(const float* o, const float* W, float* y, int n) {}
#pragma sst gpu_compute flops(8192) intops(0) read(16) write(4)
__global__ void mlp_up(const float* x, const float* W, float* h, int n) {}
#pragma sst gpu_compute flops(32768) intops(0) read(16) write(4)
__global__ void mlp_down(const float* h, const float* W, float* y, int n) {}

__global__ void layernorm(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * 0.5f + 1.0f;
}
__global__ void softmax_rows(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * 2.0f;
}
__global__ void gelu(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * 0.5f;
}

static inline int grid_for(long threads, int block) { return (int)((threads + block - 1) / block); }

static void forward_layer(float* d, cudaStream_t s) {
  const int blk = 256;
  layernorm   <<<grid_for(T * H,     blk), blk, 0, s>>>(d, d, T * H);
  qkv_proj    <<<grid_for(T * 3 * H, blk), blk, 0, s>>>(d, d, d, T * 3 * H);
  attn_scores <<<grid_for(T * T,     blk), blk, 0, s>>>(d, d, d, T * T);
  softmax_rows<<<grid_for(T * T,     blk), blk, 0, s>>>(d, d, T * T);
  attn_av     <<<grid_for(T * H,     blk), blk, 0, s>>>(d, d, d, T * H);
  attn_out    <<<grid_for(T * H,     blk), blk, 0, s>>>(d, d, d, T * H);
  layernorm   <<<grid_for(T * H,     blk), blk, 0, s>>>(d, d, T * H);
  mlp_up      <<<grid_for(T * FF,    blk), blk, 0, s>>>(d, d, d, T * FF);
  gelu        <<<grid_for(T * FF,    blk), blk, 0, s>>>(d, d, T * FF);
  mlp_down    <<<grid_for(T * H,     blk), blk, 0, s>>>(d, d, d, T * H);
}
static void backward_layer(float* d, cudaStream_t s) { forward_layer(d, s); forward_layer(d, s); }

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const char* pf = getenv("FSDP_PREFETCH");
  const bool prefetch = !pf || pf[0] == '1' || pf[0] == 't';
  int layers = getenv("LAYERS") ? atoi(getenv("LAYERS")) : L;
  if (layers < 1) layers = 1; if (layers > 64) layers = 64;

  float* d = nullptr;
  cudaMalloc((void**)&d, (size_t)T * H * sizeof(float));
  cudaStream_t compute;
  cudaStreamCreate(&compute);

  // Per-rank shard of one layer's params/grads (sharded across the DP group).
  const int shard = size > 1 ? GRAD_ELEMS / size : GRAD_ELEMS;

  // ZeRO-3 persistent footprint as cookies (mirrors capacity.py --zero 3): every
  // component sharded 1/size, plus one full-layer transient all-gather buffer.
  const long dbytes = getenv("DTYPE") ? atol(getenv("DTYPE")) : 2;
  float *w = nullptr, *g = nullptr, *opt = nullptr, *act = nullptr, *tr = nullptr;
  cudaMalloc((void**)&w,   (size_t)((long)shard * layers * dbytes));
  cudaMalloc((void**)&g,   (size_t)((long)shard * layers * dbytes));
  cudaMalloc((void**)&opt, (size_t)((long)shard * layers * 12));
  cudaMalloc((void**)&act, (size_t)(5L * T * H * dbytes * layers));
  cudaMalloc((void**)&tr,  (size_t)((long)GRAD_ELEMS * dbytes));   // transient gather
  (void)w; (void)g; (void)opt; (void)act; (void)tr;

  // Forward: each layer needs its full params gathered before compute. Prefetch
  // issues layer l+1's gather while layer l computes; otherwise it is exposed.
  MPI_Request greq;
  if (size > 1) MPI_Allgather(NULL, shard, MPI_FLOAT, NULL, shard, MPI_FLOAT, MPI_COMM_WORLD);
  for (int l = 0; l < layers; ++l) {
    if (size > 1 && prefetch && l + 1 < layers)
      MPI_Iallgather(NULL, shard, MPI_FLOAT, NULL, shard, MPI_FLOAT, MPI_COMM_WORLD, &greq);
    forward_layer(d, compute);
    cudaStreamSynchronize(compute);
    if (size > 1) {
      if (prefetch) { if (l + 1 < layers) MPI_Wait(&greq, MPI_STATUS_IGNORE); }
      else if (l + 1 < layers)
        MPI_Allgather(NULL, shard, MPI_FLOAT, NULL, shard, MPI_FLOAT, MPI_COMM_WORLD);
    }
  }

  // Backward (reverse): gather params, compute grads, reduce-scatter grads (kept as
  // each rank's shard). The reduce-scatter overlaps subsequent backward compute.
  MPI_Request rsreqs[64]; int nrs = 0;
  if (size > 1) MPI_Allgather(NULL, shard, MPI_FLOAT, NULL, shard, MPI_FLOAT, MPI_COMM_WORLD);
  for (int l = layers - 1; l >= 0; --l) {
    if (size > 1 && prefetch && l - 1 >= 0)
      MPI_Iallgather(NULL, shard, MPI_FLOAT, NULL, shard, MPI_FLOAT, MPI_COMM_WORLD, &greq);
    backward_layer(d, compute);
    cudaStreamSynchronize(compute);
    if (size > 1) {
      MPI_Ireduce_scatter_block(NULL, NULL, shard, MPI_FLOAT, MPI_SUM,
                                MPI_COMM_WORLD, &rsreqs[nrs++]);
      if (prefetch) { if (l - 1 >= 0) MPI_Wait(&greq, MPI_STATUS_IGNORE); }
      else if (l - 1 >= 0)
        MPI_Allgather(NULL, shard, MPI_FLOAT, NULL, shard, MPI_FLOAT, MPI_COMM_WORLD);
    }
  }
  if (size > 1) MPI_Waitall(nrs, rsreqs, MPI_STATUSES_IGNORE);
  cudaDeviceSynchronize();

  cudaStreamDestroy(compute);
  cudaFree(d);
  if (rank == 0)
    std::printf("mercury_fsdp: %d ranks (ZeRO-3), prefetch=%d, %d layers, shard=%d done\n",
                size, (int)prefetch, layers, shard);
  MPI_Finalize();
  return 0;
}
