// Example: 3D-parallel LLM training under SST/Mercury -- the capstone that runs
// data + tensor + pipeline parallelism together, the way real large-model
// training (Megatron-LM / DeepSpeed) is laid out. Each rank is one GPU at a
// point (dp_rank, pp_rank, tp_rank) of a DP x PP x TP grid, and all three
// communication patterns coexist and interact in one step:
//
//   * Tensor parallelism (TP): per-layer activation all-reduce over a tp_comm
//     sub-communicator, ON the forward/backward critical path (can't hide).
//   * Pipeline parallelism (PP): activations move stage-to-stage by point-to-point
//     send/recv, GPipe microbatch schedule -> the pipeline bubble.
//   * Data parallelism (DP): the weight gradients are all-reduced over a dp_comm
//     sub-communicator after backward (overlaps the pipeline drain).
//
// The question: for a fixed GPU budget N = DP*PP*TP, which factorization wins?
// The three costs pull against each other -- TP comms grows with TP (critical
// path), the PP bubble grows with PP (amortized by microbatches), and the DP
// all-reduce volume per rank SHRINKS as TP*PP shards the model. See README.md.
//
// Modeled, never executed: sharded GEMMs carry a pinned flop count (/ TP_SIZE),
// elementwise is P4 auto-derived, and every MPI message (TP/DP sub-comm
// all-reduce, PP send/recv) is a NULL-buffer call timed from the element count.

#define ssthg_app_name mercury_3d
#include <skeleton.h>

#include <mask_mpi.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

#define H 4096
#define T 2048
#define L 8
#define FF (4 * H)
#define ACT (T * H)                 // activation all-reduce / pipeline message size

#pragma sst gpu_compute flops(8192) intops(0) read(16) write(4)
__global__ void gemm(const float* a, const float* w, float* y, int n) {}

__global__ void act_norm(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * 0.5f + 1.0f;
}

static inline int grid_for(int threads, int block) { return (threads + block - 1) / block; }

// One tensor-parallel layer: sharded GEMMs + the two Megatron activation
// all-reduces over the TP group (critical path). passes: 1 forward, 2 backward.
static void tp_layer(float* d, cudaStream_t s, MPI_Comm tp, int tp_size, int passes) {
  const int blk = 256;
  for (int p = 0; p < passes; ++p) {
    gemm<<<grid_for(T * 3 * H / tp_size, blk), blk, 0, s>>>(d, d, d, T * 3 * H / tp_size);
    gemm<<<grid_for(T * H / tp_size,     blk), blk, 0, s>>>(d, d, d, T * H / tp_size);
  }
  act_norm<<<grid_for(T * H, blk), blk, 0, s>>>(d, d, T * H);
  cudaStreamSynchronize(s);
  if (tp_size > 1) MPI_Allreduce(NULL, NULL, ACT, MPI_FLOAT, MPI_SUM, tp);
  for (int p = 0; p < passes; ++p) {
    gemm<<<grid_for(T * FF / tp_size, blk), blk, 0, s>>>(d, d, d, T * FF / tp_size);
    gemm<<<grid_for(T * H / tp_size,  blk), blk, 0, s>>>(d, d, d, T * H / tp_size);
  }
  act_norm<<<grid_for(T * H, blk), blk, 0, s>>>(d, d, T * H);
  cudaStreamSynchronize(s);
  if (tp_size > 1) MPI_Allreduce(NULL, NULL, ACT, MPI_FLOAT, MPI_SUM, tp);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // 3D grid: rank = dp*(PP*TP) + pp*TP + tp. TP_SIZE and PP_SIZE from the env;
  // DP_SIZE is whatever is left of the rank budget.
  const char* tps = getenv("TP_SIZE");
  const char* pps = getenv("PP_SIZE");
  int tp_size = tps ? atoi(tps) : 1; if (tp_size < 1) tp_size = 1;
  int pp_size = pps ? atoi(pps) : 1; if (pp_size < 1) pp_size = 1;
  if (tp_size * pp_size > size) { tp_size = 1; pp_size = 1; }
  int dp_size = size / (tp_size * pp_size);
  if (dp_size < 1) dp_size = 1;

  const int tp_rank = rank % tp_size;
  const int pp_rank = (rank / tp_size) % pp_size;
  const int dp_rank = rank / (tp_size * pp_size);
  // LAYERS overrides model depth (default L) -- a timing skeleton for scaling
  // sweeps: fewer layers = less per-step work, same comms pattern.
  int nlayers = getenv("LAYERS") ? atoi(getenv("LAYERS")) : L;
  if (nlayers < 1) nlayers = 1;
  const int layers_here = (nlayers + pp_size - 1) / pp_size;

  // MICROBATCH is the GLOBAL microbatch count -- the work held fixed across the
  // sweep so every DP x PP x TP factorization processes the same batch (the
  // textbook fixed-global-batch comparison). The DP replicas split it, so each
  // pipeline runs M = global / dp_size microbatches. This makes the comparison
  // fair: DP shards the batch (less compute per replica) the way PP/TP shard the
  // model, so the only thing the partition trades is which communication it pays.
  const int M_global = getenv("MICROBATCH") ? atoi(getenv("MICROBATCH")) : 4;
  const int M = (M_global / dp_size) > 0 ? M_global / dp_size : 1;

  // tp_comm: ranks of one (dp,pp) cell, varying tp_rank.
  MPI_Comm tp_comm;
  MPI_Comm_split(MPI_COMM_WORLD, dp_rank * pp_size + pp_rank, tp_rank, &tp_comm);
  // dp_comm: ranks of one (pp,tp) cell, varying dp_rank -> the gradient group.
  MPI_Comm dp_comm;
  MPI_Comm_split(MPI_COMM_WORLD, pp_rank * tp_size + tp_rank, dp_rank, &dp_comm);
  // PP neighbours: same dp_rank and tp_rank, adjacent pp_rank -> +/- tp_size.
  const int prev = rank - tp_size;
  const int next = rank + tp_size;

  // DP gradient = this stage's layers, each sharded 1/TP -> 12 H^2 elems/layer/TP.
  const int grad_elems = layers_here * (12 * H / tp_size) * H;

  float* d = nullptr;
  cudaMalloc((void**)&d, (size_t)T * H * sizeof(float));
  cudaStream_t s;
  cudaStreamCreate(&s);

  // Persistent per-rank training footprint as device cookies (no real memory) so
  // the GPU library's high-water reports what this config needs; mirrors
  // capacity.py. ZERO shards opt(1)/+grad(2)/+weight(3) over the DP group.
  const int zero  = getenv("ZERO")  ? atoi(getenv("ZERO"))  : 0;
  const long dbytes = getenv("DTYPE") ? atol(getenv("DTYPE")) : 2;   // bf16 default
  const long prank = (long)grad_elems;                              // params/rank (TP,PP sharded)
  const long wd = zero >= 3 ? dp_size : 1;
  const long gd = zero >= 2 ? dp_size : 1;
  const long od = zero >= 1 ? dp_size : 1;
  float *w = nullptr, *g = nullptr, *opt = nullptr, *act = nullptr, *tr = nullptr;
  cudaMalloc((void**)&w,   (size_t)(prank * dbytes / wd));
  cudaMalloc((void**)&g,   (size_t)(prank * dbytes / gd));
  cudaMalloc((void**)&opt, (size_t)(prank * 12 / od));             // fp32 master+m+v
  cudaMalloc((void**)&act, (size_t)(5L * T * H * dbytes / tp_size * nlayers)); // 1F1B, selective ckpt
  if (zero >= 3) cudaMalloc((void**)&tr, (size_t)(12L * H * H * dbytes)); // transient all-gather
  (void)w; (void)g; (void)opt; (void)act; (void)tr;

  // Forward GPipe over the pipeline stages.
  for (int m = 0; m < M; ++m) {
    if (pp_rank > 0)
      MPI_Recv(NULL, ACT, MPI_FLOAT, prev, m, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int l = 0; l < layers_here; ++l) tp_layer(d, s, tp_comm, tp_size, 1);
    if (pp_rank < pp_size - 1)
      MPI_Send(NULL, ACT, MPI_FLOAT, next, m, MPI_COMM_WORLD);
  }
  // Backward GPipe (reverse).
  for (int m = 0; m < M; ++m) {
    if (pp_rank < pp_size - 1)
      MPI_Recv(NULL, ACT, MPI_FLOAT, next, M + m, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int l = 0; l < layers_here; ++l) tp_layer(d, s, tp_comm, tp_size, 2);
    if (pp_rank > 0)
      MPI_Send(NULL, ACT, MPI_FLOAT, prev, M + m, MPI_COMM_WORLD);
  }

  // Data-parallel gradient all-reduce after backward, bucketed the way real DDP
  // reduces gradients: fixed-size buckets rather than one giant message. (Real
  // DDP fires each bucket as it fills during backward to overlap; here they
  // follow backward -- the exposed gradient cost.) The whole-gradient count
  // reaches ~1.6e9 elems for pure DP (no TP/PP sharding) = a multi-GB message;
  // bucketing also keeps every collective's byte count well inside 32-bit
  // limits. Skipped when there is no DP replica.
  if (dp_size > 1) {
    const int GRAD_CHUNK = 64 * 1024 * 1024;   // 64 Mi elems = 256 MB/bucket
    for (int off = 0; off < grad_elems; off += GRAD_CHUNK) {
      int chunk = grad_elems - off < GRAD_CHUNK ? grad_elems - off : GRAD_CHUNK;
      MPI_Request grad_req;
      MPI_Iallreduce(NULL, NULL, chunk, MPI_FLOAT, MPI_SUM, dp_comm, &grad_req);
      MPI_Wait(&grad_req, MPI_STATUS_IGNORE);
    }
  }

  cudaStreamDestroy(s);
  cudaFree(d);
  MPI_Comm_free(&tp_comm);
  MPI_Comm_free(&dp_comm);
  if (rank == 0)
    std::printf("mercury_3d: DP=%d x PP=%d x TP=%d (=%d GPUs), %d global microbatches"
                " (%d/replica), %d layers/stage done\n",
                dp_size, pp_size, tp_size, size, M_global, M, layers_here);
  MPI_Finalize();
  return 0;
}
