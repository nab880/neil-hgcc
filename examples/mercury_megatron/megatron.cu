// Example: Megatron-style 3D-less (2D) parallel transformer training under
// SST/Mercury -- the second AI-at-scale workload, showing the OTHER two
// parallelism strategies and their very different comms patterns from the
// data-parallel demo (../mercury_llm_train):
//
//   * Tensor parallelism (TP): the weight matrices of each layer are sharded
//     across a TP group, so every layer does an all-reduce of the activations
//     over a SUB-communicator -- ON the forward/backward critical path (the next
//     op needs the combined result, so it CANNOT be hidden, unlike DP's gradient
//     all-reduce). This is why TP stays intra-node on NVLink.
//   * Pipeline parallelism (PP): the layers are split across stages, and
//     activations move stage-to-stage by POINT-TO-POINT send/recv. The cost is
//     the pipeline "bubble" -- idle fill/drain time -- which microbatching
//     amortizes: bubble fraction = (P-1)/(M+P-1).
//
// Ranks form a TP_SIZE x PP_SIZE grid (TP_SIZE*PP_SIZE = nranks). The question:
// for a fixed GPU budget, how should you split it between TP and PP, and how do
// the critical-path TP all-reduce and the PP bubble trade off against the fabric?
//
// Modeled, never executed: GEMMs carry a pinned flop count (sharded by TP_SIZE),
// elementwise kernels are P4 auto-derived, and every MPI message (TP all-reduce
// on a sub-comm, PP send/recv) is a NULL-buffer call timed from the element
// count -- no real payload. See README.md for the sweeps.

#define ssthg_app_name mercury_megatron
#include <skeleton.h>

#include <mask_mpi.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

#define H 4096          // hidden size
#define T 2048          // tokens per microbatch
#define L 8             // total transformer layers (split across PP stages)
#define FF (4 * H)
#define ACT (T * H)     // activation elements crossing a pipeline stage / TP all-reduce

// Sharded GEMM: with TP the weight matrix is split TP_SIZE ways, so each rank
// computes 1/TP_SIZE of the output threads (set at the launch) at the same
// flops/thread. Compute-bound (tiled), so the roofline charges flops/peak_flops.
#pragma sst gpu_compute flops(8192) intops(0) read(16) write(4)
__global__ void gemm(const float* a, const float* w, float* y, int n) {}

// Memory-bound, auto-derived (no pragma).
__global__ void act_norm(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * 0.5f + 1.0f;
}

static inline int grid_for(int threads, int block) { return (threads + block - 1) / block; }

// One tensor-parallel transformer layer: sharded GEMMs on the compute stream,
// then the Megatron all-reduce of the activations over the TP group -- twice
// (after the attention block, after the MLP block), each on the critical path.
// backward ~doubles the GEMM flops. tp_elems is the all-reduce element count.
static void tp_layer(float* d, cudaStream_t s, MPI_Comm tp, int tp_size,
                     int passes /*1 fwd, 2 bwd*/) {
  const int blk = 256;
  // attention block: sharded QKV + output projection
  for (int p = 0; p < passes; ++p) {
    gemm    <<<grid_for(T * 3 * H / tp_size, blk), blk, 0, s>>>(d, d, d, T * 3 * H / tp_size);
    gemm    <<<grid_for(T * H / tp_size,     blk), blk, 0, s>>>(d, d, d, T * H / tp_size);
  }
  act_norm<<<grid_for(T * H, blk), blk, 0, s>>>(d, d, T * H);
  cudaStreamSynchronize(s);
  if (tp_size > 1)                          // TP all-reduce #1 (after attention)
    MPI_Allreduce(NULL, NULL, ACT, MPI_FLOAT, MPI_SUM, tp);
  // MLP block: sharded up + down projection
  for (int p = 0; p < passes; ++p) {
    gemm    <<<grid_for(T * FF / tp_size, blk), blk, 0, s>>>(d, d, d, T * FF / tp_size);
    gemm    <<<grid_for(T * H / tp_size,  blk), blk, 0, s>>>(d, d, d, T * H / tp_size);
  }
  act_norm<<<grid_for(T * H, blk), blk, 0, s>>>(d, d, T * H);
  cudaStreamSynchronize(s);
  if (tp_size > 1)                          // TP all-reduce #2 (after MLP)
    MPI_Allreduce(NULL, NULL, ACT, MPI_FLOAT, MPI_SUM, tp);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Grid: TP_SIZE x PP_SIZE = size. tp_rank within the group, pp_rank = stage.
  const char* tps = getenv("TP_SIZE");
  int tp_size = tps ? atoi(tps) : 1;
  if (tp_size < 1) tp_size = 1;
  int pp_size = size / tp_size;             // pipeline stages
  if (pp_size < 1) { pp_size = 1; tp_size = size; }
  const int tp_rank = rank % tp_size;
  const int pp_rank = rank / tp_size;
  const int M = getenv("MICROBATCH") ? atoi(getenv("MICROBATCH")) : 4;
  const int layers_here = (L + pp_size - 1) / pp_size;   // layers on this stage

  // TP sub-communicator: ranks of one pipeline stage (same pp_rank).
  MPI_Comm tp_comm;
  MPI_Comm_split(MPI_COMM_WORLD, pp_rank, tp_rank, &tp_comm);
  // PP neighbours: same tp_rank, adjacent pp_rank.
  const int prev = (pp_rank - 1) * tp_size + tp_rank;
  const int next = (pp_rank + 1) * tp_size + tp_rank;

  float* d = nullptr;
  cudaMalloc((void**)&d, (size_t)T * H * sizeof(float));
  cudaStream_t s;
  cudaStreamCreate(&s);

  // GPipe: all M microbatches forward through the pipeline, then all backward.
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
