// Example: 3D-parallel LLM training (DP x PP x TP) under SST/Mercury; modeled, never executed.

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
#define ACT (T * H)

#pragma sst gpu_compute flops(8192) intops(0) read(16) write(4)
__global__ void gemm(const float* a, const float* w, float* y, int n) {}

__global__ void act_norm(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * 0.5f + 1.0f;
}

static inline int grid_for(long threads, int block) { return (int)((threads + block - 1) / block); }

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

  // rank = dp*(PP*TP) + pp*TP + tp
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
  int nlayers = getenv("LAYERS") ? atoi(getenv("LAYERS")) : L;
  if (nlayers < 1) nlayers = 1;
  const int layers_here = (nlayers + pp_size - 1) / pp_size;

  // MICROBATCH is global; each DP replica runs M = global / dp_size microbatches.
  const int M_global = getenv("MICROBATCH") ? atoi(getenv("MICROBATCH")) : 4;
  const int M = (M_global / dp_size) > 0 ? M_global / dp_size : 1;

  MPI_Comm tp_comm, dp_comm;
  MPI_Comm_split(MPI_COMM_WORLD, dp_rank * pp_size + pp_rank, tp_rank, &tp_comm);
  MPI_Comm_split(MPI_COMM_WORLD, pp_rank * tp_size + tp_rank, dp_rank, &dp_comm);
  const int prev = rank - tp_size;
  const int next = rank + tp_size;

  // DP gradient sharded 1/TP per layer -> 12 H^2 elems/layer/TP. long: at L>=16 the
  // aggregate exceeds 2^31 and would overflow int (zeroing the all-reduce entirely).
  const long grad_elems = (long)layers_here * (12 * H / tp_size) * H;

  float* d = nullptr;
  cudaMalloc((void**)&d, (size_t)T * H * sizeof(float));
  cudaStream_t s;
  cudaStreamCreate(&s);

  // Device cookies mirror capacity.py; ZERO shards opt/grad/weight over the DP group.
  const int zero  = getenv("ZERO")  ? atoi(getenv("ZERO"))  : 0;
  const long dbytes = getenv("DTYPE") ? atol(getenv("DTYPE")) : 2;
  const long prank = (long)grad_elems;
  const long wd = zero >= 3 ? dp_size : 1;
  const long gd = zero >= 2 ? dp_size : 1;
  const long od = zero >= 1 ? dp_size : 1;
  float *w = nullptr, *g = nullptr, *opt = nullptr, *act = nullptr, *tr = nullptr;
  cudaMalloc((void**)&w,   (size_t)(prank * dbytes / wd));
  cudaMalloc((void**)&g,   (size_t)(prank * dbytes / gd));
  cudaMalloc((void**)&opt, (size_t)(prank * 12 / od));
  cudaMalloc((void**)&act, (size_t)(5L * T * H * dbytes / tp_size * nlayers));
  if (zero >= 3) cudaMalloc((void**)&tr, (size_t)(12L * H * H * dbytes));
  (void)w; (void)g; (void)opt; (void)act; (void)tr;

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

  // DP gradient all-reduce in fixed-size buckets (keeps counts inside 32-bit limits).
  if (dp_size > 1) {
    const long GRAD_CHUNK = 256L * 1024 * 1024;   // 1GB/chunk, empirically the safe ceiling (1.9GB crashes the NIC)
    for (long off = 0; off < grad_elems; off += GRAD_CHUNK) {
      int chunk = (int)(grad_elems - off < GRAD_CHUNK ? grad_elems - off : GRAD_CHUNK);
      MPI_Allreduce(NULL, NULL, chunk, MPI_FLOAT, MPI_SUM, dp_comm);
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
