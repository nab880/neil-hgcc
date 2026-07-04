// Example: data-parallel transformer training with overlapped gradient all-reduce; modeled, never executed.

#define ssthg_app_name mercury_llm_train
#include <skeleton.h>

#include <mask_mpi.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

#define H 4096
#define T 2048
#define L 8
#define FF (4 * H)
#define GRAD_ELEMS (12 * H * H)

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

// One transformer layer's forward kernels on the compute stream. d is a single
// device scratch cookie -- contents are never read (model don't execute), only
// the launch dimensions (the thread count) and the pinned costs matter.
static void forward_layer(float* d, cudaStream_t s) {
  const int blk = 256;
  layernorm   <<<grid_for(T * H,       blk), blk, 0, s>>>(d, d, T * H);
  qkv_proj    <<<grid_for(T * 3 * H,   blk), blk, 0, s>>>(d, d, d, T * 3 * H);
  attn_scores <<<grid_for(T * T,       blk), blk, 0, s>>>(d, d, d, T * T);
  softmax_rows<<<grid_for(T * T,       blk), blk, 0, s>>>(d, d, T * T);
  attn_av     <<<grid_for(T * H,       blk), blk, 0, s>>>(d, d, d, T * H);
  attn_out    <<<grid_for(T * H,       blk), blk, 0, s>>>(d, d, d, T * H);
  layernorm   <<<grid_for(T * H,       blk), blk, 0, s>>>(d, d, T * H);
  mlp_up      <<<grid_for(T * FF,      blk), blk, 0, s>>>(d, d, d, T * FF);
  gelu        <<<grid_for(T * FF,      blk), blk, 0, s>>>(d, d, T * FF);
  mlp_down    <<<grid_for(T * H,       blk), blk, 0, s>>>(d, d, d, T * H);
}

// Backward is ~2x forward GEMM flops (input-grad + weight-grad per forward GEMM).
static void backward_layer(float* d, cudaStream_t s) {
  forward_layer(d, s);
  forward_layer(d, s);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const char* gd = getenv("GPUDIRECT");
  const bool gpu_direct = !gd || gd[0] == 't' || gd[0] == '1';
  const char* ov = getenv("LLM_OVERLAP");
  const bool overlap = !ov || ov[0] == '1' || ov[0] == 't';
  // LAYERS overrides the transformer depth (default L). A lighter per-step
  // "timing skeleton" for scaling sweeps: fewer layers = fewer collectives, same
  // communication pattern, so large-NRANKS runs cost proportionally less
  // wall-clock without changing what is being studied.
  int layers = getenv("LAYERS") ? atoi(getenv("LAYERS")) : L;
  if (layers < 1) layers = 1;
  if (layers > 64) layers = 64;

  float* d = nullptr;
  cudaMalloc((void**)&d, (size_t)T * H * sizeof(float));
  cudaStream_t compute, comms;
  cudaStreamCreate(&compute);
  cudaStreamCreate(&comms);

  const size_t grad_bytes = (size_t)GRAD_ELEMS * sizeof(float);

  for (int l = 0; l < layers; ++l) forward_layer(d, compute);

  MPI_Request reqs[64];
  int nreq = 0;
  for (int l = layers - 1; l >= 0; --l) {
    backward_layer(d, compute);
    if (size == 1) continue;
    if (overlap) {
      if (!gpu_direct)
        cudaMemcpyAsync(d, d, grad_bytes, cudaMemcpyDeviceToHost, comms);
      MPI_Iallreduce(NULL, NULL, GRAD_ELEMS, MPI_FLOAT, MPI_SUM,
                     MPI_COMM_WORLD, &reqs[nreq++]);
    } else {
      cudaStreamSynchronize(compute);
      if (!gpu_direct)
        cudaMemcpyAsync(d, d, grad_bytes, cudaMemcpyDeviceToHost, comms);
      MPI_Allreduce(NULL, NULL, GRAD_ELEMS, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }
  }
  if (overlap) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
  cudaDeviceSynchronize();

  cudaStreamDestroy(compute);
  cudaStreamDestroy(comms);
  cudaFree(d);
  if (rank == 0)
    std::printf("mercury_llm_train: %d ranks, gpu_direct=%d, overlap=%d, %d layers done\n",
                size, (int)gpu_direct, (int)overlap, layers);
  MPI_Finalize();
  return 0;
}
