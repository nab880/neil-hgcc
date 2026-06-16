// Example: a data-parallel transformer LLM training step under SST/Mercury --
// the canonical "AI at scale" GPU workload. Each rank holds a replica of an
// L-layer transformer block and runs forward + backward over a token batch on
// the GPU; after each layer's backward, the layer's weight gradients are
// all-reduced across ranks (data parallelism) and that all-reduce is OVERLAPPED
// with the backward compute of the remaining layers (MPI_Iallreduce on a
// non-blocking path while the compute stream keeps working). The question this
// answers -- without a single real GPU -- is the one that decides whether DDP
// scales: does the gradient all-reduce stay hidden under backward compute, and
// what does GPUDirect / overlap / rank count do to that balance?
//
// Modeled, never executed: kernels carry their cost (compute-bound GEMMs via a
// #pragma sst gpu_compute flop count; memory-bound elementwise kernels are
// auto-derived by P4), the gradient all-reduce is a NULL-buffer MPI collective
// (timing from the element count, no real payload), and gradient staging over
// PCIe (when GPUDirect is off) is a time-only cudaMemcpyAsync. Sizes below are a
// representative GPT-class layer; flop-per-thread literals in the pragmas track
// H and T, so change them together.
//
// See README.md for the build + the rank x gpu_direct x overlap sweep.

#define ssthg_app_name mercury_llm_train
#include <skeleton.h>

#include <mask_mpi.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

// ---- model / batch dimensions (a ~GPT-7B-class layer) ----------------------
#define H 4096          // hidden size
#define T 2048          // tokens in the batch (per rank)
#define L 8             // transformer layers
#define FF (4 * H)      // MLP inner dimension
// Weight gradient elements all-reduced per layer (QKV 3H^2 + out H^2 + MLP
// 8H^2 = 12 H^2). Data-parallel syncs these every step.
#define GRAD_ELEMS (12 * H * H)

// ---- kernels ---------------------------------------------------------------
// GEMMs are compute-bound: a tiled kernel does 2*K flops per output element but,
// thanks to shared-memory reuse, only touches a few bytes/thread -- so the
// roofline charges flops/peak_flops. flops(2*K) is pinned per kernel (K = the
// contraction dim of that GEMM). One thread per output element; the launch sets
// the M*N thread count.
#pragma sst gpu_compute flops(8192) intops(0) read(16) write(4)   // K = H
__global__ void qkv_proj(const float* x, const float* W, float* y, int n) {}

#pragma sst gpu_compute flops(8192) intops(0) read(16) write(4)   // K = H
__global__ void attn_scores(const float* q, const float* k, float* s, int n) {}

#pragma sst gpu_compute flops(4096) intops(0) read(16) write(4)   // K = T
__global__ void attn_av(const float* s, const float* v, float* o, int n) {}

#pragma sst gpu_compute flops(8192) intops(0) read(16) write(4)   // K = H
__global__ void attn_out(const float* o, const float* W, float* y, int n) {}

#pragma sst gpu_compute flops(8192) intops(0) read(16) write(4)   // K = H
__global__ void mlp_up(const float* x, const float* W, float* h, int n) {}

#pragma sst gpu_compute flops(32768) intops(0) read(16) write(4)  // K = 4H
__global__ void mlp_down(const float* h, const float* W, float* y, int n) {}

// Elementwise / reductions are memory-bound -- no pragma, so P4 auto-derives the
// per-thread bytes (read + write) from the body and the roofline charges
// bytes/mem_bandwidth.
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

static inline int grid_for(int threads, int block) { return (threads + block - 1) / block; }

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

// Backward is ~2x the forward GEMM flops (an input-gradient GEMM and a
// weight-gradient GEMM per forward GEMM), so issue the layer's GEMMs twice.
static void backward_layer(float* d, cudaStream_t s) {
  forward_layer(d, s);
  forward_layer(d, s);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Knobs (env, so the sweep needs no recompile): GPUDIRECT toggles PCIe
  // staging of gradients; LLM_OVERLAP toggles compute/comms overlap.
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
  cudaMalloc((void**)&d, (size_t)T * H * sizeof(float));   // single scratch cookie
  cudaStream_t compute, comms;
  cudaStreamCreate(&compute);
  cudaStreamCreate(&comms);

  const size_t grad_bytes = (size_t)GRAD_ELEMS * sizeof(float);

  // Forward: all layers on the compute stream.
  for (int l = 0; l < layers; ++l) forward_layer(d, compute);

  // Backward (reverse), with each layer's gradient all-reduce overlapping the
  // backward compute of the layers still to come.
  MPI_Request reqs[64];
  int nreq = 0;
  for (int l = layers - 1; l >= 0; --l) {
    backward_layer(d, compute);
    if (size == 1) continue;            // single GPU: no data-parallel sync
    if (overlap) {
      // GPUDirect off: stage the gradient device->host over PCIe first (time
      // only). On: the NIC reads device memory directly, no staging.
      if (!gpu_direct)
        cudaMemcpyAsync(d, d, grad_bytes, cudaMemcpyDeviceToHost, comms);
      // NULL buffers -> the collective is modeled from the element count alone.
      MPI_Iallreduce(NULL, NULL, GRAD_ELEMS, MPI_FLOAT, MPI_SUM,
                     MPI_COMM_WORLD, &reqs[nreq++]);
    } else {
      // No overlap: finish this layer's compute, then block on the all-reduce.
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
