// Example: disaggregated LLM inference under SST/Mercury -- modeled, never executed.
// Two GPU pools share one fabric: PREFILL workers process a whole prompt at once
// (compute-heavy GEMMs, quadratic in the prompt length) and ship the resulting
// KV cache to DECODE workers, which then generate tokens autoregressively. Decode
// is the opposite regime to training: its GEMMs are tiny (one token), so each step
// is dominated by a small, latency-critical tensor-parallel all-reduce -- the other
// end of the study-(E) crossover -- and the bulk KV-cache transfer is a new pattern
// that contends with decode traffic on the fabric.
//
// Modeled, never executed: kernels carry pinned flop counts; the KV transfer and the
// decode TP all-reduce are NULL-buffer MPI calls timed from their element counts; the
// per-rank weights + KV cache are device cookies (no real memory) so the decode
// footprint cross-checks examples/memory_model/capacity.py --infer. See README.md.

#define ssthg_app_name mercury_inference
#include <skeleton.h>

#include <mask_mpi.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

#define H 4096
#define L 8
#define FF (4 * H)

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

// Bucket a NULL-buffer point-to-point transfer of `count` MPI_FLOATs into chunks that fit
// the simulator's 32-bit BYTE counter (the full KV cache exceeds 2^31 at large L/S/B; the
// chunk is 1GB = <2^31 bytes). Posts non-blocking ops into reqs[]; returns how many. Both
// peers compute count identically, so the chunk decompositions match.
static int post_kv(int isend, long count, int peer, int tag, MPI_Request* reqs) {
  const long CH = 256L * 1024 * 1024;
  int n = 0;
  for (long off = 0; off < count; off += CH) {
    int cc = (int)(count - off < CH ? count - off : CH);
    if (isend) MPI_Isend(NULL, cc, MPI_FLOAT, peer, tag, MPI_COMM_WORLD, &reqs[n++]);
    else       MPI_Irecv(NULL, cc, MPI_FLOAT, peer, tag, MPI_COMM_WORLD, &reqs[n++]);
  }
  return n;
}

// Prefill: one forward pass over the whole prompt (S tokens), sharded by decode TP.
// Attention is quadratic in S -- this is the compute-heavy phase.
static void prefill_layer(float* d, cudaStream_t s, int S, int tp) {
  const int blk = 256;
  layernorm   <<<grid_for(S * H,       blk), blk, 0, s>>>(d, d, S * H);
  qkv_proj    <<<grid_for(S * 3 * H/tp,blk), blk, 0, s>>>(d, d, d, S * 3 * H / tp);
  attn_scores <<<grid_for(S * S / tp,  blk), blk, 0, s>>>(d, d, d, S * S / tp);
  softmax_rows<<<grid_for(S * S / tp,  blk), blk, 0, s>>>(d, d, S * S / tp);
  attn_av     <<<grid_for(S * H / tp,  blk), blk, 0, s>>>(d, d, d, S * H / tp);
  attn_out    <<<grid_for(S * H / tp,  blk), blk, 0, s>>>(d, d, d, S * H / tp);
  layernorm   <<<grid_for(S * H,       blk), blk, 0, s>>>(d, d, S * H);
  mlp_up      <<<grid_for(S * FF / tp, blk), blk, 0, s>>>(d, d, d, S * FF / tp);
  gelu        <<<grid_for(S * FF / tp, blk), blk, 0, s>>>(d, d, S * FF / tp);
  mlp_down    <<<grid_for(S * H / tp,  blk), blk, 0, s>>>(d, d, d, S * H / tp);
}

// Decode: one autoregressive step over B tokens, attending to S_kv cached keys.
// GEMMs are tiny (B tokens) -> memory-bound; the per-layer TP all-reduce of the B*H
// activation is small and on the critical path -> latency-bound.
static void decode_layer(float* d, cudaStream_t s, MPI_Comm tp_comm, int tp, int B, int S_kv) {
  const int blk = 256;
  qkv_proj    <<<grid_for(B * 3 * H / tp, blk), blk, 0, s>>>(d, d, d, B * 3 * H / tp);
  attn_scores <<<grid_for(B * S_kv / tp,  blk), blk, 0, s>>>(d, d, d, B * S_kv / tp); // read KV
  attn_av     <<<grid_for(B * H / tp,     blk), blk, 0, s>>>(d, d, d, B * H / tp);
  attn_out    <<<grid_for(B * H / tp,     blk), blk, 0, s>>>(d, d, d, B * H / tp);
  mlp_up      <<<grid_for(B * FF / tp,    blk), blk, 0, s>>>(d, d, d, B * FF / tp);
  mlp_down    <<<grid_for(B * H / tp,     blk), blk, 0, s>>>(d, d, d, B * H / tp);
  cudaStreamSynchronize(s);
  if (tp > 1) MPI_Allreduce(NULL, NULL, B * H, MPI_FLOAT, MPI_SUM, tp_comm); // latency-critical
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int nlayers = getenv("LAYERS") ? atoi(getenv("LAYERS")) : L;
  if (nlayers < 1) nlayers = 1; if (nlayers > 64) nlayers = 64;
  int S       = getenv("PROMPT_LEN")  ? atoi(getenv("PROMPT_LEN"))  : 2048;
  int steps   = getenv("DECODE_STEPS")? atoi(getenv("DECODE_STEPS")): 64;
  int B       = getenv("BATCH")       ? atoi(getenv("BATCH"))       : 1;
  int reqs    = getenv("REQUESTS")    ? atoi(getenv("REQUESTS"))    : 4;
  const long dbytes = getenv("DTYPE") ? atol(getenv("DTYPE")) : 2;
  // PREFILL_RANKS prefill workers; the rest are one decode tensor-parallel group.
  // PREFILL_RANKS=0 is the co-located control: decode with no KV-transfer traffic.
  int P = getenv("PREFILL_RANKS") ? atoi(getenv("PREFILL_RANKS")) : (size > 1 ? 1 : 0);
  if (P > size - 1 && size > 1) P = size - 1;
  if (P < 0) P = 0;
  const int Dn = size - P;                 // decode pool size = decode TP degree
  const bool is_prefill = rank < P;

  MPI_Comm pool;
  MPI_Comm_split(MPI_COMM_WORLD, is_prefill ? 0 : 1, rank, &pool);

  float* d = nullptr;
  cudaMalloc((void**)&d, (size_t)S * H * sizeof(float));
  cudaStream_t s;
  cudaStreamCreate(&s);

  // KV cache for one request, sharded by the decode TP degree. Transferred over the
  // fabric in MPI_FLOAT (4 B) units, so the count reproduces the modeled byte volume.
  const long kv_bytes  = (long)2 * nlayers * S * H * dbytes * B / (Dn > 0 ? Dn : 1);
  const long kv_floats = kv_bytes / 4;

  // Decode footprint as cookies: weights (TP-sharded) + KV cache; cross-checks
  // capacity.py --infer --no-embed at the same dims.
  if (!is_prefill) {
    const int tp = Dn > 0 ? Dn : 1;
    float *w = nullptr, *kv = nullptr;
    cudaMalloc((void**)&w,  (size_t)((long)nlayers * 12 * H / tp * H * dbytes));
    cudaMalloc((void**)&kv, (size_t)kv_bytes);
    (void)w; (void)kv;
  }

  for (int r = 0; r < reqs; ++r) {
    if (is_prefill) {
      for (int l = 0; l < nlayers; ++l) prefill_layer(d, s, S, 1);
      cudaStreamSynchronize(s);
      int dst = P + (rank % (Dn > 0 ? Dn : 1));      // paired decode rank
      MPI_Request sreqs[1024];
      int ns = post_kv(1, kv_floats, dst, r, sreqs);
      MPI_Waitall(ns, sreqs, MPI_STATUSES_IGNORE);
    } else {
      // Prefetch this request's KV while the previous request still decodes -> the
      // bulk transfer contends with decode's TP all-reduces on the fabric.
      int src = rank - P;
      bool recv = src < P;
      MPI_Request kvreqs[1024]; int nkv = 0;
      if (recv) nkv = post_kv(0, kv_floats, src, r, kvreqs);
      for (int t = 0; t < steps; ++t)
        for (int l = 0; l < nlayers; ++l)
          decode_layer(d, s, pool, Dn, B, S + t);
      if (recv) MPI_Waitall(nkv, kvreqs, MPI_STATUSES_IGNORE);
    }
  }
  cudaDeviceSynchronize();

  cudaStreamDestroy(s);
  cudaFree(d);
  MPI_Comm_free(&pool);
  if (rank == 0)
    std::printf("mercury_inference: %d prefill + %d decode (TP=%d), prompt=%d, %d decode steps,"
                " %d requests, %d layers done\n", P, Dn, Dn, S, steps, reqs, nlayers);
  MPI_Finalize();
  return 0;
}
