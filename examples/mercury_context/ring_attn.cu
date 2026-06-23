// Example: sequence / context parallelism (ring attention) under SST/Mercury --
// modeled, never executed. The other parallel axes (DP, TP, PP, expert) shard the
// model or the batch; context parallelism shards the SEQUENCE: each of N ranks holds
// S/N query tokens and its own S/N-token K/V block. Attention needs every query to
// see every key, so the K/V blocks are passed around a RING -- N steps, one neighbour
// hop each -- until every query has attended to all S keys (Ring Attention).
//
// The headline the simulator resolves: per ring step the K/V transfer is LINEAR in
// the block size (2*(S/N)*H) while the attention compute is QUADRATIC ((S/N)^2*H), so
// past some context length the compute dominates and the ring hides behind it. With
// CP_OVERLAP the next block's transfer is prefetched (Isend/Irecv) under the current
// step's compute -- the FSDP prefetch mechanic, but point-to-point round a ring rather
// than a reduction. Below the crossover the ring is exposed (overlap helps, bandwidth-
// bound); above it the ring is hidden (overlap stops mattering, compute-bound).
//
// Modeled, never executed: kernels carry pinned per-thread flop counts and the launch
// thread counts carry the S/N scaling, so the quadratic-vs-linear crossover is faithful
// without a recompile; the K/V exchange is a NULL-buffer point-to-point send/recv timed
// from its element count; weights + K/V block are device cookies (no real memory).
// Knobs: SEQ (context length), NRANKS (ring length), CP_OVERLAP, LINK_BW, LAYERS.

#define ssthg_app_name mercury_context
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

// Block counts use long arithmetic: at long context BQ*BQ and BQ*FF exceed 32 bits.
static inline int grid_l(long threads, int block) { return (int)((threads + block - 1) / block); }

// Bucket a NULL-buffer transfer of `count` MPI_FLOATs into chunks that fit the simulator's
// 32-bit BYTE counter (a K/V block exceeds 2^31 bytes past ~2M-token contexts; the chunk is
// 1GB). Posts non-blocking ops into reqs[]; returns how many. Both ring neighbours match.
static int post_blk(int isend, long count, int peer, int tag, MPI_Request* reqs) {
  const long CH = 256L * 1024 * 1024;
  int n = 0;
  for (long off = 0; off < count; off += CH) {
    int cc = (int)(count - off < CH ? count - off : CH);
    if (isend) MPI_Isend(NULL, cc, MPI_FLOAT, peer, tag, MPI_COMM_WORLD, &reqs[n++]);
    else       MPI_Irecv(NULL, cc, MPI_FLOAT, peer, tag, MPI_COMM_WORLD, &reqs[n++]);
  }
  return n;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const char* ov = getenv("CP_OVERLAP");
  const bool overlap = !ov || ov[0] == '1' || ov[0] == 't';
  int layers = getenv("LAYERS") ? atoi(getenv("LAYERS")) : L;
  if (layers < 1) layers = 1;
  if (layers > 64) layers = 64;
  long S = getenv("SEQ") ? atol(getenv("SEQ")) : 8192;
  if (S < (long)size) S = size;            // at least one token per rank
  const long dbytes = getenv("DTYPE") ? atol(getenv("DTYPE")) : 2;

  const int N = size;                      // ring length = number of ranks
  const long BQ = S / N;                   // local query tokens == local K/V block
  const int right = (rank + 1) % N;        // K/V blocks rotate left->right round the ring
  const int left  = (rank - 1 + N) % N;

  // K/V block shipped each ring step: K and V for BQ tokens, H wide, dtype bytes.
  // Linear in the block size -- the transfer side of the crossover. Counted in
  // MPI_FLOAT (4 B) units so the element count reproduces the modeled byte volume.
  const long kv_bytes  = (long)2 * BQ * H * dbytes;
  const long kv_floats = kv_bytes / 4;

  float* d = nullptr;
  cudaMalloc((void**)&d, (size_t)BQ * H * sizeof(float));
  cudaStream_t s;
  cudaStreamCreate(&s);

  // Footprint cookies: CP shards the sequence, not the model -- weights are full and
  // replicated; only the activations and K/V cache shrink to this rank's BQ tokens.
  float *w = nullptr, *kv = nullptr, *act = nullptr;
  cudaMalloc((void**)&w,   (size_t)((long)layers * 12 * H * H * dbytes));
  cudaMalloc((void**)&kv,  (size_t)((long)2 * layers * BQ * H * dbytes));
  cudaMalloc((void**)&act, (size_t)((long)5 * BQ * H * dbytes * layers));
  (void)w; (void)kv; (void)act;

  const int blk = 256;
  const int bq = (int)BQ;                   // kernels ignore n; only launch dims cost
  for (int l = 0; l < layers; ++l) {
    layernorm<<<grid_l(BQ * H,     blk), blk, 0, s>>>(d, d, bq * H);
    qkv_proj <<<grid_l(BQ * 3 * H, blk), blk, 0, s>>>(d, d, d, bq * 3 * H);

    // Ring attention: N steps. Each rank attends its BQ local queries against the
    // BQ-token K/V block currently in hand, then rotates blocks one hop round the
    // ring. After N steps every query has attended to all S keys. With overlap the
    // next block's transfer is issued before the step's compute and waited after, so
    // it hides behind compute; without it the exchange is exposed between steps.
    MPI_Request rreq[64], sreq[64]; int nr = 0, ns = 0;
    for (int r = 0; r < N; ++r) {
      const bool last = (r == N - 1);
      const int tag = l * N + r;
      if (overlap && !last) {
        nr = post_blk(0, kv_floats, left,  tag, rreq);
        ns = post_blk(1, kv_floats, right, tag, sreq);
      }
      const long scores = BQ * BQ;          // score matrix: quadratic in the block
      attn_scores <<<grid_l(scores, blk), blk, 0, s>>>(d, d, d, bq);
      softmax_rows<<<grid_l(scores, blk), blk, 0, s>>>(d, d, bq);
      attn_av     <<<grid_l(BQ * H, blk), blk, 0, s>>>(d, d, d, bq);
      cudaStreamSynchronize(s);
      if (!last) {
        if (overlap) {
          MPI_Waitall(nr, rreq, MPI_STATUSES_IGNORE);
          MPI_Waitall(ns, sreq, MPI_STATUSES_IGNORE);
        } else {                            // exposed: blocking rotate after compute
          const long CH = 256L * 1024 * 1024;
          for (long off = 0; off < kv_floats; off += CH) {
            int cc = (int)(kv_floats - off < CH ? kv_floats - off : CH);
            MPI_Sendrecv(NULL, cc, MPI_FLOAT, right, tag,
                         NULL, cc, MPI_FLOAT, left,  tag,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        }
      }
    }

    // Output projection, MLP, norms: all local to this rank's BQ tokens, no comm.
    attn_out<<<grid_l(BQ * H,  blk), blk, 0, s>>>(d, d, d, bq * H);
    layernorm<<<grid_l(BQ * H, blk), blk, 0, s>>>(d, d, bq * H);
    mlp_up  <<<grid_l(BQ * FF, blk), blk, 0, s>>>(d, d, d, bq * FF);
    gelu    <<<grid_l(BQ * FF, blk), blk, 0, s>>>(d, d, bq * FF);
    mlp_down<<<grid_l(BQ * H,  blk), blk, 0, s>>>(d, d, d, bq * H);
  }
  cudaDeviceSynchronize();

  cudaStreamDestroy(s);
  cudaFree(d);
  if (rank == 0)
    std::printf("mercury_context: %d ranks (ring), SEQ=%ld, block=%ld tokens/rank,"
                " overlap=%d, %d layers done\n", N, S, BQ, (int)overlap, layers);
  MPI_Finalize();
  return 0;
}
