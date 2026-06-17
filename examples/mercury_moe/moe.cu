// Example: expert-parallel Mixture-of-Experts (MoE) training step under
// SST/Mercury -- the all-to-all workload. Where the dense demos stress the fabric
// with all-reduce (DP) and pipeline send/recv, an MoE layer routes each token to
// its top-k experts, which live on different GPUs, so the dominant traffic is a
// pair of bandwidth-bound ALL-TO-ALLs per layer: a dispatch (tokens -> the ranks
// holding their experts) and a combine (results back to the origin). Expert
// parallelism puts one expert per rank.
//
// The question only a packet-level fabric can answer: real routing is uneven, so
// some experts are hotspots. A skewed routing distribution makes the all-to-all
// columns uneven -- the hot expert's rank is both an injection/fan-in hotspot on
// the fabric AND the compute straggler (it receives and processes the most
// tokens), and the slowest rank gates the step. SKEW sweeps that imbalance.
//
// Modeled, never executed: GEMMs carry a pinned flop count (dual-roofline +
// wave-quant in the GPU library), routing is a distribution (SKEW), not emergent
// data, and the all-to-alls are NULL-buffer MPI_Alltoallv calls timed purely from
// their per-peer element counts. See README.md.

#define ssthg_app_name mercury_moe
#include <skeleton.h>

#include <mask_mpi.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>

#define H 4096            // hidden size
#define T 2048            // tokens in the batch (per rank)
#define FF (4 * H)        // expert FFN inner dimension
#define TOPK 2            // experts each token is routed to

// ---- kernels (all matmul-class -> tensor-core peak; see moe.py) -------------
#pragma sst gpu_compute flops(8192) intops(0) read(16) write(4)    // K = H
__global__ void attn(const float* x, const float* w, float* y, int n) {}

#pragma sst gpu_compute flops(8192) intops(0) read(16) write(4)    // up: K = H
__global__ void expert_up(const float* x, const float* w, float* y, int n) {}

#pragma sst gpu_compute flops(32768) intops(0) read(16) write(4)   // down: K = 4H
__global__ void expert_down(const float* x, const float* w, float* y, int n) {}

static inline int grid_for(int threads, int block) { return (threads + block - 1) / block; }

// Dense attention block (replicated per rank, no comm): 4 projections over the
// token batch. passes: 1 forward, 2 backward.
static void attention(float* d, cudaStream_t s, int passes) {
  const int blk = 256;
  for (int p = 0; p < passes; ++p) {
    attn<<<grid_for(T * 3 * H, blk), blk, 0, s>>>(d, d, d, T * 3 * H); // qkv
    attn<<<grid_for(T * T,     blk), blk, 0, s>>>(d, d, d, T * T);     // scores
    attn<<<grid_for(T * H,     blk), blk, 0, s>>>(d, d, d, T * H);     // a*v
    attn<<<grid_for(T * H,     blk), blk, 0, s>>>(d, d, d, T * H);     // out proj
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // One expert per rank (expert parallelism over MPI_COMM_WORLD). LAYERS sets the
  // MoE depth; SKEW in [0,1] is the routing imbalance: 0 = uniform, 1 = every
  // token to expert 0 (a single hotspot).
  int layers = getenv("LAYERS") ? atoi(getenv("LAYERS")) : 4;
  if (layers < 1) layers = 1; if (layers > 64) layers = 64;
  double skew = getenv("SKEW") ? atof(getenv("SKEW")) : 0.0;
  if (skew < 0.0) skew = 0.0; if (skew > 1.0) skew = 1.0;

  float* d = nullptr;
  cudaMalloc((void**)&d, (size_t)T * H * sizeof(float));
  cudaStream_t s;
  cudaStreamCreate(&s);

  // Per-destination token counts for one rank's dispatch. total_slots token-expert
  // pairs are spread over the `size` experts: a fraction `skew` concentrates on
  // expert 0, the rest is uniform. Counts are in ELEMENTS (tokens x H floats).
  const long total_slots = (long)T * TOPK;
  int* sendcounts = (int*)malloc(sizeof(int) * size);
  int* sdispls    = (int*)malloc(sizeof(int) * size);
  int* recvcounts = (int*)malloc(sizeof(int) * size);
  int* rdispls    = (int*)malloc(sizeof(int) * size);
  const double uniform = (1.0 - skew) * (double)total_slots / size;
  long num_to0 = (long)(skew * total_slots + uniform);   // tokens to hot expert 0
  long num_toj = (long)uniform;                          // tokens to any other
  for (int j = 0; j < size; ++j)
    sendcounts[j] = (int)((j == 0 ? num_to0 : num_toj) * H);
  // Every rank uses the same distribution, so the amount any source sends to MY
  // expert is the count that distribution assigns to destination == rank.
  int per_src_to_me = (int)((rank == 0 ? num_to0 : num_toj) * H);
  for (int j = 0; j < size; ++j) recvcounts[j] = per_src_to_me;
  sdispls[0] = rdispls[0] = 0;
  for (int j = 1; j < size; ++j) {
    sdispls[j] = sdispls[j-1] + sendcounts[j-1];
    rdispls[j] = rdispls[j-1] + recvcounts[j-1];
  }
  // Tokens this rank's expert actually processes (drives its compute) = sum of
  // what every rank sends to it. The hot expert gets `size` x the most -> straggler.
  long recv_tokens = (long)(rank == 0 ? num_to0 : num_toj) * size;
  if (recv_tokens < 1) recv_tokens = 1;

  // One MoE layer: dense attention, then dispatch -> expert FFN -> combine.
  // passes scales the GEMM work (1 forward, 2 backward).
  auto moe_layer = [&](int passes) {
    attention(d, s, passes);
    cudaStreamSynchronize(s);
    // dispatch all-to-all: tokens to the ranks holding their experts
    if (size > 1)
      MPI_Alltoallv(NULL, sendcounts, sdispls, MPI_FLOAT,
                    NULL, recvcounts, rdispls, MPI_FLOAT, MPI_COMM_WORLD);
    // expert FFN on the received tokens (load-imbalanced by routing). Output
    // elements -> launched threads: up = tokens x FF, down = tokens x H. Block
    // counts use long arithmetic; the hot expert's count can exceed 32 bits.
    const int blk = 256;
    for (int p = 0; p < passes; ++p) {
      long up_threads   = recv_tokens * (long)FF;   // up-projection H -> FF
      long down_threads = recv_tokens * (long)H;    // down-projection FF -> H
      expert_up  <<<(int)((up_threads   + blk - 1) / blk), blk, 0, s>>>(d, d, d, (int)recv_tokens);
      expert_down<<<(int)((down_threads + blk - 1) / blk), blk, 0, s>>>(d, d, d, (int)recv_tokens);
    }
    cudaStreamSynchronize(s);
    // combine all-to-all: results back to the origin ranks (reverse pattern)
    if (size > 1)
      MPI_Alltoallv(NULL, recvcounts, rdispls, MPI_FLOAT,
                    NULL, sendcounts, sdispls, MPI_FLOAT, MPI_COMM_WORLD);
  };

  for (int l = 0; l < layers; ++l) moe_layer(1);  // forward
  for (int l = 0; l < layers; ++l) moe_layer(2);  // backward (~2x GEMM work)

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
