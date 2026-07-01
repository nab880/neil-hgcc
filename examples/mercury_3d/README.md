# mercury_3d — 3D parallelism (data × pipeline × tensor), under Mercury

The capstone of the AI-at-scale set. The two prior demos showed the strategies in
isolation — data parallelism ([`../mercury_llm_train`](../mercury_llm_train)) and
tensor + pipeline parallelism ([`../mercury_megatron`](../mercury_megatron)). This
one runs **all three at once**, on a `DP × PP × TP` grid, the way real large-model
training is actually laid out (Megatron-LM / DeepSpeed "3D parallelism"). Each
rank is one GPU at a point `(dp_rank, pp_rank, tp_rank)`, and the three
communication patterns coexist and interact in a single training step:

- **Tensor (TP)** — per-layer activation all-reduce over a `tp_comm` sub-comm, on
  the forward/backward **critical path** (cannot hide).
- **Pipeline (PP)** — activations move stage-to-stage by point-to-point
  send/recv, GPipe microbatch schedule → the pipeline **bubble**.
- **Data (DP)** — the weight gradients are all-reduced over a `dp_comm` sub-comm
  after backward, **bucketed** the way PyTorch DDP reduces gradients.

The question it answers — the one a Megatron/DeepSpeed user actually faces — is:
**for a fixed GPU budget `N = DP·PP·TP`, which factorization wins?** The three
costs pull against each other, and they're *coupled*: the DP gradient volume per
rank **shrinks as `PP·TP` shards the model**, so the right DP is a function of how
much tensor/pipeline parallelism you already chose.

## Fixed-global-batch: a fair comparison

`MICROBATCH` is the **global** microbatch count, held fixed across the whole
sweep, and the `DP` replicas split it (`M_local = MICROBATCH / DP`). So every
factorization processes the *same batch* — DP shards the batch the way PP/TP shard
the model. Without this, raising DP would silently raise the global batch and DP
configs would look slow only because they did more work. With it, the only thing a
partition trades is **which communication it pays for**: the DP gradient
all-reduce, the TP critical-path all-reduce, or the PP bubble.

## Modeled, never executed

As in the prior demos: sharded GEMMs carry a pinned per-thread flop count
(`/ TP_SIZE`), elementwise kernels are P4 auto-derived, and **every** MPI message
— the TP and DP sub-comm all-reduces, the PP send/recv — is a NULL-buffer call
timed from its element count, with no real payload. The DP gradient all-reduce is
chunked into 64 Mi-element buckets: that both models DDP's real gradient bucketing
*and* keeps each collective's byte count inside 32-bit limits (the un-sharded
gradient reaches ~1.6 G elements = multi-GB for pure DP).

## Files

| File | What it is |
|------|------------|
| [`parallel3d.cu`](parallel3d.cu) | the app: a `DP×PP×TP` grid built from two `MPI_Comm_split`s (TP and DP groups) + a GPipe pipeline; TP all-reduce on the critical path, DP gradient all-reduce bucketed |
| [`parallel3d.py`](parallel3d.py) | the SST config: A100-class GPU params + a `LINK_BW` fabric override |
| [`run.sh`](run.sh) | build with `hg++`, install the `.so`, run the three sweeps below |

## Run it

```sh
./run.sh                  # hg++ and sst on PATH
# or a single point (DP = NRANKS/(TP·PP)):
NRANKS=8 TP_SIZE=2 PP_SIZE=2 MICROBATCH=8 LINK_BW=300GB/s sst parallel3d.py
```

`NRANKS` and `LINK_BW` are read by `parallel3d.py`; `TP_SIZE`, `PP_SIZE`, and
`MICROBATCH` by the skeleton (`DP_SIZE = NRANKS / (TP·PP)`). Step time is SST's
`simulated time`.

## What it shows

**1) The 3D split** (fixed 8-GPU budget, global batch M=8, 150 GB/s). Every
factorization does equal work, so this is purely a contest of communication cost:

| layout (DP×PP×TP) | step time | what dominates |
|-------------------|-----------|----------------|
| **DP1 × PP8 × TP1** | **104 ms** | pure pipeline — least collective comms, bubble amortized at M=8 |
| DP2 × PP4 × TP1 | 104 ms | pipeline + a little DP |
| DP2 × PP2 × TP2 | 111 ms | balanced 3-way |
| DP1 × PP4 × TP2 | 113 ms | pipeline + tensor |
| DP4 × PP2 × TP1 | 114 ms | pipeline + more DP |
| DP4 × PP1 × TP2 | 114 ms | tensor + DP |
| DP8 × PP1 × TP1 | 138 ms | pure data — full un-sharded gradient all-reduce |
| DP2 × PP1 × TP4 | 155 ms | tensor-heavy |
| DP1 × PP2 × TP4 | 166 ms | tensor-heavy |
| DP1 × PP1 × TP8 | 296 ms | pure tensor — TP all-reduce on the critical path, ×2/layer, never hides |

The ordering is the whole lesson: **sharding the model (PP, then a balanced mix)
beats both extremes.** Pure tensor parallelism is worst — its all-reduce is on the
critical path and can't be hidden (2.8× the best). Pure data parallelism is
middling — its bucketed gradient all-reduce *can* overlap and the volume is
tolerable at 150 GB/s, but it's the largest single message. The sweet spot is
pipeline-dominant with just enough DP/TP to fill the budget.

**2) Pure data parallelism's gradient all-reduce vs the fabric** (DP=8, PP=1,
TP=1, M=8). With no model sharding, every rank holds the full gradient, so this is
the largest collective in the whole study — and it's bandwidth-bound:

| `LINK_BW` | 12 GB/s | 50 GB/s | 150 GB/s | 600 GB/s |
|-----------|---------|---------|----------|----------|
| step time | 1.00 s | 289 ms | 138 ms | 82 ms |

12→600 GB/s is a 12× swing — almost all of the 1.00 s at 12 GB/s is the
multi-GB gradient all-reduce. At 600 GB/s it nearly vanishes (82 ms ≈ the compute
floor), because unlike TP's all-reduce, DP's *can* overlap.

**3) The best split moves with the fabric** (M=8). Slow (cross-node) fabric
punishes whoever communicates most; fast (NVLink) fabric forgives it. So the
optimal partition shifts toward the strategy that talks least — pipeline — as the
fabric slows:

| layout | 12 GB/s | 300 GB/s | fabric sensitivity |
|--------|---------|----------|--------------------|
| DP1 × PP8 × TP1 (pipeline) | **176 ms** | 101 ms | 1.7× |
| DP1 × PP1 × TP8 (tensor) | 1.45 s | 247 ms | 5.9× |
| DP8 × PP1 × TP1 (data) | 1.00 s | **102 ms** | 9.9× |

At **12 GB/s** (a cross-node fabric) pipeline parallelism wins by a landslide:
it moves the least data on the critical path, so it's barely slowed (1.7×), while
data and tensor parallelism are *an order of magnitude* slower. At **300 GB/s**
(intra-node NVLink) the picture moves toward parity — data parallelism is the most
bandwidth-sensitive (9.9×), so fast fabric makes its bucketed gradient all-reduce
cheap and it **ties pipeline at ~101 ms**; tensor parallelism stays ~2.4× worse
because its all-reduce is on the critical path and no bandwidth fully hides it.
So the right strategy tracks the fabric: **pipeline everywhere, data parallelism
once you have NVLink, tensor parallelism only inside the fastest domain** — which
is exactly the deployment rule the big-model training stacks follow.

## The questions this answers

- **Given N GPUs, how do I split them three ways?** Shard the model first
  (pipeline is cheapest to communicate), add tensor parallelism only inside a fast
  domain, and use data parallelism for whatever GPUs are left — its gradient
  all-reduce shrinks as the model gets sharded.
- **Why not just crank up one strategy?** Each extreme has a wall: TP's
  critical-path all-reduce (worst here), the PP bubble (needs microbatches), and
  DP's full-gradient all-reduce (biggest message).
- **How do the three couple?** The DP gradient volume per rank is
  `(model params) / (PP · TP)` — so the more you shard with pipeline and tensor
  parallelism, the cheaper data parallelism gets. The factorization is a single
  joint optimization, which is exactly why this is run as one combined sweep.

## Contention & algorithm study (WS2-2d)

`factorization_sweep.sh` sweeps every DP×PP×TP factorization of a fixed GPU budget
N and finds the fastest, on a single crossbar (no contention — the control) and on
the fat-tree (real packet contention), for both all-reduce algorithms
(recursive-doubling and `SUMI_ALLREDUCE_ALG=ring`). It asks whether the network
chooses the parallelization.

Result (N=16, LAYERS=2, MICROBATCH=8; compute-bound 50 GB/s and comms-bound 12 GB/s):

| config | optimum (50 GB/s) | optimum (12 GB/s) |
|--------|-------------------|-------------------|
| crossbar, recursive-doubling | DP4×PP2×TP2 | DP4×PP2×TP2 |
| fat-tree, recursive-doubling | DP4×PP2×TP2 | DP4×PP2×TP2 |
| fat-tree, ring               | DP4×PP2×TP2 | DP4×PP2×TP2 |

Two findings:

1. **The optimum is contention-robust.** The balanced, comms-light DP4×PP2×TP2 wins
   under *every* topology × algorithm × bandwidth combination. It already minimises
   contended traffic (small DP gradient + small TP activation all-reduce), so it is
   insensitive to how the fabric handles large messages.
2. **Contention and the algorithm reshape the *penalty landscape*, not the peak.**
   The comms-heavy factorizations are punished far more on the fat-tree under ring —
   at 12 GB/s, pure-TP (1×1×16) 390→684 ms (+75%) and pure-DP (16×1×1) 268→406 ms
   (+51%) — while recursive-doubling stays within ~2% of the crossbar. Ring is much
   more contention-sensitive than recursive-doubling.

This is exactly what an analytical (no-contention) model gets wrong: it would
under-cost the comms-heavy configurations and be blind to the algorithm×topology
interaction. The robustness of the optimum *and* the steep, fabric-dependent cost
of a poor choice are both packet-level results. (A scale at which the optimum
itself shifts is plausible at larger N / lower bandwidth — `N=32 sh
factorization_sweep.sh` — but is heavier to run on one host.)

## Accuracy note

Same as the prior demos: GEMM times are roofline estimates from a pinned flop
count; for cuBLAS-exact times, calibrate the GEMM kernels on real hardware and
supply `gpu_kernel_times` (see [`../mercury_cuda_halo`](../mercury_cuda_halo) and
[`../../docs/gpu_calibration.schema.json`](../../docs/gpu_calibration.schema.json)).
The flop-per-thread literals track `H` and `T` — change them together. Message
*volumes* (activation and gradient element counts) are exact; only the per-kernel
compute is a roofline.
