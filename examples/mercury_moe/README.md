# mercury_moe — expert-parallel Mixture-of-Experts (the all-to-all workload)

The dense demos stress the fabric with the two collectives of dense training —
all-reduce (data parallelism, [`../mercury_llm_train`](../mercury_llm_train)) and
pipeline send/recv ([`../mercury_megatron`](../mercury_megatron),
[`../mercury_3d`](../mercury_3d)). A **Mixture-of-Experts** layer stresses it
through a *third* pattern: each token is routed to its top-`k` experts, which live
on different GPUs, so the dominant traffic is a pair of bandwidth-bound
**all-to-all**s per layer — a **dispatch** (tokens → the ranks holding their
experts) and a **combine** (results back to the origin). Expert parallelism puts
one expert per rank.

The question only a packet-level fabric can frame: real routing is **uneven**, so
some experts are hotspots. A skewed routing distribution makes the all-to-all
columns uneven — the hot expert's rank is *both* a fan-in hotspot on the fabric
*and* the compute straggler (it receives and processes the most tokens), and the
slowest rank gates the step. `SKEW ∈ [0,1]` sweeps that imbalance: `0` = uniform,
`1` = every token to expert 0 (a single hotspot).

## Modeled, never executed

As in the other demos: the attention and expert GEMMs carry a pinned per-thread
flop count (dual-roofline + wave-quantization in the GPU library; the matmul-class
kernels run at the tensor-core peak, see [`moe.py`](moe.py)), routing is a
**distribution** (`SKEW`), not emergent data, and both all-to-alls are NULL-buffer
`MPI_Alltoallv` calls timed purely from their per-peer element counts — no payload
ever moves. The hot expert's per-peer counts are what make its column heavy.

## Files

| File | What it is |
|------|------------|
| [`moe.cu`](moe.cu) | the app: dense attention (replicated) + `MPI_Alltoallv` dispatch → expert FFN → combine, with per-destination counts set by `SKEW`; block counts use 64-bit arithmetic (the hot expert's token count can exceed 32 bits) |
| [`moe.py`](moe.py) | the SST config: A100 dual-roofline params + a `LINK_BW` / `TOPO` fabric override |
| [`skew_sweep.sh`](skew_sweep.sh) | the study: routing-skew sweep (crossbar vs fat-tree), a scale sweep, and a bandwidth × skew contention sweep |

## Run it

```sh
# hg++ and sst on PATH; build + install the .so once:
hg++ -c moe.cu -o moe.o && hg++ moe.o -o libmercury_moe.so
cp libmercury_moe.so "$(dirname "$(dirname "$(command -v sst)")")/lib/sst-elements-library/ext/"

# a single point (run from examples/ so the shared modules are importable):
cd .. && NRANKS=8 TOPO=fattree SKEW=0.8 LAYERS=4 LINK_BW=100GB/s sst mercury_moe/moe.py

# the full study:
sh mercury_moe/skew_sweep.sh
```

`NRANKS` (= experts), `TOPO` (`single` crossbar | `fattree`), `LINK_BW`, `LAYERS`,
and `SKEW` are read by the config/skeleton, so the sweep never recompiles. Step
time is SST's `simulated time`. **Run from `examples/`** — the shared
`platform_file_hg_test` and `scale_topo` modules live there (the sweep script `cd`s
there for you).

## What it shows

All numbers: `N=8`, `LAYERS=4`, one expert/rank, top-2.

**1) Routing skew is costly — but at a realistic fabric it is the compute
straggler, not the network** (100 GB/s):

| `SKEW` | crossbar | fat-tree | fab ratio |
|--------|----------|----------|-----------|
| 0.0 (uniform) | 101 ms | 101 ms | 1.00 |
| 0.4 | 281 ms | 282 ms | 1.00 |
| 0.8 | 454 ms | 455 ms | 1.00 |
| 1.0 (one hotspot) | 478 ms | 479 ms | 1.00 |

Skew 0→0.8 is a **4.5× slowdown**, but the crossbar (no contention) and the
fat-tree (real contention) are *identical* — at 100 GB/s the all-to-all is cheap,
so the entire cost is the hot expert processing `size`× the tokens and gating the
step. This part a roofline-plus-analytical-all-to-all would also capture.

**2) The all-to-all's fabric contention appears only when bandwidth is starved**
(uniform routing, varying `LINK_BW`):

| `LINK_BW` | crossbar | fat-tree | fab ratio |
|-----------|----------|----------|-----------|
| 100 GB/s | 101 ms | 101 ms | 1.00 |
| 12 GB/s | 285 ms | 374 ms | **1.31** |
| 4 GB/s | 732 ms | 867 ms | **1.18** |

With *uniform* routing and a slow (cross-node) fabric, the all-to-all becomes
bandwidth-bound and the fat-tree's shared links cost **31% / 18%** more than a
contention-free crossbar carrying the same traffic. *This* is the packet-level
contribution an analytical all-to-all cost model structurally misses.

**3) The two effects trade — they do not stack.** Under heavy skew the straggler
dominates so completely that fabric contention disappears behind it:

| | crossbar | fat-tree | fab ratio |
|--|----------|----------|-----------|
| 12 GB/s, skew 0.8 | 1191 ms | 1193 ms | 1.00 |
| 4 GB/s, skew 0.8 | 2920 ms | 2987 ms | 1.02 |

At 12 GB/s the *uniform* fat-tree penalty was 31%, but at skew 0.8 it collapses to
~0% — the hot expert's compute straggler is now the entire critical path, and the
contended all-to-all hides behind it. So the regime where the fabric is the object
of study is **uniform / mild-skew + bandwidth-starved**; the regime where compute
dominates is **heavy skew**. Which one you are in is exactly the kind of question a
combined compute + packet-fabric model answers and a single-sided model cannot.

**4) Global skew makes the hotspot worse with scale** (skew 0.6, fat-tree):

| `N` | 4 | 8 | 16 | 32 |
|-----|----|----|----|----|
| step | 208 ms | 368 ms | 699 ms | 1376 ms |

Roughly linear in `N`: a globally-skewed routing distribution concentrates a fixed
*fraction* of every rank's tokens on expert 0, so the hot expert's absolute load —
and its straggler cost — grows with the cluster. Load-balancing the router (or
capacity-dropping) is what production MoE does to break this; the demo quantifies
what it costs not to.

## The questions this answers

- **What does routing imbalance actually cost?** At a fast fabric, a 4.5× step-time
  blow-up by skew 0.8 — and it is the hot expert's *compute*, not the network.
- **When does the all-to-all fabric matter?** Only once it is bandwidth-bound
  (cross-node links) *and* routing is near-uniform; then the fat-tree costs ~30%
  over a contention-free interconnect, which an analytical model misses.
- **Do hotspot and contention compound?** No — they trade. Heavy skew hides fabric
  contention behind the straggler; the packet model shows the crossover.

## Accuracy note

Same as the other demos: GEMM times are roofline estimates from a pinned flop
count; for cuBLAS-exact times, calibrate on real hardware and supply
`gpu_kernel_times` (see [`../mercury_cuda_halo`](../mercury_cuda_halo)). The
flop-per-thread literals track `H`, `T`, and `FF` — change them together. All
message *volumes* (the per-peer dispatch/combine element counts) are exact; only
the per-kernel compute is a roofline.
