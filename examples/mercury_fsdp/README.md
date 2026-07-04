# mercury_fsdp — fully-sharded data parallelism (FSDP / ZeRO-3)

Study (G) ([`../memory_model`](../memory_model)) showed FSDP/ZeRO-3 is what makes
data parallelism *fit* — sharding parameters, gradients, and optimizer across the DP
group turns a model that OOMs into one that runs. This demo measures what that
costs in *time*. Where plain DP ([`../mercury_llm_train`](../mercury_llm_train))
replicates the model and pays one post-backward gradient **all-reduce** (which
overlaps backward), FSDP pays, **per layer**, a parameter **all-gather** on the
forward *and* backward critical path plus a gradient **reduce-scatter**.

The question FSDP's viability rests on: **does prefetching the next layer's
all-gather hide it behind the current layer's compute?** `FSDP_PREFETCH` is that knob.

## Modeled, never executed

Same transformer kernels and dims as the DP demo (H=4096, L=8). Parameters are
sharded across the `N`-rank DP group; per layer the shard is `GRAD_ELEMS/N`. The
all-gather (`MPI_Iallgather`) and reduce-scatter (`MPI_Ireduce_scatter_block`) are
NULL-buffer collectives timed from their per-rank shard counts. The persistent
ZeRO-3 state is allocated as device cookies (no real memory), so the per-rank
`mem_footprint` cross-checks `capacity.py --zero 3` exactly. Implementing this
required a real **ring reduce-scatter** in the sumi backend (it was a stub).

## Files

| File | What it is |
|------|------------|
| [`fsdp.cu`](fsdp.cu) | per-layer all-gather → compute → (backward) reduce-scatter, with a one-layer prefetch pipeline; ZeRO-3 footprint cookies |
| [`fsdp.py`](fsdp.py) | SST config (A100 dual-roofline + `LINK_BW` + `gpu_mem_capacity`), mirrors `train.py` |
| [`run.sh`](run.sh) | build + sweep FSDP (prefetch on/off) vs the plain-DP baseline across bandwidth |

## Run it

```sh
./run.sh                                   # hg++ and sst on PATH
# one point (from examples/, where the shared modules live):
cd .. && NRANKS=8 LAYERS=8 FSDP_PREFETCH=1 LINK_BW=100GB/s GPU_MEM=80GB sst mercury_fsdp/fsdp.py
```

`NRANKS`, `LINK_BW`, `GPU_MEM` are read by `fsdp.py`; `FSDP_PREFETCH`, `LAYERS`,
`DTYPE` by the skeleton.

## What it shows

Step time (ms), N=8, 8 layers, vs the plain-DP baseline at equal config:

| `LINK_BW` | DP | FSDP (prefetch) | FSDP (no prefetch) | FSDP / DP |
|-----------|-----|-----------------|--------------------|-----------|
| 12 GB/s | 940 | 2881 | 2833 | 3.1× |
| 50 GB/s | 226 | 741 | 759 | 3.3× |
| 100 GB/s | 113 | 420 | 435 | 3.7× |
| 300 GB/s | 74.6 | 219 | 221 | 2.9× |

Three findings:

1. **FSDP's communication does not hide the way DP's does.** DP's single gradient
   all-reduce overlaps the backward pass, so at 300 GB/s DP reaches the **compute
   floor** (74.6 ms — comms fully hidden). FSDP cannot: its all-gathers sit on the
   forward *and* backward critical path, per layer, so even at NVLink bandwidth FSDP
   is **2.9× the floor**. This is the persistent time price of ZeRO-3's memory
   savings — feasibility (G) bought with bandwidth.

2. **Prefetch only partially hides the all-gather.** It overlaps the next layer's
   gather with the current layer's compute, but it can hide at most the per-layer
   *compute* window (~3 ms here), and the per-layer all-gather is larger (~7 ms at
   100 GB/s) — so prefetch saves only 1–3% (and is neutral when fully comms-bound at
   12 GB/s, where there is nothing to hide behind). Prefetch hides the gather only
   once per-layer compute exceeds the gather time — i.e. at larger batch/model or
   faster fabric.

3. **FSDP issues 3 collectives per layer** (forward gather, backward gather,
   backward reduce-scatter) versus DP's 1 all-reduce — 1.5× the *volume*, but a
   larger ~3× *time* gap because DP overlaps and FSDP does not.

**Paired with (G):** the per-rank footprint here is **4.33 GB** (ZeRO-3 shards),
matching `capacity.py --zero 3` exactly (+ the 32 MB scratch). That is the same
mechanism that fits 70B on 80 GB GPUs in (G) — and this demo is its step-time bill.

## Accuracy note

Roofline compute, exact message volumes — as in the other demos. The ring
reduce-scatter and ring all-gather are time duals (same bytes/steps); the
reduce-scatter is now a real sumi collective, not modeled by its dual. The prefetch
result is scale-dependent: it tracks the ratio of per-layer compute to per-layer
gather, so larger per-layer compute (bigger batch/model) shifts it toward full
hiding.
