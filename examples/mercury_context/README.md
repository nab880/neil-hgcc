# mercury_context — sequence / context parallelism (ring attention)

The other parallel axes shard the model ([TP/PP](../mercury_3d)) or the batch
([DP](../mercury_llm_train), [FSDP](../mercury_fsdp)). **Context parallelism** shards
the **sequence**: each of `N` ranks holds `S/N` query tokens and its own `S/N`-token
K/V block. Attention needs every query to see every key, so the K/V blocks are passed
around a **ring** — `N` steps, one neighbour hop each — until every query has attended
to all `S` keys. This is **Ring Attention**, the technique behind million-token context.

The question this demo answers: **does the K/V ring stay hidden behind attention
compute as context grows?** Per ring step the K/V transfer is **linear** in the block
size (`2·(S/N)·H`) while the attention compute is **quadratic** (`(S/N)²·H`). So there
is a context length below which the ring is *exposed* (bandwidth-bound, overlap helps)
and above which it is *hidden* (compute-bound, overlap stops mattering). The simulator
resolves that crossover directly. `CP_OVERLAP` is the knob: prefetch the next block's
transfer (`MPI_Isend`/`Irecv`) under the current step's compute — the [FSDP](../mercury_fsdp)
prefetch mechanic, but point-to-point round a ring rather than a reduction.

## Modeled, never executed

Same transformer kernels and dims as the DP demo (`H=4096`, `L=8`). The sequence `S`
(`SEQ`) is a runtime knob: the per-step attention launches `(S/N)²` threads and the
projections/MLP `(S/N)·H`, so the quadratic-vs-linear scaling is carried by the launch
thread counts — no recompile to sweep context. The K/V exchange is a NULL-buffer
point-to-point `MPI_Send`/`Recv` (`MPI_Sendrecv` exposed, `Isend`/`Irecv` overlapped)
timed from its element count. Weights, K/V block, and activations are device cookies
(no real memory); CP shards the sequence, not the model, so weights stay replicated and
only the activations/K/V shrink to this rank's `S/N` tokens. No simulator changes — the
ring is point-to-point, not a sumi collective.

## Files

| File | What it is |
|------|------------|
| [`ring_attn.cu`](ring_attn.cu) | per-layer ring attention (`N` steps), `CP_OVERLAP` prefetch pipeline, local MLP/projections |
| [`ring_attn.py`](ring_attn.py) | SST config (A100 dual-roofline + `LINK_BW`), mirrors `train.py` |
| [`run.sh`](run.sh) | build + the crossover / fabric-sensitivity / ring-scale sweeps |

## Run it

```sh
./run.sh                                   # hg++ and sst on PATH
# one point (from examples/, where the shared modules live):
cd .. && NRANKS=8 SEQ=131072 LAYERS=2 CP_OVERLAP=1 LINK_BW=100GB/s sst mercury_context/ring_attn.py
```

`NRANKS`, `LINK_BW` are read by `ring_attn.py`; `SEQ`, `CP_OVERLAP`, `LAYERS`, `DTYPE`
by the skeleton.

## What it shows

**1) The hiding crossover** (N=8 ring, 100 GB/s, 2 layers) — step time (ms) vs context:

| `SEQ` | overlap | no-overlap | overlap gain |
|-------|---------|-----------|--------------|
| 8 192 | 6.3 | 8.0 | 1.27× |
| 16 384 | 11.8 | 15.6 | 1.32× |
| 32 768 | 24.3 | 33.8 | **1.39×** |
| 65 536 | 63.6 | 82.6 | 1.30× |
| 131 072 | 190.9 | 228.6 | 1.20× |

The overlap gain is `(compute+transfer)/max(compute,transfer)` — it peaks (→2) when
compute ≈ transfer and falls to 1 when either dominates. Here it **peaks at ~32k
tokens** (1.39×), the crossover: below it the K/V ring is exposed and prefetch matters
most; above it attention compute (quadratic) overtakes the transfer (linear) and the
gain decays toward 1 — the ring is increasingly **hidden**. At 128k, overlap already
saves only 20% and is still shrinking.

**2) Fabric sensitivity** (N=8, 2 layers, overlap on) — exposed vs hidden context:

| `LINK_BW` | SEQ=16k (exposed) | SEQ=128k (hidden) |
|-----------|-------------------|-------------------|
| 25 GB/s | 24.4 | 211.5 |
| 100 GB/s | 11.8 | 190.9 |
| 400 GB/s | 10.7 | 190.9 |

Short context is **bandwidth-bound**: 16× the fabric (25→400 GB/s) cuts step time 2.3×
until it hits the compute floor (~10.7). Long context is **flat** — 190.9 ms at both
100 and 400 GB/s — the attention compute fully hides the K/V ring, so buying fabric
bandwidth does nothing. This is the headline: whether long-context training is fabric-
or compute-bound is a property of the context length, and the crossover is sharp.

**3) Ring scale** (SEQ=65 536, 100 GB/s, 2 layers, overlap on):

| `NRANKS` | step (ms) |
|----------|-----------|
| 4 | 118.0 |
| 8 | 63.6 |
| 16 | 37.4 |

More ranks = smaller blocks = more ring steps but quadratically less compute per step;
per-rank attention work is `S²/N`, so doubling the ring nearly halves step time (1.85×,
1.70× — sub-2× because the step count and per-step transfer grow with `N`).

## Accuracy note

Roofline compute, exact message volumes — as in the other demos. The crossover location
moves the expected way with every lever: faster fabric or fewer layers shifts it to
shorter context; more ranks (smaller blocks) shifts it to longer context. The model is
**non-causal** (every query attends every key); causal masking skips the "future" half
of the ring and ~halves both the compute and the steps, shifting the crossover but not
its existence. Determinism holds (a given config reproduces exactly).
