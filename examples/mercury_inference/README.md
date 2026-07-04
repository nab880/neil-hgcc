# mercury_inference — disaggregated LLM inference (the decode regime)

Every other demo here is **training**. Inference is a different fabric workload, and
disaggregated serving (separate prefill and decode GPU pools, as in
vLLM/Splitwise/DistServe) splits it into two phases that share one fabric:

- **Prefill** processes the whole prompt at once — compute-heavy GEMMs, quadratic in
  the prompt length — and produces the KV cache.
- **KV-cache transfer** ships that cache from a prefill worker to a decode worker — a
  *bulk, bandwidth-bound* point-to-point message.
- **Decode** generates tokens autoregressively: one token at a time, so GEMMs are tiny
  and each step is dominated by a small, **latency-critical** tensor-parallel
  all-reduce — the *opposite* regime to training's bandwidth-bound gradient all-reduce.

The memory half is study (G): `capacity.py --infer` gives the KV-cache footprint. This
demo is the **timing** half.

## Modeled, never executed

`MPI_COMM_WORLD` splits into a prefill pool and a decode pool (`MPI_Comm_split`); the
decode pool is one tensor-parallel group. Prefill runs a full forward over `PROMPT_LEN`
tokens (the `train.cu` kernels); the KV cache is a NULL-buffer `MPI_Send`/`Irecv`
(bulk); each decode step runs tiny GEMMs plus a NULL-buffer TP `MPI_Allreduce` of the
`BATCH x H` activation. Decode weights + KV cache are device cookies — the per-rank
`mem_footprint` cross-checks `capacity.py --infer` (532 MB sim = 0.50 GB closed form +
the 32 MB scratch, at TP=7).

## Files

| File | What it is |
|------|------------|
| [`infer.cu`](infer.cu) | prefill/decode pools, KV transfer, latency-critical decode TP all-reduce, KV footprint cookies |
| [`infer.py`](infer.py) | SST config (A100 dual-roofline + `LINK_BW` + `gpu_mem_capacity`) |
| [`run.sh`](run.sh) | build + the two sweeps below |

## Run it

```sh
./run.sh
# one point (from examples/):
cd .. && NRANKS=8 PREFILL_RANKS=1 PROMPT_LEN=2048 DECODE_STEPS=16 LINK_BW=100GB/s sst mercury_inference/infer.py
```

Knobs: `NRANKS`, `LINK_BW`, `GPU_MEM`, `TOPO` (`single`|`fattree`) read by `infer.py`;
`PREFILL_RANKS` (0 = co-located control), `PROMPT_LEN`, `DECODE_STEPS`, `BATCH`,
`REQUESTS`, `SUMI_ALLREDUCE_ALG` by the skeleton.

## What it shows

**1) Decode is latency-bound — the regime inversion.** Pure decode (TP=8, 16 steps),
sweeping fabric bandwidth and the all-reduce algorithm:

| `LINK_BW` | recursive-halving | ring |
|-----------|-------------------|------|
| 12 GB/s | 76.0 ms | 166.6 ms |
| 100 GB/s | 75.9 ms | 166.6 ms |
| 300 GB/s | 75.9 ms | 166.6 ms |

Decode time is **flat in bandwidth** (the small per-step all-reduce is latency-bound,
not bandwidth-bound) and **recursive-halving beats ring 2.2×** — the exact *inverse* of
training study (E), where the bandwidth-optimal ring wins at scale and bandwidth is the
lever. A serving stack should pick the latency-optimal collective for decode; a training
stack picks the bandwidth-optimal one. Same fabric, opposite choice.

**2) The prefill→decode handoff hides under a long enough decode.** The handoff
(prefill compute + bulk KV transfer) is a *fixed* cost; whether disaggregation is free
depends on amortizing it over the generation (fat-tree, 12 GB/s, prompt 4096, TP=4):

| `DECODE_STEPS` | decode-only | disaggregated | ratio |
|----------------|-------------|---------------|-------|
| 2 | 13.4 ms | 125.8 ms | 9.4× |
| 8 | 53.8 ms | 125.8 ms | 2.3× |
| 32 | 215.1 ms | 216.5 ms | **1.01×** |

The ~126 ms handoff dominates short generations (9.4× at 2 steps) but is **fully hidden
once decode runs long enough** (32 steps: disaggregated = decode-only). This is the
disaggregation viability condition — and the reason the bandwidth-bound transfer and the
latency-bound decode coexist on one fabric: the transfer overlaps decode rather than
competing with its latency-bound all-reduce. (On a crossbar there is no link sharing at
all; even on the fat-tree at this scale the transfer does not measurably delay the tiny
all-reduce — the handoff cost is exposure, not contention, until decode is too short to
hide it.)

## Accuracy note

Roofline compute, exact message volumes, as elsewhere. The KV cache is modeled in
`MPI_FLOAT` units sized to the `DTYPE` byte volume; decode attention reads a
representative `PROMPT_LEN + step` keys. A continuous-batching / many-request-in-flight
refinement (true steady-state serving) is a later extension; this models one
prefill→decode handoff per request.
