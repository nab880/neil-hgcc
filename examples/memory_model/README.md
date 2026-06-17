# memory_model — per-rank GPU memory & feasibility ("does it fit?")

The simulator models training/inference **time**; this models **feasibility** — the
per-rank memory footprint of a parallelization, and whether it fits in GPU memory.
It is the closed-form companion to [`../validation/analytic_baseline.py`](../validation/analytic_baseline.py)
(which is the closed-form companion for *time*), and it is **wired into the
simulator**: a run reports its per-rank high-water and flags OOM against a capacity.

Why closed-form rather than measured: the demos `cudaMalloc` only a ~32 MB scratch
buffer and reuse it (model-don't-execute), so real allocations do not reflect the
true weight/optimizer/activation footprint. The footprint is a function of model
dims and parallelism, computed the same way the time model computes step time.

## The model (`capacity.py`)

Bytes are decimal (1 GB = 1e9) to match SST `UnitAlgebra` (`"80GB"` → 80e9), so the
tool and the simulator's `gpu_mem_capacity` use identical units.

- **Params.** `12·H²` per transformer layer (matches `GRAD_ELEMS` in the demos) +
  `2·V·H` embeddings. Per rank `= P/(TP·PP)` — PP shards layers, TP shards within.
- **State bytes/param** (mixed-precision Adam): weight 2 + grad 2 + fp32 master 4 +
  momentum 4 + variance 4 = **16 B/param**, split into components so ZeRO can
  partition them.
- **ZeRO/FSDP over DP**: stage 1 shards the optimizer (12 B), stage 2 also the
  gradient, stage 3 also the weight (+ a transient all-gather buffer). Each sharded
  component is divided by the DP degree.
- **Activations** (1F1B): `act_factor·s·mb·H·dtype/TP` per layer × `L` layers'
  worth in flight; `act_factor` from `--checkpoint {none:17, selective:5, full:2}`.
- **KV cache** (`--infer`): `2·(L/PP)·s·(H/gqa)·dtype·batch/TP` — weights + KV only.

```sh
python3 capacity.py --model 70b --dp 16 --tp 1 --pp 1 --zero 3 --gpu-mem 80
python3 capacity.py --model 175b --gpus 64 --zero 1            # frontier sweep
python3 capacity.py --model 70b --infer --tp 8 --batch 32 --seq 4096 --gqa 8
sh frontier_sweep.sh                                            # MODEL/N/DEV knobs
```

Presets: `demo`(~2B), `7b`, `70b`, `175b` (sizes via the `12·H²` approximation;
SwiGLU/GQA models differ slightly). Devices: `a100-40/80`, `h100-80`, `h200-141`,
or `--gpu-mem N`.

## What it shows

**1) The memory wall opposes the communication wall.** The *memory*-optimal layout
is always pure tensor parallelism — TP shards params, grads, optimizer **and**
activations — which is exactly study B's *communication*-worst corner (TP's
all-reduce is on the critical path). The fast region and the feasible region pull
apart. 70B on 16×A100-80GB (ZeRO-1): only the model-sharded layouts fit; every
layout with DP≥2 is OOM.

| layout (DP×PP×TP) | per-rank | verdict |
|-------------------|----------|---------|
| DP16×PP1×TP1 (pure data) | 322 GB | OOM |
| DP4×PP4×TP1 | 127 GB | OOM |
| DP1×PP4×TP4 (balanced) | 68 GB | **FITS** |
| DP1×PP1×TP16 (pure tensor) | 66 GB | **FITS** (memory-optimal, but time-worst) |

**2) ZeRO buys back data parallelism.** To run data parallelism (for throughput) on
a 70B model you must shard the optimizer state. Pure DP=16, by ZeRO stage:

| stage | per-rank | verdict |
|-------|----------|---------|
| ZeRO-0 | 1053 GB | OOM |
| ZeRO-1 (optimizer) | 322 GB | OOM |
| ZeRO-2 (+ grad) | 200 GB | OOM |
| ZeRO-3 / FSDP (+ weight) | 80 GB | **FITS** |

ZeRO trades memory for communication — the all-gather/reduce-scatter of study (E) —
which is exactly the cross-study link.

**3) Inference is a different regime.** With `--infer` there is no optimizer or
gradient; the footprint is sharded weights + KV cache, and the KV cache grows with
batch × context. 70B, TP8, batch 32, seq 4096, GQA-8: weights 16.2 GB + KV 5.4 GB.

## Simulator integration

`gpu_library` reports the per-rank high-water (`cookie_end_ - kCookieBase`) in its
rank summary and, if `gpu_mem_capacity > 0`, prints `[gpu] OOM` when exceeded
(`gpu_mem_fatal=true` aborts). `mercury_3d` allocates the realistic per-rank
footprint as cookies (no real memory), driven by `ZERO`/`DTYPE` env vars matching
`capacity.py`. The two agree exactly up to the demo's 32 MB scratch buffer:

```
NRANKS=1 LAYERS=8 GPU_MEM=80GB sst ../mercury_3d/parallel3d.py
  -> mem_footprint=26474446848  (26.474 GB)
python3 capacity.py --model demo --L 8 --no-embed --checkpoint selective
  -> TOTAL 26.44 GB   (+ 32 MB scratch = 26.474 GB, exact)
```

## Accuracy note

Param counts use the `12·H²`/layer approximation (exact for the demos; ±10% for
SwiGLU/GQA production models). Activation memory follows a simplified 1F1B
Korthikanti form with a single checkpointing factor — order-of-magnitude for the
feasibility verdict, not a bytes-exact profiler. Weight/optimizer/KV accounting is
exact given the param count and dtype.
