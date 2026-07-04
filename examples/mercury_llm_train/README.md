# mercury_llm_train — data-parallel LLM training at scale, under Mercury

The canonical "AI at scale" GPU workload, made runnable **without a single real
GPU**: a data-parallel transformer training step. Each rank holds a replica of an
8-layer transformer block, runs forward + backward over a token batch on the GPU,
and all-reduces the per-layer weight gradients across ranks — with that all-reduce
**overlapped** against the backward compute of the remaining layers. Compiled with
`hg++` and run under SST/Mercury, it answers the question that decides whether
distributed training scales: *does the gradient all-reduce stay hidden under
backward compute, and how do the fabric, GPUDirect, and overlap move that line?*

The kernels never execute — their **time** is modeled (roofline from the
simulated GPU, or a calibration table), the gradient all-reduce rides the
simulated Merlin network as a real MPI collective, and GPUDirect toggles PCIe
staging of the gradients. So this demonstrates the **performance/scaling**
pipeline (kernel times, compute/comms overlap, scaling efficiency), which is
exactly what an at-scale study cares about — not numerical training output.

## What it exercises (the whole pipeline, at once)

| Pipeline piece | In this workload |
|----------------|------------------|
| Compute-bound kernels | the GEMMs (`qkv_proj`, `attn_*`, `mlp_*`) — `#pragma sst gpu_compute` pins the flop count; the roofline charges `flops/peak_flops` |
| Memory-bound kernels | `layernorm`, `softmax_rows`, `gelu` — **no pragma**, so P4 auto-derives per-thread bytes; the roofline charges `bytes/mem_bandwidth` |
| Transfers / PCIe | gradient staging when GPUDirect is off — a time-only `cudaMemcpyAsync` over `pcie_bandwidth` |
| Streams & overlap | backward compute on one stream; the gradient all-reduce issued non-blocking (`MPI_Iallreduce`) so it overlaps |
| Collectives over the network | the gradient all-reduce — a NULL-buffer MPI collective timed from the element count, riding the simulated Merlin fabric |
| Multi-rank scale | data-parallel replicas, one GPU per rank |

## Files

| File | What it is |
|------|------------|
| [`train.cu`](train.cu) | the app: forward + backward transformer kernels, per-layer gradient all-reduce overlapped with backward compute |
| [`train.py`](train.py) | the SST config: A100-class GPU params + a `LINK_BW` override so the fabric can be swept |
| [`run.sh`](run.sh) | build with `hg++`, install the `.so`, run the three sweeps below |

## Run it

```sh
# hg++ and sst on PATH (e.g. source sst-hgcc's module env)
./run.sh
```

Knobs are environment variables (no recompile): `NRANKS`, `LINK_BW` (read by
`train.py`), and `GPUDIRECT`, `LLM_OVERLAP` (read by the skeleton). A single run:

```sh
NRANKS=8 LINK_BW=300GB/s GPUDIRECT=true LLM_OVERLAP=1 sst train.py
```

The GPU compute floor is reported on the `[gpu] rank summary:` line
(`total_gpu_time` ≈ **77 ms** for the 240 kernels of one step); the end-to-end
step time is SST's `simulated time`.

## What it shows

Dimensions: a GPT-7B-class layer (hidden 4096, 2048 tokens, 8 layers), A100-class
GPU (≈312 TFLOP/s fp16, 2 TB/s HBM, 32 GB/s PCIe). Step compute ≈ 77 ms.

**1) Fabric bandwidth — the comms→compute crossover** (4 ranks, GPUDirect on,
overlap on). The gradient all-reduce is ~805 MB/layer; a slow fabric makes the
step network-bound, a fast one lets overlap hide the all-reduce entirely:

| `LINK_BW` | step time | regime |
|-----------|-----------|--------|
| 12 GB/s (PCIe/Eth) | 806 ms | comms-bound — the fabric is the wall |
| 50 GB/s (IB) | 194 ms | comms still dominates |
| 150 GB/s (NVLink) | 77 ms | **compute-bound** — all-reduce fully hidden |
| 600 GB/s | 77 ms | compute-bound |

**2) The levers at an NVLink-class fabric** (4 ranks, 150 GB/s). With a fast
fabric the bottleneck moves to PCIe gradient staging, so **GPUDirect** becomes the
dominant lever, and **overlap** hides the rest:

| GPUDirect | overlap | step time | |
|-----------|---------|-----------|--|
| on | on | 77 ms | compute-bound (best) |
| on | off | 144 ms | overlap off → compute + comms serialize (1.9×) |
| off | on | 201 ms | PCIe gradient staging exposed (2.6×) |
| off | off | 233 ms | both costs exposed |

**3) Data-parallel scaling** (GPUDirect on, overlap on, 150 GB/s). Ring
all-reduce keeps per-rank volume ~constant, and overlap hides it under compute, so
the step time is flat — **near-perfect weak scaling** — until the all-reduce
finally exceeds the compute it can hide behind:

| ranks | 1 | 2 | 4 | 8 | 16 |
|-------|----|----|----|----|----|
| step time | 77 ms | 77 ms | 77 ms | 77 ms | 80 ms |

## The questions this answers

- **At what fabric bandwidth does DDP stop being network-bound?** Here the
  crossover is ~150 GB/s (NVLink-class); below it, buy bandwidth before GPUs.
- **Is GPUDirect worth it?** At a fast fabric it's 2.6× — PCIe gradient staging is
  the bottleneck once the network isn't.
- **Does compute/comms overlap matter, and does DP scale?** Overlap is ~1.9×, and
  with it the step time is flat from 1→8 GPUs (the all-reduce hides under
  compute), starting to expose at 16.

Sweep `gpu_peak_flops` / `gpu_mem_bandwidth` (the GPU), `pcie_bandwidth` (staging),
or `LINK_BW` (the fabric) to move the crossover and re-answer for your target node.

## Accuracy note

GEMM times come from the roofline with a pinned flop count (a tiled,
compute-bound GEMM); for cuBLAS-exact times, calibrate the GEMM kernels on real
hardware and supply `gpu_kernel_times` (see
[`../mercury_cuda_halo`](../mercury_cuda_halo) and
[`../../docs/gpu_calibration.schema.json`](../../docs/gpu_calibration.schema.json)).
The flop-per-thread literals in the kernel pragmas track `H` and `T` — change them
together.
