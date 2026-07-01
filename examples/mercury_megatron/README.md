# mercury_megatron — tensor + pipeline parallel LLM training, under Mercury

The second AI-at-scale workload, and the counterpart to the data-parallel demo
([`../mercury_llm_train`](../mercury_llm_train)): a **Megatron-style** transformer
training step that splits a single model across GPUs two ways at once, exercising
the two parallelism strategies whose communication patterns are *nothing like*
data parallelism's overlappable gradient all-reduce:

- **Tensor parallelism (TP)** — each layer's weight matrices are sharded across a
  TP group, so every layer does an **all-reduce of the activations over a
  sub-communicator**, *on the forward/backward critical path*. The next op needs
  the combined result, so unlike DP this all-reduce **cannot be hidden** — which
  is exactly why TP stays inside a node on NVLink.
- **Pipeline parallelism (PP)** — the layers are split into stages, and
  activations move stage-to-stage by **point-to-point send/recv**. The cost is the
  pipeline **bubble** (idle fill/drain), which microbatching amortizes:
  bubble fraction = `(P-1)/(M+P-1)`.

Ranks form a `TP_SIZE × PP_SIZE` grid (`TP_SIZE*PP_SIZE = nranks`). The question
this answers — without a real GPU — is the one Megatron users actually face: *for
a fixed GPU budget, how do you split it between tensor and pipeline parallelism,
and how do the critical-path TP all-reduce and the PP bubble trade off?*

As in the DP demo, kernels are **modeled, not executed**: GEMMs carry a pinned
flop count (sharded by `TP_SIZE`), elementwise kernels are P4 auto-derived, and
every MPI message — the TP all-reduce on a sub-comm, the PP send/recv — is a
NULL-buffer call timed from the element count, no real payload.

## What it adds over the DP demo (different comms patterns)

| Pattern | DP (`mercury_llm_train`) | This workload |
|---------|--------------------------|---------------|
| Collective | gradient all-reduce (param-size), **overlapped** with backward | **TP**: activation all-reduce over a `MPI_Comm_split` sub-comm, **critical path** |
| Point-to-point | — | **PP**: activation `MPI_Send`/`MPI_Recv` between stages |
| At-scale concern | does the all-reduce hide under compute? | TP comms you can't hide + the PP bubble |

## Files

| File | What it is |
|------|------------|
| [`megatron.cu`](megatron.cu) | the app: TP sub-comm all-reduce per layer (critical path) + a GPipe PP schedule (point-to-point, microbatched) |
| [`megatron.py`](megatron.py) | the SST config: A100-class GPU params + a `LINK_BW` override |
| [`run.sh`](run.sh) | build with `hg++`, install the `.so`, run the three sweeps below |

## Run it

```sh
./run.sh                  # hg++ and sst on PATH
# or a single point:
NRANKS=8 TP_SIZE=2 MICROBATCH=8 LINK_BW=300GB/s sst megatron.py
```

`NRANKS` and `LINK_BW` are read by `megatron.py`; `TP_SIZE` (PP_SIZE = NRANKS/TP_SIZE)
and `MICROBATCH` by the skeleton. Step time is SST's `simulated time`.

## What it shows

**1) The pipeline bubble** (pure PP: `TP=1`, 4 stages, 150 GB/s). With one
microbatch the pipeline is mostly idle fill; microbatching amortizes it, exactly
as `(P-1)/(M+P-1)` predicts:

| microbatches M | step time | bubble fraction |
|----------------|-----------|-----------------|
| 1  | 53 ms  | 75% (=3/4) |
| 4  | 93 ms  | 43% (=3/7) |
| 16 | 253 ms | 16% (=3/19) |

(Steady-state ≈ 13 ms/microbatch; the overhead above `M × 13 ms` is the constant
`(P-1)`-stage bubble, so its *fraction* shrinks as M grows.)

**2) Tensor parallelism's all-reduce is on the critical path** (pure `TP=4`,
`PP=1`, M=4). It responds strongly to the fabric — but, unlike DP, it **never
collapses to the compute floor**, because the next layer can't start until the
all-reduce completes:

| `LINK_BW` | 12 GB/s | 50 GB/s | 150 GB/s | 600 GB/s |
|-----------|---------|---------|----------|----------|
| step time | 637 ms | 229 ms | 143 ms | 111 ms |

Compare the DP demo at 600 GB/s: the gradient all-reduce was *fully hidden* (step
= the 77 ms compute floor). Here, even at 600 GB/s, TP comms is still exposed.
**This is why you keep TP inside one NVLink node.**

**3) The TP/PP split has a real optimum, and it moves** (fixed 8-GPU budget,
150 GB/s). With few microbatches the PP bubble is large, so favour tensor
parallelism; with many, the bubble is amortized, so favour pipeline parallelism
and avoid TP's critical-path comms:

| layout | TP1×PP8 | TP2×PP4 | TP4×PP2 | TP8×PP1 |
|--------|---------|---------|---------|---------|
| **M=1** | 55 ms | 41 ms | **36 ms** | 37 ms |
| **M=8** | **104 ms** | 113 ms | 166 ms | 296 ms |

At M=1 the sweet spot is TP-heavy (4×2); at M=8 it flips to pure pipeline (1×8).

## The questions this answers

- **How many microbatches do I need to hide the pipeline bubble?** The bubble is
  `(P-1)/(M+P-1)` — for 4 stages, M=16 gets it under ~16%.
- **Why does tensor parallelism want NVLink?** Its all-reduce is on the critical
  path and never hides — 12→600 GB/s still moves the step time 6×, and it never
  reaches the compute floor.
- **How should I split a fixed GPU budget between TP and PP?** It depends on the
  microbatch count (and fabric): few microbatches favour TP, many favour PP. The
  sweep finds the knee.

## Accuracy note

Same as the DP demo: GEMM times are roofline estimates from a pinned flop count;
for cuBLAS-exact times, calibrate the GEMM kernels on real hardware and supply
`gpu_kernel_times` (see [`../mercury_cuda_halo`](../mercury_cuda_halo) and
[`../../docs/gpu_calibration.schema.json`](../../docs/gpu_calibration.schema.json)).
The flop-per-thread literals track `H` and `T` — change them together.
