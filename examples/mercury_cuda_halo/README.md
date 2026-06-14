# mercury_cuda_halo — a GPU halo-exchange under Mercury

The CUDA_PLAN §1 use case in one runnable example: a 1-D stencil where each rank
computes its interior on the GPU and exchanges a halo slab with its ring
neighbors. Compiled with `hg++` and run under SST/Mercury, it lets you study
GPU-dense scale-out — compute/comms overlap, GPUDirect, bandwidth trade-offs —
**without a real GPU**. The kernel never executes; its time is modeled.

## Files

| File | What it is |
|------|------------|
| [`halo.cu`](halo.cu) | the app: `stencil<<<>>>` on a compute stream, `MPI_Sendrecv` of device buffers, `cudaMemcpyAsync` |
| [`halo.py`](halo.py) | the SST config: GPU model params + `gpu_direct`, reads `NRANKS`/`GPUDIRECT` from the environment |
| [`gpu_calibration.json`](gpu_calibration.json) | a sample calibration table (see below) |
| [`run.sh`](run.sh) | build with `hg++`, install the `.so`, run the `gpu_direct` sweep |

## Run it

```sh
# hg++ and sst on PATH (e.g. source sst-hgcc/mymodules.sh)
NRANKS=4 ./run.sh
```

Expected: both runs finish (`mercury_cuda_halo: rank N done`) and the reported
`simulated time` is **lower with `gpu_direct=true`** — the difference is the PCIe
staging of the device send/recv buffers that GPUDirect removes. Sweep `NRANKS`
(2–8) and `pcie_bandwidth` (in `halo.py`) to map the trade-off surface.

## How the kernel time is chosen

By default the kernel time is a **roofline** estimate from the platform params
(`gpu_peak_flops`, `gpu_mem_bandwidth`, `gpu_kernel_launch_overhead`) and the
`#pragma sst gpu_compute` per-thread costs on the kernel.

For single-node accuracy at scale, **calibrate once** on one real GPU node and
reuse it everywhere. `halo.py` points `gpu_kernel_times` at
[`gpu_calibration.json`](gpu_calibration.json), which maps the kernel's mangled
name (`_Z7stencilPfi`) to measured `(threads, seconds)` samples; the model
exact-matches the launch's total thread count or log-log interpolates between
them, overriding the roofline. Delete (or rename) the JSON to fall back to the
roofline. The format is specified by
[`../../docs/gpu_calibration.schema.json`](../../docs/gpu_calibration.schema.json).

To calibrate your own kernel, find its mangled name in the rewritten
`sst.pp.halo.cu` (the `sst_hg_cuda_launch("<name>", ...)` call), measure it at a
few problem sizes on real hardware, and list the samples here.

## The questions this answers

- Does compute/comms overlap survive at thousands of ranks, or does halo
  staging serialize against the interior kernel?
- Is GPUDirect worth it for this message size and network? (the `gpu_direct`
  sweep)
- Which buys more time-to-solution: more HBM bandwidth, more NIC bandwidth, or
  GPUDirect? (sweep `gpu_mem_bandwidth` / the Merlin link bandwidth / `gpu_direct`)
