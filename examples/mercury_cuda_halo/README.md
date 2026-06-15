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

**Roofline accuracy.** With `gpu_mem_bandwidth` set to the rate the calibration
achieved (~900 GB/s here), the uncalibrated and calibrated runs agree to within
the per-launch kernel-launch overhead — about **3.5%** over this 20-step halo. At
the shipped 1500 GB/s *peak* the roofline runs **~36% faster**; that gap is the
achieved-vs-peak memory efficiency a real kernel never reaches, which is exactly
what the table supplies. So the roofline is a sound lower bound that tracks
calibration once you feed it the achieved bandwidth — calibrate when you also need
the efficiency factor. The P4 auto-derived costs (used when no `#pragma sst
gpu_compute` is present) feed the same roofline; on a pure-streaming kernel they
recover the exact DRAM traffic, matching a same-bandwidth calibration to within
that launch overhead.

To calibrate your own kernel, find its mangled name in the rewritten
`sst.pp.halo.cu` (the `sst_hg_cuda_launch("<name>", ...)` call), then measure it
on real hardware: compile the unmodified `.cu` with your own `nvcc`, `-include`
[`sst_hg_cuda_calibrate.h`](../../hgcc_include/libraries/sst_hg_cuda_calibrate.h),
and wrap each launch in a scope —

```cuda
{ SST_HG_CALIBRATE("_Z7stencilPfi", (uint64_t)b * t);
  stencil<<<b, t, 0, compute>>>(u, n); }
```

Running that build emits a `gpu_kernel_times.json` in exactly this format
(averaging repeated launches per thread count); drop it in as
`gpu_calibration.json`. The table here is an illustrative sample, not a
measurement.

## The questions this answers

- Does compute/comms overlap survive at thousands of ranks, or does halo
  staging serialize against the interior kernel?
- Is GPUDirect worth it for this message size and network? (the `gpu_direct`
  sweep)
- Which buys more time-to-solution: more HBM bandwidth, more NIC bandwidth, or
  GPUDirect? (sweep `gpu_mem_bandwidth` / the Merlin link bandwidth / `gpu_direct`)
