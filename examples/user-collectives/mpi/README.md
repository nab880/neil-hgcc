# Tier 1 — write your collective in MPI point-to-point

Prototype a collective algorithm (a custom all-reduce, gather, tree/ring/…)
purely in `MPI_Send`/`MPI_Recv`, and measure its packet-level cost on a topology
of your choosing. **No simulator edits, no `libsumi` rebuild** — your algorithm
rides the real Mercury/Iris point-to-point path, so the reported time is a full
network-model measurement.

This is the honest, zero-C++ path and is what most algorithm prototyping wants.
It measures *an algorithm expressed in MPI messages*. It does **not** replace the
MPI library's own collective implementation — for that (a native DAG algorithm
selectable by name, with full protocol fidelity), see Tier 2 in
`../../../../user-collectives.md` and `../sumi/AUTHORING.md`.

## Workflow

```
cp my_allreduce.c my_scheme.c        # start from the template
$EDITOR my_scheme.c                  # edit the >>> YOUR ALGORITHM HERE <<< region
SRC=my_scheme.c ./build.sh           # -> libmy_scheme.so
APP=my_scheme NRANKS=8 sst run.py    # run; read the reported simulated time
```

The template contributes `rank+1` from every rank and asserts the known sum
`n(n+1)/2`, so a wrong algorithm prints an error and aborts instead of reporting
a meaningless time. Keep that self-check when you swap in your algorithm.

## Knobs (all environment variables)

| Var | Meaning | Default |
|-----|---------|---------|
| `SRC` | source file for `build.sh` | `my_allreduce.c` |
| `APP` | app / library name (`lib<APP>.so`) | source basename |
| `NRANKS` | number of MPI ranks | `8` |
| `TOPO` | `single` (one crossbar) or `fattree` | `single` |
| `LINK_BW` | per-link bandwidth, e.g. `12 GB/s` | platform default |
| `LINK_LAT` | per-hop latency, e.g. `20ns` | `20ns` (single) / `100ns` (fattree) |

Sweep by changing `NRANKS` / `TOPO` / `LINK_*` between runs. `fattree` is sized
automatically to `NRANKS` by `../../scale_topo.py`, and — unlike a single
crossbar — partitions across `sst -n` threads, so large runs go faster.

## Files

- `my_allreduce.c` — documented skeleton with a worked naive-ring all-reduce and
  the `rank+1` → `n(n+1)/2` self-check. Edit the marked region.
- `build.sh` — source-agnostic build (`SRC=`/`APP=`); reuses the mv2 harness's
  MPI-wrapper shim and PMI-stripping, keeping the macOS `-bundle` / Linux
  `-shared` split.
- `run.py` — source-agnostic SST driver reading the knobs above.

## Prerequisites

A built install tree (`../../../../install` by default, or set `SST_HG_PREFIX`)
containing `hgcc`, `mvapich2/lib/libmpi.a`, and the Iris `libpmi`/`libfabric`
under `lib/sst-elements-library/ext`. Build it with the repo's `build_all.sh`.

## No MVAPICH2? Use the mask-mpi path instead

This harness targets the hgcc + MVAPICH2 stack (real MVAPICH2 collectives). If
your install has no `mvapich2/libmpi.a`, use the **mask-mpi** twin of this
template, which runs with just `sst` — no hgcc, no MVAPICH2 — because mask-mpi
is an in-tree MPI over the same sumi engine:

```
sst-elements/src/sst/elements/mask-mpi/tests/my_collective.cc      # same worked ring, edit here
sst-elements/src/sst/elements/mask-mpi/tests/test_my_collective.py # NRANKS=<n> sst test_my_collective.py
```

`my_collective.cc` carries the identical point-to-point algorithm as
`my_allreduce.c` here (same `/* YOUR ALGORITHM HERE */` region and self-check);
it is built as an SST element library by the mask-mpi `Makefile.am`. That path
is verified end-to-end (PASS at 2/4/7/8/16 ranks). For selecting a *native*
engine algorithm by name (Tier 2), see `../sumi/AUTHORING.md`.
