# Tier 2 / Tier 3 — native sumi collective algorithms

At this level you implement a collective as a DAG of send/recv actions inside the
Iris/sumi engine. It is what `MPI_Allreduce` actually runs, so it is measured
with full protocol fidelity.

- **How to write one:** read [`AUTHORING.md`](AUTHORING.md) — the
  `DagCollectiveActor` contract, keyed off the ring all-reduce, including the
  `Action::max_round` round-budget guard and custom reduce-op handling.
- **Tier 2 (in-tree):** add your `.cc`/`.h` under
  `sst-elements/src/sst/elements/iris/sumi/`, add them to
  `libsumi_la_SOURCES`/`nobase_library_include_HEADERS` in
  `src/sst/elements/iris/Makefile.am`, register with one
  `SUMI_REGISTER_COLLECTIVE(...)` line in `collective_registry_builtins.cc`, and
  rebuild `libsumi`.
- **Tier 3 (out-of-tree, no libsumi rebuild):** build your algorithm into its own
  shared library and load it at runtime. **Linux-first.**

## Tier 3 workflow

```sh
# 1. Build the example (a renamed ring registered as "myring") into libmycoll.so
./build_collective_plugin.sh

# 2. Run an mv2 collective test with the plugin loaded and selected (Linux):
cd ../../../tests/mv2
COLLECTIVE_PLUGINS=$PWD/../../examples/user-collectives/sumi/libmycoll.so \
  SST_HG_MV2_PRELOAD=1 SUMI_ALLREDUCE_ALG=myring \
  sst run_test_mv2_allreduce.py
```

The plugin's `SUMI_REGISTER_COLLECTIVE` static initializer runs at `dlopen` and
populates the same registry the engine reads — which works only because Tier 2
already made the engine registry-driven, and because the registry's storage is a
function-local static singleton in `libsumi` loaded `RTLD_GLOBAL`.

## Files

- `AUTHORING.md` — the authoring guide.
- `my_ring_allreduce.{h,cc}` — an editable copy of the in-tree ring, renamed and
  registered as `"myring"`; your starting point for a plugin.
- `build_collective_plugin.sh` — compiles a plugin `.cc` against the installed
  `sst/elements/iris` headers into a `.so`.

## Caveats

- **macOS:** the mv2 preload path this composes is Linux-only and off by default;
  macOS two-level namespaces / `RTLD_GLOBAL` symbol resolution across libraries
  is an open question (see `../../../../user-collectives.md` §8). Treat Tier 3 as
  Linux-first; for in-tree Tier 2 on macOS just rebuild `libsumi`.
- **v-variants** (`gatherv`/`scatterv`/`alltoallv`/`allgatherv`) take per-rank
  `int*` count arrays and are not covered by the registry.
