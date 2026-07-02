# Tier 2 — authoring a native sumi collective algorithm

At Tier 2 you implement a collective as a **DAG of send/recv actions** inside the
Iris/sumi engine, register it by name, and select it at runtime. Unlike Tier 1
(which expresses an algorithm in application-level MPI messages), a Tier 2
algorithm *is* what `MPI_Allreduce` runs, so it is measured with full protocol
fidelity (eager/rendezvous, NIC/router model, contention).

The worked reference is the ring all-reduce:
`sst-elements/src/sst/elements/iris/sumi/ring_allreduce.{h,cc}`. A copy you can
edit lives next to this file (`my_ring_allreduce.{h,cc}`).

## The two classes

Every algorithm is a `DagCollective` + `DagCollectiveActor` pair (see
`collective.h`, `collective_actor.h`):

- **`DagCollective`** — the factory. Holds the buffers/args and returns a fresh
  actor from `newActor()`. Boilerplate; copy the ring's.
- **`DagCollectiveActor`** — the algorithm. You override three (optionally four)
  methods below.

## The author's contract

### `initDag()` — build the schedule

This is the algorithm. Vocabulary (`collective_actor.h`):

- `new SendAction(round, partner, buf_type)` and
  `new RecvAction(round, partner, buf_type)` — one message each. Set
  `->offset` and `->nelems` (element offset/count into the buffer) after
  constructing.
- `addDependency(precursor, action)` — `action` may not start until `precursor`
  completes. This is how you order rounds and pair sends with the recvs that
  consume their result. A `nullptr` precursor is ignored, so round 0 actions can
  pass `nullptr`.
- Buffer types:
  - `SendAction::in_place` — send bytes straight out of the result buffer.
  - `RecvAction::reduce` — combine incoming bytes into the local buffer via the
    reduce function (reduce-scatter phase).
  - `RecvAction::in_place` — drop incoming bytes into the buffer (all-gather
    phase — no reduction).

The ring lays out `N-1` reduce-scatter rounds then `N-1` all-gather rounds,
chaining each round's send+recv off the previous round's send+recv.

### `initBuffers()` / `finalizeBuffers()`

Wire `result_buffer_` / `send_buffer_` and allocate any scratch with
`my_api_->allocateWorkspace(size, hint)`; free it in `finalizeBuffers()`. Copy
the ring's pattern (it works in place in the result buffer and uses one scratch
recv buffer).

### `bufferAction(dst_buffer, msg_buffer, ac)`

Called when a `RecvAction` fires. Apply the reduce (`fxn_(dst, msg, nelems)`) for
reducing rounds, or `my_api_->memcopy(dst, msg, nelems*type_size_)` for
placement rounds. The ring keys this off the round index.

## Two sharp edges (learned in the ring — the template guards both)

### 1. Round budget vs `Action::max_round`

The message-id codec packs the round index into the message id against a fixed
budget:

```cpp
// collective_actor.h
static const uint32_t max_round = 500;
static uint32_t messageId(type_t ty, int r, int p){
  const int num_enums = 6;
  return p*max_round*num_enums + r*num_enums + ty;   // r must stay < max_round
}
```

If your algorithm's round count reaches `max_round`, the round index bleeds into
the partner field and **silently corrupts message matching** (wrong messages get
paired; results are wrong or it deadlocks). `2*(N-1)` is what the *ring* happens
to use — your algorithm has its own round count. Ship a guard against **your**
count, like the ring's:

```cpp
if (my_num_rounds >= static_cast<int>(Action::max_round)){
  sst_hg_abort_printf("myalg needs %d rounds but Action::max_round is %u; "
                      "raise max_round for nproc=%d", my_num_rounds,
                      Action::max_round, N);
}
```

> Note: `max_round` is **500** on this branch (`collective_actor.h:74`). A ring
> is fine up to ~250 ranks; a log-depth algorithm never approaches it. If you
> genuinely need more rounds, raising `max_round` is a one-line change but
> widens every message id — do it deliberately.

### 2. Custom reduce ops

`reduce_fxn` is `std::function<void(void*, const void*, int)>`
(`comm_functions.h`): `(dst, src, nelems)` computes `dst = dst OP src`. For a
standard op (`MPI_SUM` etc.) the engine supplies it and you just call `fxn_`. If
you reduce with a non-standard op, provide your own `reduce_fxn` and make sure
`bufferAction` routes `RecvAction::reduce` messages through it — the actor stores
it as `fxn_` and also exposes it via `slicer_->fxn` (set in `initDag`).

## Register it

Add one line to `collective_registry_builtins.cc` (in-tree) or your plugin
`.cc` (out-of-tree, Tier 3):

```cpp
SUMI_REGISTER_COLLECTIVE(allreduce, "myalg", MyAllreduce,
    a.engine, a.dst, a.src, a.nelems, a.type_size, a.tag, a.fxn, a.cq_id, a.comm)
```

The `TYPE` is the `Collective::type_t` (e.g. `allreduce`, `bcast`); `NAME` is the
string you select by; the trailing arguments are your constructor call spelled
against the `CollectiveFactoryArgs a` — pick the fields your constructor needs
(allreduce uses `a.src`+`a.fxn` and ignores `a.root`; bcast uses `a.root` and
ignores `a.src`/`a.fxn`). The v-variants (`gatherv`/`scatterv`/`alltoallv`/
`allgatherv`) take per-rank `int*` count arrays and are **not** covered by the
registry.

## Select it

Precedence: **Python param > `SUMI_<OP>_ALG` env var > built-in default.**

```python
# run.py / a platform file: the sumi library reads app1.* params
platform.addParamSet("operating_system", {"app1.allreduce_alg": "myalg"})
```

```sh
# quick-and-dirty env path (same process, so it reaches the C++ engine)
SUMI_ALLREDUCE_ALG=myalg APP=test_mv2_allreduce sst run_test_mv2_allreduce.py
```

Selecting a name that isn't registered **aborts with a clear message** (rather
than silently falling back), so a typo can't masquerade as a real measurement.
Leaving it empty uses the op's built-in default, unchanged.

Supported ops and their registered built-ins (see
`collective_registry_builtins.cc`): `allreduce` (`recdouble`, `ring`),
`reduce` (`recdouble`), `bcast` (`btree`), `reduce_scatter` (`halving`),
`scan` (`btree`), `gather` (`btree`), `scatter` (`btree`).

## Verify and measure

There are two harnesses, depending on which MPI you have built:

**mask-mpi (recommended; no hgcc, no MVAPICH2).** mask-mpi is an in-tree MPI that
runs `MPI_*` straight over the sumi `CollectiveEngine`, so your algorithm is
exercised with only `sst`. In `sst-elements/src/sst/elements/mask-mpi/tests`:

```sh
# single run, algorithm selected by param (app1.allreduce_alg) or env
ALG=ring sst test_allreduce.py                 # self-checks the sum -> PASS
SUMI_ALLREDUCE_ALG=ring sst test_allreduce.py  # env path (param wins if both set)

# correctness + timing across algorithms x rank counts (incl. non-power-of-2):
ALGS="ring recdouble myring" NRANKS_LIST="2 4 7 8 16" ./collective_sweep.sh
```

`test_allreduce.py` contributes `rank+1` into 1024 elements and asserts the sum
`n(n+1)/2`, so a wrong DAG fails the assert (or deadlocks, caught by the
engine's `deadlockCheck()`); `collective_sweep.sh` records the simulated time to
a CSV. This is how the registry+ring were verified.

**mv2 / hgcc (full MVAPICH2 fidelity; needs the hgcc toolchain + a built
mvapich2).** For environments with the hgcc MPI stack, the Tier 1 harness drives
the same selection through real MVAPICH2 collectives:

- **Correctness:** `../correctness_matrix.sh` sweeps `check_allreduce` across rank
  counts under each algorithm; every cell must `PASS`.
- **Cost:** `../sweep.py` runs {your algorithm, built-ins} × message sizes ×
  ranks × topology into a CSV; `../plot_sweep.py` charts time vs size.
