# codesign — a Pareto-optimal 128-GPU AI datacenter (hardware × software co-design)

The other studies each answered one question; this capstone uses them as **building
blocks** of a single architect's decision: given **128 GPUs**, how do you split them
between **training** and **inference**, what **hardware** do you buy for each pool, and
what **software layout** do you run — to be Pareto-optimal across **training throughput**,
**inference latency/throughput**, and **cost**? The two pools may buy *different* hardware
(the central lever), and they pull opposite ways (study I), so it is a genuine co-design.

See [`../../../CODESIGN-EXPERIMENT.md`](../../../CODESIGN-EXPERIMENT.md) for the full plan
and the empirical pre-checks behind the axis choices.

## The two hardware axes (and why)

A pre-check settled what actually moves each workload:

- **Training → network bandwidth** (`LINK_BW`). The gradient all-reduce is the lever
  (studies A/E); HBM bandwidth barely moves step time.
- **Inference → HBM capacity** (`GPU_MEM`, GB) — *not* HBM bandwidth. Measured: decode
  latency is flat in HBM bandwidth (0–2% per 4×) and in network. The real lever is the
  **TP degree** (TP1 ≈ 4 ms vs TP8 ≈ 38 ms), and TP is forced *up* by KV-cache capacity.
  So more HBM capacity → fit the KV cache at a *lower* TP → lower latency. `capacity.py`
  confirms the unlock: long context (16 k, batch 8) needs TP≥4 at 80 GB, TP≥2 at 160 GB,
  fits at TP1 at 320 GB.

HBM *bandwidth* is therefore left at its A100 default — it is not a co-design axis.

## Files

| File | What it is |
|------|------------|
| [`sweep_train.sh`](sweep_train.sh) | Stage 1a: capacity-gated `mercury_3d` sweep over (split × HBM capacity × network × TP/PP/DP), min-ZeRO-that-fits, → `train.csv` |
| [`sweep_infer.sh`](sweep_infer.sh) | Stage 1b: gated `mercury_inference` pure-decode sweep over (capacity × TP × batch × context) → `infer.csv` |
| [`pareto.py`](pareto.py) | Stage 2+3: cost model + non-dominated fleet assembly → `frontier.csv` (+ optional plot) |

`mercury_3d`, `mercury_inference`, and `memory_model/capacity.py` are reused **unchanged**
(no backend change). Feasibility always uses the real depth `L=64`; the sims use a
`SIM_LAYERS` timing skeleton (proportional cost, same comm pattern) so a first pass is fast.

## Run it

```sh
# fast smoke grid (SIM_LAYERS=8, trimmed splits) — minutes:
./sweep_train.sh          # -> train.csv
./sweep_infer.sh          # -> infer.csv
./pareto.py --plot        # -> frontier.csv (+ frontier.png if matplotlib present)

# paper grid (SIM_LAYERS=64, full splits/hardware) — hours, gate-first keeps it ~500 runs:
FULL=1 ./sweep_train.sh
FULL=1 ./sweep_infer.sh
./pareto.py --batch 8 --ctx 16384 --plot
```

Knobs (env): `FULL=1` selects the paper grid; `SPLITS`/`CAPS`/`NETS`/`SIM_LAYERS`/`MICRO`
(train), `BATCHES`/`CTXS`/`DECODE_STEPS` (infer) override individual axes. `pareto.py`
takes `--batch`/`--ctx` (the serving workload), `--base`/`--k-cap`/`--k-net` (cost
coefficients), `--plot`.

## What it produces

`pareto.py` assembles every (split × training block × inference block) into a full
datacenter config, computes the **non-dominated set** over four objectives
(train_tput ↑, infer_tput ↑, latency ↓, cost ↓), and writes `frontier.csv`. It then tests
**hypothesis 1** — whether every *uniform*-hardware fleet is dominated by some
*differentiated* one — and reports that conclusion's robustness to a ±2× sweep of the cost
coefficients (since the dollar split is an assumption, not a measurement).

## Cost model

Normalized per-GPU cost, `= 1.0` at the reference GPU (80 GB, 300 GB/s):

```
cost_gpu(cap, net) = base + k_cap·(cap/80GB) + k_net·(net/300GB/s)
```

Defaults `base=0.55, k_cap=0.30, k_net=0.15` are illustrative — the point is the *shape*
of the frontier and its robustness, not a dollar figure, so `pareto.py` sweeps the
coefficients and reports sensitivity.

## Results (full grid, `SIM_LAYERS=64` — the model's true depth)

324 fleet configs, **23 on the Pareto front** (`frontier_L64.csv`, `frontier_L64.png`).
The first pass at `SIM_LAYERS=8` (kept as `*_L8.*`) gave the right *serving* story but a
*wrong* training one — see the depth caveat below.

1. **Capacity sets a discrete decode-latency tier** (the headline, unchanged). The
   inference pool's HBM capacity fixes the lowest feasible TP, which fixes latency:

   | inference HBM | forced TP | decode latency (L=64) | fleet serving tput |
   |---------------|-----------|-----------------------|--------------------|
   | 80 GB | ≥4 | 26.8 ms | 38–72 k tok/s |
   | 160 GB | ≥2 | 15.4 ms | 0.13–0.25 M tok/s |
   | 320 GB | 1 | 3.9 ms | 1.0–2.0 M tok/s |

   6.8× latency / ~27× throughput, entirely from capacity — HBM bandwidth never enters.
   Decode scales *exactly* 8× from L=8→L=64, so the skeleton was faithful for inference.

2. **Differentiation is one-sided: serving buys capacity, training stays cheap.** At true
   depth the throughput-optimal training layout is **pure DP**, and the gradient all-reduce
   **hides completely under 64-layer backward compute** — the best layout is identical at
   50/150/600 GB/s. So **all 23 Pareto fleets buy the cheapest training hardware**
   (80GB/50, robust across ±2× cost coeffs); training throughput scales *linearly with the
   split* (`train_tput = 2.55·N_t`), bought with GPU allocation, not fabric. 32/36 uniform
   fleets are dominated; the 4 survivors are the cheapest fleet per split.

3. **Depth caveat (a flipped conclusion).** The L=8 skeleton put faster *training* networks
   on the frontier — an artifact: 8 layers is too little compute to hide the all-reduce, so
   network appeared to bind. At L=64 it hides. Decode scaled perfectly with depth; training
   did not. **Network-bound training returns** only when per-rank tokens shrink (strong
   scaling) or the model is large enough to exceed the memory wall (study G) and *must*
   shard with TP/PP/FSDP, whose comms is on the critical path (study H) and cannot hide. A
   13B model still fits under pure DP, which is *why* the training pool is cheap here.

Re-run any depth with `SIM_LAYERS=<n>`; `*_L8.*` vs `*_L64.*` files show the difference.

## Extension (not in this scaffold)

Long-context **prefill** served by context parallelism (study J, `mercury_context`) would
add a network axis back onto the serving pool — a deliberate counterpoint to "network =
training only." It is scoped as a second result in the plan, not part of the core
decode-only sweep here.
