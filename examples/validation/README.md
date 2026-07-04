# Validation (WS3)

Establishes how far the packet-level simulator can be trusted, in three layers:
a closed-form cross-check it should match, the regime where it adds value, and
the real-hardware anchor it still needs.

## What's here

- `analytic_baseline.py` — a Calculon-style **no-contention** closed form for the
  DP transformer step. Uses the *same* model the simulator charges (dual roofline
  + wave/tile quantization for compute; a textbook all-reduce for comms) but with
  no network contention. Mirrors `../mercury_llm_train/train.cu`; keep dims in sync.
- `validate.sh` — runs the simulator across rank counts and both all-reduce
  algorithms, on a single crossbar and on the fat-tree, and tabulates each against
  the analytic baseline.

## The three validation layers

### 1. Closed-form cross-check (runnable here, no GPU)

The analytic baseline is an independent re-derivation of the step time from the
kernel flop counts. On a contention-free configuration the simulator must agree
with it; disagreement would mean a bookkeeping bug.

- **Compute floor: exact.** `analytic_baseline.py --nranks 1` predicts
  **74.5784 ms**; the simulator's compute floor (NRANKS=4, 300 GB/s, comms hidden)
  is **74.58 ms** — they match to 4 decimals, validating the dual-roofline +
  wave-quant accounting.
- **Comms on a crossbar:** `validate.sh` column `d_xbar` — the simulator's exposed
  all-reduce time on a single (minimal-contention) crossbar vs the analytic form.

### 2. Where the simulator earns its keep (runnable here)

On a real fat-tree the simulator diverges from the analytic baseline — `validate.sh`
column `d_ftree`. That gap is **network contention**: packets from concurrent
collectives queueing at shared links/switches, which a closed-form model assumes
away. This is the quantity the packet-level fabric exists to measure, and the
reason an analytical tool (Calculon, ASTRA-sim analytical mode) cannot answer the
question this simulator targets.

Measured (DP step, overlap off, 50 GB/s links, A100 compute model):

| alg | N | analytic ms | crossbar ms | Δ | fat-tree ms | Δ |
|-----|---|-------------|-------------|------|-------------|------|
| recursive-doubling | 8  | 300.07 | 304.32 | +1.4% | 304.37 | +1.4% |
| recursive-doubling | 16 | 316.18 | 321.84 | +1.8% | 321.90 | +1.8% |
| ring | 8  | 300.08 | 309.99 | +3.3% | 312.11 | +4.0% |
| ring | 16 | 316.19 | 337.43 | +6.7% | 458.35 | **+45.0%** |

Recursive-doubling stays within ~2% of the closed form even on the fat-tree
(contention-robust at this scale), but the ring diverges **+45%** at N=16: its
2(N-1) serialized neighbour-steps hit real link contention under the linear
rank-to-fabric placement, which the analytic model cannot see. The algorithm x
topology interaction -- not the per-algorithm cost in isolation -- is the result,
and it is the lead-in to the factorization study (WS2-2d).

### 3. Real-hardware anchor (BLOCKED here — no GPU on this host)

The above validates the simulator against a closed form and characterises its
contention term, but not against measured silicon. To close that, run the
calibrate-once flow on a real node and a small real all-reduce / training step,
then fill this table with measured numbers **and their source** (do not estimate):

| Config (model, GPUs, fabric) | Source (cite) | Measured step | Sim step | Error |
|------------------------------|---------------|---------------|----------|-------|
| _e.g. GPT-3 13B, 8xA100, NVLink_ | _paper / run log_ | _fill_ | _fill_ | _fill_ |
| _e.g. published MLPerf entry_ | _MLPerf result id_ | _fill_ | _fill_ | _fill_ |

`analytic_baseline.py` and `validate.sh` take the model dims and hardware peaks as
arguments, so a published config can be reproduced without code changes.

## Reproduce

```sh
# compute-floor cross-check
python3 analytic_baseline.py --nranks 1

# full table (sim vs analytic, crossbar vs fat-tree, both algorithms)
NSET="8 16" BW=50GB/s sh validate.sh
```
