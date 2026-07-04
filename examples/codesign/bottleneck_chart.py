#!/usr/bin/env python3
#
# Copyright 2009-2026 NTESS. Under the terms
# of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# Copyright (c) 2009-2026, NTESS
# All rights reserved.
#
# This file is part of the SST software package. For license
# information, see the LICENSE file in the top level directory of the
# distribution.
#
# Render the bottleneck figure from bottleneck.csv (bottleneck.sh). Two panels that show
# WHY each pool sits where it does, not just the outcome:
#   (A) per-config time decomposition: compute (GPU-busy) vs exposed communication
#       (step - GPU-busy). A tall compute bar with ~no exposed comm = compute-bound;
#       a large exposed-comm bar = the critical-path collective is the bottleneck.
#   (B) network sensitivity: each config's step time across LINK_BW, as % change from the
#       slowest network. Everything ~flat (0%) => nothing is network-BANDWIDTH-bound; the
#       decode slowdown at low capacity is the all-reduce LATENCY, not bandwidth.
# Together: training is compute-bound (fabric hidden); decode's bottleneck shifts from
# compute (TP1, high capacity) to a latency-bound all-reduce (TP>=2, low capacity), and
# capacity is the lever because it buys the pool down to TP1 where the all-reduce vanishes.

import csv
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, "bottleneck.csv")
if not os.path.exists(src):
    sys.exit("missing bottleneck.csv -- run ./bottleneck.sh first")
rows = list(csv.DictReader(open(src)))
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib not available")


def by_label(pool):
    out = {}
    for r in rows:
        if r["pool"] != pool:
            continue
        out.setdefault(r["label"], []).append(
            (float(r["net_gbps"]), float(r["sim_ms"]), float(r["gpu_ms"])))
    for k in out:
        out[k].sort()
    return out


train, decode = by_label("train"), by_label("decode")
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.6))

# (A) compute vs exposed-comm as a 100%-stacked decomposition, at the mid network --
# scale-independent (training step ~392 ms and decode per-token ~4-27 ms share one axis).
# The exposed-comm FRACTION is the bottleneck; the absolute step/token time is annotated.
items = ([("decode", k, v) for k, v in sorted(decode.items())]
        + [("train", k, v) for k, v in sorted(train.items(), key=lambda kv: kv[0])])
labels, comp_f, exp_f, totals = [], [], [], []
for pool, label, pts in items:
    _, sim, gpu = pts[len(pts) // 2]
    ex = max(0.0, sim - gpu)
    labels.append(label); totals.append(sim)
    comp_f.append(100.0 * gpu / sim); exp_f.append(100.0 * ex / sim)
y = range(len(labels))
axA.barh(list(y), comp_f, color="tab:blue", label="compute (GPU-busy)")
axA.barh(list(y), exp_f, left=comp_f, color="tab:red", label="exposed comm (critical-path collective)")
for i, tot in enumerate(totals):
    unit = "ms/tok" if labels[i].startswith("cap") else "ms/step"
    axA.text(101, i, f"{tot:.0f} {unit}", va="center", fontsize=7)
axA.set_yticks(list(y)); axA.set_yticklabels(labels, fontsize=7)
axA.set_xlim(0, 130); axA.set_xticks([0, 25, 50, 75, 100])
axA.set_xlabel("share of step time (%)")
axA.set_title("(A) bottleneck = the exposed-comm (red) share\n"
              "decode TP>=2 is all-reduce-bound; TP1 & training are compute-bound")
axA.legend(fontsize=8, loc="lower left")

# (B) network sensitivity: % change in step time from the slowest (50 GB/s) network.
for pool, d, marker, col in (("decode", decode, "s", "tab:red"),
                             ("train", train, "o", "tab:blue")):
    first = True
    for label, pts in d.items():
        base = pts[0][1]
        nets = [p[0] for p in pts]
        pct = [100.0 * (p[1] - base) / base for p in pts]
        axB.plot(nets, pct, marker + "-", color=col, alpha=0.7,
                 label=(pool if first else None))
        first = False
axB.axhline(0, ls=":", color="k", lw=0.8)
axB.set_xscale("log")
axB.set_xlabel("network bandwidth (GB/s)")
axB.set_ylabel("step-time change from 50 GB/s (%)")
axB.set_ylim(-5, 5)
axB.set_title("(B) ~0% everywhere => nothing is network-BANDWIDTH-bound\n"
              "(decode slowdown at low capacity is all-reduce latency)")
axB.legend(fontsize=9)

fig.tight_layout()
out = os.path.join(here, "bottleneck.png")
fig.savefig(out, dpi=130)
print(f"# wrote {out}")
print(f"  {'config':22} {'compute_ms':>11} {'exposed_ms':>11} {'bottleneck':>14}")
for (pool, label, pts) in items:
    _, sim, gpu = pts[len(pts) // 2]
    ex = max(0.0, sim - gpu)
    kind = "compute" if ex < 0.15 * sim else "exposed-comm"
    print(f"  {label:22} {gpu:>11.2f} {ex:>11.2f} {kind:>14}")
