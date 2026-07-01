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
# Co-design Stage 2+3: cost model + Pareto assembly over the two building-block CSVs
# (train.csv from sweep_train.sh, infer.csv from sweep_infer.sh). A fleet config is
# (split N_t, a training block on hardware HW_t, an inference block on hardware HW_i);
# the two pools buy hardware independently, which is what makes the co-design a co-design.
# Objectives: train throughput up, inference throughput up, decode latency down, cost down.
# Emits frontier.csv (the non-dominated set), checks whether differentiated hardware
# dominates a uniform fleet, and reports robustness to the cost coefficients (+/-2x).
# matplotlib plots are drawn only if it is importable; the CSV is always written.

import argparse
import csv
import itertools
import os
import sys

TOTAL_GPUS = 128
REF_CAP_GB = 80.0       # cost reference: 80 GB HBM
REF_NET_GBPS = 300.0    # cost reference: 300 GB/s network


def cost_gpu(cap_gb, net_gbps, base, k_cap, k_net):
    """Normalized per-GPU cost: fixed base + capacity term + network term (=1.0 at ref)."""
    return base + k_cap * (cap_gb / REF_CAP_GB) + k_net * (net_gbps / REF_NET_GBPS)


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def best_train_blocks(rows):
    """Per (N_t, cap_t, net_t), the layout with the highest train throughput."""
    best = {}
    for r in rows:
        key = (int(r["Nt"]), float(r["cap_gb"]), float(r["net_gbps"]))
        tput = float(r["train_tput"])
        if key not in best or tput > best[key]["train_tput"]:
            best[key] = {"Nt": key[0], "cap": key[1], "net": key[2], "train_tput": tput,
                         "layout": f"TP{r['TP']}xPP{r['PP']}xDP{r['DP']} z{r['zero']} {r['alg']}",
                         "t_step_ms": float(r["t_step_ms"])}
    return best


def best_infer_blocks(rows, batch, ctx):
    """For the target serving workload (batch, ctx): per (cap_i, net_i), the min-latency
    feasible serving config -- i.e. the lowest TP that fits, which is the latency win."""
    best = {}
    for r in rows:
        if int(r["batch"]) != batch or int(r["ctx"]) != ctx:
            continue
        key = (float(r["cap_gb"]), float(r["net_gbps"]))
        lat = float(r["latency_ms"])
        if key not in best or lat < best[key]["latency_ms"]:
            best[key] = {"cap": key[0], "net": key[1], "TP": int(r["TP"]),
                         "latency_ms": lat, "per_replica_tput": float(r["per_replica_tput"])}
    return best


def assemble(train_blocks, infer_blocks, base, k_cap, k_net):
    """Every (split x training block at that split x inference block) -> a fleet config."""
    by_split = {}
    for b in train_blocks.values():
        by_split.setdefault(b["Nt"], []).append(b)

    configs = []
    for Nt, tbs in by_split.items():
        Ni = TOTAL_GPUS - Nt
        for tb, ib in itertools.product(tbs, infer_blocks.values()):
            replicas = Ni // ib["TP"]
            if replicas < 1:
                continue
            infer_tput = replicas * ib["per_replica_tput"]
            cost = (Nt * cost_gpu(tb["cap"], tb["net"], base, k_cap, k_net)
                    + Ni * cost_gpu(ib["cap"], ib["net"], base, k_cap, k_net))
            configs.append({
                "Nt": Nt, "Ni": Ni,
                "train_tput": tb["train_tput"], "train_layout": tb["layout"],
                "hw_t": f"{tb['cap']:.0f}GB/{tb['net']:.0f}GBps",
                "infer_tput": infer_tput, "latency_ms": ib["latency_ms"],
                "infer_TP": ib["TP"], "replicas": replicas,
                "hw_i": f"{ib['cap']:.0f}GB/{ib['net']:.0f}GBps",
                "uniform": (tb["cap"] == ib["cap"] and tb["net"] == ib["net"]),
                "cost": cost,
            })
    return configs


def pareto_front(configs):
    """Non-dominated set. Objectives as higher-is-better: train_tput, infer_tput,
    -latency, -cost. X dominated iff some Y is >= on all and > on one."""
    def vec(c):
        return (c["train_tput"], c["infer_tput"], -c["latency_ms"], -c["cost"])
    front = []
    for x in configs:
        vx = vec(x)
        dominated = False
        for y in configs:
            if y is x:
                continue
            vy = vec(y)
            if all(a >= b for a, b in zip(vy, vx)) and any(a > b for a, b in zip(vy, vx)):
                dominated = True
                break
        if not dominated:
            front.append(x)
    return front


def differentiated_dominates_uniform(configs):
    """Is every uniform-hardware fleet dominated by some differentiated one? (the
    headline hypothesis). Returns (claim_holds, n_uniform, n_uniform_dominated)."""
    def vec(c):
        return (c["train_tput"], c["infer_tput"], -c["latency_ms"], -c["cost"])
    uniform = [c for c in configs if c["uniform"]]
    diff = [c for c in configs if not c["uniform"]]
    dominated = 0
    for u in uniform:
        vu = vec(u)
        if any(all(a >= b for a, b in zip(vec(d), vu)) and
               any(a > b for a, b in zip(vec(d), vu)) for d in diff):
            dominated += 1
    return (uniform and dominated == len(uniform)), len(uniform), dominated


def main():
    p = argparse.ArgumentParser(description="Co-design Pareto assembly over the sweep CSVs")
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--train", default=os.path.join(here, "train.csv"))
    p.add_argument("--infer", default=os.path.join(here, "infer.csv"))
    p.add_argument("--out", default=os.path.join(here, "frontier.csv"))
    p.add_argument("--batch", type=int, default=8, help="target serving batch")
    p.add_argument("--ctx", type=int, default=16384, help="target serving context")
    p.add_argument("--base", type=float, default=0.55)
    p.add_argument("--k-cap", type=float, default=0.30)
    p.add_argument("--k-net", type=float, default=0.15)
    p.add_argument("--plot", action="store_true", help="draw plots if matplotlib present")
    a = p.parse_args()

    for path in (a.train, a.infer):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            sys.exit(f"missing/empty {path} -- run sweep_train.sh and sweep_infer.sh first")

    tb = best_train_blocks(load_rows(a.train))
    ib = best_infer_blocks(load_rows(a.infer), a.batch, a.ctx)
    if not ib:
        sys.exit(f"no inference rows for batch={a.batch} ctx={a.ctx} "
                 f"(have: {sorted({(r['batch'], r['ctx']) for r in load_rows(a.infer)})})")

    configs = assemble(tb, ib, a.base, a.k_cap, a.k_net)
    if not configs:
        sys.exit("no feasible fleet configs assembled")
    front = pareto_front(configs)

    cols = ["Nt", "Ni", "hw_t", "train_layout", "train_tput", "hw_i", "infer_TP",
            "replicas", "infer_tput", "latency_ms", "cost", "uniform"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in sorted(front, key=lambda c: c["cost"]):
            w.writerow(c)

    print(f"# co-design frontier (serving workload: batch={a.batch}, ctx={a.ctx})")
    print(f"#   {len(configs)} fleet configs, {len(front)} on the Pareto front "
          f"-> {a.out}")
    print(f"#   cost coeffs: base={a.base} k_cap={a.k_cap} k_net={a.k_net}")
    print(f"  {'split':>7} {'HW_t':>13} {'HW_i':>13} {'tr_tput':>9} "
          f"{'inf_tput':>10} {'lat_ms':>8} {'cost':>7}  kind")
    for c in sorted(front, key=lambda c: c["cost"]):
        print(f"  {c['Nt']}:{c['Ni']:<5} {c['hw_t']:>13} {c['hw_i']:>13} "
              f"{c['train_tput']:9.2f} {c['infer_tput']:10.1f} {c['latency_ms']:8.2f} "
              f"{c['cost']:7.1f}  {'uniform' if c['uniform'] else 'diff'}")

    holds, nu, nud = differentiated_dominates_uniform(configs)
    print()
    if nu == 0:
        print("# hypothesis 1 -- NOT TESTABLE on these CSVs: no uniform-hardware fleet is")
        print("#   constructible (the train and inference sweeps share no (capacity,network)")
        print("#   point). Re-run with overlapping CAPS/NETS across both sweeps, e.g. FULL=1.")
        return
    print(f"# hypothesis 1 -- differentiated dominates uniform: "
          f"{nud}/{nu} uniform fleets dominated -> {'HOLDS' if holds else 'does NOT hold'}")

    # Robustness: re-rank under +/-2x on each cost coefficient; does the claim survive?
    print("# cost-coefficient robustness (+/-2x on k_cap, k_net):")
    for kc, kn in [(a.k_cap * 2, a.k_net), (a.k_cap / 2, a.k_net),
                   (a.k_cap, a.k_net * 2), (a.k_cap, a.k_net / 2)]:
        cfg2 = assemble(tb, ib, a.base, kc, kn)
        h2, _, d2 = differentiated_dominates_uniform(cfg2)
        print(f"    k_cap={kc:.3f} k_net={kn:.3f}: {d2} dominated -> "
              f"{'HOLDS' if h2 else 'no'}")

    if a.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("# matplotlib not available -- skipping plots (CSV written)")
            return
        xs = [c["latency_ms"] for c in configs]
        ys = [c["train_tput"] for c in configs]
        cs = [c["cost"] for c in configs]
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=18)
        fx = [c["latency_ms"] for c in front]
        fy = [c["train_tput"] for c in front]
        ax.scatter(fx, fy, edgecolor="red", facecolor="none", s=60, label="Pareto front")
        ax.set_xlabel("decode latency (ms, lower better)")
        ax.set_ylabel("train throughput (lower-left is cheap+slow)")
        fig.colorbar(sc, label="fleet cost")
        ax.legend()
        fig.tight_layout()
        out_png = os.path.splitext(a.out)[0] + ".png"   # frontier.csv -> frontier.png
        fig.savefig(out_png, dpi=130)
        print(f"# wrote {out_png}")


if __name__ == "__main__":
    main()
