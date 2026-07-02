#!/usr/bin/env python3
"""Plot a sweep.py CSV: simulated collective time vs message size, one line per
algorithm, one subplot per (topology, rank count).

Usage:  ./plot_sweep.py sweep.csv [-o sweep.png]
"""
import argparse
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "PASS" or not r.get("time_s"):
                continue
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        print("no PASS rows with a time in %s" % args.csv, file=sys.stderr)
        sys.exit(1)

    # facet key = (topo, nranks); series key = algorithm
    facets = sorted({(r["topo"], int(r["nranks"])) for r in rows})
    series = defaultdict(lambda: defaultdict(list))  # facet -> alg -> [(bytes, t)]
    for r in rows:
        facets_key = (r["topo"], int(r["nranks"]))
        series[facets_key][r["algorithm"]].append(
            (int(r["bytes"]), float(r["time_s"])))

    n = len(facets)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2), squeeze=False)
    for ax, facet in zip(axes[0], facets):
        topo, nranks = facet
        for alg in sorted(series[facet]):
            pts = sorted(series[facet][alg])
            xs = [b for b, _ in pts]
            ys = [t for _, t in pts]
            ax.plot(xs, ys, marker="o", label=alg)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("message size (bytes)")
        ax.set_ylabel("simulated time (s)")
        ax.set_title("%s, %d ranks" % (topo, nranks))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title="algorithm")

    fig.tight_layout()
    out = args.out or (args.csv.rsplit(".", 1)[0] + ".png")
    fig.savefig(out, dpi=130)
    print("wrote %s" % out, file=sys.stderr)


if __name__ == "__main__":
    main()
