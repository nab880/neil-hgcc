#!/usr/bin/env python3
"""Tier 2 comparison sweep: {algorithms} x {message sizes} x {rank counts} x
{topology} -> one CSV of simulated collective time.

Each cell runs the self-checking check_allreduce app (so a wrong result is
caught, not silently timed) and records the simulator's reported simulated time.
Message size is set at compile time via -DNELEMS, so the app is rebuilt once per
size and reused across algorithms and rank counts.

Usage:
  ./sweep.py                                   # sane defaults -> sweep.csv
  ./sweep.py --algs ring recdouble --sizes 16 256 4096 --nranks 8 16 --out my.csv
  ./sweep.py --topo fattree --nranks 16 64

Then plot with:  ./plot_sweep.py sweep.csv
"""
import argparse
import csv
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MPI_DIR = os.path.join(HERE, "mpi")

# "... simulated time: 2.111 us"
_TIME_RE = re.compile(r"simulated time:\s*([0-9.eE+-]+)\s*([a-zµμ]+)")
_UNIT_S = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "μs": 1e-6,
           "ns": 1e-9, "ps": 1e-12}


def parse_time_seconds(text):
    m = None
    for m in _TIME_RE.finditer(text):
        pass  # keep the last match (the final "simulation complete" line)
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2)
    return val * _UNIT_S.get(unit, float("nan"))


def resolve_prefix(p):
    p = p or os.path.join(HERE, "..", "..", "..", "install")
    return os.path.abspath(p)


def build_size(prefix, nelems, env):
    e = dict(env)
    e["SRC"] = "check_allreduce.c"
    e["EXTRA_CFLAGS"] = "-DNELEMS=%d" % nelems
    subprocess.run(["./build.sh", prefix], cwd=MPI_DIR, env=e, check=True,
                   stdout=subprocess.DEVNULL)


def run_cell(sst, nranks, alg, topo, env):
    e = dict(env)
    e["APP"] = "check_allreduce"
    e["NRANKS"] = str(nranks)
    e["ALG"] = alg
    e["TOPO"] = topo
    out = subprocess.run([sst, "run.py"], cwd=MPI_DIR, env=e,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    ok = out.returncode == 0 and "PASS: check_allreduce" in out.stdout
    return ok, parse_time_seconds(out.stdout), out.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algs", nargs="+", default=["ring", "recdouble"],
                    help="algorithm names; '' (the built-in default) is always included")
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[16, 256, 4096, 65536], help="message sizes in int elements")
    ap.add_argument("--nranks", nargs="+", type=int, default=[8])
    ap.add_argument("--topo", default="single")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "sweep.csv"))
    args = ap.parse_args()

    prefix = resolve_prefix(args.prefix)
    sst = os.path.join(prefix, "bin", "sst")
    if not os.path.isfile(sst):
        sst = "sst"
    env = dict(os.environ)

    # The built-in default (empty ALG) plus each named algorithm, de-duplicated.
    algs = [""]
    for a in args.algs:
        if a not in algs:
            algs.append(a)

    rows = []
    for nelems in args.sizes:
        print("building check_allreduce with NELEMS=%d ..." % nelems, file=sys.stderr)
        build_size(prefix, nelems, env)
        for nranks in args.nranks:
            for alg in algs:
                label = alg or "<default>"
                ok, secs, log = run_cell(sst, nranks, alg, args.topo, env)
                status = "PASS" if ok else "FAIL"
                print("  nelems=%-6d nranks=%-3d %-12s %s %s" %
                      (nelems, nranks, label, status,
                       ("%.3e s" % secs) if secs is not None else "no-time"),
                      file=sys.stderr)
                if not ok:
                    sys.stderr.write("".join("      | " + l + "\n"
                                             for l in log.splitlines()[-6:]))
                rows.append({
                    "algorithm": label, "nranks": nranks, "nelems": nelems,
                    "bytes": nelems * 4, "topo": args.topo,
                    "status": status,
                    "time_s": ("%.9e" % secs) if secs is not None else "",
                })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["algorithm", "nranks", "nelems",
                                          "bytes", "topo", "status", "time_s"])
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (args.out, len(rows)), file=sys.stderr)


if __name__ == "__main__":
    main()
