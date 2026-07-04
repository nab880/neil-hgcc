#!/usr/bin/env python
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
# Shared topology helper for the AI-at-scale demos. Lets a demo run on either the
# original single crossbar (small, simple) or a real multi-router fat-tree that
# scales past 32 endpoints and -- unlike one crossbar -- partitions across SST
# host threads/ranks (sst -n / mpirun), so parallel execution actually speeds up.
# Selected by the TOPO environment variable so the stock examples are unchanged.

import os
import math
from sst.merlin.topology import *


def fattree_shape(n):
    """A balanced 2-level fat-tree shape (down,up:down) with exactly n hosts.
    Perfect squares give d,d:d (16->4,4:4 ... 1024->32,32:32); other n factor as
    a*b nearest the square root; primes fall back to a single fat router."""
    d = math.isqrt(n)
    if d * d == n:
        return "%d,%d:%d" % (d, d, d)
    a = d
    while a > 1 and n % a:
        a -= 1
    if a <= 1:
        return "%d" % n
    return "%d,%d:%d" % (a, n // a, n // a)


def make_topology(nranks):
    """Return a configured Merlin topology for nranks endpoints.

    TOPO=single  (default) -- one crossbar router (the original behaviour; the
                  port cap is raised to nranks so it still builds past 32).
    TOPO=fattree           -- a fat-tree sized to nranks (LINK_LAT sets the
                  per-hop latency, default 100ns)."""
    if os.environ.get("TOPO", "single").startswith("fattree"):
        topo = topoFatTree()
        topo.shape = fattree_shape(nranks)
        lat = os.environ.get("LINK_LAT", "100ns")
        topo.link_latency = lat
        topo.host_link_latency = lat
    else:
        topo = topoSingle()
        topo.num_ports = max(32, nranks)
    return topo
