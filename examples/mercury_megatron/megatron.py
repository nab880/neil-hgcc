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

import os
import sys
import sst
from sst.merlin.base import *
from sst.merlin.endpoint import *
from sst.merlin.interface import *
from sst.merlin.topology import *
from sst.hg import *

if __name__ == "__main__":
    example_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(example_dir))

    PlatformDefinition.loadPlatformFile("platform_file_hg_test")
    PlatformDefinition.setCurrentPlatform("platform_hg_test")
    platform = PlatformDefinition.getCurrentPlatform()

    # nranks = TP_SIZE * PP_SIZE. TP_SIZE / MICROBATCH are read by the skeleton
    # from the environment; run.sh sweeps them along with NRANKS and LINK_BW.
    nranks = int(os.environ.get("NRANKS", "4"))

    params = {
        "app1.name": "mercury_megatron",
        "app1.exe_library_name": "mercury_megatron",
        "app1.dependencies": ["sumi", ],
        "app1.libraries": ["gpulibrary:GpuLibrary",
                            "computelibrary:ComputeLibrary",
                            "mask_mpi:MpiApi", ],
        "app1.gpu_peak_flops":    "3.0e14",
        "app1.gpu_mem_bandwidth": "2000GB/s",
        "app1.pcie_bandwidth":    "32GB/s",
        "app1.pcie_latency":      "1us",
        "app1.gpu_kernel_launch_overhead": "2us",
    }
    platform.addParamSet("operating_system", params)

    # Fabric bandwidth: TP all-reduce is on the critical path, so this is the
    # lever for tensor parallelism (NVLink-class intra-node fabrics).
    link_bw = os.environ.get("LINK_BW", "150GB/s")
    platform.addParamSet("network_interface", {"link_bw": link_bw})
    platform.addParamSet("router", {"link_bw": link_bw, "xbar_bw": link_bw})
    platform.addParamSet("node", {"channel_bandwidth": link_bw, "num_channels": "1"})

    from scale_topo import make_topology
    topo = make_topology(nranks)   # TOPO=single (default) | TOPO=fattree (scales past 32)

    ep = HgJob(0, nranks)

    system = System()
    system.setTopology(topo)
    system.allocateNodes(ep, "linear")

    system.build()
