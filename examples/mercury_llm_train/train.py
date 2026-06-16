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
    # The shared platform file lives in examples/. loadPlatformFile imports by
    # module name (not path), so put that directory on sys.path and load by name.
    sys.path.insert(0, os.path.dirname(example_dir))

    PlatformDefinition.loadPlatformFile("platform_file_hg_test")
    PlatformDefinition.setCurrentPlatform("platform_hg_test")
    platform = PlatformDefinition.getCurrentPlatform()

    # nranks = data-parallel replicas (one GPU per rank). GPUDIRECT / LLM_OVERLAP
    # are read by the skeleton itself from the environment; run.sh sweeps them.
    nranks = int(os.environ.get("NRANKS", "4"))

    params = {
        "app1.name": "mercury_llm_train",
        "app1.exe_library_name": "mercury_llm_train",
        "app1.dependencies": ["sumi", ],
        "app1.libraries": ["gpulibrary:GpuLibrary",
                            "computelibrary:ComputeLibrary",
                            "mask_mpi:MpiApi", ],
        # GPU model knobs -- an A100-class node (fp16 tensor peak, HBM2e).
        "app1.gpu_peak_flops":    "3.0e14",
        "app1.gpu_mem_bandwidth": "2000GB/s",
        "app1.pcie_bandwidth":    "32GB/s",
        "app1.pcie_latency":      "1us",
        "app1.gpu_kernel_launch_overhead": "2us",
    }
    platform.addParamSet("operating_system", params)

    # Fabric bandwidth is the lever that moves DDP between comms-bound and
    # compute-bound. Override the platform's default 12 GB/s link with LINK_BW
    # (e.g. 50 GB/s InfiniBand, 300 GB/s NVLink) so the sweep can cross over.
    link_bw = os.environ.get("LINK_BW", "12GB/s")
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
