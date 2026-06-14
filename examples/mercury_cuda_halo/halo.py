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

    nranks     = int(os.environ.get("NRANKS", "4"))
    gpu_direct = os.environ.get("GPUDIRECT", "false")

    params = {
        "app1.name" : "mercury_cuda_halo",
        "app1.exe_library_name" : "mercury_cuda_halo",
        "app1.dependencies" : ["sumi", ],
        "app1.libraries" : ["gpulibrary:GpuLibrary",
                            "computelibrary:ComputeLibrary",
                            "mask_mpi:MpiApi",],
        # GPU model knobs (CUDA_PLAN §4). Tune to your target node.
        "app1.gpu_mem_bandwidth" : "1500GB/s",
        "app1.gpu_peak_flops"    : "1.5e13",
        "app1.pcie_bandwidth"    : "32GB/s",
        "app1.pcie_latency"      : "1us",
        # The GPUDirect study: flip this and re-run to see the staging cost.
        "app1.gpu_direct" : gpu_direct,
    }
    # Calibrate once on a real node, then drop the JSON in here for single-node
    # accuracy at any scale. Comment out to use the roofline instead.
    cal = os.path.join(example_dir, "gpu_calibration.json")
    if os.path.exists(cal):
        params["app1.gpu_kernel_times"] = cal

    platform.addParamSet("operating_system", params)

    topo = topoSingle()
    topo.num_ports = 32

    ep = HgJob(0, nranks)

    system = System()
    system.setTopology(topo)
    system.allocateNodes(ep, "linear")

    system.build()
