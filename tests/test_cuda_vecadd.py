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

import sst, os
from sst.merlin.base import *
from sst.merlin.endpoint import *
from sst.merlin.interface import *
from sst.merlin.topology import *
from sst.hg import *

if __name__ == "__main__":

    PlatformDefinition.loadPlatformFile("platform_file_hg_test")
    PlatformDefinition.setCurrentPlatform("platform_hg_test")
    platform = PlatformDefinition.getCurrentPlatform()

    # The roofline knobs and the optional calibration table are swept from the
    # environment by run_test_cuda_vecadd.sh; defaults give the plain run.
    params = {
        "app1.name" : "test_cuda_vecadd",
        "app1.exe_library_name"  : "test_cuda_vecadd",
        # The GPU library is an optional capability declared per-app, like
        # ComputeLibrary / mask_mpi. Without it the first sst_hg_cuda_* call
        # aborts with "Library GpuLibrary not found".
        "app1.libraries" : ["gpulibrary:GpuLibrary",
                            "computelibrary:ComputeLibrary",],
        "app1.gpu_mem_bandwidth" : os.environ.get("GPU_MEM_BANDWIDTH", "900GB/s"),
    }
    if os.environ.get("GPU_KERNEL_TIMES"):
        params["app1.gpu_kernel_times"] = os.environ["GPU_KERNEL_TIMES"]
    platform.addParamSet("operating_system", params)

    topo = topoSingle()
    topo.num_ports = 32

    ep = HgJob(0,1)

    system = System()
    system.setTopology(topo)
    system.allocateNodes(ep,"linear")

    system.build()
