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

    params = {
        "app1.name" : "test_cuda_vecadd",
        "app1.exe_library_name"  : "test_cuda_vecadd",
        "app1.libraries" : ["gpulibrary:GpuLibrary",
                            "computelibrary:ComputeLibrary",],
        "app1.gpu_mem_bandwidth" : os.environ.get("GPU_MEM_BANDWIDTH", "900GB/s"),
    }
    if os.environ.get("GPU_KERNEL_TIMES"):
        params["app1.gpu_kernel_times"] = os.environ["GPU_KERNEL_TIMES"]
    if os.environ.get("GPU_SM_COUNT"):
        params["app1.gpu_sm_count"] = os.environ["GPU_SM_COUNT"]
    if os.environ.get("GPU_TENSOR_PEAK_FLOPS"):
        params["app1.gpu_tensor_peak_flops"] = os.environ["GPU_TENSOR_PEAK_FLOPS"]
    if os.environ.get("GPU_TENSOR_KERNELS"):
        params["app1.gpu_tensor_kernels"] = os.environ["GPU_TENSOR_KERNELS"]
    platform.addParamSet("operating_system", params)

    topo = topoSingle()
    topo.num_ports = 32

    ep = HgJob(0,1)

    system = System()
    system.setTopology(topo)
    system.allocateNodes(ep,"linear")

    system.build()
