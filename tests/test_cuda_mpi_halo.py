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

    nranks = int(os.environ.get("NRANKS", "2"))
    gpu_direct = os.environ.get("GPUDIRECT", "false")

    PlatformDefinition.loadPlatformFile("platform_file_hg_test")
    PlatformDefinition.setCurrentPlatform("platform_hg_test")
    platform = PlatformDefinition.getCurrentPlatform()

    platform.addParamSet("operating_system", {
        "app1.name" : "test_cuda_mpi_halo",
        "app1.exe_library_name"  : "test_cuda_mpi_halo",
        "app1.dependencies" : ["sumi", ],
        "app1.libraries" : ["gpulibrary:GpuLibrary",
                            "computelibrary:ComputeLibrary",
                            "mask_mpi:MpiApi",],
        "app1.gpu_direct" : gpu_direct,
    })

    topo = topoSingle()
    topo.num_ports = 32

    ep = HgJob(0, nranks)

    system = System()
    system.setTopology(topo)
    system.allocateNodes(ep,"linear")

    system.build()
