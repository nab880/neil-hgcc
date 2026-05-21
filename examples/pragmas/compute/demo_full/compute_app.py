#!/usr/bin/env python

import os
import sst
from sst.merlin.base import *
from sst.merlin.endpoint import *
from sst.merlin.interface import *
from sst.merlin.topology import *
from sst.hg import *

if __name__ == "__main__":
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    platform_file = os.path.join(examples_dir, "..", "..", "platform_file_hg_test.py")

    PlatformDefinition.loadPlatformFile(platform_file)
    PlatformDefinition.setCurrentPlatform("platform_hg_test")
    platform = PlatformDefinition.getCurrentPlatform()

    platform.addParamSet("operating_system", {
        "app1.name" : "compute_demo",
        "app1.exe_library_name" : "compute_demo",
        "app1.dependencies" : ["sumi", ],
        "app1.libraries" : ["computelibrary:ComputeLibrary",
                            "mask_mpi:MpiApi",],
    })

    topo = topoSingle()
    topo.num_ports = 32

    ep = HgJob(0, 2)

    system = System()
    system.setTopology(topo)
    system.allocateNodes(ep, "linear")

    system.build()
