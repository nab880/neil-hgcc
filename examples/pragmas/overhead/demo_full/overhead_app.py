#!/usr/bin/env python

import os
import sys
import sst
from sst.merlin.base import *
from sst.merlin.endpoint import *
from sst.merlin.interface import *
from sst.merlin.topology import *
from sst.hg import *

if __name__ == "__main__":
    examples_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    sys.path.insert(0, examples_dir)

    PlatformDefinition.loadPlatformFile("platform_file_hg_test")
    PlatformDefinition.setCurrentPlatform("platform_hg_test")
    platform = PlatformDefinition.getCurrentPlatform()

    platform.addParamSet("operating_system", {
        "app1.name" : "overhead_demo",
        "app1.exe_library_name" : "overhead_demo",
        "app1.dependencies" : ["sumi", ],
        "app1.libraries" : ["computelibrary:ComputeLibrary",
                            "mask_mpi:MpiApi",],
        # Read by `#pragma sst overhead launch_overhead` in overhead_app.cc.
        "app1.launch_overhead" : "10us",
    })

    topo = topoSingle()
    topo.num_ports = 32

    ep = HgJob(0, 2)

    system = System()
    system.setTopology(topo)
    system.allocateNodes(ep, "linear")

    system.build()
