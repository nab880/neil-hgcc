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
# SST driver for disaggregated inference (infer.cu). nranks = prefill workers +
# decode pool; PREFILL_RANKS / PROMPT_LEN / DECODE_STEPS / BATCH / REQUESTS are read
# by the skeleton; SUMI_ALLREDUCE_ALG selects the decode all-reduce algorithm.

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

    nranks = int(os.environ.get("NRANKS", "8"))

    gpu_flops        = os.environ.get("GPU_PEAK_FLOPS", "1.95e13")          # A100 fp32
    gpu_tensor_flops = os.environ.get("GPU_TENSOR_PEAK_FLOPS", "3.12e14")   # A100 bf16 TC
    gpu_bw           = os.environ.get("GPU_MEM_BW", "2000GB/s")
    gpu_sm_count     = os.environ.get("GPU_SM_COUNT", "108")                # A100, 0 = off
    gpu_mem_capacity = os.environ.get("GPU_MEM", "80GB")                    # A100-80GB; 0B = off
    gpu_mem_fatal    = os.environ.get("GPU_MEM_FATAL", "false")
    tensor_kernels = ",".join([
        "_Z8qkv_projPKfS0_Pfi", "_Z11attn_scoresPKfS0_Pfi",
        "_Z7attn_avPKfS0_Pfi",  "_Z8attn_outPKfS0_Pfi",
        "_Z6mlp_upPKfS0_Pfi",   "_Z8mlp_downPKfS0_Pfi",
    ])

    params = {
        "app1.name": "mercury_inference",
        "app1.exe_library_name": "mercury_inference",
        "app1.dependencies": ["sumi", ],
        "app1.libraries": ["gpulibrary:GpuLibrary",
                            "computelibrary:ComputeLibrary",
                            "mask_mpi:MpiApi", ],
        "app1.gpu_peak_flops":        gpu_flops,
        "app1.gpu_tensor_peak_flops": gpu_tensor_flops,
        "app1.gpu_tensor_kernels":    tensor_kernels,
        "app1.gpu_sm_count":          gpu_sm_count,
        "app1.gpu_mem_capacity":      gpu_mem_capacity,
        "app1.gpu_mem_fatal":         gpu_mem_fatal,
        "app1.gpu_max_threads_per_sm": "2048",
        "app1.gpu_mem_bandwidth": gpu_bw,
        "app1.pcie_bandwidth":    "32GB/s",
        "app1.pcie_latency":      "1us",
        "app1.gpu_kernel_launch_overhead": "2us",
    }
    platform.addParamSet("operating_system", params)

    link_bw = os.environ.get("LINK_BW", "100GB/s")
    platform.addParamSet("network_interface", {"link_bw": link_bw})
    platform.addParamSet("router", {"link_bw": link_bw, "xbar_bw": link_bw})
    platform.addParamSet("node", {"channel_bandwidth": link_bw, "num_channels": "1"})

    from scale_topo import make_topology
    topo = make_topology(nranks)   # TOPO=single (default) | TOPO=fattree

    ep = HgJob(0, nranks)

    system = System()
    system.setTopology(topo)
    system.allocateNodes(ep, "linear")

    system.build()
