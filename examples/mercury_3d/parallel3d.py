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

    # nranks = DP * PP * TP. TP_SIZE / PP_SIZE / MICROBATCH are read by the
    # skeleton (DP_SIZE = nranks / (TP*PP)); run.sh sweeps the factorization.
    nranks = int(os.environ.get("NRANKS", "8"))

    # A100 dual-roofline + wave-quant compute model (WS1), matching mercury_llm_train.
    # The single sharded GEMM kernel runs on the tensor cores; act_norm is the
    # memory-bound CUDA-core path. All env-overridable.
    gpu_flops        = os.environ.get("GPU_PEAK_FLOPS", "1.95e13")       # A100 fp32
    gpu_tensor_flops = os.environ.get("GPU_TENSOR_PEAK_FLOPS", "3.12e14") # A100 bf16 TC
    gpu_sm_count     = os.environ.get("GPU_SM_COUNT", "108")             # A100, 0 = off
    gpu_mem_capacity = os.environ.get("GPU_MEM", "80GB")                 # A100-80GB; 0B = off
    gpu_mem_fatal    = os.environ.get("GPU_MEM_FATAL", "false")          # warn-only by default

    params = {
        "app1.name": "mercury_3d",
        "app1.exe_library_name": "mercury_3d",
        "app1.dependencies": ["sumi", ],
        "app1.libraries": ["gpulibrary:GpuLibrary",
                            "computelibrary:ComputeLibrary",
                            "mask_mpi:MpiApi", ],
        "app1.gpu_peak_flops":         gpu_flops,
        "app1.gpu_tensor_peak_flops":  gpu_tensor_flops,
        "app1.gpu_tensor_kernels":     "_Z4gemmPKfS0_Pfi",
        "app1.gpu_sm_count":           gpu_sm_count,
        "app1.gpu_mem_capacity":       gpu_mem_capacity,
        "app1.gpu_mem_fatal":          gpu_mem_fatal,
        "app1.gpu_max_threads_per_sm": "2048",
        "app1.gpu_mem_bandwidth": "2000GB/s",
        "app1.pcie_bandwidth":    "32GB/s",
        "app1.pcie_latency":      "1us",
        "app1.gpu_kernel_launch_overhead": "2us",
    }
    platform.addParamSet("operating_system", params)

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
