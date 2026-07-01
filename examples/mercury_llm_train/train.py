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

    # GPU model knobs default to an A100-class node. The compute model is a dual
    # roofline (WS1-1a): the GEMM kernels run on the tensor cores (fp16/bf16 peak,
    # gpu_tensor_peak_flops) while the elementwise kernels run on the CUDA cores
    # (fp32 peak, gpu_peak_flops); wave/tile quantization (WS1-1b, gpu_sm_count)
    # charges the ceil(blocks/SMs) underutilization the smooth roofline misses.
    # All are env-overridable for the sensitivity study -- note the GEMM compute
    # floor now moves with GPU_TENSOR_PEAK_FLOPS, not GPU_PEAK_FLOPS.
    gpu_flops        = os.environ.get("GPU_PEAK_FLOPS", "1.95e13")   # A100 fp32
    gpu_tensor_flops = os.environ.get("GPU_TENSOR_PEAK_FLOPS", "3.12e14")  # A100 bf16 TC
    gpu_bw           = os.environ.get("GPU_MEM_BW", "2000GB/s")
    gpu_sm_count     = os.environ.get("GPU_SM_COUNT", "108")         # A100, 0 = off
    # The matmul-class kernels charged at the tensor peak (mangled names).
    tensor_kernels = ",".join([
        "_Z8qkv_projPKfS0_Pfi", "_Z11attn_scoresPKfS0_Pfi",
        "_Z7attn_avPKfS0_Pfi",  "_Z8attn_outPKfS0_Pfi",
        "_Z6mlp_upPKfS0_Pfi",   "_Z8mlp_downPKfS0_Pfi",
    ])

    params = {
        "app1.name": "mercury_llm_train",
        "app1.exe_library_name": "mercury_llm_train",
        "app1.dependencies": ["sumi", ],
        "app1.libraries": ["gpulibrary:GpuLibrary",
                            "computelibrary:ComputeLibrary",
                            "mask_mpi:MpiApi", ],
        "app1.gpu_peak_flops":        gpu_flops,
        "app1.gpu_tensor_peak_flops": gpu_tensor_flops,
        "app1.gpu_tensor_kernels":    tensor_kernels,
        "app1.gpu_sm_count":          gpu_sm_count,
        "app1.gpu_max_threads_per_sm": "2048",
        "app1.gpu_mem_bandwidth": gpu_bw,
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
