# Installing SST-HGCC

For full documentation (Mercury app examples, pragma reference, configuration
options, and code layout), see [README.md](README.md).

## Prerequisites

Install these **before** sst-hgcc:

1. **SST Core** — provides `sst-config` on your `PATH`
2. **sst-elements** — Mercury/HG element (built with C++17)
3. **LLVM 22** — with libTooling (required for the `ssthg_clang` rewriter)

## Build and install

```bash
curl -LO https://github.com/llvm/llvm-project/releases/download/llvmorg-22/llvm-project-22.src.tar.xz
tar -xf llvm-project-22.src.tar.xz
cd llvm-project-22.src
mkdir build && cd build

ccmake -S ../llvm \
-DLLVM_ENABLE_PROJECTS="clang;compiler-rt;lld" \
-DLLVM_ENABLE_RUNTIMES=all \
-DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_INSTALL_PREFIX=$HOME/llvm-project-22.src/install \
-DLLVM_INCLUDE_TESTS=OFF \
-DLLVM_ENABLE_ZSTD=OFF \
-DLLVM_TARGETS_TO_BUILD=host \
-G Ninja

ninja -j 6 && ninja install

export PATH=$HOME/llvm-project-22.src/install/bin:$PATH
# Required on Mac only!
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)

# use llvm lld; sometimes regular ld fails
export LDFLAGS="-fuse-ld=lld"

cd

# Install SST Core (required — sst-config must be on PATH)
git clone https://github.com/sstsimulator/sst-core.git
cd sst-core
./autogen.sh
mkdir build && cd build
../configure CXX=clang++ CC=clang \
  --with-std=17 \
  --prefix=$HOME/sst-core/install
make V=1 && make install
export PATH=$HOME/sst-core/install/bin:$PATH

cd

# Install sst-elements
git clone https://github.com/sstsimulator/sst-elements.git
cd sst-elements
./autogen.sh
mkdir build && cd build

../configure CXX=clang++ CC=clang \
--with-std=17 \
--prefix=$HOME/sst-elements/install

make V=1 && make install

cd

# Install sst-hgcc
git clone https://github.com/sstsimulator/sst-hgcc.git
cd sst-hgcc
./autogen.sh
mkdir build && cd build

../configure CXX=clang++ CC=clang \
--with-std=17 \
--prefix=$HOME/sst-hgcc/install \
--with-sst-core=$HOME/sst-core/install \
--with-sst-elements=$HOME/sst-elements/install \
--with-clang=$HOME/llvm-project-22.src/install

make V=1 && make install

export PATH=$HOME/sst-hgcc/install/bin:$PATH

# Verify the install
hg++ --version
make check          # lit rewriter tests (optional)
make installcheck   # SST integration test (requires sst on PATH)
```

The above command should generate (roughly) the following output:
```bash
./build.sh 
+ cat test_tls.cc
#define ssthg_app_name test_tls
#include <mercury/common/skeleton.h>

#include <mask_mpi.h>
#include <iostream>

int my_global=0;

int main(int argc, char* argv[]) {

  MPI_Init(&argc,&argv);
  
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

  ++my_global;

  MPI_Barrier(MPI_COMM_WORLD);

  std::cerr << "my_global: " << my_global << std::endl;

  MPI_Finalize();

  return 0;
}

+ hg++ -c test_tls.cc
+ hg++ test_tls.o -o test_tls.so
+ cat platform_file_hg_test.py 
import sst
from sst.merlin.base import *

platdef = PlatformDefinition("platform_hg_test")
PlatformDefinition.registerPlatformDefinition(platdef)

platdef.addParamSet("node",{
    "verbose"                  : "0",
    "name"                     : "hg.NodeCL",
    "negligible_compute_bytes" : "64B",
    "parallelism"              : "1.0",
    "frequency"                : "2.1GHz",
    "flow_mtu"                 : "512",
    "channel_bandwidth"        : "11.2 GB/s",
    "num_channels"             : "4",
})

platdef.addParamSet("nic",{
    "verbose" : "0",
    "mtu" : "4096",
})

platdef.addParamSet("operating_system",{
    "verbose" : "0",
    "name"    : "hg.OperatingSystemCL"
})

platdef.addParamSet("topology",{
    "link_latency" : "20ns",
    "num_ports" : "32"
})

platdef.addClassType("network_interface","sst.merlin.interface.ReorderLinkControl")

platdef.addParamSet("network_interface",{
    "link_bw" : "12 GB/s",
    "input_buf_size" : "64 kB",
    "output_buf_size" : "64 kB"
})

platdef.addParamSet("router",{
    "link_bw" : "12 GB/s",
    "flit_size" : "8B",
    "xbar_bw" : "50 GB/s",
    "input_latency" : "20ns",
    "output_latency" : "20ns",
    "input_buf_size" : "64 kB",
    "output_buf_size" : "64 kB",
    "num_vns" : 1,
    "xbar_arb" : "merlin.xbar_arb_lru",
})

platdef.addParamSet("operating_system", {
    "ncores" : "24",
    "nsockets" : "4",
    "app1.post_rdma_delay" : "88us",
    "app1.post_header_delay" : "0.36us",
    "app1.poll_delay" : "0us",
    "app1.rdma_pin_latency" : "5.43us",
    "app1.rdma_page_delay" : "50.50ns",
    "app1.rdma_page_size" : "4096",
    "app1.max_vshort_msg_size" : "512 B",
    "app1.max_eager_msg_size" : "65536 B",
    "app1.use_put_window" : "true",
    "app1.compute_library_access_width" : "512",
    "app1.compute_library_loop_overhead" : "1.0",
})

platdef.addClassType("router","sst.merlin.base.hr_router")

+ cat test_tls.py 
#!/usr/bin/env python
#
# Copyright 2009-2024 NTESS. Under the terms
# of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# Copyright (c) 2009-2024, NTESS
# All rights reserved.
#
# This file is part of the SST software package. For license
# information, see the LICENSE file in the top level directory of the
# distribution.

import sst
from sst.merlin.base import *
from sst.merlin.endpoint import *
from sst.merlin.interface import *
from sst.merlin.topology import *
from sst.hg import *

if __name__ == "__main__":

    PlatformDefinition.loadPlatformFile("platform_file_hg_test")
    PlatformDefinition.setCurrentPlatform("platform_hg_test")
    platform = PlatformDefinition.getCurrentPlatform()

    platform.addParamSet("operating_system", {
        "app1.name" : "test_tls",
        "app1.exe_library_name" : "test_tls",
        "app1.dependencies" : ["sumi", ],
        "app1.libraries" : ["computelibrary:ComputeLibrary",
                            "mask_mpi:MpiApi",],
    })

    topo = topoSingle()
    topo.num_ports = 32

    ep = HgJob(0,2)

    system = System()
    system.setTopology(topo)
    system.allocateNodes(ep,"linear")

    system.build()

+ sst test_tls.py 
my_global: 1
my_global: 1
Simulation is complete, simulated time: 1.76579 us
