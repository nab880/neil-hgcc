#!/usr/bin/env python
#
# Source-agnostic SST driver for a user MPI collective (Tier 1 of the
# user-collectives tool). Generalization of tests/mv2/run_test_mv2_allreduce.py:
# the app name, rank count, topology, and link parameters all come from the
# environment, so one script sweeps any single-file MPI program across fabrics.
#
# Environment knobs:
#   APP       app / library name (default my_allreduce) -> loads lib<APP>.so
#   NRANKS    number of MPI ranks (default 8)
#   TOPO      single (default) | fattree            (see ../../scale_topo.py)
#   LINK_BW   per-link bandwidth, e.g. "12 GB/s"     (default from platform)
#   LINK_LAT  per-hop latency, e.g. "20ns"           (default 20ns / 100ns)
#   ALG       sumi engine algorithm name for allreduce (e.g. ring, recdouble);
#             empty => the built-in default. Only affects the *engine* path,
#             not a hand-rolled point-to-point algorithm (that is T1's point).
#
# Usage:  APP=my_allreduce NRANKS=8 TOPO=fattree sst run.py

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MV2 = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "tests", "mv2"))
_EXAMPLES = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _MV2, _EXAMPLES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_APP_NAME = os.environ.get("APP", "my_allreduce")
_NRANKS = int(os.environ.get("NRANKS", "8"))
_LINK_BW = os.environ.get("LINK_BW", "")
_LINK_LAT = os.environ.get("LINK_LAT", "")
_ALG = os.environ.get("ALG", "")


def _preload_hg_and_user_app():
    """Optionally dlopen Mercury + deps + lib<APP>.so before import sst.

    Disabled by default; set SST_HG_MV2_PRELOAD=1 to enable the Linux-only
    workaround that forces userSkeletonMainInitFxn registration before SST
    requireLibrary() runs. Same rationale as run_test_mv2_sendrecv.py.
    """
    if sys.platform == "darwin":
        return
    flag = (os.environ.get("SST_HG_MV2_PRELOAD") or "").strip().lower()
    if flag not in ("1", "yes", "true", "on"):
        return
    try:
        import ctypes
    except ImportError:
        return
    pfx = (os.environ.get("SST_HG_PREFIX") or "").strip()
    if not pfx:
        cand = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", "install"))
        libhg = os.path.join(cand, "lib", "sst-elements-library", "libhg.so")
        libhg_dy = os.path.join(cand, "lib", "sst-elements-library", "libhg.dylib")
        if os.path.isfile(libhg) or os.path.isfile(libhg_dy):
            pfx = cand
    if not pfx:
        return
    lib_dir = os.path.join(pfx, "lib", "sst-elements-library")
    ext_dir = os.path.join(lib_dir, "ext")
    sumi_so = os.path.join(ext_dir, "libsumi.so")
    app_so = os.path.join(_HERE, "lib%s.so" % _APP_NAME)
    hg = None
    for name in ("libhg.so", "libhg.dylib"):
        p = os.path.join(lib_dir, name)
        if os.path.isfile(p):
            hg = p
            break
    if not hg:
        return
    # Out-of-tree collective plugins (Tier 3): each is dlopen'd RTLD_GLOBAL
    # *after* libsumi so its SUMI_REGISTER_COLLECTIVE static initializers land
    # in the engine's registry. Comma-separated absolute paths.
    plugins = [p for p in (os.environ.get("COLLECTIVE_PLUGINS") or "").split(",") if p.strip()]
    try:
        ctypes.CDLL(hg, mode=ctypes.RTLD_GLOBAL)
        if os.path.isfile(sumi_so):
            ctypes.CDLL(sumi_so, mode=ctypes.RTLD_GLOBAL)
        for plugin in plugins:
            ctypes.CDLL(plugin.strip(), mode=ctypes.RTLD_GLOBAL)
        if os.path.isfile(app_so):
            ctypes.CDLL(app_so, mode=ctypes.RTLD_GLOBAL)
    except OSError as err:
        raise RuntimeError(
            "Could not dlopen Mercury, a collective plugin, or lib%s.so "
            "(check SST_HG_PREFIX=%r, COLLECTIVE_PLUGINS, and LD_LIBRARY_PATH): %s"
            % (_APP_NAME, pfx, err)
        ) from err


_preload_hg_and_user_app()

import sst
from sst.merlin.base import *
from sst.merlin.endpoint import *
from sst.merlin.interface import *
from sst.merlin.topology import *
from sst.hg import *

from scale_topo import make_topology

if __name__ == "__main__":
    PlatformDefinition.loadPlatformFile("platform_file_mv2_test")
    PlatformDefinition.setCurrentPlatform("platform_mv2_test")
    platform = PlatformDefinition.getCurrentPlatform()

    _os_params = {
        "app1.name": _APP_NAME,
        "app1.exe_library_name": _APP_NAME,
        "app1.dependencies": ["sumi", "fabric"],
        "app1.libraries": [
            "computelibrary:ComputeLibrary",
            "macro:libfabric",
        ],
    }
    # Engine algorithm selection (Python param path; ALG env feeds it here).
    # The sumi library reads app1.* params, so app1.allreduce_alg is the knob.
    if _ALG:
        _os_params["app1.allreduce_alg"] = _ALG
    platform.addParamSet("operating_system", _os_params)

    # Optional link-bandwidth override applied to both endpoints and routers.
    if _LINK_BW:
        platform.addParamSet("network_interface", {"link_bw": _LINK_BW})
        platform.addParamSet("router", {"link_bw": _LINK_BW})

    topo = make_topology(_NRANKS)
    if _LINK_LAT:
        topo.link_latency = _LINK_LAT

    ep = HgJob(0, _NRANKS)

    system = System()
    system.setTopology(topo)
    system.allocateNodes(ep, "linear")

    system.build()
