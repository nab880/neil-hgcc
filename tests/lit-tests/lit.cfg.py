import os
import shutil

import lit.formats

config.name = "sst-hgcc"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".cc", ".c", ".cu"]
config.test_source_root = os.path.dirname(os.path.realpath(__file__))
config.test_exec_root = os.path.join(config.test_source_root, "Output")

_here = os.path.dirname(os.path.realpath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
_install_bin = os.path.join(_repo_root, "install", "bin")
_llvm_bin = os.environ.get("LLVM_ROOT", "/opt/homebrew/opt/llvm@18") + "/bin"
_extra_path = os.pathsep.join(p for p in (_install_bin, _llvm_bin) if os.path.isdir(p))
if _extra_path:
    os.environ["PATH"] = _extra_path + os.pathsep + os.environ.get("PATH", "")


def _resolve(env_name, tool_name):
    path = os.environ.get(env_name)
    if path and os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    return shutil.which(tool_name)


hgcc = _resolve("HGCC", "hgcc")
filecheck = _resolve("FILECHECK", "FileCheck")

# hg++: $HGXX, else the hg++ next to the resolved hgcc, else PATH. The
# sibling must win over PATH or an older installed hg++ shadows the build
# tree under test (install/bin sits on PATH above).
hgxx = os.environ.get("HGXX")
if hgxx and not (os.path.isfile(hgxx) and os.access(hgxx, os.X_OK)):
    hgxx = None
if not hgxx and hgcc:
    _hgxx_cand = os.path.join(os.path.dirname(os.path.realpath(hgcc)), "hg++")
    if os.path.isfile(_hgxx_cand) and os.access(_hgxx_cand, os.X_OK):
        hgxx = _hgxx_cand
if not hgxx:
    hgxx = shutil.which("hg++")

missing = []
if not hgcc:
    missing.append("hgcc (install sst-hgcc or set $HGCC)")
if not hgxx:
    missing.append("hg++ (sits next to hgcc in the build dir, or set $HGXX)")
if not filecheck:
    missing.append("FileCheck (install LLVM tools or set $FILECHECK)")

if missing:
    config.unsupported = True
    lit_config.warning(
        "sst-hgcc lit-tests disabled; missing: " + ", ".join(missing)
    )
else:
    config.substitutions.append(("%hgcc", hgcc))
    config.substitutions.append(("%hgxx", hgxx))
    config.substitutions.append(("%FileCheck", filecheck))

    for var in (
        "PATH",
        "HOME",
        "TMPDIR",
        "CC",
        "CXX",
        "LLVM_ROOT",
        "SST_HG_INSTALL_DIR",
        "SST_HG_CONFIG",
        "SDKROOT",
        "MACOSX_DEPLOYMENT_TARGET",
        "SSTHG_SYSTEM_INCLUDES",
        "CPATH",
        "LIBRARY_PATH",
        "PYTHONPATH",
        "DYLD_LIBRARY_PATH",
        "LD_LIBRARY_PATH",
        "SST_LIB_PATH",
        "SST_HG_PREFIX",
        "HGCC",
        "HGXX",
        "FILECHECK",
    ):
        if var in os.environ:
            config.environment[var] = os.environ[var]

    _hgcc_parent = os.path.dirname(os.path.realpath(hgcc))
    _sst_hgcc_src = os.path.abspath(os.path.join(_here, "..", ".."))
    _build_vars = os.path.join(_hgcc_parent, "hgccvars.py")
    _py_paths = [_sst_hgcc_src]
    if os.path.isfile(_build_vars):
        _py_paths.insert(0, _hgcc_parent)
    existing_pp = config.environment.get("PYTHONPATH", "")
    config.environment["PYTHONPATH"] = os.pathsep.join(
        [p for p in _py_paths if p] + ([existing_pp] if existing_pp else [])
    )

    if "SST_HG_CC" not in os.environ:
        _clang_c = os.path.join(_llvm_bin, "clang")
        if os.path.isfile(_clang_c) and os.access(_clang_c, os.X_OK):
            config.environment["SST_HG_CC"] = _clang_c
        else:
            _sys_clang = shutil.which("clang")
            if _sys_clang:
                config.environment["SST_HG_CC"] = _sys_clang

    config.environment["SST_HG_DELETE_TEMP_SOURCES"] = "0"
    config.environment["SST_HG_DELETE_TEMP_OBJECTS"] = "0"
    config.environment["SST_HG_VERBOSE"] = "0"
    config.environment["SST_HG_SKELETONIZE"] = "1"
