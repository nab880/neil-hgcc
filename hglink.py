"""
Copyright 2009-2023 National Technology and Engineering Solutions of Sandia,
LLC (NTESS).  Under the terms of Contract DE-NA-0003525, the U.S. Government
retains certain rights in this software.

Sandia National Laboratories is a multimission laboratory managed and operated
by National Technology and Engineering Solutions of Sandia, LLC., a wholly
owned subsidiary of Honeywell International, Inc., for the U.S. Department of
Energy's National Nuclear Security Administration under contract DE-NA0003525.

Copyright (c) 2009-2023, NTESS

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Questions? Contact sst-macro-help@sandia.gov
"""

import os
import platform
import sys


def _add_mv2_link(ctx, ldTarget, args, cmds, objects):
  """Append MV2-specific compile + link commands for a --mpi=mvapich2 .so.

  Mirrors the manual steps in tests/mv2/run_test_mv2_sendrecv.sh so a single
  hgcc invocation replaces the hand-rolled shell pipeline.

  Step order (appended to `cmds`, which is executed sequentially later):
    1. gen_register_shim.py: nm scans user .o for the hgcc-mangled
       sst_hg_user_main symbol, writes a tiny C shim, compiles to shim.o.
    2. Compile mv2_ld_glibc_stub.c (Linux) and mv2_mpi_wrappers.c from the
       installed support dir.
    3. Link the final .so with libmpi_nopmi.a whole-archive (+ -force_load on
       macOS), Iris libpmi / libfabric, libhg (Linux), user objects, and the
       shim / stub objects.
  """
  from hgccvars import prefix
  try:
    from hgccvars import mv2SupportDir as _mv2SupportDir
  except ImportError:
    _mv2SupportDir = os.path.join(prefix, "share", "hgcc", "mv2")
  try:
    from hgccvars import hgElementsLibDir as _hgElementsLibDir
  except ImportError:
    _hgElementsLibDir = os.path.join(prefix, "lib", "sst-elements-library")

  mpi_prefix = ctx.mpi_prefix
  app_name = ctx.app_name
  mpi_lib = os.path.join(mpi_prefix, "lib")
  mpi_inc = os.path.join(mpi_prefix, "include")
  libmpi_nopmi = os.path.join(mpi_lib, "libmpi_nopmi.a")
  if not os.path.isfile(libmpi_nopmi):
    sys.exit(
        "hgcc: %s not found; rebuild mvapich2 so the post-install "
        "simple_pmi strip runs (produces libmpi_nopmi.a)." % libmpi_nopmi)

  ext_lib = os.path.join(_hgElementsLibDir, "ext")
  if not os.path.isdir(ext_lib):
    sys.exit(
        "hgcc: Iris ext lib dir not found: %s (build sst-elements first)."
        % ext_lib)

  support_wrappers = os.path.join(_mv2SupportDir, "mv2_mpi_wrappers.c")
  support_glibc_stub = os.path.join(_mv2SupportDir, "mv2_ld_glibc_stub.c")
  gen_shim_py = os.path.join(_mv2SupportDir, "gen_register_shim.py")
  for f in (support_wrappers, support_glibc_stub, gen_shim_py):
    if not os.path.isfile(f):
      sys.exit("hgcc: MV2 support file missing: %s" % f)

  out_dir = os.path.dirname(os.path.abspath(ldTarget)) or os.getcwd()
  shim_o = os.path.join(out_dir, "register_app_gen.%s.o" % app_name)
  sst_elements_inc = os.path.join(prefix, "include", "sst", "elements")

  user_objs = [o for o in objects if o.endswith(".o")]
  shim_cmd = [
      sys.executable, gen_shim_py, app_name, ctx.cxx, sst_elements_inc, shim_o,
      "--",
  ] + user_objs
  cmds.append([None, shim_cmd, [shim_o]])

  glibc_stub_o = os.path.join(out_dir, "mv2_ld_glibc_stub.o")
  if platform.system() != "Darwin":
    stub_cmd = [ctx.cc, "-fPIC", "-c", support_glibc_stub, "-o", glibc_stub_o]
    cmds.append([None, stub_cmd, [glibc_stub_o]])

  wrappers_o = os.path.join(out_dir, "mv2_mpi_wrappers.o")
  # PROTOTYPE: HGCC_MV2_GEN_WRAPPERS=1 auto-generates the MPI_* wrapper set from
  # the MVAPICH2 archive instead of hand-maintaining it. gen_symbol_wrappers.py
  # emits a strong tail-call trampoline for every weak-aliased public MPI symbol
  # so the set can never be incomplete. The host stubs in mv2_mpi_wrappers.c
  # (hgcc_gethostname etc.) are real stubs, not aliases, so they cannot be
  # generated -- we still compile that file, with -DHGCC_MV2_GENERATED_MPI_WRAPPERS
  # so only the stubs (not the duplicate MPI_* wrappers) are emitted. The
  # generated trampolines are linked as an extra object. Darwin only (the
  # dynamic_lookup-bundle hazard this addresses is macOS-specific).
  gen_wrappers = (os.environ.get("HGCC_MV2_GEN_WRAPPERS") not in (None, "", "0")
                  and platform.system() == "Darwin")
  gen_o = None
  if gen_wrappers:
    gen_py = os.path.join(os.path.dirname(_mv2SupportDir),
                          "gen_symbol_wrappers.py")
    gen_s = os.path.join(out_dir, "mv2_mpi_wrappers_gen.S")
    gen_o = os.path.join(out_dir, "mv2_mpi_wrappers_gen.o")
    cmds.append([None, [sys.executable, gen_py, "--archive", libmpi_nopmi,
                        "--include", r"^_MPI[X]?_", "--out", gen_s, "--quiet"],
                 [gen_s]])
    cmds.append([None, [ctx.cc, "-fPIC", "-c", gen_s, "-o", gen_o], [gen_o]])
    wrappers_cmd = [
        ctx.cc, "-fPIC", "-c", support_wrappers, "-DHGCC_MV2_GENERATED_MPI_WRAPPERS",
        "-I%s" % mpi_inc, "-o", wrappers_o,
    ]
  else:
    wrappers_cmd = [
        ctx.cc, "-fPIC", "-c", support_wrappers,
        "-I%s" % mpi_inc, "-o", wrappers_o,
    ]
  cmds.append([None, wrappers_cmd, [wrappers_o]])

  is_darwin = (platform.system() == "Darwin")
  link = [ctx.cc]
  if is_darwin:
    link += [
        "-bundle", "-undefined", "dynamic_lookup",
        "-Wl,-rpath,%s" % os.path.join(prefix, "lib"),
        "-Wl,-rpath,%s" % ext_lib,
        "-Wl,-force_load,%s" % libmpi_nopmi,
        "-L%s" % ext_lib, "-lpmi", "-lfabric",
        wrappers_o,
        shim_o,
    ]
    if gen_o:
      link += [gen_o]
    link += objects
    link += ["-o", ldTarget]
  else:
    link += [
        "-shared",
        "-Wl,-Bsymbolic-functions",
        "-Wl,-rpath,%s" % os.path.join(prefix, "lib"),
        "-Wl,-rpath,%s" % _hgElementsLibDir,
        "-Wl,-rpath,%s" % ext_lib,
        "-Wl,--whole-archive", libmpi_nopmi, "-Wl,--no-whole-archive",
        "-L%s" % ext_lib, "-lpmi", "-lfabric",
        "-L%s" % _hgElementsLibDir,
        "-Wl,--no-as-needed", "-lhg", "-Wl,--as-needed",
        glibc_stub_o,
        wrappers_o,
        shim_o,
    ]
    link += objects
    link += ["-o", ldTarget]
  cmds.append([None, link, []])

  # Post-link guard: fail loudly if any weak MPI_* symbol was left unwrapped
  # (undefined dynamic_lookup + local force_loaded def => binds to garbage at
  # load on macOS). Runs after the link; nonzero exit aborts the build.
  checker = os.path.join(_mv2SupportDir, "check_mpi_wrappers.py")
  if os.path.isfile(checker):
    cmds.append([None, [sys.executable, checker, ldTarget], []])


def _conftest_strip_sst_pmi_link(flags, lib_dirs, libs):
  """
  Nested autoconf (e.g. hwloc under MVAPICH) inherits LDFLAGS with -lpmi and
  -L.../sst-elements-library/ext; linking trivial conftests then loads libpmi.so
  and fails on unresolved SST/Mercury symbols. Drop those for conftest only.
  """
  ext_mark = "sst-elements-library" + os.sep + "ext"
  ext_alt = "sst-elements-library/ext"

  def bad_libpath(p):
    if not p:
      return False
    norm = p.replace("\\", "/")
    return ext_alt in norm or ext_mark in p

  out_flags = []
  for f in flags:
    if f == "-lpmi":
      continue
    if f.startswith("-L") and len(f) > 2 and bad_libpath(f[2:]):
      continue
    out_flags.append(f)

  out_L = [p for p in (lib_dirs or []) if not bad_libpath(p)]
  out_l = [x for x in (libs or []) if x != "pmi"]
  return out_flags, out_L, out_l


def addLink(ctx, ldTarget, args, cmds, objects, toExe=False):
  #xargs includes the list of object files
  from hgccvars import prefix
  from hgccvars import soFlagsStr

  is_conftest = getattr(ctx, 'is_conftest', False)

  # --mpi=mvapich2: skip the generic link builder entirely and emit the MV2
  # shim + support stubs + whole-archive link that used to live in the test
  # driver shell script.
  if (not is_conftest and not toExe
      and getattr(ctx, "mpi_impl", None) == "mvapich2"
      and (ldTarget.endswith(".so") or ldTarget.endswith(".dylib"))):
    _add_mv2_link(ctx, ldTarget, args, cmds, objects)
    return

  wlFlags = getattr(ctx, 'wlFlags', [])

  if is_conftest:
    # Conftest: minimal link; drop -flat_namespace (dyld/SST issues on macOS).
    # Note: _conftest_strip_sst_pmi_link only scrubs -lpmi / sst-elements-library/ext
    # from args.l / args.L / ctx.compilerFlags. It does NOT strip ctx.ldFlags because
    # the sstLdFlags population loop in hgcclib skips conftest, so ctx.ldFlags is
    # empty here. If that ever changes (e.g. sstLdFlags is added unconditionally),
    # extend _conftest_strip_sst_pmi_link to scrub wlFlags / ctx.ldFlags too.
    bad_flags = {"-Wl,-flat_namespace", "-flat_namespace"}
    linkCmdArr = [ctx.ld]
    for f in wlFlags:
      if f not in bad_flags:
        linkCmdArr.append(f)
    cflags, use_L, use_l = _conftest_strip_sst_pmi_link(
        ctx.compilerFlags, args.L, args.l)
    for f in cflags:
      if f not in bad_flags:
        linkCmdArr.append(f)
    linkCmdArr.extend(map(lambda x: "-L%s" % x, use_L))
    linkCmdArr.extend(map(lambda x: "-l%s" % x, use_l))
  else:
    ldpathMaker = "-Wl,-rpath,%s/lib" % prefix
    linkCmdArr = [ctx.ld, ldpathMaker] 
    linkCmdArr.extend(wlFlags)
    linkCmdArr.extend(ctx.ldFlags)
    if not toExe:
      linkCmdArr.extend(soFlagsStr.split())
    linkCmdArr.extend(ctx.libs)
    linkCmdArr.extend(ctx.compilerFlags)
    linkCmdArr.extend(map(lambda x: "-L%s" % x, args.L))
    linkCmdArr.extend(map(lambda x: "-l%s" % x, args.l))
  
  # Add conditional replacement libraries based on --replacements parameter
  if hasattr(ctx, 'requestedReplacements') and ctx.requestedReplacements:
    replacementsLibPath = prefix + "/lib"
    
    # pthread.h has no hgcc_* lib (Mercury provides pthread).
    replacementLibMap = {
      'omp.h': 'hgcc_omp',
      'blas.h': 'hgcc_blas',
      'cblas.h': 'hgcc_blas',
      'bgp.h': 'hgcc_machines'
    }
    
    for replacement in ctx.requestedReplacements:
      if replacement in replacementLibMap:
        libName = replacementLibMap[replacement]
        libPath = replacementsLibPath + "/lib%s.a" % libName
        try:
          with open(libPath, 'r'):
            pass
          libExists = True
        except:
          libExists = False
        if libExists:
          linkCmdArr.append(libPath)
        else:
          # Try as shared library
          libPath = replacementsLibPath + "/lib%s.so" % libName
          try:
            with open(libPath, 'r'):
              pass
            libExists = True
          except:
            libExists = False
          if libExists:
            linkCmdArr.append(libPath)
          else:
            print("Warning: Replacement library for '%s' not found" % replacement)
  
  linkCmdArr.extend(objects)
  linkCmdArr.append("-o")
  linkCmdArr.append(ldTarget)
  cmds.append([None,linkCmdArr,[]])
