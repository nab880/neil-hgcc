#!/usr/bin/env python3
"""Unit tests for hgcompile AST argv helpers."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hgcompile as hc
import hglink as hl  


class _FakeArgs:
  def __init__(self):
    self.D = ["EXTRA=1"]
    self.I = ["/tmp/include"]
    self.std = None


class _FakeCtx:
  def __init__(self):
    self.defines = ["A", "B=2"]
    self.cppFlags = ["-I/sys/cpp"]
    self.compilerFlags = ["-fPIC", "-std=c++17", "-c", "-o", "out.o"]
    self.directIncludes = ["/skel.h"]


class TestHgcompileAst(unittest.TestCase):
  def test_filter_ast_host_flags(self):
    flags = ["-c", "-O2", "-stdlib=libc++", "-o", "x.o", "-pipe", "foo.cc"]
    got = hc._filter_ast_host_flags(flags)
    self.assertEqual(got, ["-O2", "foo.cc"])

  def test_trim_trailing_source_arg(self):
    self.assertEqual(
        hc._trim_trailing_source_arg(["-std=c++11", "/a/b.cc"]),
        ["-std=c++11"],
    )
    self.assertEqual(hc._trim_trailing_source_arg(["-v"]), ["-v"])

  def test_host_compile_argv_for_ast(self):
    argv = hc.host_compile_argv_for_ast(_FakeCtx(), _FakeArgs())
    self.assertIn("-DA", argv)
    self.assertIn("-DB=2", argv)
    self.assertIn("-DEXTRA=1", argv)
    self.assertIn("-I/tmp/include", argv)
    self.assertIn("-I/sys/cpp", argv)
    self.assertIn("-fPIC", argv)
    self.assertIn("-std=c++17", argv)
    self.assertIn("-include", argv)
    self.assertIn("/skel.h", argv)
    self.assertNotIn("-c", argv)
    self.assertNotIn("-o", argv)

  def test_strip_include_directives(self):
    argv = ["-DA=1", "-include", "/skel.h", "-I/tmp", "-include-pch", "x.pch"]
    got = hc._strip_include_directives(argv)
    self.assertEqual(got, ["-DA=1", "-I/tmp", "-include-pch", "x.pch"])

  def test_apply_ast_gnuxx_remap(self):
    self.assertEqual(
        hc.apply_ast_gnuxx_remap(["-std=c++11", "-DZ"], True),
        ["-std=gnu++17", "-DZ"],
    )
    self.assertEqual(
        hc.apply_ast_gnuxx_remap(["-std=c++11"], False),
        ["-std=c++11"],
    )

  def test_pp_compiler_flags_match_ast_std(self):
    ctx = _FakeCtx()
    ctx.typ = "c++"
    ctx.compilerFlags = ["-fPIC", "-std=c++11", "-c", "-o", "x.o"]
    ast = ["-std=c++14", "-DZ"]
    got = hc._pp_compiler_flags_match_ast_std(ctx, _FakeArgs(), ast, "-std=c++17")
    self.assertEqual(got[0], "-std=c++14")
    self.assertIn("-fPIC", got)
    self.assertNotIn("-std=c++11", got)

  def test_apply_ast_libstdcxx_cpp14_floor(self):
    self.assertEqual(
        hc.apply_ast_libstdcxx_cpp14_floor(["-std=c++11", "-DZ"], True),
        ["-std=c++14", "-DZ"],
    )
    self.assertEqual(
        hc.apply_ast_libstdcxx_cpp14_floor(["-std=gnu++0x"], True),
        ["-std=gnu++14"],
    )
    self.assertEqual(
        hc.apply_ast_libstdcxx_cpp14_floor(["-std=c++11"], False),
        ["-std=c++11"],
    )
    self.assertEqual(
        hc.apply_ast_libstdcxx_cpp14_floor(["-std=c++17"], True),
        ["-std=c++17"],
    )

  def test_compile_commands_argv(self):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cc", delete=False) as tu:
      tu.write("//x\n")
      src = tu.name
    try:
      db_path = src + ".json"
      entry = {
        "file": src,
        "arguments": [
            "g++",
            "-std=c++14",
            "-stdlib=libstdc++",
            "-c",
            src,
        ],
      }
      with open(db_path, "w", encoding="utf-8") as fh:
        json.dump([entry], fh)
      os.environ["HGCC_COMPILE_COMMANDS"] = db_path
      argv = hc.compile_commands_argv_for_source(src)
      self.assertIsNotNone(argv)
      self.assertIn("-std=c++14", argv)
      self.assertNotIn("-c", argv)
      self.assertFalse(any(a.endswith(".cc") and not a.startswith("-") for a in argv))
    finally:
      os.environ.pop("HGCC_COMPILE_COMMANDS", None)
      try:
        os.unlink(src)
      except OSError:
        pass
      try:
        os.unlink(db_path)
      except OSError:
        pass

  def test_conftest_strip_sst_pmi_link(self):
    flags = ["-fPIC", "-L/prefix/sst-elements-library/ext", "-lpmi", "-O0"]
    L = ["/prefix/sst-elements-library/ext", "/usr/lib"]
    l = ["pmi", "m"]
    out_f, out_L, out_l = hl._conftest_strip_sst_pmi_link(flags, L, l)
    self.assertEqual(out_f, ["-fPIC", "-O0"])
    self.assertEqual(out_L, ["/usr/lib"])
    self.assertEqual(out_l, ["m"])


class _FakeCudaArgs(_FakeArgs):
  def __init__(self):
    _FakeArgs.__init__(self)
    self.O = None
    self.g = False
    self.cuda_gpu_arch = "sm_80"


class _FakeHgccVars:
  """Stands in for the hgccvars module in _build_src2src_plan."""
  useLibcxxForAst = True
  astGnuxxRemap = False
  clangLibtoolingCxxResourceStr = ""


_CUDA_FLAGS_SM80 = ["-x", "cuda", "--cuda-host-only", "-nocudainc",
                    "-nocudalib", "--cuda-gpu-arch=sm_80",
                    "-D__host__=__attribute__((host))",
                    "-D__device__=__attribute__((device))"]


def _cuda_test_ctx():
  ctx = _FakeCtx()
  ctx.typ = "c++"
  ctx.clangArgs = []
  ctx.src2srcDebug = False
  return ctx


def _cuda_test_plan(src):
  ctx = _cuda_test_ctx()
  args = _FakeCudaArgs()
  plan = hc._build_src2src_plan(
      ctx, src, args, _FakeHgccVars(), "-I/res", "-I/cres", "-std=c++17")
  plan.use_clangxx_host = True
  plan.clang_cxx_bin = "/llvm/bin/clang++"
  return ctx, args, plan


class TestHgcompileCuda(unittest.TestCase):
  # _build_src2src_plan resolves clang++ through `import hgccvars`, which only
  # exists in a configured build tree; stand in an empty module so the
  # getattr-with-default paths run instead of ModuleNotFoundError.
  @classmethod
  def setUpClass(cls):
    import types
    cls._saved_hgccvars = sys.modules.get("hgccvars")
    sys.modules["hgccvars"] = types.ModuleType("hgccvars")

  @classmethod
  def tearDownClass(cls):
    if cls._saved_hgccvars is not None:
      sys.modules["hgccvars"] = cls._saved_hgccvars
    else:
      sys.modules.pop("hgccvars", None)

  def test_cuda_lang_flags(self):
    self.assertEqual(hc._cuda_lang_flags(_FakeCudaArgs()), _CUDA_FLAGS_SM80)
    # import-safe default when args has no cuda_gpu_arch (stale callers)
    self.assertIn("--cuda-gpu-arch=sm_70", hc._cuda_lang_flags(_FakeArgs()))

  def test_is_cuda_source(self):
    self.assertTrue(hc._is_cuda_source("vecadd.cu"))
    self.assertTrue(hc._is_cuda_source("/b/sst.pp.vecadd.cu"))
    self.assertFalse(hc._is_cuda_source("a.cc"))
    self.assertFalse(hc._is_cuda_source("a.cu.cc"))
    self.assertFalse(hc._is_cuda_source(None))

  def test_src2src_plan_cuda_flags(self):
    _, _, plan = _cuda_test_plan("vec.cu")
    self.assertEqual(plan.cuda_flags, _CUDA_FLAGS_SM80)
    _, _, plan_cc = _cuda_test_plan("vec.cc")
    self.assertEqual(plan_cc.cuda_flags, [])

  def test_preprocess_argv_has_cuda_flags_for_cu_only(self):
    for src, want in (("vec.cu", True), ("vec.cc", False)):
      ctx, args = _cuda_test_ctx(), _FakeCudaArgs()
      cmds = []
      hc.addPreprocess(
          ctx, src, "pp." + src, args, cmds,
          compiler_flags=["-std=c++17"],
          use_clang_cpp_for_e=True,
          clang_exe_for_preprocess="/llvm/bin/clang++",
          stdlib_flag="libc++")
      argv = cmds[0][1]
      self.assertIn("-E", argv)
      if want:
        # contiguous, right after the driver + -stdlib pair
        self.assertEqual(argv[2:2 + len(_CUDA_FLAGS_SM80)], _CUDA_FLAGS_SM80)
      else:
        self.assertNotIn("-nocudainc", argv)
        self.assertNotIn("-x", argv)

  def test_preprocess_cuda_hard_errors_without_clangxx(self):
    ctx, args = _cuda_test_ctx(), _FakeCudaArgs()
    ctx.compiler = "g++"
    with self.assertRaises(SystemExit):
      hc.addPreprocess(
          ctx, "vec.cu", "pp.vec.cu", args, [],
          compiler_flags=["-std=c++17"],
          use_clang_cpp_for_e=True,
          clang_exe_for_preprocess=None,
          stdlib_flag="libc++")

  def test_preprocess_cuda_rejects_c_driver(self):
    ctx, args = _cuda_test_ctx(), _FakeCudaArgs()
    ctx.typ = "c"
    ctx.compiler = "gcc"
    with self.assertRaises(SystemExit):
      hc.addPreprocess(ctx, "vec.cu", "pp.vec.cu", args, [],
                       use_clang_cpp_for_e=False)

  def test_host_obj_cmd_cuda_host_only_for_cu(self):
    # The rewritten sst.pp.*.cu compiles in CUDA host-only mode (not -x c++):
    # CUDA-mode preprocessing baked in cuda_wrappers shims that only recompile
    # in CUDA mode. The rewriter lowers every <<< and strips device bodies.
    ctx, args, plan = _cuda_test_plan("vec.cu")
    cmd = hc._build_host_obj_cmd(ctx, args, plan, "sst.pp.vec.cu", "tmp.o")
    self.assertIn("--cuda-host-only", cmd)
    self.assertEqual(cmd[cmd.index("-x") + 1], "cuda")
    self.assertNotIn("c++", cmd)
    # -x cuda must precede the source file so it applies to it
    self.assertLess(cmd.index("-x"), cmd.index("sst.pp.vec.cu"))
    ctx, args, plan = _cuda_test_plan("vec.cc")
    cmd = hc._build_host_obj_cmd(ctx, args, plan, "sst.pp.vec.cc", "tmp.o")
    self.assertNotIn("-x", cmd)

  def test_ssthg_clang_cmd_cuda_flags_after_separator(self):
    ctx, _, plan = _cuda_test_plan("vec.cu")
    cmd = hc._build_ssthg_clang_cmd(ctx, plan, "/prefix", "pp.vec.cu",
                                    True, clangMajorVersion=22)
    sep = cmd.index("--")
    self.assertEqual(cmd[sep + 1:sep + 1 + len(_CUDA_FLAGS_SM80)],
                     _CUDA_FLAGS_SM80)
    ctx, _, plan = _cuda_test_plan("vec.cc")
    cmd = hc._build_ssthg_clang_cmd(ctx, plan, "/prefix", "pp.vec.cc",
                                    True, clangMajorVersion=22)
    self.assertNotIn("-nocudainc", cmd)

  def test_emit_llvm_gets_x_cxx_but_llvm_compile_does_not(self):
    ctx, args, plan = _cuda_test_plan("vec.cu")
    cmds = []
    hc.addEmitLlvm(ctx, "sst.pp.vec.cu", "out.ll", args, cmds, plan=plan)
    argv = cmds[0][1]
    i = argv.index("-x")
    self.assertEqual(argv[i:i + 3], ["-x", "c++", "sst.pp.vec.cu"])
    cmds = []
    hc.addLlvmCompile(ctx, "out.ll", "tmp.o", args, cmds, plan=plan)
    self.assertNotIn("-x", cmds[0][1])

  def test_filter_ast_host_flags_keeps_cuda_flags(self):
    flags = hc._cuda_lang_flags(_FakeCudaArgs())
    self.assertEqual(hc._filter_ast_host_flags(list(flags)), flags)


if __name__ == "__main__":
  unittest.main()
