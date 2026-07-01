# Phase-0 findings: toolkit-free CUDA host-only parsing

Recorded 2026-06-12, macOS arm64 (Darwin 25.4), Homebrew LLVM.
Gating toolchain: **llvm@22** (22.1.4 — what hgcc is configured against,
`hgccvars.clangDir=/opt/homebrew/opt/llvm@22`). Run `./run.sh` to reproduce;
`LLVMS="llvm@18 llvm@21 llvm@22" ./run.sh` for the wider sweep.

## Matrix (llvm@22, libc++)

| probe | what | result |
|-------|------|--------|
| p1 | host-only `-fsyntax-only` parse vs stubs | pass |
| p2 | parse without `--cuda-gpu-arch` | pass — flag optional |
| p3a/b | `-E` then reparse of `.cu`-suffixed pp file | pass |
| p4 | `CUDAGlobalAttr` + `CUDAKernelCallExpr` in host-only AST | pass |
| p5 | hand-written rewriter output builds/links/runs vs stub ABI | pass |
| p6 | Apple clang 21 parse (informational) | pass |

Informational: probe 1 also passes on llvm@18 and llvm@21 (not gated;
project targets LLVM 22 only).

## Answers the implementation plan needed

1. **`-nocudainc` is self-contained on LLVM 22** given our stubs. The only
   compiler-provided piece used is `__clang_cuda_builtin_vars.h`, which
   resolves from the **LLVM resource dir**
   (`.../lib/clang/22/include/`) — ships with LLVM, not the CUDA toolkit.
2. **The qualifier macros are ours to own.** `__global__` et al. are *not*
   predefined in clang's CUDA mode (`unknown type name '__global__'`
   without our headers); the P2 replacement `cuda_runtime.h` must define
   them, as the spike stub does.
3. **Sema resolves the legacy launch hook.** With no CUDA SDK detected,
   `<<<>>>` requires a declaration of `cudaConfigureCall`
   (deleting it: `use of undeclared identifier cudaConfigureCall`;
   deleting `__cudaPushCallConfiguration` instead: no effect). P2
   replacement headers must declare `cudaConfigureCall`; keep the
   push/pop-config declarations too in case a future LLVM flips the
   default launch ABI.
4. **`--cuda-gpu-arch` is optional for host-only parse** (p2). The P1
   driver may pass it anyway for determinism of `__CUDA_ARCH__`-dependent
   user code, but absence is not an error.
5. **Preprocess-then-reparse works** (p3) — the `addPreprocess` →
   `ssthg_clang` pipeline shape holds for CUDA, provided the pp file keeps
   the `.cu` suffix (it does: `addPrefixAndRebase` preserves suffixes) and
   `-x cuda` stays on the reparse. No `-fpreprocessed` issue observed.
6. **Host-only codegen also works on stubs alone** (`-S` clean): no fatbin
   registration machinery is demanded when no device code is embedded.
   Note for P1's negative test (pipeline fails at host `-c` on the
   surviving `<<<`): that step compiles `-x c++`, where `<<<` is indeed a
   parse error, so the planned assertion holds — but the test must put
   `spike/stubs` on `-I`, because the replacement headers that satisfy
   `#include <cuda_runtime.h>` only land in P2 and preprocessing would
   otherwise fail first.
7. **ABI vets clean** (p5): the hand-written rewriter output compiles as
   plain C++ against `hg_cuda.h` and the run prints the expected
   malloc → memcpy×2 → launch(mangled name, grid 256×256) → sync →
   memcpy → free×3 sequence.

## Open items

- **Linux + libstdc++ legs not run** (no local Linux). Add a CI leg before
  P1 merges; expect the glibc `-fgnuc-version` wrinkles the C path already
  handles in `hgcompile.py` to be the risk area, not CUDA parsing.
- Finding 3 (legacy launch hook) should be re-probed when the LLVM floor
  moves past 22.

## Verdict

**GREEN — proceed to Phase 1.** Design decision D2 of CUDA_PLAN.md
(toolkit-free host-only parse with shipped stub headers) is confirmed on
the project's target toolchain.
