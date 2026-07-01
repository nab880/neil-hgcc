# String-codegen hardening plan

The CUDA launch rewrite (`clang/astVisitor.cc::rewriteCudaLaunch`) and the
skeletonizer rewrites (`replaceStmt`, `nullifyIfStmt`, the pragma rewrites) all
generate replacement C++ by concatenating source text into a `std::stringstream`
and handing the result to `::replace(stmt, str)`. Nothing checks that string
until the *next* compiler parses it, so a malformed emission surfaces as a clang
diagnostic pointing at machine-generated `sst.pp.*` source the user never wrote.

The cost-model refactor (`clang/gpuCostModel.h`) already removed the worst
*drift* hazard (dimension names / ABI order duplicated across sites). This doc
covers two further hardenings that reduce the blast radius of the string-codegen
paradigm **without** replacing it (an AST-to-AST rewrite is a different, much
larger project and would make this function inconsistent with the rest of the
~90 KB `astVisitor.cc`):

1. A scoped emitter (`CodeBuilder`) that tracks brace/paren depth and asserts
   balance, so an unbalanced emission fails *in the plugin* with a clear message
   instead of downstream.
2. Negative lit tests that pin the invariants the emitter can't check by
   construction (param-name collisions, unresolved kernels), so a regression
   fails a test rather than shipping.

Neither changes generated output; both are pure guardrails.

---

## Part 1 — `CodeBuilder`: a brace-balance-checking emitter

### Goal

Turn silent structural bugs (missing `}`, stray `)`, unclosed statement-expr)
into a loud, located failure at emit time. Today the only signal is a downstream
clang error like:

```
sst.pp.foo.cu:82644:3: error: expected '}'
```

with no hint that the plugin produced it. After this change the plugin itself
aborts with:

```
foo.cu:12:3: error: internal codegen error in rewriteCudaLaunch:
  unbalanced '{' (depth 1) at end of emission
```

### Design

A thin wrapper over `std::string` that (a) offers the same `<<` ergonomics the
code already uses, and (b) exposes explicit scope open/close calls that maintain
a depth counter. `str()` (or the destructor in debug) asserts the counter is
zero. It does **not** try to parse C++ — it only balances the three bracket
kinds and tracks that scopes opened via the helper are closed via the helper.

Deliberately minimal: it is a balance checker, not a pretty-printer or an AST.

### Header: `clang/codeBuilder.h`

```cpp
#ifndef bin_clang_codeBuilder_h
#define bin_clang_codeBuilder_h

#include <sstream>
#include <string>
#include "util.h"          // errorAbort
#include "clangHeaders.h"  // clang::SourceLocation

// Accumulates generated source text while tracking bracket balance. This does
// NOT parse C++ -- it only guards the structural invariant that every scope the
// rewrite opens is closed, catching the most common string-codegen bug (a
// missing/extra brace) at emit time instead of in the downstream compiler.
//
// Usage mirrors the existing `os << "..."` style; use the scope helpers for
// anything that opens a block so the balance counter stays accurate:
//
//   CodeBuilder b(getStart(expr), "rewriteCudaLaunch");
//   b.openStmtExpr();                       // "({ "
//   b << "dim3 g = (" << gridStr << "); ";
//   b.openBlock();                          // "{ "
//   ...
//   b.closeBlock();                         // " }"
//   b.closeStmtExpr();                      // " })"
//   ::replace(expr, b.take());              // asserts balance == 0
class CodeBuilder {
 public:
  CodeBuilder(clang::SourceLocation loc, const char* where)
    : loc_(loc), where_(where) {}

  // Raw append; use for leaf tokens that don't change scope depth. Appended
  // text is NOT scanned for brackets -- callers must use the scope helpers for
  // block/paren structure. (Scanning arbitrary text would false-positive on
  // brackets inside string/char literals in embedded expressions.)
  template <class T>
  CodeBuilder& operator<<(const T& v) { os_ << v; return *this; }

  void openBlock()     { os_ << "{ ";  ++brace_; }
  void closeBlock()    { failIf(brace_ == 0, "closeBlock with no open '{'");
                         os_ << " }"; --brace_; }

  void openStmtExpr()  { os_ << "({ "; ++paren_; ++brace_; }
  void closeStmtExpr() { failIf(paren_ == 0 || brace_ == 0,
                                "closeStmtExpr with no open '({'");
                         os_ << " })"; --paren_; --brace_; }

  // A lambda body: "[&]{ " ... "}" ; the trailing "()" call is caller-emitted.
  void openLambda()    { os_ << "[&]{ "; ++brace_; }
  void closeLambda()   { failIf(brace_ == 0, "closeLambda with no open '{'");
                         os_ << " }"; --brace_; }

  // Finalize: assert everything opened was closed, then hand off the string.
  std::string take() {
    failIf(brace_ != 0, "unbalanced '{' (depth " + std::to_string(brace_) + ")");
    failIf(paren_ != 0, "unbalanced '(' (depth " + std::to_string(paren_) + ")");
    return os_.str();
  }

 private:
  void failIf(bool bad, const std::string& msg) {
    if (bad) errorAbort(loc_, "internal codegen error in " + std::string(where_)
                              + ": " + msg);
  }
  std::ostringstream os_;
  clang::SourceLocation loc_;
  const char* where_;
  int brace_ = 0;
  int paren_ = 0;
};

#endif
```

### What it catches / does not catch

Catches (the common, high-value cases):
- A scope opened with `openBlock`/`openStmtExpr`/`openLambda` and never closed.
- A close call with no matching open (double-close, wrong nesting order at the
  helper level).
- `take()` called while any scope is still open.

Does **not** catch (out of scope by design — would require real parsing):
- Brackets typed directly into `operator<<` text (`b << "if (x) {"`). The rule
  is: structural braces go through the helpers; only leaf tokens go through
  `<<`. `rewriteCudaLaunch` already has exactly this shape, so it fits cleanly.
- Brackets embedded inside spliced expressions (`printWithGlobalsReplaced(arg)`,
  `derivedAccum`). These are opaque strings; the helper intentionally does not
  scan them (that would false-positive on `[]`, `()` inside the expression and
  on brackets inside string literals).

The value is bounding the *plugin-authored* structure, which is where the bugs
we actually hit (the param-collision redeclare, the shadow-block nesting) lived.

### Migrating `rewriteCudaLaunch`

Mechanical, ~1 hour, no output change. The current `std::stringstream os` becomes
a `CodeBuilder b(getStart(expr), "rewriteCudaLaunch")`. Concretely:

| Current text | Becomes |
|---|---|
| `os << "({ dim3 __sst_g_" ...` | `b.openStmtExpr(); b << "dim3 __sst_g_" ...` |
| the derived cost lambda `... [&]{ ...` | `b.openLambda();` around the body |
| the nested shadow block `{ derivedAccum ... }` | `b.openBlock(); b << derivedAccum ...; b.closeBlock();` |
| final `... ); })` | `b << "...);"; b.closeStmtExpr();` |
| `::replace(expr, os.str())` | `::replace(expr, b.take())` |

The two `struct { ... }` and lambda-return braces are leaf-level text inside a
single statement and can stay as `<<` (they're matched within one emission and
don't nest user scopes) — or be wrapped too, for completeness. Keep the
migration conservative: wrap the *outer* statement-expr, the lambda, and the
shadow block (the three real nesting levels), leave inline struct literals as
text.

### Verification

- Build the plugin; run the full `tests/lit-tests/cuda/` and
  `tests/lit-tests/pragmas/` suites. Output is byte-identical, so every existing
  `CHECK` still matches — that is the regression guard for "no behavior change."
- The guard was verified by temporarily omitting `os.closeStmtExpr()` in
  `rewriteCudaLaunch`, rebuilding, and confirming the plugin aborts *before*
  emitting `sst.pp.*` with a located message:

  ```
  derived-cost-vecadd.cu:21:3: error: internal codegen error in
    rewriteCudaLaunch: unbalanced '{' (depth 1) at end of emission
  ```

  The imbalance was then reverted (no test-only code path ships). A permanent
  self-test would need a gtest harness the plugin doesn't currently have; the
  manual check is the pragmatic equivalent until then.

### Follow-on: skeletonizer compute emitter (done)

`ComputeVisitor::replaceStmt` / `addLoopContribution` / `emitCostAccumulation`
were migrated to `CodeBuilder` (they emit the same nested-brace loop-tree
structure as the CUDA cost path and share `emitCostAccumulation`). This also
brace-guards the CUDA `deriveKernelCost` accumulation for free. Output is
token-identical; all 26 pragma + 18 CUDA lit tests pass.

The remaining ~13 codegen `replace()` sites (flat single-brace fragments in
`pragmas.cc`, the var-template/global-var emits, and the `pp.os` PrettyPrinter
uses) were intentionally *not* migrated: they have no nested-scope structure, so
`CodeBuilder` adds churn without guard value. Adopt it there only if one is
touched for another reason.

---

## Part 2 — Negative lit tests for the invariants the emitter can't check

`CodeBuilder` guards *structure*. It cannot guard *semantics* of the generated
code (a name that collides, an unresolved kernel, a shadow that redeclares). Those
are pinned by tests. The repo already uses the `RUN: not %hgxx ... / CHECK:
error` pattern (see `tests/lit-tests/cuda/launch-dependent.cu`); extend it.

### Coverage today

Positive tests exist for the derived path, param binding, side-effect capture,
and one collision (`derived-param-collision.cu`, added with the collision fix).
Gaps worth a negative or edge test:

### T-a — param named exactly like a cost accumulator (regression pin)

`derived-param-collision.cu` covers `gridDim`/`flops`. Add the remaining
accumulator names so a future emitter change that stops shadowing them fails
loudly. New `tests/lit-tests/cuda/derived-collision-accum.cu`:

```cuda
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgxx -c %s -o %t.d/out.o > %t.d/log 2>&1
// RUN: ls %t.d/out.o
#include <cuda_runtime.h>
// Params named after EVERY cost accumulator (intops/readBytes/writeBytes) must
// still compile -- the derived accumulation shadows them in a nested block.
__global__ void k(float* out, int intops, int readBytes, int writeBytes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = intops + readBytes + writeBytes;
}
int main() {
  float* d; cudaMalloc((void**)&d, 64);
  k<<<1, 1>>>(d, 1, 2, 3);
  cudaFree(d); return 0;
}
// Compiles clean (out.o exists) -> the shadow-block nesting held.
```

The success criterion is simply that `out.o` is produced; if the shadow-block
invariant regresses, the generated code redeclares and host compile fails.

### T-b — every builtin name as a param (grid/block/thread/blockIdx)

One test per builtin is overkill; one test using all four as params exercises the
`paramNames.count(...)` suppression branch fully. New
`tests/lit-tests/cuda/derived-collision-builtins.cu`:

```cuda
// RUN: ... (same harness)
__global__ void k(float* out, int gridDim, int blockDim,
                  int threadIdx, int blockIdx) {
  out[0] = gridDim + blockDim + threadIdx + blockIdx;
}
// Each synthetic dim3 builtin is suppressed because a param shadows it;
// CHECK-NOT: dim3 gridDim =
// CHECK-NOT: dim3 blockDim =
// CHECK-NOT: dim3 threadIdx(
// CHECK-NOT: dim3 blockIdx(
```

Note: `threadIdx`/`blockIdx` as parameter names shadow CUDA builtins, which is
unusual but legal in host-visible signatures; if the front-end rejects them,
downgrade this test to `gridDim`/`blockDim` only and document why.

### T-c — unresolved / dependent kernel is a hard error (already partially covered)

`launch-dependent.cu` covers the dependent-template case. No change needed; noted
here so the invariant ("unresolved callee => errorAbort naming gpu_compute") is
recorded as intentionally tested.

### T-d — `CodeBuilder` self-test (pairs with Part 1)

If Part 1 lands, add the self-test described in its Verification section so the
guard itself is covered. Without Part 1 this row is N/A.

### Harness notes

- Positive compile tests assert `ls %t.d/out.o` (object produced).
- Negative tests use `RUN: not %hgxx ...` and `CHECK: error:` against the log,
  matching `launch-dependent.cu`.
- All new tests are picked up automatically by `tests/lit-tests/` (glob); no
  Makefile change. `EXTRA_DIST = clang` already ships new headers, so
  `codeBuilder.h` needs no Makefile edit either.

---

## Effort / sequencing

| Step | Effort | Risk | Payoff |
|---|---|---|---|
| Part 1 `CodeBuilder` + migrate `rewriteCudaLaunch` | ~1 h | low (output unchanged, pinned by existing CHECKs) | brace-class bugs fail in-plugin, located |
| Part 2 T-a, T-b | ~30 min | none (additive tests) | shadow/suppression invariants regression-pinned |
| Part 1 self-test (T-d) | ~15 min | none | proves the guard fires |
| Skeletonizer adoption of `CodeBuilder` | opportunistic | low | same win across the plugin |

Recommended order: Part 2 first (pure additive safety, no code change), then
Part 1 (guarded by the now-stronger test net). Do the skeletonizer adoption lazily.

## Explicitly out of scope

- Replacing string codegen with structured AST emission (different project;
  inconsistent with the rest of the plugin; large).
- Making `CodeBuilder` a real C++ parser/validator (false-positive minefield on
  embedded expressions and string literals; not worth it).
- Touching `printWithGlobalsReplaced` pretty-print round-tripping (inherent to
  the source-to-source design of the whole tool, not this function).
