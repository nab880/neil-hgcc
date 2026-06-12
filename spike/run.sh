#!/usr/bin/env bash
# Phase-0 probe matrix (CUDA-IMPL-PLAN.md): prove toolkit-free CUDA host-only
# parsing on LLVM 22 (the toolchain hgcc is configured against), the
# preprocess-then-reparse pattern hgcompile uses, and AST visibility of the
# nodes the P2 rewriter hooks.
#
# Exit 0 iff every gating cell passes. Apple clang (probe 6) is recorded but
# never gates. Probe other toolchains ad hoc with LLVMS="llvm@18 ..." env var.
set -u
cd "$(dirname "$0")"

LLVMS=${LLVMS:-"llvm@22"}
BASE="-x cuda --cuda-host-only -nocudainc -nocudalib -std=c++17 -Istubs"
TMP=$(mktemp -d /tmp/hgcc-cuda-spike.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

fail=0
results=()

record() { # record <cell-name> <pass|FAIL|skip>
  results+=("$(printf '%-34s %s' "$1" "$2")")
  [ "$2" = FAIL ] && fail=1
}

run_probe() { # run_probe <cell-name> <cmd...>
  local name=$1; shift
  if "$@" >"$TMP/out" 2>&1; then
    record "$name" pass
  else
    record "$name" FAIL
    echo "---- $name failed:"
    sed -n '1,15p' "$TMP/out"
  fi
}

for v in $LLVMS; do
  CXX=/opt/homebrew/opt/$v/bin/clang++
  if [ ! -x "$CXX" ]; then
    record "$v (toolchain missing)" FAIL
    continue
  fi

  # 1. plain host-only parse
  run_probe "$v p1-parse" $CXX $BASE --cuda-gpu-arch=sm_70 -fsyntax-only vecadd.cu

  # 2. is --cuda-gpu-arch required for host-only? (informational either way,
  #    but record it: P1 driver flag construction depends on the answer)
  if $CXX $BASE -fsyntax-only vecadd.cu >/dev/null 2>&1; then
    record "$v p2-no-arch-flag" pass
  else
    record "$v p2-no-arch-flag" "skip (arch flag required)"
  fi

  # 3. preprocess then reparse, the hgcompile addPreprocess pattern;
  #    pp file keeps the .cu suffix exactly as addPrefixAndRebase will
  run_probe "$v p3a-preprocess" $CXX $BASE --cuda-gpu-arch=sm_70 -E vecadd.cu -o "$TMP/pp.vecadd.cu"
  run_probe "$v p3b-reparse-pp" $CXX $BASE --cuda-gpu-arch=sm_70 -fsyntax-only "$TMP/pp.vecadd.cu"

  # 4. AST nodes the P2 rewriter hooks must be visible in host-only mode
  $CXX $BASE --cuda-gpu-arch=sm_70 -fsyntax-only -Xclang -ast-dump vecadd.cu >"$TMP/ast" 2>/dev/null
  if grep -q CUDAGlobalAttr "$TMP/ast" && grep -q CUDAKernelCallExpr "$TMP/ast"; then
    record "$v p4-ast-nodes" pass
  else
    record "$v p4-ast-nodes" FAIL
  fi

  # 5. ABI vet: hand-written rewriter output builds as plain C++ against the
  #    stub runtime and runs (added with the expected/ + stub commit)
  if [ -f expected/sst.pp.vecadd.cu.cc ]; then
    if $CXX -std=c++17 -x c++ -I. expected/sst.pp.vecadd.cu.cc stub_hg_cuda.cc \
          -o "$TMP/vecadd_sim" >"$TMP/out" 2>&1 \
        && "$TMP/vecadd_sim" >"$TMP/sim" 2>&1 \
        && grep -q sst_hg_cuda_launch "$TMP/sim"; then
      record "$v p5-abi-vet" pass
    else
      record "$v p5-abi-vet" FAIL
      sed -n '1,15p' "$TMP/out" "$TMP/sim" 2>/dev/null
    fi
  fi
done

# 6. Apple clang, informational only (users may have CC=AppleClang; hgcc
#    always drives the LLVM clang, so this never gates)
if /usr/bin/clang++ $BASE --cuda-gpu-arch=sm_70 -fsyntax-only vecadd.cu >/dev/null 2>&1; then
  record "apple-clang p1 (info)" pass
else
  record "apple-clang p1 (info)" "skip (unsupported)"
fi

echo
echo "==== Phase-0 probe matrix ===="
printf '%s\n' "${results[@]}"
echo "=============================="
[ $fail -eq 0 ] && echo RESULT: GREEN || echo RESULT: RED
exit $fail
