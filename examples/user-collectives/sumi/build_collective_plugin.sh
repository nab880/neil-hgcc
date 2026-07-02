#!/usr/bin/env bash
#
# Build an out-of-tree sumi collective algorithm into a shared library you load
# at runtime -- no libsumi rebuild (Tier 3). The SUMI_REGISTER_COLLECTIVE static
# initializer in your .cc runs at dlopen and populates the engine's registry.
#
# Usage:
#   ./build_collective_plugin.sh                         # my_ring_allreduce.cc -> libmycoll.so
#   SRC=my_alg.cc OUT=libmyalg.so ./build_collective_plugin.sh
#   ./build_collective_plugin.sh /path/to/install        # override PREFIX
#
# Then load it (Linux; off by default -- see run.py / README.md):
#   COLLECTIVE_PLUGINS=$PWD/libmycoll.so SST_HG_MV2_PRELOAD=1 \
#     SUMI_ALLREDUCE_ALG=myring APP=test_mv2_allreduce sst run_test_mv2_allreduce.py
#
# Linux-first. macOS uses two-level namespaces; a -bundle build is attempted but
# RTLD_GLOBAL cross-library symbol resolution there is an open question (see the
# platform caveat in ../../../../user-collectives.md §8).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SRC="${SRC:-my_ring_allreduce.cc}"
OUT="${OUT:-libmycoll.so}"
if [[ ! -f "$SRC" ]]; then
  echo "Error: SRC not found: $SRC"
  exit 1
fi

PREFIX="${1:-${SST_HG_PREFIX:-$SCRIPT_DIR/../../../../install}}"
if [[ ! -d "$PREFIX" ]]; then
  echo "Error: PREFIX not found: $PREFIX"
  exit 1
fi
PREFIX="$(cd "$PREFIX" && pwd)"

INC="$PREFIX/include"
if [[ ! -f "$INC/sst/elements/iris/sumi/collective_registry.h" ]]; then
  echo "Error: installed sumi headers not found under $INC/sst/elements/iris."
  echo "Build sst-elements (with the collective registry) and install first."
  exit 1
fi

# Base compile flags: prefer sst-config so we match the core build's C++
# standard / defines; add the include roots the sumi headers need
# (<iris/...> and <mercury/...> resolve from include/sst/elements; the bare
# <output.h> in collective.h resolves from include/sst/core).
CXX="${CXX:-}"
if [[ -z "$CXX" ]]; then
  if command -v clang++ >/dev/null 2>&1; then CXX=clang++; else CXX=c++; fi
fi

CXXFLAGS_BASE="-std=c++17 -fPIC"
if [[ -x "$PREFIX/bin/sst-config" ]]; then
  CXXFLAGS_BASE="$("$PREFIX/bin/sst-config" --ELEMENT_CXXFLAGS 2>/dev/null || echo "$CXXFLAGS_BASE")"
fi

INCLUDES=(-I "$INC/sst/elements" -I "$INC/sst/core" -I "$INC")

echo "Compiling $SRC -> $OUT ..."
if [[ "$(uname -s)" == Darwin ]]; then
  $CXX $CXXFLAGS_BASE "${INCLUDES[@]}" -bundle -undefined dynamic_lookup \
    "$SRC" -o "$OUT"
else
  $CXX $CXXFLAGS_BASE "${INCLUDES[@]}" -shared "$SRC" -o "$OUT"
fi

echo "Built: $SCRIPT_DIR/$OUT"
echo "Load with COLLECTIVE_PLUGINS=$SCRIPT_DIR/$OUT and select by its registered name."
