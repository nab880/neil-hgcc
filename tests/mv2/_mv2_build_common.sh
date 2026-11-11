#!/usr/bin/env bash
# Shared build body for MV2 tests. Callers set TEST_NAME (e.g. test_mv2_bcast)
# and then `source` this file from their per-test build_${TEST_NAME}.sh wrapper.
#
# Mirrors the standalone build_test_mv2_sendrecv.sh (fallback / legacy path):
#   hgcc -c ${TEST_NAME}.c
#   generate register_app_gen.cpp shim (re-registers ssthg_app_name under the
#     short app name ${TEST_NAME})
#   link lib${TEST_NAME}.so against libmpi.a (simple_pmi stripped) + Iris pmi/fabric.

set -e

if [[ -z "${TEST_NAME:-}" ]]; then
  echo "Error: _mv2_build_common.sh requires TEST_NAME to be set."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PREFIX="${1:-${SST_HG_PREFIX:-$SCRIPT_DIR/../../../install}}"
if [[ ! -d "$PREFIX" ]]; then
  echo "Error: PREFIX not found: $PREFIX"
  echo "Usage: build_${TEST_NAME}.sh [PREFIX]"
  exit 1
fi
PREFIX="$(cd "$PREFIX" && pwd)"

export PATH="$PREFIX/bin:$PATH"

if ! command -v hgcc >/dev/null 2>&1; then
  echo "Error: hgcc not found. Add $PREFIX/bin to PATH or set PREFIX."
  exit 1
fi

MV2_LIB="$PREFIX/mvapich2/lib"
MV2_INC="$PREFIX/mvapich2/include"
HG_ELEMS_LIB="$PREFIX/lib/sst-elements-library"
OFI_EXT_LIB="$HG_ELEMS_LIB/ext"

if [[ ! -f "$MV2_LIB/libmpi.a" ]]; then
  echo "Error: $MV2_LIB/libmpi.a not found. Build mvapich2 with hgcc first."
  exit 1
fi
if [[ ! -d "$OFI_EXT_LIB" ]] || ! ls "$OFI_EXT_LIB"/libfabric.* 1>/dev/null 2>&1; then
  echo "Error: Iris libfabric not found under $OFI_EXT_LIB. Build sst-elements first."
  exit 1
fi
if ! ls "$OFI_EXT_LIB"/libpmi.* 1>/dev/null 2>&1; then
  echo "Error: Iris PMI (libpmi) not found under $OFI_EXT_LIB. Build sst-elements (iris) first."
  exit 1
fi

echo "Building lib${TEST_NAME} ..."
hgcc --disable-mpi --replacements pthread.h -std=gnu11 -c "${TEST_NAME}.c" -I "$MV2_INC"

# hgcc embeds app registration under a path-mangled name inside the .o.
# Generate a tiny C++ shim that re-registers the same function under the
# short app name the SST Python driver expects ("${TEST_NAME}").
if [[ "$(uname -s)" == Darwin ]]; then
  _nm_def=(nm -gU)
else
  _nm_def=(nm -g --defined-only)
fi
MAIN_SYM=$("${_nm_def[@]}" "${TEST_NAME}.o" \
  | awk '/sst_hg_user_main/{print $3; exit}' | sed 's/^_//')
if [[ -z "$MAIN_SYM" ]]; then
  echo "Error: could not find sst_hg_user_main symbol in ${TEST_NAME}.o"
  exit 1
fi
cat > register_app_gen.cpp <<CPPEOF
extern "C" int ${MAIN_SYM}(int, char**);
typedef int (*main_fxn)(int, char**);
extern int userSkeletonMainInitFxn(const char* name, main_fxn fxn);
static int _reg = userSkeletonMainInitFxn("${TEST_NAME}", ${MAIN_SYM});
CPPEOF
clang++ -std=c++11 -c register_app_gen.cpp \
  -I"$PREFIX/include/sst/elements"

# Strip MVAPICH2/MPICH embedded simple PMI client so the SST PMI library is used.
LIBMPI_NOPMI="$SCRIPT_DIR/libmpi_nopmi.a"
cp "$MV2_LIB/libmpi.a" "$LIBMPI_NOPMI"
while IFS= read -r _mem; do
  [[ -z "$_mem" ]] && continue
  ar -d "$LIBMPI_NOPMI" "$_mem" 2>/dev/null || true
done < <(ar t "$LIBMPI_NOPMI" 2>/dev/null | grep -i 'simple_pmi' || true)

if [[ "$(uname -s)" == Darwin ]]; then
  gcc -c mv2_mpi_wrappers.c -I "$MV2_INC"
  gcc -bundle -undefined dynamic_lookup \
    -Wl,-rpath,"$PREFIX/lib" \
    -Wl,-rpath,"$OFI_EXT_LIB" \
    -Wl,-force_load,"$LIBMPI_NOPMI" \
    -L "$OFI_EXT_LIB" -lpmi -lfabric \
    mv2_mpi_wrappers.o register_app_gen.o "${TEST_NAME}.o" \
    -o "lib${TEST_NAME}.so"
  rm -f mv2_mpi_wrappers.o
else
  _stub_cflags=(-fPIC)
  if [[ "$(uname -s)" == Linux ]] && command -v gcc >/dev/null 2>&1; then
    _dM=$(echo | gcc -dM -E - 2>/dev/null || true)
    if echo "$_dM" | grep -q '#define __linux__' && ! echo "$_dM" | grep -q '#define __GLIBC__'; then
      _stub_cflags+=(-D__GLIBC__=1)
    fi
  fi
  gcc "${_stub_cflags[@]}" -c "$SCRIPT_DIR/mv2_ld_glibc_stub.c" -o "$SCRIPT_DIR/mv2_ld_glibc_stub.o"
  gcc -fPIC -c mv2_mpi_wrappers.c -I "$MV2_INC" -o "$SCRIPT_DIR/mv2_mpi_wrappers.o"
  gcc -shared \
    -Wl,-Bsymbolic-functions \
    -Wl,-rpath,"$PREFIX/lib" \
    -Wl,-rpath,"$HG_ELEMS_LIB" \
    -Wl,-rpath,"$OFI_EXT_LIB" \
    -Wl,--whole-archive "$LIBMPI_NOPMI" -Wl,--no-whole-archive \
    -L "$OFI_EXT_LIB" -lpmi -lfabric \
    -L "$HG_ELEMS_LIB" -Wl,--no-as-needed -lhg -Wl,--as-needed \
    "$SCRIPT_DIR/mv2_ld_glibc_stub.o" \
    "$SCRIPT_DIR/mv2_mpi_wrappers.o" \
    register_app_gen.o "${TEST_NAME}.o" -o "lib${TEST_NAME}.so"
  rm -f "$SCRIPT_DIR/mv2_ld_glibc_stub.o" "$SCRIPT_DIR/mv2_mpi_wrappers.o"
fi
rm -f "$LIBMPI_NOPMI"
rm -f "${TEST_NAME}.o" register_app_gen.o register_app_gen.cpp

echo "Built: $SCRIPT_DIR/lib${TEST_NAME}.so"
echo "Run:   sst --verbose run_${TEST_NAME}.py  (or ./run_${TEST_NAME}.sh to rebuild + run)"
