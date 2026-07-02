#!/usr/bin/env bash
#
# Build any single-file MPI program into lib<APP>.so for the Mercury/Iris
# simulator. Source-agnostic generalization of tests/mv2/_mv2_build_common.sh:
# instead of a per-test TEST_NAME wrapper, set SRC= (and optionally APP=).
#
# Usage:
#   ./build.sh                              # builds my_allreduce.c -> libmy_allreduce.so
#   SRC=my_scheme.c ./build.sh              # APP defaults to the src basename
#   SRC=my_scheme.c APP=fancy ./build.sh    # override the app/library name
#   ./build.sh /path/to/install             # override the install PREFIX
#
# Keeps the macOS -bundle / Linux -shared split and the simple-PMI stripping
# from the mv2 harness verbatim; only the source/app names are parameterized.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# The mv2 harness ships the MPI wrapper shim and the glibc loader stub we reuse.
MV2_DIR="$(cd "$SCRIPT_DIR/../../../tests/mv2" && pwd)"

SRC="${SRC:-my_allreduce.c}"
if [[ ! -f "$SRC" ]]; then
  echo "Error: SRC not found: $SRC (run from $SCRIPT_DIR or set SRC=)"
  exit 1
fi
# APP defaults to the source file's basename without extension.
APP="${APP:-$(basename "${SRC%.*}")}"

PREFIX="${1:-${SST_HG_PREFIX:-$SCRIPT_DIR/../../../../install}}"
if [[ ! -d "$PREFIX" ]]; then
  echo "Error: PREFIX not found: $PREFIX"
  echo "Usage: [SRC=x.c] [APP=name] ./build.sh [PREFIX]"
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

echo "Building lib${APP} from ${SRC} ..."
# EXTRA_CFLAGS lets a sweep vary compile-time knobs, e.g. -DNELEMS=256.
hgcc --replacements pthread.h -std=gnu11 ${EXTRA_CFLAGS:-} -c "$SRC" -o "${APP}.o" -I "$MV2_INC"

# hgcc embeds app registration under a path-mangled name inside the .o.
# Generate a tiny C++ shim that re-registers the same function under the short
# app name the SST Python driver expects ("${APP}").
if [[ "$(uname -s)" == Darwin ]]; then
  _nm_def=(nm -gU)
else
  _nm_def=(nm -g --defined-only)
fi
MAIN_SYM=$("${_nm_def[@]}" "${APP}.o" \
  | awk '/sst_hg_user_main/{print $3; exit}' | sed 's/^_//')
if [[ -z "$MAIN_SYM" ]]; then
  echo "Error: could not find sst_hg_user_main symbol in ${APP}.o"
  echo "Hint: your source must contain '#define ssthg_app_name ${APP}' and a main()."
  exit 1
fi
cat > register_app_gen.cpp <<CPPEOF
extern "C" int ${MAIN_SYM}(int, char**);
typedef int (*main_fxn)(int, char**);
extern int userSkeletonMainInitFxn(const char* name, main_fxn fxn);
static int _reg = userSkeletonMainInitFxn("${APP}", ${MAIN_SYM});
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
  gcc -c "$MV2_DIR/mv2_mpi_wrappers.c" -I "$MV2_INC" -o mv2_mpi_wrappers.o
  gcc -bundle -undefined dynamic_lookup \
    -Wl,-rpath,"$PREFIX/lib" \
    -Wl,-rpath,"$OFI_EXT_LIB" \
    -Wl,-force_load,"$LIBMPI_NOPMI" \
    -L "$OFI_EXT_LIB" -lpmi -lfabric \
    mv2_mpi_wrappers.o register_app_gen.o "${APP}.o" \
    -o "lib${APP}.so"
  rm -f mv2_mpi_wrappers.o
else
  _stub_cflags=(-fPIC)
  if [[ "$(uname -s)" == Linux ]] && command -v gcc >/dev/null 2>&1; then
    _dM=$(echo | gcc -dM -E - 2>/dev/null || true)
    if echo "$_dM" | grep -q '#define __linux__' && ! echo "$_dM" | grep -q '#define __GLIBC__'; then
      _stub_cflags+=(-D__GLIBC__=1)
    fi
  fi
  gcc "${_stub_cflags[@]}" -c "$MV2_DIR/mv2_ld_glibc_stub.c" -o "$SCRIPT_DIR/mv2_ld_glibc_stub.o"
  gcc -fPIC -c "$MV2_DIR/mv2_mpi_wrappers.c" -I "$MV2_INC" -o "$SCRIPT_DIR/mv2_mpi_wrappers.o"
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
    register_app_gen.o "${APP}.o" -o "lib${APP}.so"
  rm -f "$SCRIPT_DIR/mv2_ld_glibc_stub.o" "$SCRIPT_DIR/mv2_mpi_wrappers.o"
fi
rm -f "$LIBMPI_NOPMI"
rm -f "${APP}.o" register_app_gen.o register_app_gen.cpp

echo "Built: $SCRIPT_DIR/lib${APP}.so"
echo "Run:   APP=${APP} NRANKS=8 sst run.py"
