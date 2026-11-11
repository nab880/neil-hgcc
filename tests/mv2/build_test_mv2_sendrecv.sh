#!/usr/bin/env bash
# Legacy / fallback build script for libtest_mv2_sendrecv.so.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PREFIX="${1:-${SST_HG_PREFIX:-$SCRIPT_DIR/../../../install}}"
if [[ ! -d "$PREFIX" ]]; then
  echo "Error: PREFIX not found: $PREFIX"
  echo "Usage: $0 [PREFIX]"
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

# Step 1: Compile with hgcc (injects skeleton.h, pthread replacement headers).
# Step 2: Compile MPI wrappers (macOS only - provides strong global MPI_*
#         symbols to override the local weak aliases from #pragma weak).
# Step 3: Link manually with -force_load to pull all archive members and
#         the strong wrappers that ensure MPI_* resolves inside the bundle.
echo "Building libtest_mv2_sendrecv ..."
hgcc --disable-mpi --replacements pthread.h -std=gnu11 -c test_mv2_sendrecv.c -I "$MV2_INC"

# hgcc embeds app registration under a path-mangled name inside the .o.
# Generate a tiny C++ shim that re-registers the same function under the
# short app name the SST Python driver expects ("test_mv2_sendrecv").
# Darwin: nm -gU = externals that are defined. GNU nm: -U means --unicode; use
# --defined-only instead.
if [[ "$(uname -s)" == Darwin ]]; then
  _nm_def=(nm -gU)
else
  _nm_def=(nm -g --defined-only)
fi
MAIN_SYM=$("${_nm_def[@]}" test_mv2_sendrecv.o \
  | awk '/sst_hg_user_main/{print $3; exit}' | sed 's/^_//')
if [[ -z "$MAIN_SYM" ]]; then
  echo "Error: could not find sst_hg_user_main symbol in test_mv2_sendrecv.o"
  exit 1
fi
cat > register_app_gen.cpp <<CPPEOF
extern "C" int ${MAIN_SYM}(int, char**);
typedef int (*main_fxn)(int, char**);
extern int userSkeletonMainInitFxn(const char* name, main_fxn fxn);
static int _reg = userSkeletonMainInitFxn("test_mv2_sendrecv", ${MAIN_SYM});
CPPEOF
clang++ -std=c++11 -c register_app_gen.cpp \
  -I"$PREFIX/include/sst/elements"

# Strip MVAPICH2/MPICH embedded simple PMI client so the SST PMI library is used.
# Object names inside libmpi.a vary by version and platform (e.g. lib_libmpi_la-* vs
# other prefixes); remove every archive member whose name contains simple_pmi.
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
    mv2_mpi_wrappers.o register_app_gen.o test_mv2_sendrecv.o \
    -o libtest_mv2_sendrecv.so
  rm -f mv2_mpi_wrappers.o
else
  # Link libhg so userSkeletonMainInitFxn resolves and the loader can find Mercury
  # (register_app_gen calls into libhg.so). Without this, dlopen(libtest_...) can fail
  # silently during SST requireLibrary and no main is registered ("no main_fxns_").
  _stub_cflags=(-fPIC)
  # Some hosts define __linux__ but not __GLIBC__ when preprocessing the stub; keep body.
  if [[ "$(uname -s)" == Linux ]] && command -v gcc >/dev/null 2>&1; then
    _dM=$(echo | gcc -dM -E - 2>/dev/null || true)
    if echo "$_dM" | grep -q '#define __linux__' && ! echo "$_dM" | grep -q '#define __GLIBC__'; then
      _stub_cflags+=(-D__GLIBC__=1)
    fi
  fi
  gcc "${_stub_cflags[@]}" -c "$SCRIPT_DIR/mv2_ld_glibc_stub.c" -o "$SCRIPT_DIR/mv2_ld_glibc_stub.o"
  # mv2_mpi_wrappers.c provides hgcc_gethostname / hgcc_gethostid / hgcc_usleep
  # stubs that hgcc-rewritten code in libmpi.a references via replacements/unistd.h.
  # (The MPI_* -> PMPI_* wrappers in the same file are no-ops on Linux once the
  # MVAPICH2 weak-alias #pragmas take effect, but they are harmless.)
  gcc -fPIC -c mv2_mpi_wrappers.c -I "$MV2_INC" -o "$SCRIPT_DIR/mv2_mpi_wrappers.o"
  # -Bsymbolic-functions: prefer our own (MVAPICH2) MPI_* function symbols over
  # any MPI_* the host sst process already loaded (e.g. Open MPI pulled in as an
  # MCA component). Without this the DSO's MPI_Init call resolves to Open MPI's
  # MPI_Init, which is already initialized and errors "MPI_INIT called more than
  # once". Use the -functions variant rather than plain -Bsymbolic so extern DATA
  # symbols (e.g. SST::Hg::GlobalVariable::glblCtx, sst_hg_global_stacksize) keep
  # resolving to libhg's single-instance definitions instead of getting a local
  # copy in the DSO (which would make the per-rank globals segment diverge from
  # the one libhg uses to allocate it, causing OOB writes).
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
    register_app_gen.o test_mv2_sendrecv.o -o libtest_mv2_sendrecv.so
  rm -f "$SCRIPT_DIR/mv2_ld_glibc_stub.o" "$SCRIPT_DIR/mv2_mpi_wrappers.o"
fi
rm -f "$LIBMPI_NOPMI"
rm -f test_mv2_sendrecv.o register_app_gen.o register_app_gen.cpp

echo "Built: $SCRIPT_DIR/libtest_mv2_sendrecv.so"
echo "Run:   sst --verbose run_test_mv2_sendrecv.py  (or ./run_test_mv2_sendrecv.sh to rebuild + run)"
