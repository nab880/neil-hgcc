#!/usr/bin/env bash
# Shared build+run body for MV2 tests. Callers set TEST_NAME (e.g. test_mv2_bcast)
# and then `source` this file from their per-test run_${TEST_NAME}.sh wrapper.
#
# Mirrors run_test_mv2_sendrecv.sh: rebuilds via hgcc, sanity-checks loader deps,
# sets SST env, runs `sst --verbose run_${TEST_NAME}.py`.

set -e

if [[ -z "${TEST_NAME:-}" ]]; then
  echo "Error: _mv2_run_common.sh requires TEST_NAME to be set."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PREFIX="${1:-${SST_HG_PREFIX:-$SCRIPT_DIR/../../../install}}"
if [[ ! -d "$PREFIX" ]]; then
  echo "Error: PREFIX not found: $PREFIX"
  echo "Usage: run_${TEST_NAME}.sh [PREFIX]"
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
hgcc --mpi=mvapich2 --app-name="${TEST_NAME}" \
     -std=gnu11 -fPIC "${TEST_NAME}.c" \
     -o "lib${TEST_NAME}.so"

if [[ "$(uname -s)" == Linux ]] && command -v ldd >/dev/null 2>&1; then
  _bad=$(ldd "$SCRIPT_DIR/lib${TEST_NAME}.so" 2>/dev/null | grep 'not found' || true)
  if [[ -n "$_bad" ]]; then
    echo "Error: lib${TEST_NAME}.so has unresolved loader dependencies:"
    echo "$_bad"
    echo "Hint: ensure LD_LIBRARY_PATH includes $HG_ELEMS_LIB (Mercury libhg.so)."
    exit 1
  fi
fi

echo "Running SST ..."
export SST_HG_PREFIX="$PREFIX"
export FI_PROVIDER=sstmac
export SST_LIB_PATH="$SCRIPT_DIR:$OFI_EXT_LIB:${SST_LIB_PATH:-}"
export LD_LIBRARY_PATH="$HG_ELEMS_LIB:$PREFIX/lib:$OFI_EXT_LIB:$SCRIPT_DIR:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$PREFIX/lib:$OFI_EXT_LIB:$SCRIPT_DIR:${DYLD_LIBRARY_PATH:-}"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

if ! command -v sst >/dev/null 2>&1; then
  echo "Error: sst not found in PATH."
  exit 1
fi

if [[ "$(uname -s)" == Linux ]] && command -v ldd >/dev/null 2>&1; then
  _sst_bin=$(command -v sst)
  _ldd_bad=$(ldd "$_sst_bin" 2>/dev/null | grep 'not found' || true)
  if [[ -n "$_ldd_bad" ]]; then
    echo "Error: sst cannot load one or more shared libraries:"
    echo "$_ldd_bad"
    if echo "$_ldd_bad" | grep -q libpython; then
      echo "Fix: prepend the directory containing that libpython*.so to LD_LIBRARY_PATH (the Python tree SST was built with)."
      echo "  Example:  export LD_LIBRARY_PATH=\"/path/to/python/lib:\${LD_LIBRARY_PATH}\""
    fi
    exit 1
  fi
fi


_min_mem_mb="${SST_HG_MV2_MIN_MEM_MB:-1024}"
if [[ "${SST_HG_MV2_SKIP_PREFLIGHT:-0}" != 1 ]]; then
  _pf_warn=""
  _pf_fatal=""

  _avail_mb=""
  if [[ -r /proc/meminfo ]]; then
    _avail_kb=$(awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo 2>/dev/null || true)
    [[ -n "$_avail_kb" ]] && _avail_mb=$(( _avail_kb / 1024 ))
  elif [[ "$(uname -s)" == Darwin ]] && command -v vm_stat >/dev/null 2>&1; then
    _page_bytes=$(vm_stat | awk '/page size of/ {print $8; exit}' 2>/dev/null || echo 4096)
    _free_pages=$(vm_stat | awk '/Pages free/ {gsub(/\./,""); print $3; exit}' 2>/dev/null || echo 0)
    _inact_pages=$(vm_stat | awk '/Pages inactive/ {gsub(/\./,""); print $3; exit}' 2>/dev/null || echo 0)
    _avail_mb=$(( (_free_pages + _inact_pages) * _page_bytes / 1024 / 1024 ))
  fi
  if [[ -n "$_avail_mb" && "$_avail_mb" -lt "$_min_mem_mb" ]]; then
    _pf_fatal+="  available RAM ${_avail_mb} MB < required ${_min_mem_mb} MB"$'\n'
  fi

  _vlim=$(ulimit -v 2>/dev/null || echo unlimited)
  if [[ "$_vlim" != unlimited && -n "$_vlim" ]]; then
    _vlim_mb=$(( _vlim / 1024 ))
    if [[ "$_vlim_mb" -lt "$_min_mem_mb" ]]; then
      _pf_fatal+="  ulimit -v = ${_vlim_mb} MB < required ${_min_mem_mb} MB"$'\n'
    fi
  fi

  if [[ -r /sys/fs/cgroup/memory.max ]]; then
    _cg_max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)
    if [[ "$_cg_max" =~ ^[0-9]+$ ]]; then
      _cg_mb=$(( _cg_max / 1024 / 1024 ))
      if [[ "$_cg_mb" -lt "$_min_mem_mb" ]]; then
        _pf_fatal+="  cgroup memory.max = ${_cg_mb} MB < required ${_min_mem_mb} MB"$'\n'
      fi
    fi
  fi


  if [[ -n "$_pf_fatal" ]]; then
    echo "Error: preflight resource check failed:"
    echo -n "$_pf_fatal"
    echo "Fix: run inside a compute allocation (e.g. salloc -N1 --mem=4G)."
    echo "     Or override:  SST_HG_MV2_MIN_MEM_MB=<mb>  or  SST_HG_MV2_SKIP_PREFLIGHT=1"
    exit 1
  fi
  if [[ -n "$_pf_warn" ]]; then
    echo "Warning: preflight resource check:"
    echo -n "$_pf_warn"
    [[ -n "$_avail_mb" ]] && echo "  (MemAvailable ${_avail_mb} MB, threshold ${_min_mem_mb} MB)"
  fi
fi

# SST-HG global-var runtime is currently ASLR-sensitive on Linux: with ASLR on,
# MPI_Init flakes ~30% of runs at MPIU_Handle_direct_init / MPID_nem_choose_netmod.
# Run with ASLR disabled (setarch -R) to make the per-app data layout deterministic.
# Honor SST_HG_MV2_NO_SETARCH=1 to opt out.
if [[ "${SST_HG_MV2_NO_SETARCH:-0}" != 1 ]] && command -v setarch >/dev/null 2>&1; then
  setarch "$(uname -m)" -R sst --verbose "run_${TEST_NAME}.py"
else
  sst --verbose "run_${TEST_NAME}.py"
fi
echo "PASS: run_${TEST_NAME} completed."
