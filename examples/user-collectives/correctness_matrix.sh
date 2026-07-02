#!/usr/bin/env bash
#
# Tier 2 correctness matrix: run a self-checking MPI_Allreduce app at several
# rank counts (including a non-power-of-2, where a ring's remainder logic bites)
# under the built-in default and each named algorithm. Every cell must print
# PASS -- the app asserts the known sum n(n+1)/2, so a wrong DAG either mis-sums
# (caught by the assert) or deadlocks (caught by the engine's deadlockCheck()).
#
# Because every algorithm computes the same all-reduce, "PASS everywhere" is the
# diff against the default: a selected algorithm that disagrees fails the assert.
#
# Usage:
#   ./correctness_matrix.sh
#   ALGS="ring recdouble myring" NRANKS_LIST="2 3 4 7 8 16" ./correctness_matrix.sh
#   COLLECTIVE_PLUGINS=/abs/libmycoll.so ALGS="myring" ./correctness_matrix.sh   # Linux
#
# Env:
#   ALGS         algorithm names to test (default "ring recdouble"); the empty
#                default is always tested too and shown as "<default>".
#   NRANKS_LIST  rank counts (default "2 4 7 8")
#   PREFIX       install tree (default ../../../../install)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MPI_DIR="$SCRIPT_DIR/mpi"

PREFIX="${PREFIX:-$SCRIPT_DIR/../../../install}"
PREFIX="$(cd "$PREFIX" 2>/dev/null && pwd || echo "$PREFIX")"
SST="${SST:-$PREFIX/bin/sst}"
command -v "$SST" >/dev/null 2>&1 || SST="sst"

ALGS="${ALGS:-ring recdouble}"
NRANKS_LIST="${NRANKS_LIST:-2 4 7 8}"

echo "Building check_allreduce ..."
( cd "$MPI_DIR" && SRC=check_allreduce.c ./build.sh "$PREFIX" ) || { echo "build failed"; exit 1; }

run_one() {  # $1=nranks  $2=alg-label  $3=ALG-env
  local n="$1" label="$2" alg="$3" out rc
  out="$( cd "$MPI_DIR" && APP=check_allreduce NRANKS="$n" ALG="$alg" \
          "$SST" run.py 2>&1 )"
  rc=$?
  if [[ $rc -eq 0 ]] && grep -q "PASS: check_allreduce" <<<"$out"; then
    printf "  n=%-3s %-12s PASS\n" "$n" "$label"
    return 0
  fi
  printf "  n=%-3s %-12s FAIL (rc=%d)\n" "$n" "$label" "$rc"
  echo "$out" | tail -8 | sed 's/^/      | /'
  return 1
}

fails=0
echo "Correctness matrix (each cell must PASS):"
for n in $NRANKS_LIST; do
  run_one "$n" "<default>" "" || fails=$((fails+1))
  for alg in $ALGS; do
    run_one "$n" "$alg" "$alg" || fails=$((fails+1))
  done
done

echo
if [[ $fails -eq 0 ]]; then
  echo "ALL PASS"
  exit 0
fi
echo "$fails cell(s) FAILED"
exit 1
