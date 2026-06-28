#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

missing=()
[[ -z "${LLVM_ROOT:-}" ]] && missing+=("LLVM_ROOT")
[[ -z "${HGCC:-}" ]] && missing+=("HGCC")

if (( ${#missing[@]} > 0 )); then
  echo "error: required environment variable(s) not set: ${missing[*]}" >&2
  echo "Set them before running this script, for example:" >&2
  echo "  export LLVM_ROOT=/path/to/llvm    # e.g. /opt/homebrew/opt/llvm@22" >&2
  echo "  export HGCC=/path/to/hgcc         # e.g. .../sst-hgcc/build/hgcc" >&2
  exit 1
fi

cd "$SCRIPT_DIR"

if command -v lit >/dev/null 2>&1; then
  exec lit -v "$@" .
elif python3 -m lit --help >/dev/null 2>&1; then
  exec python3 -m lit -v "$@" .
else
  echo "error: lit not found (install with: pip install lit)" >&2
  exit 1
fi
