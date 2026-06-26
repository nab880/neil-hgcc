#!/usr/bin/env python3
"""Post-link guard for `hgcc --mpi=mvapich2` application bundles.

Why this exists
---------------
On macOS the app `.so` is a `-bundle -undefined dynamic_lookup` image, and
MVAPICH2's `MPI_*` entry points are *weak* aliases of `PMPI_*`. The MV2 link
`-force_load`s `libmpi_nopmi.a`, which pulls those weak `MPI_*` defs in as
**local (non-external)** symbols. Any `MPI_*` the application calls that does
*not* have a strong wrapper in `mv2_mpi_wrappers.c` is then left as an
**undefined `dynamic_lookup`** reference. dyld resolves dynamic_lookup against
*exported* symbols only, so the bundle's own local `MPI_*` is invisible to it:
the reference binds to garbage at load time and the program jumps off into the
weeds (observed: `MPI_Irecv` jumping to the MPI-handle-shaped address
`0x4c000111` and SIGSEGV'ing deep in finalize/p2p).

The defect is silent -- the link succeeds, the crash happens at runtime far
from the cause. This script turns it into a loud, named build-time error.

The invariant
-------------
A symbol that is simultaneously **undefined** *and* **defined-locally** in the
same image is an unwrapped weak MPI symbol. A correct strong wrapper makes the
symbol external-defined, which removes it from *both* sets. So:

    BUG = undefined(MPI_*) ∩ local_defined(MPI_*)

This is exact and has no false positives: a symbol only lands in the
intersection when force_load supplied a private copy *and* the call site was
left dynamic. (Symbols that genuinely resolve at runtime -- e.g. `MPI_Wtime`,
which is strong, hence external -- are never in the intersection.)

Usage:  check_mpi_wrappers.py <linked.so>
Exit 1 (fail the build) if the intersection is non-empty, naming each symbol
and the wrapper to add. Set HGCC_ALLOW_UNWRAPPED_MPI=1 to downgrade to a
warning (escape hatch; not recommended).
"""

import os
import re
import subprocess
import sys

# Match MPI / PMPI / MPIX / PMPIX symbols, with or without the Mach-O leading '_'.
_MPI_RE = re.compile(r"^_?(P?MPI[X]?_\w+)$")


def _classify(libpath):
    """Return (undefined, local_defined) sets of MPI symbol base names, or None
    if the symbol table could not be read."""
    # `nm -m` gives unambiguous Mach-O descriptions:
    #   <addr> (__TEXT,__text) external      _foo      -> exported definition
    #   <addr> (__TEXT,__text) non-external  _foo      -> local definition
    #          (undefined) ... [dynamically looked up] _foo  -> undefined ref
    try:
        out = subprocess.run(["nm", "-m", libpath], capture_output=True,
                             text=True)
    except OSError:
        return None
    if out.returncode != 0:
        # Fall back to a plain `nm` (e.g. non-Darwin); type letters: U =
        # undefined, lowercase = local defined, uppercase = external.
        return _classify_plain(libpath)

    undefined, local_def = set(), set()
    for line in out.stdout.splitlines():
        # The symbol name is not always the last token: an undefined line ends
        # with "(dynamically looked up)". Scan tokens for the MPI-shaped name.
        base = None
        for tok in line.split():
            m = _MPI_RE.match(tok)
            if m:
                base = m.group(1)
                break
        if base is None:
            continue
        if "(undefined)" in line:
            undefined.add(base)
        elif "non-external" in line:
            local_def.add(base)
        # "external" (without "non-") => a real exported definition: safe.
    return undefined, local_def


def _classify_plain(libpath):
    try:
        out = subprocess.run(["nm", libpath], capture_output=True, text=True)
    except OSError:
        return None
    if out.returncode != 0:
        return None
    undefined, local_def = set(), set()
    for line in out.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        # "<addr> <type> <name>" or "<type> <name>" (undefined has no addr)
        if len(parts) >= 2 and len(parts[-2]) == 1:
            typ, name = parts[-2], parts[-1]
        else:
            continue
        m = _MPI_RE.match(name)
        if not m:
            continue
        base = m.group(1)
        if typ == "U":
            undefined.add(base)
        elif typ in "tdbsrn":          # lowercase, defined locally
            local_def.add(base)
    return undefined, local_def


def main(argv):
    if len(argv) != 2:
        sys.stderr.write("usage: check_mpi_wrappers.py <linked.so>\n")
        return 2
    libpath = argv[1]
    if not os.path.isfile(libpath):
        # Nothing to check (e.g. link produced no file); let the real link
        # error surface instead of masking it.
        return 0

    result = _classify(libpath)
    if result is None:
        sys.stderr.write(
            "hgcc: warning: could not read symbol table of %s; skipping "
            "MPI-wrapper check\n" % libpath)
        return 0
    undefined, local_def = result
    missing = sorted(undefined & local_def)
    if not missing:
        return 0

    allow = os.environ.get("HGCC_ALLOW_UNWRAPPED_MPI", "") not in ("", "0")
    label = "warning" if allow else "error"
    msg = [
        "",
        "hgcc: %s: unwrapped weak MPI symbol(s) in %s" % (label, os.path.basename(libpath)),
        "",
        "  The following MPI symbols are referenced as undefined yet exist only",
        "  as local (force_loaded weak) definitions. In the macOS",
        "  -undefined dynamic_lookup bundle they will bind to GARBAGE at load",
        "  time and crash at runtime (e.g. a jump to an MPI-handle address):",
        "",
    ]
    for sym in missing:
        pmpi = sym if sym.startswith("P") else "P" + sym
        msg.append("      %-24s  -> add:  int %s(...) { return %s(...); }"
                    % (sym, sym, pmpi))
    msg += [
        "",
        "  Add a strong forwarding wrapper for each to:",
        "      support/mv2/mv2_mpi_wrappers.c   (and the installed copy",
        "      share/hgcc/mv2/mv2_mpi_wrappers.c). Copy the signature from",
        "      <mpi.h>; the body just forwards to the PMPI_ variant.",
        "  (Escape hatch: set HGCC_ALLOW_UNWRAPPED_MPI=1 to build anyway.)",
        "",
    ]
    sys.stderr.write("\n".join(msg))
    return 0 if allow else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
