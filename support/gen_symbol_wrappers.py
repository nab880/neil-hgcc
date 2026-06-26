#!/usr/bin/env python3
"""Generate strong exported wrappers for the weak-aliased public symbols of a
static library, so they survive a macOS `-undefined dynamic_lookup` bundle link.

The problem (generic)
---------------------
Libraries that implement the "profiling interface" idiom (MPI is the canonical
example) define each public entry point `FOO` as a *weak* alias of a *strong*
implementation `PFOO`. When such an archive is `-force_load`ed into a macOS
bundle, the weak `FOO` lands as a **local (non-external)** symbol. Any `FOO`
the application calls is then left as an **undefined dynamic_lookup** reference
that dyld can only satisfy from *exported* symbols -- and the local `FOO` is
invisible to it. The reference binds to garbage and the program crashes at load
/ first call. (See check_mpi_wrappers.py for the matching build-time guard.)

The fix is to provide a strong, *exported* definition of every such public
symbol. This tool generates them automatically from the archive's symbol table
so the set can never be incomplete (no hand-maintained list to forget).

How pairs are discovered (library-agnostic)
-------------------------------------------
The weak-alias idiom makes `FOO` and `PFOO` the *same code*, hence the same
address within the same object. So, with no naming convention required:

    for each (archive-member, address) in the __text section:
        if there is >=1 external symbol AND >=1 non-external symbol there,
        emit an exported wrapper for each non-external ("public") name that
        tail-calls the external ("implementation") name.

`--include` / `--exclude` regexes (matched against the public symbol name) let
you scope it to a library's public surface (e.g. `^_MPI_` for MPI). Default is
all such pairs.

How wrappers forward (signature-free)
-------------------------------------
Each wrapper is a one-instruction tail-call trampoline (`b` on arm64, `jmp` on
x86-64). A tail call preserves every argument register and the stack, and lets
the implementation return straight to the original caller -- so it is correct
for *any* signature (including variadic and struct-by-value returns) without
parsing a single header. Being a strong global, it overrides the weak local the
same way a hand-written forwarding wrapper would.

Usage
-----
    gen_symbol_wrappers.py --archive libfoo.a [--archive ...] \
        --out wrappers.S [--include REGEX] [--exclude REGEX] \
        [--emit trampoline|alias-list] [--nm nm]

`--emit alias-list` instead writes a macOS `ld -alias_list` file (`<impl>
<alias>` per line) -- an alternative, also signature-free, mechanism.
"""

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict


# nm -mA line:
#   <archive>:<member>: <addr> (<sect>) <scope> <name>
# undefined lines have "(undefined)" and no address; we skip those.
_LINE = re.compile(
    r"^.*?:(?P<member>[^:]+):\s+"
    r"(?P<addr>[0-9a-fA-F]+)\s+\((?P<sect>[^)]*)\)\s+"
    r"(?P<scope>non-external|external|private external)\s+"
    r"(?P<name>\S+)\s*$"
)


def discover_pairs(archives, nm="nm", text_only=True):
    """Return {public_name: impl_name} for every same-address external/local
    pair found across the given archives."""
    # (member, addr) -> {"ext": [...], "loc": [...]}
    groups = defaultdict(lambda: {"ext": [], "loc": []})
    for arc in archives:
        try:
            out = subprocess.run([nm, "-mA", arc], capture_output=True,
                                 text=True, check=True)
        except (OSError, subprocess.CalledProcessError) as e:
            sys.exit("gen_symbol_wrappers: nm failed on %s: %s" % (arc, e))
        for line in out.stdout.splitlines():
            m = _LINE.match(line)
            if not m:
                continue
            if text_only and "__text" not in m.group("sect"):
                continue
            key = (m.group("member"), m.group("addr"))
            if m.group("scope") == "external":
                groups[key]["ext"].append(m.group("name"))
            elif m.group("scope") == "non-external":
                groups[key]["loc"].append(m.group("name"))

    pairs = {}
    ambiguous = []
    for (member, addr), g in groups.items():
        if not g["ext"] or not g["loc"]:
            continue
        if len(g["ext"]) > 1:
            # More than one external at this address: can't pick an impl
            # unambiguously. Record and skip (rare; warn the caller).
            ambiguous.append((member, addr, g["ext"], g["loc"]))
            continue
        impl = g["ext"][0]
        for pub in g["loc"]:
            # Don't alias a symbol to itself (shouldn't happen for distinct
            # names, but be safe).
            if pub != impl:
                pairs[pub] = impl
    return pairs, ambiguous


def _filter(pairs, include, exclude):
    inc = re.compile(include) if include else None
    exc = re.compile(exclude) if exclude else None
    out = {}
    for pub, impl in pairs.items():
        if inc and not inc.search(pub):
            continue
        if exc and exc.search(pub):
            continue
        out[pub] = impl
    return out


def emit_trampolines(pairs, out_path, archives, include, exclude):
    lines = [
        "/* AUTO-GENERATED by gen_symbol_wrappers.py -- do not edit. */",
        "/* Strong exported tail-call trampolines for weak-aliased public",
        " * symbols, so they survive a -undefined dynamic_lookup bundle link. */",
        "/* archives: %s */" % ", ".join(os.path.basename(a) for a in archives),
        "/* include=%r exclude=%r  count=%d */" % (include, exclude, len(pairs)),
        "",
        "#if defined(__arm64__) || defined(__aarch64__)",
        "# define NSX_TAIL(impl) b impl",
        "# define NSX_ALIGN .p2align 2",
        "#elif defined(__x86_64__)",
        "# define NSX_TAIL(impl) jmp impl",
        "# define NSX_ALIGN .p2align 4",
        "#else",
        '# error "gen_symbol_wrappers: unsupported architecture"',
        "#endif",
        "",
        "    .text",
    ]
    for pub in sorted(pairs):
        impl = pairs[pub]
        lines += [
            "    .globl %s" % pub,
            "    NSX_ALIGN",
            "%s:" % pub,
            "    NSX_TAIL(%s)" % impl,
        ]
    lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def emit_alias_list(pairs, out_path):
    # macOS `ld -alias_list` format: "<real_name> <alias_name>" per line.
    with open(out_path, "w") as f:
        for pub in sorted(pairs):
            f.write("%s %s\n" % (pairs[pub], pub))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", action="append", required=True,
                    help="static library/object to scan (repeatable)")
    ap.add_argument("--out", required=True, help="output file (.S or alias list)")
    ap.add_argument("--include", default=None,
                    help="regex; only public symbols matching it are wrapped")
    ap.add_argument("--exclude", default=None,
                    help="regex; public symbols matching it are skipped")
    ap.add_argument("--emit", choices=("trampoline", "alias-list"),
                    default="trampoline")
    ap.add_argument("--nm", default="nm", help="nm binary to use")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv[1:])

    for a in args.archive:
        if not os.path.isfile(a):
            sys.exit("gen_symbol_wrappers: archive not found: %s" % a)

    pairs, ambiguous = discover_pairs(args.archive, nm=args.nm)
    pairs = _filter(pairs, args.include, args.exclude)
    if not pairs:
        sys.exit("gen_symbol_wrappers: no weak/strong symbol pairs matched "
                 "(check --archive / --include).")

    if args.emit == "trampoline":
        emit_trampolines(pairs, args.out, args.archive, args.include, args.exclude)
    else:
        emit_alias_list(pairs, args.out)

    if not args.quiet:
        sys.stderr.write("gen_symbol_wrappers: %d wrapper(s) -> %s\n"
                         % (len(pairs), args.out))
        sample = sorted(pairs)[:5]
        for s in sample:
            sys.stderr.write("    %s -> %s\n" % (s, pairs[s]))
        if len(pairs) > len(sample):
            sys.stderr.write("    ... (%d more)\n" % (len(pairs) - len(sample)))
        if ambiguous:
            sys.stderr.write("gen_symbol_wrappers: %d ambiguous address(es) "
                             "skipped (multiple external symbols)\n" % len(ambiguous))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
