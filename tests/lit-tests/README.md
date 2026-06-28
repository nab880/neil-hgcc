# sst-hgcc lit tests

Self-contained LLVM lit test
suite for the `sst-hgcc` source-to-source rewriter. Each test runs `hgcc -c`
with `SST_HG_DELETE_TEMP_SOURCES=0`, then uses `FileCheck` to verify the
rewritten `sst.pp.<basename>.cc` intermediate.

## Requirements
* `hgcc` on `$PATH` (or pointed at via `$HGCC`)
* `FileCheck` on `$PATH` (or pointed at via `$FILECHECK`)
* `lit` (`pip install lit`)

### Installing `FileCheck`

If you have an LLVM build with `FileCheck` (e.g. `llvm-project/install/bin/FileCheck`),
add its `bin` to `$PATH` or export `$FILECHECK` to its absolute path:

```bash
export FILECHECK=/path/to/llvm/install/bin/FileCheck
```

If you don't have LLVM tools locally, install the pure-Python
[`filecheck`](https://pypi.org/project/filecheck/) package and point
`$FILECHECK` at its entrypoint:

```bash
pip install filecheck
export FILECHECK="$(command -v filecheck)"
```



## Running

From the sst-hgcc source tree (after `make` in the build directory):

```bash
lit -v tests/lit-tests
```

Or from the build directory:

```bash
make check
```

## Layout

```
core/      tests for core src2src transforms (main refactor, globals, TLS,
           replacement-header baseline)
pragmas/   one test per supported #pragma sst directive
```
Each test creates a scratch dir at `%t.d` (e.g.
`tests/lit-tests/Output/pragmas/delete.cc.tmp.d/`). Inspect the preserved
`sst.pp.*.cc` file there to see exactly what the rewriter emitted.
