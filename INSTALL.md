# Installing SST-HGCC

For full documentation (Mercury app examples, pragma reference, configuration
options, and code layout), see [README.md](README.md).

## Prerequisites

Install these **before** sst-hgcc:

1. **Autotools** — `autoconf`, `automake`, `libtool` (for `./autogen.sh`)
2. **SST Core** — provides `sst-config` on your `PATH`
3. **sst-elements** — Mercury/HG element (built with C++17)
4. **LLVM 22** — with libTooling (required for the `ssthg_clang` rewriter)

Use any current LLVM **22.x** source release (example below pins 22.1.8), or on
macOS install LLVM 22 from Homebrew (see below). After installing LLVM, keep its
`bin` directory on your `PATH` while configuring and building SST Core,
sst-elements, and sst-hgcc.

### Homebrew LLVM 22 (macOS alternative)

Skip the source build below if Homebrew already provides LLVM 22 with libTooling:

```bash
brew install llvm@22

LLVM_PREFIX="$(brew --prefix llvm@22)"
export PATH="${LLVM_PREFIX}/bin:$PATH"
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
export LDFLAGS="-fuse-ld=lld"

# Optional: lit tests and make check
export FILECHECK="${LLVM_PREFIX}/bin/FileCheck"
export LLVM_ROOT="${LLVM_PREFIX}"
```

When configuring sst-hgcc, pass `--with-clang="${LLVM_PREFIX}"` (same as in the
full recipe below). Homebrew installs `clang`, `clang++`, `lld`, and `FileCheck`
under that prefix; use those compilers for SST Core and sst-elements as well.

## Build and install

```bash
LLVM_VER=22.1.8
LLVM_SRC=llvm-project-${LLVM_VER}.src
LLVM_PREFIX=$HOME/${LLVM_SRC}/install

curl -LO "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VER}/${LLVM_SRC}.tar.xz"
tar -xf "${LLVM_SRC}.tar.xz"
cd "${LLVM_SRC}"
mkdir build && cd build

cmake -S ../llvm -B . \
  -DLLVM_ENABLE_PROJECTS="clang;compiler-rt;lld" \
  -DLLVM_ENABLE_RUNTIMES=all \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX="${LLVM_PREFIX}" \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_TARGETS_TO_BUILD=host \
  -G Ninja

ninja -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" && ninja install

export PATH="${LLVM_PREFIX}/bin:$PATH"
# macOS only — set before configuring SST Core, sst-elements, and sst-hgcc
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
export LDFLAGS="-fuse-ld=lld"

cd

# Install SST Core (required — sst-config must be on PATH)
git clone https://github.com/sstsimulator/sst-core.git
cd sst-core
./autogen.sh
mkdir build && cd build
../configure CXX=clang++ CC=clang \
  --with-std=17 \
  --prefix=$HOME/sst-core/install
make -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" && make install
export PATH=$HOME/sst-core/install/bin:$PATH

cd

# Install sst-elements
git clone https://github.com/sstsimulator/sst-elements.git
cd sst-elements
./autogen.sh
mkdir build && cd build

../configure CXX=clang++ CC=clang \
  --with-std=17 \
  --prefix=$HOME/sst-elements/install \
  --with-sst-core=$HOME/sst-core/install

make -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" && make install

cd

# Install sst-hgcc
git clone https://github.com/sstsimulator/sst-hgcc.git
cd sst-hgcc
./autogen.sh
mkdir build && cd build

../configure CXX=clang++ CC=clang \
  --with-std=17 \
  --prefix=$HOME/sst-hgcc/install \
  --with-sst-core=$HOME/sst-core/install \
  --with-sst-elements=$HOME/sst-elements/install \
  --with-clang="${LLVM_PREFIX}"

make -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" && make install

export PATH=$HOME/sst-hgcc/install/bin:$PATH

# Verify the install
hg++ --version
make check          # lit rewriter tests (optional; needs lit + FileCheck)
make install        # installs libtest_tls.so for the integration test
make installcheck   # SST integration test (requires sst on PATH)
```

### Expected `make installcheck` output

From the build directory, after `make install`:

```
my_global: 1
my_global: 1
Simulation is complete, simulated time: ...
```

The integration test runs `tests/test_tls.py` with `libtest_tls.so` installed to
the path reported by `sst-config SST_ELEMENT_LIBRARY SST_ELEMENT_LIBRARY_EXT_LIBDIR`.
