

dnl CHECK_CUDA_HOSTONLY: probe whether the --with-clang clang++ can parse CUDA
dnl host-only with no CUDA toolkit installed (CUDA_PLAN.md design decision D2,
dnl proven by the Phase-0 spike). The conftest TU exercises the full P0 surface:
dnl the qualifier attribute macros, dim3, the legacy cudaConfigureCall launch
dnl hook (the one Sema resolves when no CUDA SDK is detected), a __global__
dnl kernel, and a <<<1,1>>> launch. Substitutes HAVE_CUDA_SUPPORT (True/False
dnl for hgccvars.py) and DEFAULT_CUDA_ARCH.

AC_DEFUN([CHECK_CUDA_HOSTONLY], [

  AC_ARG_WITH([cuda-arch],
    [AS_HELP_STRING(
        [--with-cuda-arch@<:@=ARCH@:>@],
        [Default --cuda-gpu-arch hg++ passes for CUDA host-only parsing (default: sm_70)]
      )
    ],
    [default_cuda_arch=$withval],
    [default_cuda_arch=sm_70]
  )
  AC_SUBST([DEFAULT_CUDA_ARCH], [$default_cuda_arch])

  AC_MSG_CHECKING([whether $CLANG_INSTALL_DIR/bin/clang++ parses CUDA host-only without a toolkit])
  have_cuda_support=no
  if test "X$found_clang" = "Xyes" && test -x "$CLANG_INSTALL_DIR/bin/clang++"; then
    cudaprobe=conftest_cuda$$.cu
    trap 'rm -f "$cudaprobe"' 0 1 2 13 15
    cat > "$cudaprobe" <<'_HGCC_CUDA_PROBE_EOF'
#include <stddef.h>
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
struct dim3 {
  unsigned x, y, z;
  dim3(unsigned vx = 1, unsigned vy = 1, unsigned vz = 1)
      : x(vx), y(vy), z(vz) {}
};
typedef enum cudaError { cudaSuccess = 0 } cudaError_t;
typedef struct CUstream_st* cudaStream_t;
extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem = 0,
                                         cudaStream_t stream = 0);
__global__ void hgconf_kern(int* p) { *p = 1; }
int main() {
  int* p = 0;
  hgconf_kern<<<1, 1>>>(p);
  return 0;
}
_HGCC_CUDA_PROBE_EOF
    if "$CLANG_INSTALL_DIR/bin/clang++" -x cuda --cuda-host-only -nocudainc \
         -nocudalib "--cuda-gpu-arch=$default_cuda_arch" -std=c++17 \
         -fsyntax-only "$cudaprobe" >/dev/null 2>&1; then
      have_cuda_support=yes
    fi
    rm -f "$cudaprobe"
  fi
  AC_MSG_RESULT([$have_cuda_support])

  if test "X$have_cuda_support" = "Xyes"; then
    AC_SUBST([HAVE_CUDA_SUPPORT], [True])
  else
    AC_SUBST([HAVE_CUDA_SUPPORT], [False])
  fi
  dnl Makefile gate for the CUDA integration test (tests/Makefile.am).
  AM_CONDITIONAL([HAVE_CUDA_SUPPORT], [test "X$have_cuda_support" = "Xyes"])
])
