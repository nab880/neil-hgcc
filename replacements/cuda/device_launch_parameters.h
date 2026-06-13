/* Replacement <device_launch_parameters.h> for CUDA host-only simulation
 * builds: builtin variables + the bare device intrinsics a kernel body needs
 * to *parse*. Kernel bodies are stripped by the rewriter, so nothing here is
 * ever codegen'd or executed. Promoted from spike/stubs.
 *
 * threadIdx/blockIdx/blockDim/gridDim come from clang's own resource-dir
 * header __clang_cuda_builtin_vars.h, which ships with LLVM (not the CUDA
 * toolkit) and is available under -nocudainc. If a clang lacks it, fall back
 * to plain extern declarations — parse-only equivalence is all we need. */
#ifndef HGCC_CUDA_DEVICE_LAUNCH_PARAMETERS_H
#define HGCC_CUDA_DEVICE_LAUNCH_PARAMETERS_H

#if defined(__clang__) && defined(__CUDA__)
#if __has_include(<__clang_cuda_builtin_vars.h>)
#include <__clang_cuda_builtin_vars.h>
#else
#include "vector_types.h"
extern const uint3 threadIdx;
extern const uint3 blockIdx;
extern const dim3 blockDim;
extern const dim3 gridDim;
extern const int warpSize;
#endif
#endif

/* Device intrinsics only need to *parse* inside kernel bodies (CUDA mode).
 * The rewriter strips those bodies before host -c, so guarding to __CUDA__
 * keeps the __device__ attribute off the plain-C++ host compile (where clang
 * warns 'device attribute ignored'). */
#if defined(__CUDA__)
extern "C" __device__ void __syncthreads(void);
extern "C" __device__ void __threadfence(void);
#endif

#endif
