/* Spike stub for device_launch_parameters.h: builtin variables + the bare
 * device intrinsics a kernel body needs to *parse* (bodies are stripped by
 * the rewriter; nothing here is ever codegen'd or executed).
 *
 * threadIdx/blockIdx/blockDim/gridDim come from clang's own resource-dir
 * header __clang_cuda_builtin_vars.h, which ships with LLVM (not the CUDA
 * toolkit) and is therefore available under -nocudainc. If a probed LLVM
 * version lacks it, fall back to plain extern declarations — parse-only
 * equivalence is all the spike needs.
 */
#ifndef SPIKE_DEVICE_LAUNCH_PARAMETERS_H
#define SPIKE_DEVICE_LAUNCH_PARAMETERS_H

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

extern "C" __device__ void __syncthreads(void);
extern "C" __device__ void __threadfence(void);

#endif
