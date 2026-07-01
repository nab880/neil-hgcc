/* Builtin vars + device intrinsics for parse-only kernel bodies. */
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

/* Guard device intrinsics to __CUDA__ so host -c does not warn. */
#if defined(__CUDA__)
extern "C" __device__ void __syncthreads(void);
extern "C" __device__ void __threadfence(void);
#endif

#endif
