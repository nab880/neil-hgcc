/* Spike stub for <cuda_runtime.h> — the umbrella header user code includes.
 *
 * Owns everything the CUDA toolkit's headers normally provide, because the
 * Phase-0 premise (CUDA_PLAN.md D2) is parsing with -nocudainc -nocudalib
 * and no toolkit installed:
 *
 *   1. the function-type qualifier macros (__global__ et al.) that normally
 *      come from host_defines.h — clang's CUDA mode treats them as plain
 *      attribute macros, not keywords;
 *   2. the types (driver_types.h / vector_types.h stubs);
 *   3. the builtin variables (device_launch_parameters.h stub);
 *   4. the runtime API declarations, all trivially cudaSuccess;
 *   5. the launch-configuration hooks Sema/CodeGen resolve for <<<>>>.
 *      Both the legacy (cudaConfigureCall) and the new
 *      (__cudaPushCallConfiguration) launch ABIs are declared; which one a
 *      given LLVM picks without a detected CUDA SDK is exactly what the
 *      run.sh matrix records.
 */
#ifndef SPIKE_CUDA_RUNTIME_H
#define SPIKE_CUDA_RUNTIME_H

#ifndef __global__
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#endif

#include "driver_types.h"
#include "vector_types.h"
#include "device_launch_parameters.h"

#include <stddef.h>

extern "C" {

cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count,
                       cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaStreamCreate(cudaStream_t* stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaGetLastError(void);
const char* cudaGetErrorString(cudaError_t error);

} /* extern "C" */

/* Launch-configuration hooks (C++ default args, so outside the block above).
 * Sema resolves one of these when it builds a CUDAKernelCallExpr. */
extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem = 0,
                                         cudaStream_t stream = 0);
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                size_t sharedMem = 0,
                                                cudaStream_t stream = 0);
extern "C" cudaError_t __cudaPopCallConfiguration(dim3* gridDim,
                                                  dim3* blockDim,
                                                  size_t* sharedMem,
                                                  void* stream);

#endif
