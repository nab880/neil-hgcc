/* Spike stub <cuda_runtime.h> for toolkit-free host-only parsing. */
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

/* Launch-config hooks (C++ default args, outside block above); Sema resolves one for CUDAKernelCallExpr. */
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
