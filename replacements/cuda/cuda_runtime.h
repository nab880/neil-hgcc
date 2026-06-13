/* Replacement <cuda_runtime.h> — the umbrella header CUDA user code includes,
 * for toolkit-free host-only simulation builds (CUDA_PLAN.md D2). It owns:
 *
 *   1. the function-type qualifier macros (__global__ et al.); clang's CUDA
 *      mode treats them as plain attribute macros, not keywords;
 *   2. the CUDA types (driver_types.h / vector_types.h);
 *   3. the builtin variables (device_launch_parameters.h);
 *   4. the launch-configuration hooks Sema resolves for <<<>>> (hg_cuda_shims.h);
 *   5. the runtime API, lowered to the sst_hg_cuda ABI (hg_cuda.h) by the
 *      static inline wrappers below. The rewriter (Phase 2 Track 2) handles
 *      only <<<>>> and kernel bodies; ALL runtime API lowering happens here.
 *
 * __host__ and __device__ are pre-defined on the command line by the hgcc
 * driver (_cuda_lang_flags); the macro block below redefines them to the same
 * tokens (a benign identical redefinition) and adds the rest. Keep these
 * spellings in sync with the driver. */
#ifndef HGCC_CUDA_RUNTIME_H
#define HGCC_CUDA_RUNTIME_H

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

#include <stddef.h>

#include "driver_types.h"
#include "vector_types.h"
#include "device_launch_parameters.h"
#include "hg_cuda_shims.h"

#include <hgcc/libraries/hg_cuda.h>

/* Runtime API -> sst_hg_cuda ABI. static inline so the host C++ compile
 * inlines each call away (e.g. cudaMalloc((void**)&p, n) nets
 * p = sst_hg_cuda_malloc(n)); no external cuda* symbol is emitted. All return
 * cudaSuccess: the model has no error paths (CUDA_PLAN.md §8). */

static inline cudaError_t cudaMalloc(void** devPtr, size_t size) {
  *devPtr = sst_hg_cuda_malloc(size);
  return cudaSuccess;
}

static inline cudaError_t cudaFree(void* devPtr) {
  sst_hg_cuda_free(devPtr);
  return cudaSuccess;
}

static inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count,
                                     cudaMemcpyKind kind) {
  sst_hg_cuda_memcpy(dst, src, count, (int)kind, 0);
  return cudaSuccess;
}

static inline cudaError_t cudaMemcpyAsync(void* dst, const void* src,
                                          size_t count, cudaMemcpyKind kind,
                                          cudaStream_t stream) {
  sst_hg_cuda_memcpy(dst, src, count, (int)kind, (void*)stream);
  return cudaSuccess;
}

static inline cudaError_t cudaDeviceSynchronize(void) {
  sst_hg_cuda_device_sync();
  return cudaSuccess;
}

static inline cudaError_t cudaStreamCreate(cudaStream_t* stream) {
  *stream = (cudaStream_t)sst_hg_cuda_stream_create();
  return cudaSuccess;
}

static inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  sst_hg_cuda_stream_destroy((void*)stream);
  return cudaSuccess;
}

static inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  sst_hg_cuda_stream_sync((void*)stream);
  return cudaSuccess;
}

static inline cudaError_t cudaEventCreate(cudaEvent_t* event) {
  *event = (cudaEvent_t)sst_hg_cuda_event_create();
  return cudaSuccess;
}

/* No sst_hg_cuda_event_destroy in the v1 ABI: event ids are trivial and not
 * reclaimed, so destruction is a no-op. */
static inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
  (void)event;
  return cudaSuccess;
}

static inline cudaError_t cudaEventRecord(cudaEvent_t event,
                                          cudaStream_t stream = 0) {
  sst_hg_cuda_event_record((void*)event, (void*)stream);
  return cudaSuccess;
}

static inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  sst_hg_cuda_event_sync((void*)event);
  return cudaSuccess;
}

static inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start,
                                               cudaEvent_t stop) {
  *ms = sst_hg_cuda_event_elapsed_ms((void*)start, (void*)stop);
  return cudaSuccess;
}

static inline cudaError_t cudaGetDeviceCount(int* count) {
  *count = sst_hg_cuda_get_device_count();
  return cudaSuccess;
}

static inline cudaError_t cudaSetDevice(int dev) {
  sst_hg_cuda_set_device(dev);
  return cudaSuccess;
}

static inline cudaError_t cudaGetLastError(void) {
  return cudaSuccess;
}

static inline const char* cudaGetErrorString(cudaError_t error) {
  (void)error;
  return "no error";
}

#endif
