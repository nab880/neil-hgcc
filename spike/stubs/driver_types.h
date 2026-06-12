/* Spike stub for the CUDA toolkit's driver_types.h: just the types vecadd.cu
 * and the launch machinery need. Error paths do not exist in the model
 * (CUDA_PLAN.md §8), so cudaError_t is minimal. */
#ifndef SPIKE_DRIVER_TYPES_H
#define SPIKE_DRIVER_TYPES_H

typedef enum cudaError {
  cudaSuccess = 0,
  cudaErrorUnknown = 999
} cudaError_t;

enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
};

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

#endif
