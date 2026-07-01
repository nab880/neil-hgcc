/* Minimal CUDA driver types for the spike. */
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
