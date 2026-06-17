/* Minimal CUDA driver types for toolkit-free host-only builds. */
#ifndef HGCC_CUDA_DRIVER_TYPES_H
#define HGCC_CUDA_DRIVER_TYPES_H

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

/* Minimal cudaDeviceProp for nominal runtime wrapper values. */
struct cudaDeviceProp {
  char name[256];
  unsigned long totalGlobalMem;
  unsigned long sharedMemPerBlock;
  int warpSize;
  int maxThreadsPerBlock;
  int multiProcessorCount;
  int clockRate;
  int major;
  int minor;
};

#endif
