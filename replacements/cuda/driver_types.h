/* Replacement <driver_types.h> for CUDA host-only simulation builds: the
 * types the runtime API and launch machinery need. The model has no error
 * paths (CUDA_PLAN.md §8), so cudaError_t is minimal. Promoted from the
 * Phase-0 spike stub (spike/stubs/driver_types.h), plus cudaDeviceProp. */
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

/* Minimal cudaDeviceProp: the commonly-read fields. The simulator does not
 * model a real device, so the runtime wrapper fills only nominal values. */
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
