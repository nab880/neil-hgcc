/* Launch-config hooks Sema needs for <<<>>>; rewriter lowers launches. */
#ifndef HGCC_CUDA_SHIMS_H
#define HGCC_CUDA_SHIMS_H

#include "driver_types.h"   /* cudaError_t, cudaStream_t */
#include "vector_types.h"   /* dim3 */
#include <stddef.h>         /* size_t */

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
