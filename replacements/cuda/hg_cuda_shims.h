/* Launch-config hooks (C++ default args); Sema resolves one for CUDAKernelCallExpr. */
#ifndef HGCC_CUDA_SHIMS_H
#define HGCC_CUDA_SHIMS_H

#include "driver_types.h"   /* cudaError_t, cudaStream_t */
#include "vector_types.h"   /* dim3 */
#include <stddef.h>         /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                              size_t sharedMem = 0,
                              cudaStream_t stream = 0);
unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                     size_t sharedMem = 0,
                                     cudaStream_t stream = 0);
cudaError_t __cudaPopCallConfiguration(dim3* gridDim,
                                       dim3* blockDim,
                                       size_t* sharedMem,
                                       void* stream);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif
