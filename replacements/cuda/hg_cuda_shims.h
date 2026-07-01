/* Launch-config hooks (C++ default args); Sema resolves one for CUDAKernelCallExpr. */
#ifndef HGCC_CUDA_SHIMS_H
#define HGCC_CUDA_SHIMS_H

#include "driver_types.h"   /* cudaError_t, cudaStream_t */
#include "vector_types.h"   /* dim3 */
#include <stddef.h>         /* size_t */

/* These hooks use C++ default arguments, so the whole block is C++-only. The
 * header is reached only from cuda_runtime.h under -x cuda (always C++); guard
 * the body so a stray C include is a clean no-op rather than a parse error. */
#ifdef __cplusplus
extern "C" {

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

}  /* extern "C" */
#endif

#endif
