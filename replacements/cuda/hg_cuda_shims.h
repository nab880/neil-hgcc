/* Launch-configuration hooks for CUDA host-only parsing. With no CUDA SDK
 * detected, Sema resolves cudaConfigureCall when it builds a
 * CUDAKernelCallExpr for <<<>>> (FINDINGS.md finding 3). These are
 * declarations only: the rewriter (Phase 2 Track 2) lowers every launch to
 * sst_hg_cuda_launch, so a configure call never survives to codegen or link.
 * Both the legacy (cudaConfigureCall) and newer
 * (__cudaPush/PopCallConfiguration) launch ABIs are declared so a future LLVM
 * flipping its default launch lowering still parses. */
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
