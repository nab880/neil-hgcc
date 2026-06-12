/*
 * Hand-written expected output of the Phase-2 rewriter for spike/vecadd.cu —
 * the contract the ssthg_clang work is built to satisfy. Compiled as plain
 * C++ (-x c++) against spike/stub_hg_cuda.cc by probe 5.
 *
 * What the rewriter did (per CUDA_PLAN.md §5/§7):
 *   - __global__ vecAdd: body stripped after cost extraction; nothing of the
 *     kernel survives into the C++ output.
 *   - vecAdd<<<grid, block>>>(da, db, dc, n) -> sst_hg_cuda_launch(...) with
 *     the kernel's mangled name and per-thread costs. Costs here are
 *     hand-derived (P4 wires ComputeVisitor): c[i] = a[i] + b[i] under a
 *     bounds check is 1 flop, ~3 int ops (index math + compare), 8 bytes
 *     read, 4 bytes written per thread.
 *   - cudaMalloc/cudaMemcpy/cudaDeviceSynchronize/cudaFree arrive here
 *     already mapped to sst_hg_cuda_* by the replacement <cuda_runtime.h>,
 *     which -E inlines; dim3 below stands in for the same inlining.
 *   - main is NOT renamed here: the sst_hg_user_main_* rename and the
 *     sstGlobals companion TU are existing pipeline behavior, untouched by
 *     CUDA mode and out of spike scope.
 */
#include "hg_cuda.h"

#include <cstdio>
#include <cstdlib>

/* inlined by -E from the replacement vector_types.h */
struct dim3 {
  unsigned int x, y, z;
  constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
      : x(vx), y(vy), z(vz) {}
};

/* __global__ void vecAdd(const float*, const float*, float*, int)
   { body stripped by ssthg_clang } */

int main(int argc, char** argv)
{
  const int n = 1 << 16;
  const size_t bytes = n * sizeof(float);

  float* ha = (float*)malloc(bytes);
  float* hb = (float*)malloc(bytes);
  float* hc = (float*)malloc(bytes);
  for (int i = 0; i < n; ++i) {
    ha[i] = (float)i;
    hb[i] = (float)(n - i);
  }

  float* da;
  float* db;
  float* dc;
  da = (float*)sst_hg_cuda_malloc(bytes);
  db = (float*)sst_hg_cuda_malloc(bytes);
  dc = (float*)sst_hg_cuda_malloc(bytes);

  sst_hg_cuda_memcpy(da, ha, bytes, 1 /*cudaMemcpyHostToDevice*/, 0);
  sst_hg_cuda_memcpy(db, hb, bytes, 1 /*cudaMemcpyHostToDevice*/, 0);

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  sst_hg_cuda_launch("_Z6vecAddPKfS0_Pfi",
                     grid.x, grid.y, grid.z,
                     block.x, block.y, block.z,
                     0, 0,
                     /*flops=*/1, /*intops=*/3,
                     /*bytesRead=*/8, /*bytesWritten=*/4);
  sst_hg_cuda_device_sync();

  sst_hg_cuda_memcpy(hc, dc, bytes, 2 /*cudaMemcpyDeviceToHost*/, 0);

  sst_hg_cuda_free(da);
  sst_hg_cuda_free(db);
  sst_hg_cuda_free(dc);

  printf("vecadd done n=%d c[0]=%f\n", n, hc[0]);
  free(ha);
  free(hb);
  free(hc);
  return 0;
}
