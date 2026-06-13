/* Replacement <vector_types.h> for CUDA host-only simulation builds.
 * dim3 with the usual 1-filling constructor so `dim3 block(256)` and the
 * implicit conversions in <<<>>> configs parse. Promoted from the Phase-0
 * spike stub (spike/stubs/vector_types.h). */
#ifndef HGCC_CUDA_VECTOR_TYPES_H
#define HGCC_CUDA_VECTOR_TYPES_H

struct uint3 {
  unsigned int x, y, z;
};

struct dim3 {
  unsigned int x, y, z;
#if defined(__cplusplus)
  constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
      : x(vx), y(vy), z(vz) {}
  constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
  constexpr operator uint3() const { return uint3{x, y, z}; }
#endif
};

#endif
