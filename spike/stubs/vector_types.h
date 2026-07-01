/* dim3 with 1-filling constructor. */
#ifndef SPIKE_VECTOR_TYPES_H
#define SPIKE_VECTOR_TYPES_H

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
