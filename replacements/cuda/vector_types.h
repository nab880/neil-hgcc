/* dim3 with 1-filling constructor for <<<>>> configs. */
#ifndef HGCC_CUDA_VECTOR_TYPES_H
#define HGCC_CUDA_VECTOR_TYPES_H

struct uint3 {
  unsigned int x, y, z;
};

/* Common CUDA vector types. Simulation never executes device arithmetic, so
 * plain aggregates suffice -- they exist only so host-visible signatures using
 * them (kernel params, sizeof) parse. Extend as workloads require. */
struct float1 { float x; };
struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
struct double1 { double x; };
struct double2 { double x, y; };
struct double3 { double x, y, z; };
struct double4 { double x, y, z, w; };
struct int1 { int x; };
struct int2 { int x, y; };
struct int3 { int x, y, z; };
struct int4 { int x, y, z, w; };

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
