/**
Copyright 2009-2023 National Technology and Engineering Solutions of Sandia,
LLC (NTESS).  Under the terms of Contract DE-NA-0003525, the U.S. Government
retains certain rights in this software.

Sandia National Laboratories is a multimission laboratory managed and operated
by National Technology and Engineering Solutions of Sandia, LLC., a wholly
owned subsidiary of Honeywell International, Inc., for the U.S. Department of
Energy's National Nuclear Security Administration under contract DE-NA0003525.

Copyright (c) 2009-2023, NTESS

All rights reserved.

Questions? Contact sst-macro-help@sandia.gov
*/
#ifndef bin_clang_gpuCostModel_h
#define bin_clang_gpuCostModel_h

#include <array>
#include <cstddef>

// THE single source of truth for GPU per-thread cost dimensions.
//
// To add a dimension (e.g. tensor-core ops), add one row to kGpuCostDims below;
// every table-driven emitter -- the #pragma sst gpu_compute parser, the derived
// cost accumulation, and the CUDA launch rewrite -- picks it up automatically.
//
// The row ORDER defines the trailing argument order of sst_hg_cuda_launch (the
// cost args after the stream pointer). That C ABI is the ONE consumer that can't
// iterate this table; a static_assert next to it (hg_cuda.h) trips if the count
// changes, forcing a deliberate ABI bump rather than silent drift.
struct GpuCostDim {
  const char* accumVar;   // accumulator variable name in generated code
  const char* pragmaKey;  // clause key in "#pragma sst gpu_compute <key>(...)"
};

inline constexpr std::array<GpuCostDim, 4> kGpuCostDims{{
  {"flops",      "flops"},
  {"intops",     "intops"},
  {"readBytes",  "read"},
  {"writeBytes", "write"},
}};

inline constexpr std::size_t kNumGpuCostDims = kGpuCostDims.size();

// Named indices into a per-dimension array (kept in sync with the rows above).
// These let the ComputeVisitor increment counts by meaning ("this is a flop")
// while everything else iterates the table.
enum GpuCostIdx {
  GPU_COST_FLOPS = 0,
  GPU_COST_INTOPS,
  GPU_COST_READ_BYTES,
  GPU_COST_WRITE_BYTES,
};

#endif
