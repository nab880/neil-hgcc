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
#ifndef bin_clang_gpuPragma_h
#define bin_clang_gpuPragma_h

#include "pragmas.h"
#include "gpuCostModel.h"

#include <array>
#include <list>
#include <map>
#include <string>

/* Per-thread cost expressions passed to sst_hg_cuda_launch, one per
 * kGpuCostDims entry, in table order (default "0"). */
struct GpuComputeCost {
  std::array<std::string, kNumGpuCostDims> exprs;
  GpuComputeCost() { exprs.fill("0"); }
};

/* #pragma sst gpu_compute: attach manual costs to a kernel decl or launch stmt. */
class SSTGpuComputePragma : public SSTPragma {
 public:
  SSTGpuComputePragma(clang::SourceLocation loc,
                      std::map<std::string, std::list<std::string>>&& args);

  bool firstPass() const override { return true; }

  void activate(clang::Stmt* s) override;
  void activate(clang::Decl* d) override;

  static const GpuComputeCost* costForDecl(const clang::Decl* d);
  static const GpuComputeCost* costForStmt(const clang::Stmt* s);

 private:
  GpuComputeCost cost_;
};

#endif
