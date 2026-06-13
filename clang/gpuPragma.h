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

#include <list>
#include <map>
#include <string>

/** Per-launch cost vector the rewriter passes to sst_hg_cuda_launch. Each field
 * is a source expression string over kernel parameters/builtins (P2 commonly a
 * literal; P4 fills derived expressions). Default "0". */
struct GpuComputeCost {
  std::string flops = "0";
  std::string intops = "0";
  std::string bytesRead = "0";
  std::string bytesWritten = "0";
};

/** #pragma sst gpu_compute flops(<e>) intops(<e>) read(<e>) write(<e>)
 *
 * Attaches a manual cost vector to a kernel FunctionDecl or to a launch
 * statement; TraverseCUDAKernelCallExpr reads it when lowering the launch.
 * Runs in the first pass so a kernel's costs are recorded before any launch of
 * it is rewritten in the main pass. */
class SSTGpuComputePragma : public SSTPragma {
 public:
  SSTGpuComputePragma(clang::SourceLocation loc,
                      std::map<std::string, std::list<std::string>>&& args);

  bool firstPass() const override { return true; }

  void activate(clang::Stmt* s) override;
  void activate(clang::Decl* d) override;

  /** Cost attached to a kernel decl (keyed by canonical decl), or null. */
  static const GpuComputeCost* costForDecl(const clang::Decl* d);
  /** Cost attached to a specific launch statement, or null. */
  static const GpuComputeCost* costForStmt(const clang::Stmt* s);

 private:
  GpuComputeCost cost_;
};

#endif
