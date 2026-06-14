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

#include "gpuPragma.h"
#include "clangHeaders.h"
#include "util.h"

using namespace clang;

namespace {
/* Costs outlive the pragma objects: activate() runs in the first pass, the
 * launch rewrite reads them in the main pass. Keyed by canonical decl (a kernel
 * may be redeclared) and by launch statement. */
std::map<const Decl*, GpuComputeCost> g_declCosts;
std::map<const Stmt*, GpuComputeCost> g_stmtCosts;

std::string takeFront(std::map<std::string, std::list<std::string>>& args,
                      const char* key) {
  auto it = args.find(key);
  if (it == args.end()) return std::string();
  std::string v = it->second.empty() ? std::string() : it->second.front();
  args.erase(it);
  return v;
}
}

SSTGpuComputePragma::SSTGpuComputePragma(
    SourceLocation loc, std::map<std::string, std::list<std::string>>&& in_args)
{
  auto args = in_args;
  std::string f = takeFront(args, "flops");
  std::string i = takeFront(args, "intops");
  std::string r = takeFront(args, "read");
  std::string w = takeFront(args, "write");
  if (!f.empty()) cost_.flops = f;
  if (!i.empty()) cost_.intops = i;
  if (!r.empty()) cost_.bytesRead = r;
  if (!w.empty()) cost_.bytesWritten = w;
  if (!args.empty()){
    errorAbort(loc, "invalid #pragma sst gpu_compute clause; "
                    "allowed: flops, intops, read, write");
  }
}

void SSTGpuComputePragma::activate(Decl* d)
{
  g_declCosts[d->getCanonicalDecl()] = cost_;
}

void SSTGpuComputePragma::activate(Stmt* s)
{
  g_stmtCosts[s] = cost_;
}

const GpuComputeCost* SSTGpuComputePragma::costForDecl(const Decl* d)
{
  if (!d) return nullptr;
  auto it = g_declCosts.find(d->getCanonicalDecl());
  return it == g_declCosts.end() ? nullptr : &it->second;
}

const GpuComputeCost* SSTGpuComputePragma::costForStmt(const Stmt* s)
{
  auto it = g_stmtCosts.find(s);
  return it == g_stmtCosts.end() ? nullptr : &it->second;
}

// Active in every mode: the launch rewrite (TraverseCUDAKernelCallExpr) reads
// these costs in any CUDA build, including the default ENCAPSULATE mode, so the
// pragma must record them there too -- not only under skeletonize.
static PragmaRegister<SSTArgMapPragmaShim, SSTGpuComputePragma, true>
    gpuComputePragma("sst", "gpu_compute",
                     modes::ENCAPSULATE | modes::MEMOIZE | modes::SKELETONIZE
                     | modes::SHADOWIZE | modes::PUPPETIZE);
