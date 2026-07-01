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
#ifndef bin_clang_codeBuilder_h
#define bin_clang_codeBuilder_h

#include <sstream>
#include <string>

#include "clangHeaders.h"  // clang::SourceLocation
#include "util.h"          // errorAbort

// Accumulates generated source text while tracking bracket balance.
//
// This does NOT parse C++ -- it only guards the structural invariant that every
// scope the rewrite *opens through this class* is closed, catching the most
// common string-codegen bug (a missing/extra brace) at emit time with a located
// error instead of a downstream "expected '}'" in machine-generated sst.pp.*.
//
// Usage mirrors the existing `os << "..."` style; route anything that opens a
// block/statement-expr/lambda through the scope helpers so the balance counters
// stay accurate. Leaf tokens (identifiers, expressions, spliced strings) go
// through operator<< and are NOT scanned -- scanning them would false-positive on
// brackets inside embedded expressions and string literals.
//
//   CodeBuilder b(getStart(expr), "rewriteCudaLaunch");
//   b.openStmtExpr();                        // "({ "
//   b << "dim3 g = (" << gridStr << "); ";
//   b.openLambda();                          // "[&]{ "
//     b.openBlock(); b << derivedAccum; b.closeBlock();
//     b << "return ...; ";
//   b.closeLambda(); b << "(); ";            // caller emits the "()" call
//   b << "sst_hg_cuda_launch(...); ";
//   b.closeStmtExpr();                       // " })"
//   ::replace(expr, b.take());               // asserts all scopes closed
class CodeBuilder {
 public:
  CodeBuilder(clang::SourceLocation loc, const char* where)
    : loc_(loc), where_(where) {}

  // Raw append for leaf tokens; not scanned for brackets (see class comment).
  template <class T>
  CodeBuilder& operator<<(const T& v) { os_ << v; return *this; }

  void openBlock()  { os_ << "{ ";  ++brace_; }
  void closeBlock() {
    failIf(brace_ == 0, "closeBlock() with no open '{'");
    os_ << "} "; --brace_;
  }

  // GNU statement-expression: opens one paren and one brace together.
  void openStmtExpr()  { os_ << "({ "; ++paren_; ++brace_; }
  void closeStmtExpr() {
    failIf(paren_ == 0 || brace_ == 0, "closeStmtExpr() with no open '({'");
    os_ << "})"; --paren_; --brace_;
  }

  // Lambda body "[&]{ ... }"; the trailing "()" invocation is caller-emitted.
  void openLambda()  { os_ << "[&]{ "; ++brace_; }
  void closeLambda() {
    failIf(brace_ == 0, "closeLambda() with no open '{'");
    os_ << "} "; --brace_;
  }

  // Finalize: assert everything opened was closed, then hand off the string.
  std::string take() {
    failIf(brace_ != 0, "unbalanced '{' (depth " + std::to_string(brace_)
                        + ") at end of emission");
    failIf(paren_ != 0, "unbalanced '(' (depth " + std::to_string(paren_)
                        + ") at end of emission");
    return os_.str();
  }

 private:
  void failIf(bool bad, const std::string& msg) {
    if (bad) errorAbort(loc_, "internal codegen error in "
                              + std::string(where_) + ": " + msg);
  }

  std::ostringstream os_;
  clang::SourceLocation loc_;
  const char* where_;
  int brace_ = 0;
  int paren_ = 0;
};

#endif
