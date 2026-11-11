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

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Questions? Contact sst-macro-help@sandia.gov
*/

#include "computePragma.h"
#include "clangHeaders.h"
#include "computeVisitor.h"
#include "dataFlow.h"
#include "computeLoops.h"
#include "validateScope.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/Sema/Sema.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/Specifiers.h"
#include <sstream>

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

#if CLANG_VERSION_MAJOR >= 7
  #define UnaryOperatorCtor(...) UnaryOperator(__VA_ARGS__, false)
#else
  #define UnaryOperatorCtor(...) UnaryOperator(__VA_ARGS__)
#endif

#if CLANG_VERSION_MAJOR >= 5
  #define BinaryOperatorCtor(...) BinaryOperator(__VA_ARGS__, FPOptions())
#else
  #define BinaryOperatorCtor(...) BinaryOperator(__VA_ARGS__, false)
#endif

SSTLoopCountPragma::SSTLoopCountPragma(SourceLocation loc, const std::list<clang::Token>& tokens)
{
  if (tokens.size() != 1){
    errorAbort(loc, "loop_count pragma should taken a single token as the pragma argument");
  }
  loopCountToken_ = tokens.front();
  loopCount_ = getSingleString(tokens);
}

void
SSTLoopCountPragma::transformWhileLoop(Stmt* s)
{
  WhileStmt* ws = cast<WhileStmt>(s);
  FunctionDecl* fd = CompilerGlobals::astContextLists.enclosingFunctionDecls.back();
  Expr* loop_count = tokenToExpr(fd, loopCountToken_, getStart(ws->getCond()));
  IdentifierInfo* II = &CompilerGlobals::CI().getPreprocessor().getIdentifierTable().get("loop_variable");
  TypeSourceInfo *TInfo = fd->getASTContext().getTrivialTypeSourceInfo(loop_count->getType(), getStart(ws->getCond()));
  VarDecl* loopVar = VarDecl::Create(fd->getASTContext(), fd,
                                     getStart(ws->getCond()), getEnd(ws->getCond()),
                                     II, fd->getASTContext().IntTy, TInfo,
                                     SC_None);
  llvm::APInt api(32,0,true);
  IntegerLiteral* zero = IntegerLiteral::Create(fd->getASTContext(), api,
                                                fd->getASTContext().IntTy, getEnd(ws->getCond()));
  loopVar->setInit(zero);
  DeclGroupRef dgr(loopVar);
  DeclStmt* init_stmt = new (fd->getASTContext()) DeclStmt(dgr, getStart(ws->getCond()), getEnd(ws->getCond()));

  DeclRefExpr* cond_ref = DeclRefExpr::Create(
     fd->getASTContext(),
     loopVar->getQualifierLoc(),
     SourceLocation(),
     loopVar,
     false,
     getStart(ws->getCond()),
     loopVar->getType(),
     VK_LValue,
     loopVar);

  auto &Ctx = fd->getASTContext();

  BinaryOperator *cond_stmt = BinaryOperator::Create(Ctx,
    cond_ref,
    loop_count,
    BO_LT,
    Ctx.IntTy,
    VK_PRValue,
    OK_Ordinary,
    getStart(ws->getCond()),
    FPOptions());

  DeclRefExpr* inc_ref = DeclRefExpr::Create(
     fd->getASTContext(),
     loopVar->getQualifierLoc(),
     SourceLocation(),
     loopVar,
     false,
     getStart(ws->getCond()),
     loopVar->getType(),
     VK_LValue,
     loopVar);

  UnaryOperator *inc_stmt =
      UnaryOperator::Create(Ctx,
        inc_ref,
        UO_PostInc,
        loop_count->getType(),
        VK_PRValue,
        OK_Ordinary,
        getStart(ws->getCond()),
        false,
        FPOptions());

  ForStmt* fs = new (fd->getASTContext()) ForStmt(fd->getASTContext(), init_stmt, cond_stmt, loopVar, inc_stmt,
                     ws->getBody(), getStart(ws), getStart(ws->getCond()), getEnd(ws->getCond()));

  auto iter = CompilerGlobals::astContextLists.findStmtBlockMatch(ws);
  if (iter == CompilerGlobals::astContextLists.currentStmtBlockEnd()){
    internalError(ws, "while-stmt is not part of an enclosing CompountStmt");
  }
  *iter = fs;
}

void
SSTLoopCountPragma::transformForLoop(Stmt* s)
{
  ForStmt* fs = cast<ForStmt>(s);
  FunctionDecl* fd = CompilerGlobals::astContextLists.enclosingFunctionDecls.back();
  Expr* loop_count = tokenToExpr(fd, loopCountToken_, getStart(fs->getCond()));
  IdentifierInfo* II = &CompilerGlobals::CI().getPreprocessor().getIdentifierTable().get("loop_variable");
  TypeSourceInfo *TInfo = fd->getASTContext().getTrivialTypeSourceInfo(loop_count->getType(), getStart(fs->getInit()));
  VarDecl* loopVar = VarDecl::Create(fd->getASTContext(), fd,
                                     getStart(fs->getInit()), getEnd(fs->getInit()),
                                     II, fd->getASTContext().IntTy, TInfo,
                                     SC_None);
  llvm::APInt api(32,0,true);
  IntegerLiteral* zero = IntegerLiteral::Create(fd->getASTContext(), api,
                                                fd->getASTContext().IntTy, getEnd(fs->getInit()));
  loopVar->setInit(zero);
  DeclGroupRef dgr(loopVar);
  DeclStmt* init_stmt = new (fd->getASTContext()) DeclStmt(dgr, getStart(fs->getInit()), getEnd(fs->getInit()));

  DeclRefExpr* cond_ref = DeclRefExpr::Create(
     fd->getASTContext(),
     loopVar->getQualifierLoc(),
     SourceLocation(),
     loopVar,
     false,
     getStart(fs->getCond()),
     loopVar->getType(),
     VK_LValue,
     loopVar);

  auto &Ctx = fd->getASTContext();

  BinaryOperator *cond_stmt = BinaryOperator::Create(Ctx,
    cond_ref,
    loop_count,
    BO_LT,
    Ctx.IntTy,
    VK_PRValue,
    OK_Ordinary,
    getStart(fs->getCond()),
    FPOptions());

  DeclRefExpr* inc_ref = DeclRefExpr::Create(
     fd->getASTContext(),
     loopVar->getQualifierLoc(),
     SourceLocation(),
     loopVar,
     false,
     getStart(fs->getInc()),
     loopVar->getType(),
     VK_LValue,
     loopVar);

  UnaryOperator *inc_stmt = UnaryOperator::Create(Ctx,
    inc_ref,
    UO_PostInc,
    loop_count->getType(),
    VK_PRValue,
    OK_Ordinary,
    getStart(fs->getInc()),
    false,
    FPOptions());

  fs->setInit(init_stmt);
  fs->setCond(cond_stmt);
  fs->setInc(inc_stmt);
  fs->setConditionVariable(fd->getASTContext(), loopVar);
}

void
SSTLoopCountPragma::activate(Stmt *s)
{
  if (isa<WhileStmt>(s)){
    transformWhileLoop(s);
  } else if (isa<ForStmt>(s)){
    transformForLoop(s);
  } else {
    errorAbort(s, "loop_count pragma not applied to ForStmt or WhileStmt");
  }
}

void
SSTComputePragma::visitAndReplaceStmt(Stmt* stmt)
{
  Loop loop(0); //just treat this is a loop of size 1
  loop.tripCount = "1";
  ComputeVisitor vis{};
  vis.setContext(stmt);
  vis.addOperations(stmt,loop.body);
  vis.replaceStmt(stmt,loop,nthread_);
}

void
SSTComputePragma::activate(Stmt *stmt)
{
#define scase(type,s) \
  case(clang::Stmt::type##Class): \
    visit##type(clang::cast<type>(s)); break
  switch(stmt->getStmtClass()){
    scase(ForStmt,stmt);
    scase(IfStmt,stmt);
    default:
      defaultAct(stmt);
      break;
  }
#undef scase
  throw StmtDeleteException(stmt);
}

void
SSTComputePragma::activate(Decl* d)
{
#define dcase(type,d) \
  case(clang::Decl::type): \
    visit##type##Decl(clang::cast<type##Decl>(d)); break
  switch(d->getKind()){
    dcase(Function,d);
    dcase(CXXMethod,d);
    default:
      break;
  }
#undef dcase
  throw DeclDeleteException(d);
}

void
SSTComputePragma::defaultAct(Stmt *stmt)
{
  visitAndReplaceStmt(stmt);
}


void
SSTComputePragma::visitCXXMethodDecl(CXXMethodDecl* decl)
{
  if (decl->hasBody()){
    visitAndReplaceStmt(decl->getBody());
  }
}

void
SSTComputePragma::visitFunctionDecl(FunctionDecl* decl)
{
  if (decl->hasBody()){
    visitAndReplaceStmt(decl->getBody());
  }
}


void
SSTComputePragma::visitIfStmt(IfStmt* stmt)
{
  visitAndReplaceStmt(stmt->getThen());
  if (stmt->getElse()){
    visitAndReplaceStmt(stmt->getElse());
  }
}

void
SSTComputePragma::replaceForStmt(clang::ForStmt* stmt, const std::string& nthread)
{
  ComputeVisitor vis{}; //null, no parent
  Loop loop(0); //depth zeros
  vis.setContext(stmt);
  vis.visitLoop(stmt,loop);
  vis.replaceStmt(stmt,loop,nthread);
  //cfg.skipNextStmt = true;
}

void
SSTComputePragma::visitForStmt(ForStmt *stmt)
{
  replaceForStmt(stmt, nthread_);
}

void
SSTMemoryPragma::activate(Stmt *s)
{
  CompilerGlobals::astNodeMetadata.computeMemoryOverrides[s] = memSpec_;
}


std::string
SSTOpenMPParallelPragma::numThreads(SourceLocation loc, const std::list<Token> &tokens)
{
  // Clause names sit outside parentheses; inside them num_threads can be a
  // variable, as in shared(num_threads).
  auto iter = tokens.begin();
  int depth = 0;
  for (; iter != tokens.end(); ++iter){
    if (iter->is(tok::l_paren)){
      ++depth;
    } else if (iter->is(tok::r_paren)){
      --depth;
    } else if (depth == 0 && iter->is(tok::identifier)
               && iter->getIdentifierInfo()->getName() == "num_threads"){
      break;
    }
  }
  if (iter == tokens.end()){
    return "";
  }
  if (++iter == tokens.end() || iter->isNot(tok::l_paren)){
    errorAbort(loc, "num_threads clause must be followed by '('");
  }

  // num_threads takes any integer expression, e.g. nt + 1 or
  // omp_get_max_threads(), so keep every token up to the matching ')'. It
  // becomes the hook's last argument, which converts it to int.
  Preprocessor& pp = CompilerGlobals::CI().getPreprocessor();
  std::string expr;
  depth = 1;
  int nested = 0;  // inside [] or {}, as in a lambda's capture list
  bool listTail = false;
  for (++iter; iter != tokens.end(); ++iter){
    if (iter->isOneOf(tok::l_square, tok::l_brace)){
      ++nested;
    } else if (iter->isOneOf(tok::r_square, tok::r_brace)){
      --nested;
    }
    bool top = depth == 1 && nested == 0;
    if (iter->is(tok::l_paren)){
      ++depth;
    } else if (iter->is(tok::r_paren) && --depth == 0){
      break;
    } else if (top && !listTail && iter->is(tok::colon) && expr == "strict"){
      // Drop the OpenMP 6.0 modifier in num_threads(strict: n).
      expr.clear();
      continue;
    } else if (top && iter->is(tok::comma)){
      // An OpenMP 6.0 list, num_threads(4, 2), gives the outermost team
      // first; that is the region modelled here.
      listTail = true;
    }
    if (listTail){
      continue;
    }
    if (!expr.empty() && iter->hasLeadingSpace()){
      expr += " ";
    }
    expr += pp.getSpelling(*iter);
  }
  if (iter == tokens.end() || expr.empty()){
    errorAbort(loc, "num_threads clause needs an expression and a closing ')'");
  }
  return expr;
}

SSTOpenMPParallelPragma::SSTOpenMPParallelPragma(SourceLocation loc, const std::list<Token> &tokens)
  : SSTComputePragma(numThreads(loc,tokens))
{ 
}

using namespace modes;

static PragmaRegister<SSTTokenListPragmaShim, SSTLoopCountPragma, true> loopCountPragma(
    "sst", "loop_count", SKELETONIZE | PUPPETIZE | SHADOWIZE);
static PragmaRegister<SSTStringPragmaShim, SSTMemoryPragma, true> memoryPragma(
    "sst", "memory", SKELETONIZE | SHADOWIZE);
static PragmaRegister<SSTNoArgsPragmaShim, SSTComputePragma, true> computePragma(
    "sst", "compute", SKELETONIZE | SHADOWIZE);
static PragmaRegister<SSTNoArgsPragmaShim, SSTComputePragma, true> alwaysComputePragma(
    "sst", "always_compute", SKELETONIZE | SHADOWIZE | ENCAPSULATE);
// static PragmaRegister<SSTArgMapPragmaShim, SSTMemoizeComputePragma, true> memoizePragma(
//     "sst", "memoize", MEMOIZE | SKELETONIZE);
static PragmaRegister<SSTTokenListPragmaShim, SSTOpenMPParallelPragma, true> ompPragmaSkel(
    "omp", "parallel", SKELETONIZE);
static PragmaRegister<SSTTokenListPragmaShim, SSTOpenMPParallelPragma, false> ompPragmaPup(
    "omp", "parallel", PUPPETIZE);
