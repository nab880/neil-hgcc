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

#ifndef bin_clang_replAstVisitor_h
#define bin_clang_replAstVisitor_h

#include "clangHeaders.h"
#include "clang/AST/Mangle.h"
#include "pragmas.h"
#include "globalVarNamespace.h"

#include <unordered_set>

#define visitFxn(cls) \
  bool Visit##cls(clang::cls* c){ return TestStmtMacro(c); }

static constexpr int IndexResetter = -1;

// RAII push/pop for context lists; cleans up on exception.
template <class T>
struct PushGuard {
  template <class U>
  PushGuard(std::list<T>& theList, U&& t) : myList(theList) {
    myList.push_back(std::forward<U>(t));
  }

  ~PushGuard(){ myList.pop_back(); }

  void swap(T&& t){
    myList.pop_back();
    myList.push_back(t);
  }

  std::list<T>& myList;
};

template <class T, class U>
struct InsertGuard {
  InsertGuard(std::map<T*,U*>& theMap, T* t, U* u) :
    myMap(theMap), myKey(t) {
    myMap.emplace(t,u);
  }

  ~InsertGuard(){ myMap.erase(myKey); }

  std::map<T*,U*>& myMap;
  T* myKey;
};

template <class T>
struct VectorPushGuard {
  template <class... Args>
  VectorPushGuard(std::vector<T>& theVec, Args&& ...args) : myVec(theVec) {
    myVec.emplace_back(std::forward<Args>(args)...);
  }

  template <class... Args>
  void swap(Args&& ...args){
    myVec.pop_back();
    myVec.emplace_back(std::forward<Args>(args)...);
  }

  ~VectorPushGuard(){ myVec.pop_back(); }

  std::vector<T>& myVec;
};

struct IndexGuard {
  IndexGuard(int& t, int value) {
    if (t < 0) {
      t = value;
      myPtr = &t;
    } else { //don't overwrite
      myPtr = nullptr;
    }
  }

  ~IndexGuard(){ if (myPtr) *myPtr = IndexResetter; }

  int* myPtr;
};

template <class T>
struct EmplaceGuard {
  EmplaceGuard(std::list<T>& theList) : myList(theList) {
    myList.emplace_back();
  }

  ~EmplaceGuard(){ myList.pop_back(); }

  std::list<T>& myList;
};

struct IncrementGuard {
  IncrementGuard(int& idx) : myIdx(idx)
  {
    ++myIdx;
  }
  ~IncrementGuard(){ --myIdx; }
  int& myIdx;
};

struct StmtDeleteException : public std::runtime_error
{
  StmtDeleteException(clang::Stmt* deld) :
    std::runtime_error("deleted expression"), deleted (deld){
  }
  clang::Stmt* deleted;
};

struct DeclDeleteException : public std::runtime_error
{
  DeclDeleteException(clang::Decl* deld) :
    std::runtime_error("deleted expression"), deleted (deld){
  }
  clang::Decl* deleted;
};

class FirstPassASTVisitor : public clang::RecursiveASTVisitor<FirstPassASTVisitor>
{
  using Parent = clang::RecursiveASTVisitor<FirstPassASTVisitor>;
 public:
  friend struct PragmaActivateGuard;

  FirstPassASTVisitor(SSTPragmaList& pragmas);

  void preVisitTopLevelDecl(clang::Decl* /*d*/){}
  void postVisitTopLevelDecl(clang::Decl* /*d*/){}
  void finalizePass(){}

  bool VisitDecl(clang::Decl* d);
  bool VisitStmt(clang::Stmt* s);

  bool TraverseFunctionDecl(clang::FunctionDecl* fd, DataRecursionQueue* queue = nullptr);

  bool TraverseCompoundStmt(clang::CompoundStmt* cs, DataRecursionQueue* queue = nullptr);

  SSTPragmaList& getPragmas(){
    return pragmas_;
  }

 private:
  SSTPragmaList& pragmas_;

};

class SkeletonASTVisitor : public clang::RecursiveASTVisitor<SkeletonASTVisitor> {
  friend class SkeletonASTConsumer;
  friend struct PragmaActivateGuard;
  using Parent=clang::RecursiveASTVisitor<SkeletonASTVisitor>;
 private:
  struct AnonRecord {
    clang::RecordDecl* decl;
    bool typeNameAdded;
    std::string structType;  //union or struct
    std::string retType;
    bool isFxnStatic;
    //struct X_anonymous_type - gives unique typename to anonymous truct
    std::string typeName;
    AnonRecord() : 
      decl(nullptr), 
      typeNameAdded(false),
      isFxnStatic(false)
    {}
  };

  struct ArrayInfo {
    bool needsTypedef() const {
      return !typedefDeclString.empty();
    }

    std::string typedefDeclString;
    std::string typedefName;
    std::string fqTypedefName;
    std::string retType;
    bool isFxnStatic;
    bool implicitSize;
    bool needsDeref;

    ArrayInfo() : isFxnStatic(false), needsDeref(true), implicitSize(false) {}
  };


  struct GlobalReplacement {
    std::string reusableText;
    std::string inlineUseText;
    bool append;
    GlobalReplacement(const std::string& reusable,
                      const std::string& oneOff,
                      bool app) :
      reusableText(reusable),
      inlineUseText(oneOff),
      append(app)
    {}
  };

  struct GlobalVariableReplacement {
    ArrayInfo* arrayInfo;
    AnonRecord* anonRecord;
    std::string scopeUniqueVarName;
    bool useAccessor;
    bool isFxnStatic;
    bool needFullNamespace;
    std::string typeStr;
    std::string retType;
    std::string classScope;
    bool threadLocal;

    GlobalVariableReplacement(const std::string& uniqueName,
                              bool useAcc, bool isStatic, bool needNs, bool tls) :
      arrayInfo(nullptr),
      anonRecord(nullptr),
      scopeUniqueVarName(uniqueName),
      useAccessor(useAcc),
      isFxnStatic(isStatic),
      needFullNamespace(needNs),
      threadLocal(tls)
    {
    }

    ~GlobalVariableReplacement(){
      if (arrayInfo) delete arrayInfo;
      if (anonRecord) delete anonRecord;
    }

  };

  struct GlobalStandin {
    bool fxnStatic;
    bool threadLocal;
    std::string replText;
    GlobalStandin() : fxnStatic(false), threadLocal(false) {}
  };

  typedef enum {
    Global, //regular global variable (C-style)
    FileStatic,
    CxxStatic, //c++ static class variable
    FxnStatic
  } GlobalVariable_t;

  struct cArrayConfig {
    std::string fundamentalTypeString;
    clang::QualType fundamentalType;
    std::stringstream arrayIndices;
  };

  struct ReplaceGlobalsPrinterHelper : public clang::PrinterHelper {
    ReplaceGlobalsPrinterHelper(SkeletonASTVisitor* parent) :
      parent_(parent)
    {
    }

    bool handledStmt(clang::Stmt *E, clang::raw_ostream &OS) override;

   private:
    SkeletonASTVisitor* parent_;
  };

  static bool indexIsSet(int idx){
    return idx != IndexResetter;
  }

 public:
  SkeletonASTVisitor(SSTPragmaList& pragmas,
      GlobalVarNamespace& ns) :
    pragmas_(pragmas),
    activeBinOpIdx_(-1),
    foundCMain_(false), 
    refactorMain_(true),
    insideCxxMethod_(0), 
    globalNs_(ns), 
    currentNs_(&ns),
    visitingGlobal_(false),
    keepGlobals_(false)
  {
    initHeaders();
    initReservedNames();
    initMPICalls();
  }

  // Delay overlapping Rewriter inserts to avoid Clang segfaults.
  void delayedInsertAfter(clang::VarDecl* vd, const std::string& repl){
    unsigned pos = getStart(vd).getRawEncoding();
    auto iter = declsToInsertAfter_.find(pos);
    if (iter == declsToInsertAfter_.end()){
      declsToInsertAfter_[pos] = {vd,repl};
    } else {
      auto& pair = declsToInsertAfter_[pos];
      pair.first = vd;
      pair.second += repl;
    }
  }

  enum class ExprPeelMode {
    CastsAndUnary,
    CastsAndTemporaries,
    ExprCleanupsOnly,
    LambdaCaptureInit,
  };

  struct ExprPeelResult {
    clang::Expr* leaf;
    bool nonRefLeaf;
  };

  static ExprPeelResult peelExpr(clang::Expr* e, ExprPeelMode mode);

  static clang::Expr* getUnderlyingExpr(clang::Expr *e);

  bool isGlobal(const clang::DeclRefExpr* expr) const {
    return globals_.find(mainDecl(expr)) != globals_.end();
  }

  friend struct ReplaceGlobalsPrinterHelper;
  std::string printWithGlobalsReplaced(clang::Stmt* stmt);

  void registerNewKeywords(std::ostream& os);

  GlobalVarNamespace* getActiveNamespace() const {
    return currentNs_;
  }

  bool isCxx() const {
    return CompilerGlobals::CI().getLangOpts().CPlusPlus;
  }

  std::string needGlobalReplacement(clang::NamedDecl* decl) {
    const clang::Decl* md = mainDecl(decl);
    if (globalsTouched_.empty()){
      errorAbort(decl, "internal error: globals touched array is empty");
    }
    globalsTouched_.back().insert(md);
    auto iter = globals_.find(md);
    if (iter == globals_.end()){
      errorAbort(decl, "getting global replacement for non-global variable");
    }
    return iter->second.reusableText;
  }

  bool VisitStmt(clang::Stmt* S);

  bool TraverseDecl(clang::Decl* D);

  bool VisitTypedefDecl(clang::TypedefDecl* D);

  bool VisitDeclRefExpr(clang::DeclRefExpr* expr);

  bool TraverseReturnStmt(clang::ReturnStmt* stmt, DataRecursionQueue* queue = nullptr);

  bool TraverseMemberExpr(clang::MemberExpr* expr, DataRecursionQueue* queue = nullptr);

  bool VisitCXXNewExpr(clang::CXXNewExpr* expr);

  bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* expr);

  bool TraverseArraySubscriptExpr(clang::ArraySubscriptExpr* expr, DataRecursionQueue* = nullptr);

  bool TraverseCXXDeleteExpr(clang::CXXDeleteExpr* expr, DataRecursionQueue* = nullptr);

  bool TraverseLambdaExpr(clang::LambdaExpr* expr);

  bool TraverseCXXMemberCallExpr(clang::CXXMemberCallExpr* expr, DataRecursionQueue* queue = nullptr);

  bool visitVarDecl(clang::VarDecl* D);

  bool TraverseVarDecl(clang::VarDecl* D);

  bool TraverseVarTemplateDecl(clang::VarTemplateDecl* D);

  bool TraverseCallExpr(clang::CallExpr* expr, DataRecursionQueue* queue = nullptr);

  // Lower <<<>>> to sst_hg_cuda_launch; honors gpu_compute and delete pragmas.
  bool TraverseCUDAKernelCallExpr(clang::CUDAKernelCallExpr* expr,
                                  DataRecursionQueue* queue = nullptr);

  bool TraverseUnresolvedLookupExpr(clang::UnresolvedLookupExpr* expr,
                                    DataRecursionQueue* queue = nullptr);

  bool VisitCXXDependentScopeMemberExpr(clang::CXXDependentScopeMemberExpr* expr);

  bool VisitDependentScopeDeclRefExpr(clang::DependentScopeDeclRefExpr* expr);

  bool TraverseNamespaceDecl(clang::NamespaceDecl* D);

  bool TraverseCXXRecordDecl(clang::CXXRecordDecl* D);

  bool TraverseFunctionDecl(clang::FunctionDecl* D);

  bool TraverseForStmt(clang::ForStmt* S, DataRecursionQueue* queue = nullptr);

  bool TraverseDoStmt(clang::DoStmt* S, DataRecursionQueue* queue = nullptr);

  bool TraverseDecltypeTypeLoc(clang::DecltypeTypeLoc loc, bool TraverseQualifier);

  bool TraverseWhileStmt(clang::WhileStmt* S, DataRecursionQueue* queue = nullptr);

  bool TraverseUnaryOperator(clang::UnaryOperator* op, DataRecursionQueue* queue = nullptr);

  bool TraverseBinaryOperator(clang::BinaryOperator* op, DataRecursionQueue* queue = nullptr);

  bool TraverseCompoundAssignOperator(clang::CompoundAssignOperator* op, DataRecursionQueue* queue = nullptr);

  bool TraverseIfStmt(clang::IfStmt* S, DataRecursionQueue* queue = nullptr);

  bool TraverseCompoundStmt(clang::CompoundStmt* S, DataRecursionQueue* queue = nullptr);

  bool TraverseDeclStmt(clang::DeclStmt* op, DataRecursionQueue* queue = nullptr);

  bool TraverseFieldDecl(clang::FieldDecl* fd, DataRecursionQueue* queue = nullptr);

  bool TraverseInitListExpr(clang::InitListExpr* expr, DataRecursionQueue* queue = nullptr);

#define OPERATOR(NAME) \
  bool TraverseBin##NAME(clang::BinaryOperator* op, DataRecursionQueue* queue = nullptr){ \
    return TraverseBinaryOperator(op,queue); \
  }
  BINOP_LIST()
#undef OPERATOR

#define OPERATOR(NAME) \
  bool TraverseUnary##NAME(clang::UnaryOperator* op, DataRecursionQueue* queue = nullptr){ \
    return TraverseUnaryOperator(op,queue); \
  }
  UNARYOP_LIST()
#undef OPERATOR

#define OPERATOR(NAME) \
  bool TraverseBin##NAME##Assign(clang::CompoundAssignOperator* op, DataRecursionQueue* queue = nullptr){ \
    return TraverseCompoundAssignOperator(op,queue); \
  }
  CAO_LIST()
#undef OPERATOR

  bool TraverseFunctionTemplateDecl(clang::FunctionTemplateDecl* D);

  bool TraverseCXXMethodDecl(clang::CXXMethodDecl *D);

  bool TraverseCXXConstructorDecl(clang::CXXConstructorDecl* D);

  bool TraverseCXXDestructorDecl(clang::CXXDestructorDecl* D);

  clang::SourceLocation getVariableNameLocationEnd(clang::VarDecl* D);

  SSTPragmaList& getPragmas(){
    return pragmas_;
  }

  void preVisitTopLevelDecl(clang::Decl* d);
  void postVisitTopLevelDecl(clang::Decl* d);
  void finalizePass();

  void setVisitingGlobal(bool flag){
    visitingGlobal_ = flag;
  }

  void setTopLevelScope(clang::Decl* d){
    currentTopLevelScope_ = d;
  }

  clang::Decl* getTopLevelScope() const {
    return currentTopLevelScope_;
  }

  bool hasCStyleMain() const {
    return foundCMain_;
  }

  const std::string& getAppName() const {
    return mainName_;
  }

 private:
  clang::NamespaceDecl* getOuterNamespace(clang::Decl* D);

  void getTemplatePrefixString(std::ostream& os, clang::TemplateParameterList* theList);

  void getTemplateParamsString(std::ostream& os, clang::TemplateParameterList* theList);

  bool shouldVisitDecl(clang::VarDecl* D);

  void initHeaders();

  void initReservedNames();

  void initMPICalls();

  void replaceMain(clang::FunctionDecl* mainFxn);

  std::string getCleanTypeName(clang::QualType ty);

  std::string getCleanName(const std::string& in);

  std::string eraseAllStructQualifiers(const std::string& name);

  void addInContextGlobalDeclarations(clang::Stmt* body);

  clang::CXXConstructExpr* getCtor(clang::VarDecl* vd);

 private:
  static inline const clang::Decl* mainDecl(const clang::Decl* d){
    return d->getCanonicalDecl();
  }

  static inline const clang::Decl* mainDecl(const clang::DeclRefExpr* dr){
    return dr->getDecl()->getCanonicalDecl();
  }

  std::string activeGlobalScopedName_;

  bool activeGlobal() const {
    return !activeGlobalScopedName_.empty();
  }

  void clearActiveGlobal() {
    activeGlobalScopedName_.clear();
  }


  void propagateNullness(clang::Decl* dest, clang::Decl* src);

  bool deleteMemberExpr(SSTNullVariablePragma* prg, clang::MemberExpr* expr,
                        clang::NamedDecl* decl);

  void replaceNullWithEmptyType(clang::QualType type, clang::Expr* toRepl);

  void deleteNullVariableExpr(clang::Expr* expr);

  clang::Stmt* replaceNullVariableStmt(clang::Stmt* stmt, const std::string& repl);

  clang::Stmt* checkNullAssignments(clang::NamedDecl* nd, bool hasReplacement);

  void nullifyIfStmt(clang::IfStmt* if_stmt, clang::Decl* d);

  void addTransitiveNullInformation(clang::NamedDecl* nd, std::ostream& os,
                                    SSTNullVariablePragma* prg);

  void visitNullVariable(clang::Expr* expr, clang::NamedDecl* nd);

  void tryVisitNullVariable(clang::Expr* expr, clang::NamedDecl* nd);

  void nullDereferenceError(clang::Expr* expr, const std::string& varName);

  bool deleteNullVariableMember(clang::NamedDecl* nullVarDecl, clang::MemberExpr* expr);

  void setActiveGlobalScopedName(const std::string& str) {
    activeGlobalScopedName_ = str;
  }

  const std::string& activeGlobalScopedName() const {
    return activeGlobalScopedName_;
  }

  void executeCurrentReplacements();

  void replace(clang::SourceRange rng, const std::string& repl);

  void replace(clang::Expr* expr, const std::string& repl){
    replace(expr->getSourceRange(), repl);
  }

  void replace(clang::Decl* decl, const std::string& repl){
    replace(decl->getSourceRange(), repl);
  }

  bool insideTemplateFxn() const {
    if (CompilerGlobals::astContextLists.enclosingFunctionDecls.empty()) return false;
    clang::FunctionDecl* fd = CompilerGlobals::astContextLists.enclosingFunctionDecls.back();
    return fd->isDependentContext();
  }

  bool isThreadLocal(clang::VarDecl* D) const {
    switch (D->getTSCSpec()){
      case clang::TSCS___thread:
      case clang::TSCS_thread_local:
      case clang::TSCS__Thread_local:
        return true;
      default:
        return false;
    }
  }

  GlobalVariableReplacement setupGlobalReplacement(clang::VarDecl* vd, const std::string& namePrefix,
                          bool useAccessor, bool isFxnStatic, bool needFullNs);

  bool isGlobalDefinition(clang::VarDecl* D, GlobalVariableReplacement* var);
  void registerGlobalReplacement(clang::VarDecl* D, GlobalVariableReplacement* repl);
  bool setupClassStaticVarDecl(clang::VarDecl* D);
  bool setupCGlobalVar(clang::VarDecl* D, const std::string& scopePrefix);
  bool setupCppGlobalVar(clang::VarDecl* D, const std::string& scopePrefix);
  bool setupFunctionStaticCpp(clang::VarDecl* D, const std::string& scopePrefix);
  bool setupFunctionStaticC(clang::VarDecl* D, const std::string& scopePrefix);

  template <class Lambda>
  void goIntoContext(clang::Stmt* stmt, Lambda&& l){
    stmtContexts_.push_back(stmt);
    stmtReplacements_.emplace_back();
    bool deleted = false;
    try {
      l();
    } catch (StmtDeleteException& e) {
      if (stmt != e.deleted){
        stmtContexts_.pop_back(); //must pop back now
        stmtReplacements_.pop_back();
        //nope! not me - pass it along
        throw e;
      }
      deleted = true;
    }
    stmtContexts_.pop_back();
    if (!deleted) executeCurrentReplacements();
    stmtReplacements_.pop_back();
  }

  bool isNullVariable(clang::Decl* d) const {
    return CompilerGlobals::astNodeMetadata.nullVariables.find(d) !=
            CompilerGlobals::astNodeMetadata.nullVariables.end();
  }

  bool isValidAssignment(clang::Decl* lhs, clang::Expr* rhs);

  bool isNullSafeFunction(const clang::DeclContext* dc) const {
    return CompilerGlobals::astNodeMetadata.nullSafeFunctions.find(dc) !=
          CompilerGlobals::astNodeMetadata.nullSafeFunctions.end();
  }

  SSTNullVariablePragma* getNullVariable(clang::Decl* d) const {
    auto iter = CompilerGlobals::astNodeMetadata.nullVariables.find(d);
    if (iter != CompilerGlobals::astNodeMetadata.nullVariables.end()){
      return iter->second;
    }
    return nullptr;
  }

  void maybeReplaceGlobalUse(clang::DeclRefExpr* expr, clang::SourceRange rng);

  clang::Expr* getFinalExpr(clang::Expr *e);

  void replaceNullVariableConnectedContext(clang::Expr* expr, const std::string& repl);

  void deleteNullVariableStmt(clang::Stmt* stmt);
  void visitCollective(clang::CallExpr* expr);
  void visitReduce(clang::CallExpr* expr);
  void visitPt2Pt(clang::CallExpr* expr);
  bool checkDeclStaticClassVar(clang::VarDecl* D);
  bool checkInstanceStaticClassVar(clang::VarDecl* D);
  bool checkStaticFxnVar(clang::VarDecl* D);
  bool checkGlobalVar(clang::VarDecl* D);
  bool checkStaticFileVar(clang::VarDecl* D);
  bool haveActiveFxnParam() const {
    if (activeFxnParams_.empty()) return false;
    return activeFxnParams_.back();
  }
  clang::SourceLocation getEndLoc(clang::SourceLocation startLoc);

  bool insideClass() const {
    return !classContexts_.empty();
  }

  bool insideFxn() const {
    return !CompilerGlobals::astContextLists.enclosingFunctionDecls.empty();
  }

   AnonRecord* checkAnonStruct(clang::VarDecl* D);

   clang::RecordDecl* checkCombinedStructVarDecl(clang::VarDecl* D);

   ArrayInfo* checkArray(clang::VarDecl* D);

  void deleteStmt(clang::Stmt* s);

  void declareSSTExternVars(clang::SourceLocation insertLoc);

  void traverseFunctionBody(clang::Stmt* s);

  bool doTraverseLambda(clang::LambdaExpr* expr);

  void getArrayType(const clang::Type* ty, cArrayConfig& cfg);

  void setFundamentalTypes(clang::QualType qt, cArrayConfig& cfg);

  bool maybePrintGlobalReplacement(clang::VarDecl* vd, llvm::raw_ostream& os);

  void arrayFxnPointerTypedef(clang::VarDecl* D, SkeletonASTVisitor::ArrayInfo* info,
                              std::stringstream& sstr);


  const clang::Decl* getOriginalDeclaration(clang::VarDecl* vd);

  bool isInSystemHeader(clang::SourceLocation loc);

 private:
  SSTPragmaList& pragmas_;
  clang::Decl* currentTopLevelScope_;

  std::unordered_set<std::string> validHeaders_;
  std::set<std::string> ignoredHeaders_;

  // Call-expr lookahead deletions (most deletions use exceptions).
  std::set<clang::Expr*> deletedArgsCurrentCallExpr_;
  std::list<clang::MemberExpr*> memberAccesses_;
  std::map<clang::Stmt*,clang::Stmt*> extendedReplacements_;
  typedef enum { LHS, RHS } BinOpSide;
  std::vector<std::pair<clang::BinaryOperator*,BinOpSide>> binOps_;
  int activeBinOpIdx_;

  // Template statics hidden behind CXXDependentScopeMemberExpr.
  std::map<std::string,clang::VarDecl*> dependentStaticMembers_;

  bool foundCMain_;
  bool refactorMain_;
  std::string mainName_;

  std::map<unsigned, std::pair<clang::VarDecl*,std::string>> declsToInsertAfter_;
  std::list<std::list<std::pair<clang::SourceRange,std::string>>> stmtReplacements_;

  std::list<clang::ParmVarDecl*> activeFxnParams_;
  std::list<int> initIndices_;
  std::list<clang::FieldDecl*> activeFieldDecls_;
  std::list<clang::CXXConstructorDecl*> ctorContexts_;
  std::list<std::set<const clang::Decl*>> globalsTouched_;
  std::list<clang::VarDecl*> activeDecls_;
  std::list<clang::Expr*> activeInits_;
  std::list<clang::CXXRecordDecl*> classContexts_;
  std::list<clang::Stmt*> loopContexts_; //both fors and whiles
  std::list<clang::Stmt*> stmtContexts_;
  std::list<clang::Expr*> activeDerefs_;
  std::list<clang::IfStmt*> activeIfs_;
  int insideCxxMethod_;

  int cudaLaunchCounter_ = 0;
  std::unique_ptr<clang::MangleContext> cudaMangler_;
  void rewriteCudaLaunch(clang::CUDAKernelCallExpr* expr);
  std::string mangleKernelName(clang::FunctionDecl* kernel);

  std::set<std::string> ssthgFxnPrepends_;
  typedef void (SkeletonASTVisitor::*MPI_Call)(clang::CallExpr* expr);
  std::map<std::string, MPI_Call> mpiCalls_;
  std::set<std::string> reservedNames_;
  std::set<std::string> globalVarWhitelist_;

  GlobalVarNamespace& globalNs_;
  GlobalVarNamespace* currentNs_;
  bool visitingGlobal_;
  std::map<const clang::Decl*,GlobalReplacement> globals_;
  std::set<const clang::Decl*> variableTemplates_;
  std::map<const clang::Decl*,std::string> scopedNames_;
  bool keepGlobals_;
  std::set<clang::DeclRefExpr*> alreadyReplaced_;
  std::map<const clang::Decl*,GlobalStandin> globalStandins_;
  //C-style structs that have typedef'd names we can use doing global replacements
  std::map<clang::RecordDecl*,clang::TypedefDecl*> typedefStructs_;
  //used to assign a unique int ID to each static function variable
  std::map<clang::FunctionDecl*, std::map<std::string, int>> staticFxnVarCounts_;
};

struct PragmaActivateGuard {
  template <class T> //either decl/stmt
  PragmaActivateGuard(T* t, SkeletonASTVisitor* visitor, bool doVisit = true) :
    PragmaActivateGuard(t, visitor->pragmas_, doVisit, false/*2nd pass*/)
  {
  }

  template <class T>
  PragmaActivateGuard(T* t, FirstPassASTVisitor* visitor, bool doVisit = true) :
    PragmaActivateGuard(t, visitor->pragmas_, doVisit, true/*1st pass*/)
  {
  }

  ~PragmaActivateGuard();

  bool skipVisit() const {
    return skipVisit_;
  }

 private:
  template <class T> //either decl/stmt
  PragmaActivateGuard(T* t,
       SSTPragmaList& pragmas,
       bool doVisit, bool firstPass) :
    skipVisit_(false),
    pragmas_(pragmas)
  {
    myPragmas_ = [&]{
      auto tmp = pragmas_.getMatches<T>(t, firstPass);
      if(doVisit){
        return tmp;
      } else {
        return decltype(tmp){};
      }
    }();

    //this removes all inactivate pragmas from myPragmas_
    for (SSTPragma* prg : myPragmas_){
      if (prg->deleteOnUse){
        deletePragmaText(prg);
      }
      prg->activate(t);
      if (CompilerGlobals::pragmaConfig.makeNoChanges){
        skipVisit_ = true;
        CompilerGlobals::pragmaConfig.makeNoChanges = false;
      }
    }

  }

  void deletePragmaText(SSTPragma* prg);

  bool skipVisit_;
  std::list<SSTPragma*> myPragmas_;
  SSTPragmaList& pragmas_;

};

class GlobalVariableVisitor : public clang::RecursiveASTVisitor<GlobalVariableVisitor> {
 public:
  GlobalVariableVisitor(clang::VarDecl*  /*D*/, SkeletonASTVisitor* parent) :
    visitedGlobals_(false),
    parent_(parent)
  {
  }

  bool visitedGlobals() const {
    return visitedGlobals_;
  }

  bool VisitDeclRefExpr(clang::DeclRefExpr* expr);

  bool VisitCallExpr(clang::CallExpr* expr);

 private:
  bool visitedGlobals_;
  SkeletonASTVisitor* parent_;
};


#endif
