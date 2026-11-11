// Core: a free call and a global access in a class template's destructor and
// conversion operator are rewritten once, in the template. Rewriting the same
// source again for each instantiation, as constructors and methods already
// avoid, corrupted it. Conversion operators now follow the method rule, so a
// static local in one, even in a plain class, is not privatized. The compile
// must succeed; it needs sst_hg_free declared by Mercury's skeleton.h.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.template-members.cc

#include <cstdlib>

int sst_hg_lit_count = 0;

template <class T> struct sst_hg_lit_holder {
  void* p = nullptr;
  ~sst_hg_lit_holder() {
    free(p);
    ++sst_hg_lit_count;
  }
  operator bool() const {
    free(nullptr);
    return sst_hg_lit_count > 0;
  }
};

struct sst_hg_lit_plain {
  operator int() const {
    static int calls = 0;
    return ++calls;
  }
};

int sst_hg_lit_use() {
  sst_hg_lit_holder<int> a;
  sst_hg_lit_holder<double> b;
  return bool(a) + bool(b);
}

// CHECK-LABEL: struct sst_hg_lit_holder {
// CHECK: ~sst_hg_lit_holder() {
// CHECK-NEXT: {{^ *}}sst_hg_free(p);
// CHECK-NEXT: {{^ *}}++(*((int *)(get_sst_hg_global_data() + __offset_sst_hg_lit_count)));
// CHECK-NEXT: {{^ *}}}
// CHECK-NEXT: {{^ *}}operator bool() const {
// CHECK: {{^ *}}sst_hg_free(nullptr);
// CHECK-NEXT: {{^ *}}return (*sst_hg_sst_hg_lit_count) > 0;
// CHECK-LABEL: struct sst_hg_lit_plain {
// CHECK-NEXT: {{^ *}}operator int() const {
// CHECK-NEXT: {{^ *}}static int calls = 0;
// CHECK-NEXT: {{^ *}}return ++calls;
