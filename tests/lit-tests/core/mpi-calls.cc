// Core: in skeletonize mode the MPI rewrite nulls the buffer arguments of the
// sst_hg_ MPI functions and of send, allgather, ... called on an mpi_api
// object. A user's own function with a bare MPI name, or std::reduce, is not
// an MPI call: nulling its pointers silently changed its arguments, broke the
// build for std::reduce and crashed the rewriter for a two-argument
// allgather. Nor is an MPI call whose buffers are missing, not pointers, or
// filled in from default arguments; nulling those broke the build or crashed
// the rewriter. The compile must succeed.
// No Mercury header declares these MPI names today, so the test declares them.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.mpi-calls.cc

#include <numeric>

int send(const int* buf, int n);
int reduce(const int* in, int* out, int n);
int allgather(const int* buf, int n);

int sst_hg_lit_user_calls(const int* a, int* b, int n) {
  int r = send(a, n);
  r += reduce(a, b, n);
  r += allgather(a, n);
  r += std::reduce(a, a + n);
  return r;
}

extern "C" int sst_hg_send(const void* buf, int count, int dest);
extern "C" int sst_hg_allgather(const void* sendbuf, int count, int type = 0,
                                void* recvbuf = nullptr);

struct mpi_api {
  int send(const void* buf, int count, int dest);
  int allgather(const void* sendbuf, int count, int type, void* recvbuf);
  int allgather(const void* sendbuf, int count);
  int reduce(const void* sendbuf, int count);
  int allreduce(const void* sendbuf, void* recvbuf = nullptr, int count = 0);
};

int sst_hg_lit_mpi_calls(mpi_api* api, const int* a, int* b, int n) {
  int r = sst_hg_send(a, n, 0);
  r += api->send(a, n, 0);
  r += api->allgather(a, n, 0, b);
  r += sst_hg_allgather(a, n, 0, b);
  r += api->allreduce(a, b);
  r += api->allgather(a, n);
  r += api->reduce(a, n);
  r += sst_hg_allgather(a, n);
  r += api->allreduce(a);
  return r;
}

// CHECK-LABEL: sst_hg_lit_user_calls
// CHECK: {{^ *}}int r = send(a, n);
// CHECK-NEXT: {{^ *}}r += reduce(a, b, n);
// CHECK-NEXT: {{^ *}}r += allgather(a, n);
// CHECK-NEXT: {{^ *}}r += std::reduce(a, a + n);
// CHECK-LABEL: sst_hg_lit_mpi_calls
// CHECK: {{^ *}}int r = sst_hg_send(nullptr, n, 0);
// CHECK-NEXT: {{^ *}}r += api->send(nullptr, n, 0);
// CHECK-NEXT: {{^ *}}r += api->allgather(nullptr, n, 0, nullptr);
// CHECK-NEXT: {{^ *}}r += sst_hg_allgather(nullptr, n, 0, nullptr);
// CHECK-NEXT: {{^ *}}r += api->allreduce(nullptr, nullptr);
// CHECK-NEXT: {{^ *}}r += api->allgather(a, n);
// CHECK-NEXT: {{^ *}}r += api->reduce(a, n);
// CHECK-NEXT: {{^ *}}r += sst_hg_allgather(a, n);
// CHECK-NEXT: {{^ *}}r += api->allreduce(a);
