// Out-of-tree collective plugin example (Tier 3). See my_ring_allreduce.h.
//
// The SUMI_REGISTER_COLLECTIVE at the bottom runs at dlopen and populates the
// same CollectiveRegistry the engine consults, so no simulator edit is needed.

#include "my_ring_allreduce.h"
#include <iris/sumi/collective_registry.h>
#include <iris/sumi/transport.h>
#include <iris/sumi/communicator.h>
#include <mercury/common/errors.h>
#include <mercury/common/stl_string.h>
#include <cstring>
#include <vector>

namespace SST::Iris::sumi {

void
MyRingAllreduceActor::finalizeBuffers()
{
  long buffer_size = nelems_ * type_size_;
  my_api_->freeWorkspace(recv_buffer_, buffer_size);
  //send buffer aliases the result buffer
}

void
MyRingAllreduceActor::initBuffers()
{
  void* dst = result_buffer_;
  void* src = send_buffer_;
  long size = static_cast<long>(nelems_) * type_size_; // matches finalizeBuffers

  result_buffer_ = dst;
  //work in place in the result buffer; copy src in first if distinct
  if (src != dst)
    my_api_->memcopy(dst, src, size);
  //a temporary recv buffer to land each neighbour chunk before reduce/copy
  recv_buffer_ = my_api_->allocateWorkspace(size, src);
  send_buffer_ = result_buffer_;
}

void
MyRingAllreduceActor::initDag()
{
  slicer_->fxn = fxn_;

  const int N = dom_nproc_;
  const int me = dom_me_;
  if (N <= 1){
    num_reducing_rounds_ = 0;
    num_total_rounds_ = 0;
    return;
  }

  // Guard your own round count against the message-id budget (see AUTHORING.md).
  if (2 * (N - 1) >= static_cast<int>(Action::max_round)){
    sst_hg_abort_printf("myring allreduce needs %d rounds but Action::max_round "
                        "is %u; raise max_round for nproc=%d",
                        2 * (N - 1), Action::max_round, N);
  }

  // One chunk per rank; spread remainder over the first rem chunks.
  std::vector<int> off(N), cnt(N);
  int base = nelems_ / N, rem = nelems_ % N, o = 0;
  for (int c = 0; c < N; ++c){
    cnt[c] = base + (c < rem ? 1 : 0);
    off[c] = o;
    o += cnt[c];
  }

  const int right = (me + 1) % N;
  const int left  = (me - 1 + N) % N;

  Action* prev_send = nullptr;
  Action* prev_recv = nullptr;

  auto chain = [&](Action* send_ac, Action* recv_ac){
    addDependency(prev_send, send_ac);
    addDependency(prev_send, recv_ac);
    addDependency(prev_recv, send_ac);
    addDependency(prev_recv, recv_ac);
    prev_send = send_ac;
    prev_recv = recv_ac;
  };

  // Reduce-scatter: N-1 rounds; send chunk (me-r) right, recv (me-r-1) left.
  for (int r = 0; r < N - 1; ++r){
    int send_chunk = ((me - r) % N + N) % N;
    int recv_chunk = ((me - r - 1) % N + N) % N;
    Action* send_ac = new SendAction(r, right, SendAction::in_place);
    send_ac->offset = off[send_chunk];
    send_ac->nelems = cnt[send_chunk];
    Action* recv_ac = new RecvAction(r, left, RecvAction::reduce);
    recv_ac->offset = off[recv_chunk];
    recv_ac->nelems = cnt[recv_chunk];
    chain(send_ac, recv_ac);
  }

  // All-gather: N-1 rounds; send chunk (me-r+1) right, copy chunk (me-r) left.
  for (int r = 0; r < N - 1; ++r){
    int my_round   = (N - 1) + r;
    int send_chunk = ((me - r + 1) % N + N) % N;
    int recv_chunk = ((me - r) % N + N) % N;
    Action* send_ac = new SendAction(my_round, right, SendAction::in_place);
    send_ac->offset = off[send_chunk];
    send_ac->nelems = cnt[send_chunk];
    Action* recv_ac = new RecvAction(my_round, left, RecvAction::in_place);
    recv_ac->offset = off[recv_chunk];
    recv_ac->nelems = cnt[recv_chunk];
    chain(send_ac, recv_ac);
  }

  num_reducing_rounds_ = N - 1;
  num_total_rounds_ = 2 * (N - 1);
}

void
MyRingAllreduceActor::bufferAction(void *dst_buffer, void *msg_buffer, Action* ac)
{
  int rnd = num_total_rounds_ ? (ac->round % num_total_rounds_) : 0;
  if (rnd < num_reducing_rounds_){
    (fxn_)(dst_buffer, msg_buffer, ac->nelems);  // reduce-scatter
  } else {
    my_api_->memcopy(dst_buffer, msg_buffer, ac->nelems * type_size_); // all-gather
  }
}

// Registered at dlopen into the engine's CollectiveRegistry: select with
// SUMI_ALLREDUCE_ALG=myring or app1.allreduce_alg=myring.
SUMI_REGISTER_COLLECTIVE(allreduce, "myring", MyRingAllreduce,
    a.engine, a.dst, a.src, a.nelems, a.type_size, a.tag, a.fxn, a.cq_id, a.comm)

}
