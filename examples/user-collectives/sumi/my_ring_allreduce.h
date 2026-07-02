// Out-of-tree collective plugin example (Tier 3).
//
// A copy of the in-tree ring all-reduce, renamed so it can coexist with the
// built-in "ring", and registered under the name "myring". Build it into your
// own libmycoll.so with build_collective_plugin.sh -- no libsumi rebuild -- and
// select it with SUMI_ALLREDUCE_ALG=myring (or app1.allreduce_alg=myring).
//
// This is the same DagCollectiveActor contract documented in AUTHORING.md;
// edit initDag()/bufferAction() in the .cc to make it your own algorithm.

#pragma once

#include <iris/sumi/collective.h>
#include <iris/sumi/collective_actor.h>
#include <iris/sumi/collective_message.h>
#include <iris/sumi/comm_functions.h>

namespace SST::Iris::sumi {

class MyRingAllreduceActor :
  public DagCollectiveActor
{
 public:
  MyRingAllreduceActor(CollectiveEngine* engine, void* dst, void* src,
                       int nelems, int type_size, int tag, reduce_fxn fxn,
                       int cq_id, Communicator* comm) :
    DagCollectiveActor(Collective::allreduce, engine, dst, src, type_size, tag, cq_id, comm, fxn),
    fxn_(fxn), nelems_(nelems), num_reducing_rounds_(0), num_total_rounds_(0)
  {
  }

  std::string toString() const override {
    return "my ring all reduce actor";
  }

  void bufferAction(void *dst_buffer, void *msg_buffer, Action* ac) override;

  Output output;

 private:
  void finalizeBuffers() override;
  void initBuffers() override;
  void initDag() override;

 private:
  reduce_fxn fxn_;
  int nelems_;
  int num_reducing_rounds_;
  int num_total_rounds_;
};

class MyRingAllreduce :
  public DagCollective
{
 public:
  MyRingAllreduce(CollectiveEngine* engine, void* dst, void* src,
                  int nelems, int type_size, int tag, reduce_fxn fxn,
                  int cq_id, Communicator* comm)
    : DagCollective(allreduce, engine, dst, src, type_size, tag, cq_id, comm),
      fxn_(fxn), nelems_(nelems)
  {
  }

  std::string toString() const override {
    return "my sumi ring allreduce";
  }

  DagCollectiveActor* newActor() const override {
    return new MyRingAllreduceActor(engine_, dst_buffer_, src_buffer_,
                                    nelems_, type_size_, tag_, fxn_, cq_id_, comm_);
  }

 private:
  reduce_fxn fxn_;
  int nelems_;
};

}
