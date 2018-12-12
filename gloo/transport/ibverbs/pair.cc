/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/transport/ibverbs/pair.h"
#include "gloo/transport/ibverbs/buffer.h"

#include <stdlib.h>
#include <string.h>

#include "gloo/common/common.h"
#include "gloo/common/error.h"
#include "gloo/common/logging.h"

namespace gloo {
namespace transport {
namespace ibverbs {

Pair::Pair(
    const std::shared_ptr<Device>& dev,
    std::chrono::milliseconds timeout)
    : dev_(dev),
      sync_(false),
      busyPoll_(false),
      timeout_(timeout),
      completionEventsHandled_(0),
      recvPosted_(0),
      ex_(nullptr) {
  int rv;

  // Create completion queue
  {
    // Have to register this completion queue with the device's
    // completion channel to support asynchronous completion handling.
    // Pairs use asynchronous completion handling by default so
    // we call ibv_req_notify_cq(3) to request the first notification.
    cq_ = ibv_create_cq(
      dev_->context_,
      kCompletionQueueCapacity,
      this,
      dev_->comp_channel_,
      0);
    GLOO_ENFORCE(cq_);

    // Arm notification mechanism for completion queue.
    rv = ibv_req_notify_cq(cq_, kNotifyOnAnyCompletion);
    GLOO_ENFORCE_EQ(rv, 0);
  }

  // Create queue pair
  {
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(struct ibv_qp_init_attr));
    attr.send_cq = cq_;
    attr.recv_cq = cq_;
    attr.cap.max_send_wr = Pair::kSendCompletionQueueCapacity;
    attr.cap.max_recv_wr = Pair::kRecvCompletionQueueCapacity;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.qp_type = IBV_QPT_RC;
    qp_ = ibv_create_qp(dev->pd_, &attr);
    GLOO_ENFORCE(qp_);
  }

  // Init queue pair
  {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = dev_->attr_.port;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    rv = ibv_modify_qp(
        qp_,
        &attr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    GLOO_ENFORCE_EQ(rv, 0);
  }

  // Populate local address.
  // The Packet Sequence Number field (PSN) is random which makes that
  // the remote end of this pair needs to have the contents of the
  // full address struct in order to connect, and vice versa.
  {
    struct ibv_port_attr attr;
    memset(&attr, 0, sizeof(struct ibv_port_attr));
    rv = ibv_query_port(dev_->context_, dev_->attr_.port, &attr);
    GLOO_ENFORCE_EQ(rv, 0);
    rv = ibv_query_gid(
        dev_->context_,
        dev_->attr_.port,
        dev_->attr_.index,
        &self_.addr_.ibv_gid);
    GLOO_ENFORCE_EQ(rv, 0);
    self_.addr_.lid = attr.lid;
    self_.addr_.qpn = qp_->qp_num;
    self_.addr_.psn = rand() & 0xffffff;
  }

  // Post receive requests before connecting.
  // Whenever the remote side of this pair registers receive buffers,
  // this triggers their memory registration to be sent to this side.
  // Since these sends are one-sided, we always need a full bench of
  // receive work requests. Memory region receives can be interleaved
  // with regular buffer writes, so we proactively include a memory
  // region in every receive work request.
  for (int i = 0; i < kMaxBuffers; ++i) {
    mappedRecvRegions_[i] = make_unique<MemoryRegion>(dev_->pd_);
    postReceive();
  }
}

Pair::~Pair() {
  int rv;

  // Acknowledge number of completion events handled by this
  // pair's completion queue (also see ibv_get_cq_event(3)).
  ibv_ack_cq_events(cq_, completionEventsHandled_);

  rv = ibv_destroy_qp(qp_);
  GLOO_ENFORCE_EQ(rv, 0);

  rv = ibv_destroy_cq(cq_);
  GLOO_ENFORCE_EQ(rv, 0);
}

void Pair::close() {
  if (closed_) {
    // TODO: add proper handling of duplicate closes T21171834
    return;
  }

  closed_ = true;
}

const Address& Pair::address() const {
  return self_;
}

void Pair::connect(const std::vector<char>& bytes) {
  struct ibv_qp_attr attr;
  int rv;
  checkErrorState();

  peer_ = Address(bytes);

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_1024;
  attr.dest_qp_num = peer_.addr_.qpn;
  attr.rq_psn = peer_.addr_.psn;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 20;
  attr.ah_attr.is_global = 0;
  attr.ah_attr.dlid = peer_.addr_.lid;
  attr.ah_attr.port_num = dev_->attr_.port;
  if (peer_.addr_.ibv_gid.global.interface_id) {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.grh.dgid = peer_.addr_.ibv_gid;
    attr.ah_attr.grh.sgid_index = dev_->attr_.index;
  }

  // Move to Ready To Receive (RTR) state
  rv = ibv_modify_qp(
      qp_,
      &attr,
      IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
          IBV_QP_AV | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  GLOO_ENFORCE_EQ(rv, 0);

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = self_.addr_.psn;
  attr.ah_attr.is_global = 1;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7; /* infinite */
  attr.max_rd_atomic = 1;

  // Move to Ready To Send (RTS) state
  rv = ibv_modify_qp(
      qp_,
      &attr,
      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
  GLOO_ENFORCE_EQ(rv, 0);
}

// Switches the pair into synchronous mode.
//
// Note: busy polling is NOT optional. Currently, since all pairs
// share a single completion channel, busy polling is mandatory
// through ibv_poll_cq(3). If a use case comes up for supporting
// synchronous mode where the calling thread should be suspended, this
// can be revisited and we can add a completion channel per pair.
//
void Pair::setSync(bool sync, bool busyPoll) {
  checkErrorState();
  if (!sync) {
    GLOO_THROW_INVALID_OPERATION_EXCEPTION("Can only switch to sync mode");
  }
  if (!busyPoll) {
    GLOO_THROW_INVALID_OPERATION_EXCEPTION(
        "The ibverbs transport only supports busy polling in sync mode");
  }

  // The notification mechanism for this pair's completion queue is
  // still armed. This means the device thread will still call
  // handleCompletions() one more time, but this is ignored.
  //
  // No need to lock a mutex; these are atomics.
  //
  sync_ = true;
  busyPoll_ = true;
}

void Pair::sendMemoryRegion(struct ibv_mr* src, int slot) {
  auto mr = make_unique<MemoryRegion>(dev_->pd_, src);
  struct ibv_sge list = mr->sge();
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = slot;
  wr.sg_list = &list;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = slot;

  // The work request is serialized and sent to the driver so it
  // doesn't need to be valid after the ibv_post_send call.
  struct ibv_send_wr* bad_wr = nullptr;
  int rv = ibv_post_send(qp_, &wr, &bad_wr);
  if (rv != 0) {
    signalIoFailure(GLOO_ERROR_MSG("ibv_post_send: ", rv));
  }

  // Keep memory region around until this send operation completes.
  // They are posted in FIFO order, but may complete in arbitrary order.
  // Therefore we store them in a map keyed on the buffer slot.
  GLOO_ENFORCE_EQ(mappedSendRegions_.count(slot), 0);
  mappedSendRegions_[slot] = std::move(mr);
}

const struct ibv_mr* Pair::getMemoryRegion(int slot) {
  std::unique_lock<std::mutex> lock(m_);
  if (sync_) {
    auto it = peerMemoryRegions_.find(slot);
    auto start = std::chrono::steady_clock::now();
    while (it == peerMemoryRegions_.end()) {
      lock.unlock();
      pollCompletions();
      lock.lock();
      if (timeout_ != kNoTimeout &&
          (std::chrono::steady_clock::now() - start) >= timeout_) {
        lock.unlock();
        signalIoFailure(
            GLOO_ERROR_MSG(
                "Timeout waiting for memory region from ",
                peer_.str()));
        GLOO_ENFORCE(false, "Unexpected code path");
      }
      it = peerMemoryRegions_.find(slot);
    }
    return &it->second;
  } else {
    auto pred = [&]{
      return peerMemoryRegions_.find(slot) != peerMemoryRegions_.end();
    };
    if (timeout_ == kNoTimeout) {
      // No timeout set. Wait for read to complete.
      cv_.wait(lock, pred);
    } else {
      auto done = cv_.wait_for(lock, timeout_, pred);
      if (!done) {
        signalIoFailure(
            GLOO_ERROR_MSG(
                "Timeout waiting for memory region from ",
                peer_.str()));
        GLOO_ENFORCE(false, "Unexpected code path");
      }
    }
    auto it = peerMemoryRegions_.find(slot);
    GLOO_ENFORCE(it != peerMemoryRegions_.end());
    return &it->second;
  }
}

void Pair::postReceive() {
  const auto& mr = mappedRecvRegions_[recvPosted_++ % kMaxBuffers];
  struct ibv_sge list = mr->sge();
  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.sg_list = &list;
  wr.num_sge = 1;

  // The work request is serialized and sent to the driver so it
  // doesn't need to be valid after the ibv_post_recv call.
  struct ibv_recv_wr* bad_wr = nullptr;
  auto rv = ibv_post_recv(qp_, &wr, &bad_wr);
  if (rv != 0) {
    signalIoFailure(GLOO_ERROR_MSG("ibv_post_recv: ", rv));
  }
}

std::unique_ptr<::gloo::transport::Buffer>
Pair::createSendBuffer(int slot, void* ptr, size_t size) {
  std::unique_lock<std::mutex> lock(m_);
  GLOO_ENFORCE_EQ(sendCompletionHandlers_.count(slot), 0);
  auto buffer = new Buffer(this, slot, ptr, size);
  sendCompletionHandlers_[slot] = buffer;
  return std::unique_ptr<::gloo::transport::Buffer>(buffer);
}

std::unique_ptr<::gloo::transport::Buffer>
Pair::createRecvBuffer(int slot, void* ptr, size_t size) {
  std::unique_lock<std::mutex> lock(m_);
  GLOO_ENFORCE_EQ(recvCompletionHandlers_.count(slot), 0);
  auto buffer = new Buffer(this, slot, ptr, size);
  recvCompletionHandlers_[slot] = buffer;
  sendMemoryRegion(buffer->mr_, buffer->slot_);
  return std::unique_ptr<::gloo::transport::Buffer>(buffer);
}

// Send from the specified buffer to remote side of pair.
void Pair::send(
    transport::UnboundBuffer* tbuf,
    uint64_t /* unused */,
    size_t /* unused */,
    size_t /* unused */) {
  GLOO_THROW_INVALID_OPERATION_EXCEPTION(
      "Unbound buffers not supported yet for ibverbs transport");
}

// Receive into the specified buffer from the remote side of pair.
void Pair::recv(
    transport::UnboundBuffer* tbuf,
    uint64_t /* unused */,
    size_t /* unused */,
    size_t /* unused */) {
  GLOO_THROW_INVALID_OPERATION_EXCEPTION(
      "Unbound buffers not supported yet for ibverbs transport");
}

// handleCompletionEvent is called by the device thread when it
// received an event for this pair's completion queue on its
// completion channel.
void Pair::handleCompletionEvent() {
  int rv;

  completionEventsHandled_++;

  // If in sync mode, the pair was just switched and this is
  // the last notification from the device thread because
  // the notification mechanism is not re-armed below.
  if (sync_) {
    return;
  }

  try {
    checkErrorState();

    // Arm notification mechanism for completion queue.
    rv = ibv_req_notify_cq(cq_, kNotifyOnAnyCompletion);
    GLOO_ENFORCE_EQ(rv, 0);

    // Now poll for work completions to drain the completion queue.
    std::unique_lock<std::mutex> lock(m_);
    pollCompletions();
  } catch (const ::gloo::IoException&) {
    // Catch IO exceptions on the event handling thread. The exception has
    // already been saved and user threads signaled.
  }
}

// Polls this pair's completion queue for work completions. When
// called from the device thread, this pair's mutex has already been
// acquired. When called from the user thread, the mutex won't be
// acquired (since there's only a single thread using this pair).
void Pair::pollCompletions() {
  std::array<struct ibv_wc, kCompletionQueueCapacity> wc;

  // Invoke handler for every work completion.
  for (;;) {
    auto nwc = ibv_poll_cq(cq_, wc.size(), wc.data());
    GLOO_ENFORCE_GE(nwc, 0);

    // Handle work completions
    for (int i = 0; i < nwc; i++) {
      checkErrorState();
      handleCompletion(&wc[i]);
    }

    // Break unless wc was filled
    if (nwc == 0 || nwc < wc.size()) {
      break;
    }
  }
}

void Pair::handleCompletion(struct ibv_wc* wc) {
  if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
    // Incoming RDMA write completed.
    // Slot is encoded in immediate data on receive work completion.
    // It is set in the Pair::send function.
    auto slot = wc->imm_data;
    GLOO_ENFORCE_EQ(
      wc->status,
      IBV_WC_SUCCESS,
      "Recv for slot ",
      slot,
      ": ",
      ibv_wc_status_str(wc->status));

    GLOO_ENFORCE(recvCompletionHandlers_[slot] != nullptr);
    recvCompletionHandlers_[slot]->handleCompletion(wc);

    // Backfill receive work requests.
    postReceive();
  } else if (wc->opcode == IBV_WC_RDMA_WRITE) {
    // Outbound RDMA write completed.
    // Slot is encoded in wr_id fields on send work request. Unlike
    // the receive work completions, the immediate data field on send
    // work requests are not pass to the respective work completion.
    auto slot = wc->wr_id;
    GLOO_ENFORCE_EQ(
      wc->status,
      IBV_WC_SUCCESS,
      "Send for slot ",
      slot,
      ": ",
      ibv_wc_status_str(wc->status));

    GLOO_ENFORCE(sendCompletionHandlers_[slot] != nullptr);
    sendCompletionHandlers_[slot]->handleCompletion(wc);
  } else if (wc->opcode == IBV_WC_RECV) {
    // Memory region recv completed.
    //
    // Only used by the remote side of the pair to pass ibv_mr's.
    // They are written to in FIFO order, so we can pick up
    // and use the first MemoryRegion instance in the list of
    // mapped receive regions.
    //
    // The buffer trying to write to this slot might be waiting for
    // the other side of this pair to send its memory region.
    // Lock access, and notify anybody waiting afterwards.
    //
    // Slot is encoded in immediate data on receive work completion.
    // It is set in the Pair::sendMemoryRegion function.
    auto slot = wc->imm_data;
    GLOO_ENFORCE_EQ(
      wc->status,
      IBV_WC_SUCCESS,
      "Memory region recv for slot ",
      slot,
      ": ",
      ibv_wc_status_str(wc->status));

    // Move ibv_mr from memory region 'inbox' to final slot.
    const auto& mr = mappedRecvRegions_[recvPosted_ % kMaxBuffers];
    peerMemoryRegions_[slot] = mr->mr();

    // Notify any buffer waiting for the details of its remote peer.
    cv_.notify_all();

    // Backfill receive work requests.
    postReceive();
  } else if (wc->opcode == IBV_WC_SEND) {
    // Memory region send completed.
    auto slot = wc->wr_id;
    GLOO_ENFORCE_EQ(
      wc->status,
      IBV_WC_SUCCESS,
      "Memory region send for slot ",
      slot,
      ": ",
      ibv_wc_status_str(wc->status));

    GLOO_ENFORCE_GT(mappedSendRegions_.size(), 0);
    GLOO_ENFORCE_EQ(mappedSendRegions_.count(slot), 1);
    mappedSendRegions_.erase(slot);
  } else {
    GLOO_ENFORCE(false, "Unexpected completion with opcode: ", wc->opcode);
  }
}

void Pair::send(Buffer* buffer, size_t offset, size_t length, size_t roffset) {
  struct ibv_sge list;
  list.addr = (uint64_t)buffer->ptr_ + offset;
  list.length = length;
  list.lkey = buffer->mr_->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = buffer->slot_;
  wr.sg_list = &list;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = buffer->slot_;

  const struct ibv_mr* peer = getMemoryRegion(buffer->slot_);
  GLOO_ENFORCE_NE(peer, (const struct ibv_mr*)nullptr);
  wr.wr.rdma.remote_addr = (uint64_t)peer->addr + roffset;
  wr.wr.rdma.rkey = peer->rkey;

  struct ibv_send_wr* bad_wr;
  auto rv = ibv_post_send(qp_, &wr, &bad_wr);
  if (rv != 0) {
    signalIoFailure(GLOO_ERROR_MSG("ibv_post_send: ", rv));
  }
}

void Pair::signalIoFailure(const std::string& msg) {
  std::lock_guard<std::mutex> lock(m_);
  auto ex = ::gloo::IoException(msg);
  if (ex_ == nullptr) {
    // If we haven't seen an error yet, store the exception to throw on future
    // calling threads.
    ex_ = std::make_exception_ptr(ex);
    // Loop through the completion handlers and signal that an error has
    // occurred.
    for (auto& it : recvCompletionHandlers_) {
      GLOO_ENFORCE(it.second != nullptr);
      it.second->signalError(ex_);
    }
    for (auto& it : sendCompletionHandlers_) {
      GLOO_ENFORCE(it.second != nullptr);
      it.second->signalError(ex_);
    }
  }
  // Finally, throw the exception on this thread.
  throw ex;
};

void Pair::checkErrorState() {
  // If we previously encountered an error, rethrow here.
  if (ex_ != nullptr) {
    std::rethrow_exception(ex_);
  }
}

} // namespace ibverbs
} // namespace transport
} // namespace gloo
