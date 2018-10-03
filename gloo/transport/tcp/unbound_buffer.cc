/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/transport/tcp/unbound_buffer.h"

#include <stdexcept>

#include "gloo/common/logging.h"
#include "gloo/transport/tcp/context.h"

namespace gloo {
namespace transport {
namespace tcp {

UnboundBuffer::UnboundBuffer(
    const std::shared_ptr<Context>& context,
    void* ptr,
    size_t size)
    : ::gloo::transport::UnboundBuffer(ptr, size),
      context_(context),
      recvCompletions_(0),
      recvRank_(-1),
      sendCompletions_(0),
      sendRank_(-1) {}

UnboundBuffer::~UnboundBuffer() {}

void UnboundBuffer::handleRecvCompletion(int rank) {
  std::lock_guard<std::mutex> lock(m_);
  recvCompletions_++;
  recvRank_ = rank;
  recvCv_.notify_one();
}

void UnboundBuffer::waitRecv(int* rank) {
  std::unique_lock<std::mutex> lock(m_);
  auto pred = [&]{
    return recvCompletions_ > 0;
  };
  recvCv_.wait(lock, pred);
  recvCompletions_--;
  if (rank != nullptr) {
    *rank = recvRank_;
  }
}

void UnboundBuffer::handleSendCompletion(int rank) {
  std::lock_guard<std::mutex> lock(m_);
  sendCompletions_++;
  sendRank_ = rank;
  sendCv_.notify_one();
}

void UnboundBuffer::waitSend(int* rank) {
  std::unique_lock<std::mutex> lock(m_);
  auto pred = [&]{
    return sendCompletions_ > 0;
  };
  sendCv_.wait(lock, pred);
  sendCompletions_--;
  if (rank != nullptr) {
    *rank = sendRank_;
  }
}

void UnboundBuffer::send(
    int dstRank,
    uint64_t slot,
    size_t offset,
    size_t nbytes) {
  if (nbytes == 0) {
    GLOO_ENFORCE_LT(offset, this->size);
    nbytes = this->size - offset;
  }
  context_->getPair(dstRank)->send(this, slot, offset, nbytes);
}

void UnboundBuffer::recv(
    int srcRank,
    uint64_t slot,
    size_t offset,
    size_t nbytes) {
  if (nbytes == 0) {
    GLOO_ENFORCE_LT(offset, this->size);
    nbytes = this->size - offset;
  }
  context_->getPair(srcRank)->recv(this, slot, offset, nbytes);
}

void UnboundBuffer::recv(
    std::vector<int> srcRanks,
    uint64_t slot,
    size_t offset,
    size_t nbytes) {
  if (nbytes == 0) {
    GLOO_ENFORCE_LT(offset, this->size);
    nbytes = this->size - offset;
  }
  context_->recvFromAny(this, slot, offset, nbytes, srcRanks);
}

} // namespace tcp
} // namespace transport
} // namespace gloo
