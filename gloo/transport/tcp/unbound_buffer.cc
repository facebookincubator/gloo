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
      sendCompletions_(0) {}

UnboundBuffer::~UnboundBuffer() {}

void UnboundBuffer::handleRecvCompletion() {
  std::lock_guard<std::mutex> lock(m_);
  recvCompletions_++;
  recvCv_.notify_one();
}

void UnboundBuffer::waitRecv() {
  std::unique_lock<std::mutex> lock(m_);
  auto pred = [&]{
    return recvCompletions_ > 0;
  };
  recvCv_.wait(lock, pred);
  recvCompletions_--;
}

void UnboundBuffer::handleSendCompletion() {
  std::lock_guard<std::mutex> lock(m_);
  sendCompletions_++;
  sendCv_.notify_one();
}

void UnboundBuffer::waitSend() {
  std::unique_lock<std::mutex> lock(m_);
  auto pred = [&]{
    return sendCompletions_ > 0;
  };
  sendCv_.wait(lock, pred);
  sendCompletions_--;
}

void UnboundBuffer::send(int dstRank, uint64_t slot) {
  context_->getPair(dstRank)->send(this, slot);
}

void UnboundBuffer::recv(int srcRank, uint64_t slot) {
  context_->getPair(srcRank)->recv(this, slot);
}

} // namespace tcp
} // namespace transport
} // namespace gloo
