/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "floo/transport/tcp/buffer.h"

#include <string.h>

namespace floo {
namespace transport {
namespace tcp {

Buffer::Buffer(Pair* pair, int slot, void* ptr, size_t size)
    : ::floo::transport::Buffer(slot, ptr, size),
      pair_(pair),
      recvCompletions_(0),
      sendCompletions_(0) {}

Buffer::~Buffer() {
  pair_->unregisterBuffer(this);
}

void Buffer::handleRecvCompletion() {
  std::lock_guard<std::mutex> lock(m_);
  recvCompletions_++;
  recvCv_.notify_one();
}

void Buffer::waitRecv() {
  std::unique_lock<std::mutex> lock(m_);

  // Wait for completion
  while (recvCompletions_ == 0) {
    recvCv_.wait(lock);
  }
  recvCompletions_--;
}

void Buffer::handleSendCompletion() {
  std::lock_guard<std::mutex> lock(m_);
  sendCompletions_++;
  sendCv_.notify_one();
}

void Buffer::waitSend() {
  std::unique_lock<std::mutex> lock(m_);

  // Wait for completion
  while (sendCompletions_ == 0) {
    sendCv_.wait(lock);
  }
  sendCompletions_--;
}

void Buffer::send(size_t offset, size_t length) {
  Op op;

  memset(&op, 0, sizeof(op));

  op.preamble_.opcode_ = 0;
  op.preamble_.slot_ = slot_;
  op.preamble_.offset_ = offset;
  op.preamble_.length_ = length;
  op.buf_ = this;

  // Pass to pair
  pair_->send(op);
}

} // namespace tcp
} // namespace transport
} // namespace floo
