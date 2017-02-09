/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <stddef.h>
#include <string.h>

#include "floo/allreduce.h"
#include "floo/context.h"

namespace floo {

template <typename T>
class AllreduceRing : public Allreduce<T> {
 public:
  AllreduceRing(
      const std::shared_ptr<Context>& context,
      std::vector<T*> ptrs,
      int dataSize,
      typename Allreduce<T>::ReduceFunction fn = nullptr)
      : Allreduce<T>(context, fn),
        ptrs_(ptrs),
        dataSize_(dataSize),
        dataSizeBytes_(dataSize * sizeof(T)),
        leftPair_(this->getLeftPair()),
        rightPair_(this->getRightPair()) {
    inbox_ = static_cast<T*>(malloc(dataSizeBytes_));
    outbox_ = static_cast<T*>(malloc(dataSizeBytes_));

    // Buffer to send to (rank+1).
    sendDataBuf_ = rightPair_->createSendBuffer(0, outbox_, dataSizeBytes_);

    // Buffer that (rank-1) writes to.
    recvDataBuf_ = leftPair_->createRecvBuffer(0, inbox_, dataSizeBytes_);

    // Dummy buffers for localized barrier.
    // Before sending to the right, we only need to know that the node
    // on the right is done using the inbox that's about to be written
    // into. No need for a global barrier.
    sendNotificationBuf_ =
        leftPair_->createSendBuffer(1, &dummy_, sizeof(dummy_));
    recvNotificationBuf_ =
        rightPair_->createRecvBuffer(1, &dummy_, sizeof(dummy_));
  }

  virtual ~AllreduceRing() {
    if (inbox_ != nullptr) {
      free(inbox_);
    }
    if (outbox_ != nullptr) {
      free(outbox_);
    }
  }

  void run() {
    // Reduce specified pointers into ptrs_[0]
    for (int i = 1; i < ptrs_.size(); i++) {
      this->fn_(ptrs_[0], ptrs_[i], dataSize_);
    }

    // Intialize outbox with locally reduced values
    memcpy(outbox_, ptrs_[0], dataSizeBytes_);

    int numRounds = this->contextSize_ - 1;
    for (int round = 0; round < numRounds; round++) {
      // Initiate write to inbox of node on the right
      sendDataBuf_->send();

      // Wait for inbox write from node on the left
      recvDataBuf_->waitRecv();

      // Reduce
      this->fn_(ptrs_[0], inbox_, dataSize_);

      // Wait for outbox write to complete
      sendDataBuf_->waitSend();

      // Prepare for next round if necessary
      if (round < (numRounds - 1)) {
        memcpy(outbox_, inbox_, dataSizeBytes_);
      }

      // Send notification to node on the left that
      // this node is ready for an inbox write.
      sendNotificationBuf_->send();

      // Wait for notification from node on the right
      recvNotificationBuf_->waitRecv();
    }

    // Broadcast ptrs_[0]
    for (int i = 1; i < ptrs_.size(); i++) {
      memcpy(ptrs_[i], ptrs_[0], dataSizeBytes_);
    }
  }

 protected:
  std::vector<T*> ptrs_;
  int dataSize_;
  int dataSizeBytes_;

  std::unique_ptr<transport::Pair>& leftPair_;
  std::unique_ptr<transport::Pair>& rightPair_;

  T* inbox_;
  T* outbox_;
  std::unique_ptr<transport::Buffer> sendDataBuf_;
  std::unique_ptr<transport::Buffer> recvDataBuf_;

  int dummy_;
  std::unique_ptr<transport::Buffer> sendNotificationBuf_;
  std::unique_ptr<transport::Buffer> recvNotificationBuf_;
};

} // namespace floo
