/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
#include <map>
#include <mutex>
#include <queue>

#include "gloo/transport/buffer.h"
#include "gloo/transport/tcp/device.h"
#include "gloo/transport/tcp/pair.h"

namespace gloo {
namespace transport {
namespace tcp {

class Buffer : public ::gloo::transport::Buffer {
 public:
  virtual ~Buffer();

  virtual void send(size_t offset, size_t length, size_t roffset = 0) override;

  virtual void waitRecv() override;
  virtual void waitSend() override;

  void handleRecvCompletion();
  void handleSendCompletion();

 protected:
  // May only be constructed from helper function in pair.cc
  Buffer(Pair* pair, int slot, void* ptr, size_t size);

  Pair* pair_;

  std::mutex m_;
  std::condition_variable recvCv_;
  std::condition_variable sendCv_;

  int recvCompletions_;
  int sendCompletions_;
  std::atomic<int> sendPending_;

  std::exception_ptr ex_;

  // Throws if an exception if set.
  void throwIfException();

  // Set exception and wake up any waitRecv/waitSend threads.
  void signalException(std::exception_ptr);

  friend class Pair;
};

} // namespace tcp
} // namespace transport
} // namespace gloo
