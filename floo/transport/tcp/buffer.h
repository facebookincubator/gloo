/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <condition_variable>
#include <map>
#include <mutex>
#include <queue>

#include "floo/transport/buffer.h"
#include "floo/transport/tcp/device.h"
#include "floo/transport/tcp/pair.h"

namespace floo {
namespace transport {
namespace tcp {

class Buffer : public ::floo::transport::Buffer {
 public:
  virtual ~Buffer();

  virtual void send(size_t offset, size_t length) override;

  virtual void waitRecv() override;
  virtual void waitSend() override;

 protected:
  // May only be constructed from helper function in pair.cc
  Buffer(Pair* pair, int slot, void* ptr, size_t size);

  void handleRecvCompletion();
  void handleSendCompletion();

  Pair* pair_;

  std::mutex m_;
  std::condition_variable recvCv_;
  std::condition_variable sendCv_;

  int recvCompletions_;
  int sendCompletions_;

  friend class Pair;
};

} // namespace tcp
} // namespace transport
} // namespace floo
