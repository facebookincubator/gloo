/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/transport/unbound_buffer.h"

#include <condition_variable>
#include <mutex>

namespace gloo {
namespace transport {
namespace tcp {

// Forward declaration
class Pair;

class UnboundBuffer : public ::gloo::transport::UnboundBuffer {
 public:
  UnboundBuffer(void* ptr, size_t size);

  virtual ~UnboundBuffer();

  virtual void waitRecv() override;
  virtual void waitSend() override;

 protected:
  void handleRecvCompletion();
  void handleSendCompletion();

  std::mutex m_;
  std::condition_variable recvCv_;
  std::condition_variable sendCv_;

  int recvCompletions_;
  int sendCompletions_;

  friend class Pair;
};

} // namespace tcp
} // namespace transport
} // namespace gloo
