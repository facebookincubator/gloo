/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "gloo/transport/unbound_buffer.h"

#include <condition_variable>
#include <memory>
#include <mutex>

namespace gloo {
namespace transport {
namespace tcp {

// Forward declaration
class Context;
class Pair;

class UnboundBuffer : public ::gloo::transport::UnboundBuffer {
 public:
  UnboundBuffer(
      const std::shared_ptr<Context>& context,
      void* ptr,
      size_t size);

  virtual ~UnboundBuffer();

  // If specified, the source of this recv is stored in the rank pointer.
  void waitRecv(int* rank, std::chrono::milliseconds timeout) override;

  // If specified, the destination of this send is stored in the rank pointer.
  void waitSend(int* rank, std::chrono::milliseconds timeout) override;

  void send(int dstRank, uint64_t slot, size_t offset, size_t nbytes = 0)
      override;

  void recv(int srcRank, uint64_t slot, size_t offset, size_t nbytes = 0)
      override;

  void recv(
      std::vector<int> srcRanks,
      uint64_t slot,
      size_t offset,
      size_t nbytes) override;

 protected:
  void handleRecvCompletion(int rank);
  void handleSendCompletion(int rank);

  std::shared_ptr<Context> context_;

  std::mutex m_;
  std::condition_variable recvCv_;
  std::condition_variable sendCv_;

  int recvCompletions_;
  int recvRank_;
  int sendCompletions_;
  int sendRank_;

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
