/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/transport/context.h"

#include <deque>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace gloo {
namespace transport {
namespace tcp {

// Forward declaration
class Device;
class Pair;
class UnboundBuffer;

class Context : public ::gloo::transport::Context,
                public std::enable_shared_from_this<Context> {
 public:
  Context(std::shared_ptr<Device> device, int rank, int size);

  virtual ~Context();

  std::unique_ptr<transport::Pair>& createPair(
      int rank,
      std::chrono::milliseconds timeout) override;

  std::unique_ptr<transport::UnboundBuffer> createUnboundBuffer(
      void* ptr,
      size_t size) override;

 protected:
  std::shared_ptr<Device> device_;

  std::mutex m_;

  using pendingRecvTuple = std::tuple<UnboundBuffer*, std::unordered_set<int>>;

  // Buffers with pending receive operation by slot.
  std::unordered_map<uint64_t, std::deque<pendingRecvTuple>> pendingRecv_;

  // Per slot, map of rank to the number of pending send operations.
  std::unordered_map<uint64_t, std::unordered_map<int, int>> pendingRemoteSend_;

  // This function registers the specified unbound buffer for a receive
  // operation from any of the specified ranks.
  void recvFromAny(
      UnboundBuffer* buf,
      uint64_t slot,
      std::vector<int> srcRanks);

  int recvFromAnyFindRank(
      UnboundBuffer* buf,
      uint64_t slot,
      std::vector<int> srcRanks);

  UnboundBuffer* recvFromAnyCallback(
      int rank,
      uint64_t slot);

  friend class Pair;

  friend class UnboundBuffer;
};

} // namespace tcp
} // namespace transport
} // namespace gloo
