/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "gloo/common/memory.h"
#include "gloo/transport/context.h"

namespace gloo {
namespace transport {
namespace tcp {

// Forward declaration
class Context;
class Device;
class Pair;
class UnboundBuffer;

class Context : public ::gloo::transport::Context,
                      public std::enable_shared_from_this<Context> {
 public:
  Context(std::shared_ptr<Device> device, int rank, int size);

  virtual ~Context();

  std::unique_ptr<transport::Pair>& createPair(int rank) override;

  std::unique_ptr<transport::UnboundBuffer> createUnboundBuffer(
      void* ptr,
      size_t size) override;

 protected:
  std::shared_ptr<Device> device_;

  using pendingRecvTuple = std::tuple<
      WeakNonOwningPtr<UnboundBuffer>,
      size_t,
      size_t,
      std::unordered_set<int>>;

  // Buffers with pending receive operation by slot.
  std::unordered_map<uint64_t, std::deque<pendingRecvTuple>> pendingRecv_;

  // This function registers the specified unbound buffer for a receive
  // operation from any of the specified ranks.
  void recvFromAny(
      UnboundBuffer* buf,
      uint64_t slot,
      size_t offset,
      size_t nbytes,
      std::vector<int> srcRanks);

  int recvFromAnyFindRank(
      UnboundBuffer* buf,
      uint64_t slot,
      size_t offset,
      size_t nbytes,
      const std::vector<int>& srcRanks);

  // Allowed to be called only by ContextMutator::findRecvFromAny,
  // where the context lock is already held.
  bool findRecvFromAny(
      uint64_t slot,
      int rank,
      WeakNonOwningPtr<tcp::UnboundBuffer>* buf,
      size_t* offset,
      size_t* nbytes);

  // Set exception on every pair in this context. This is called when
  // waiting for a send or recv operation on an unbound buffer times
  // out. All pairs should be signaled and closed in that event.
  void signalException(const std::string& msg);

  friend class ContextMutator;

  friend class UnboundBuffer;

  friend class Pair;
};

} // namespace tcp
} // namespace transport
} // namespace gloo
