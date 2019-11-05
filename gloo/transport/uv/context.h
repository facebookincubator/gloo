/**
 * Copyright (c) 2019-present, Facebook, Inc.
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

#include <gloo/common/memory.h>
#include <gloo/transport/context.h>

//
// READ THIS FIRST!
//
// This file is a copy of "gloo/transport/tcp/context.h".
//
// Any modifications should be made to that file as well. This file is
// not a top level construct because it assumes the transport supports
// unbound buffers, which is not yet the case for the ibverbs
// transport. When that is done, the functionality in this file may be
// promoted to the top level context class.
//

namespace gloo {
namespace transport {
namespace uv {

// PendingOpCount keeps track of the number of remote pending
// operations (both pending send and receive operations) for a single
// slot, across all pairs. This used to be tracked in the pairs
// themselves, but to support receive-from-any we need a centralized
// view into all pending send operations. This class facilitaties that
// centralized view for both pending send and receive operations.
//
// The count for a slot is accessed through the transport context
// object defined further down in this file.
//
class PendingOpCount final {
 public:
  using count_t = int8_t;

 private:
  // Counts either pending send or pending recv operations.
  // Keeps track of number of non-zero entries such that
  // instances with all zero counts can be cleaned up.
  class Count final {
   public:
    explicit Count(size_t length) : count_(length), nonzero_(0) {}

    bool empty() const {
      return nonzero_ == 0;
    }

    count_t get(size_t rank) const {
      return count_[rank];
    }

    // Update the count for the specified rank.
    // If the count for this rank changes from zero to non-zero, it
    // increments the non-zero counter. If it changes from non-zero to
    // zero, it decrements the non-zero counter.
    count_t update(size_t rank, count_t v) {
      auto cur = count_[rank];
      if (cur == 0) {
        cur += v;
        if (cur != 0) {
          nonzero_++;
        }
      } else {
        cur += v;
        if (cur == 0) {
          nonzero_--;
        }
      }
      count_[rank] = cur;
      return cur;
    }

   private:
    std::vector<count_t> count_;
    ssize_t nonzero_;
  };

 public:
  explicit PendingOpCount(size_t length) : send_(length), recv_(length) {}

  bool empty() {
    return send_.empty() && recv_.empty();
  }

  count_t getSend(size_t rank) {
    return send_.get(rank);
  }

  count_t getRecv(size_t rank) {
    return recv_.get(rank);
  }

  count_t updateSend(size_t rank, count_t v) {
    return send_.update(rank, v);
  }

  count_t updateRecv(size_t rank, count_t v) {
    return recv_.update(rank, v);
  }

 private:
  Count send_;
  Count recv_;
};

// Forward declaration
class Context;
class Device;
class Pair;
class UnboundBuffer;

// Short lived object that is returned to Pair functions. It has a
// lock on the context object so that it can atomically retrieve and
// mutate the pending operation count as well as check for pending
// send or receive operations.
//
// Slots are often ephemeral identifiers. This object lazily creates
// pending op count entries for new slots, and removes them if they
// are no longer needed. Destroyed entries may be recreated later.
//
// The object is expected to be destructed as soon as it leaves scope.
//
class ContextMutator {
  using count_t = PendingOpCount::count_t;
  using PendingOpCountMap = std::unordered_map<uint64_t, PendingOpCount>;
  using PendingOpCountIterator = PendingOpCountMap::iterator;

 public:
  ContextMutator(Context& context, uint64_t slot, uint64_t rank);

  ~ContextMutator();

  // Current number of remote pending recv operations for rank.
  count_t getRemotePendingRecv();

  // Current number of remote pending send operations for rank.
  count_t getRemotePendingSend();

  // Update number of remote pending recv operations by `v`.
  count_t updateRemotePendingRecv(count_t v);

  // Update number of remote pending send operations by `v`.
  count_t updateRemotePendingSend(count_t v);

  // Find buffer for which we should execute a recv operation.
  bool findRecvFromAny(
      WeakNonOwningPtr<UnboundBuffer>* buf,
      size_t* offset,
      size_t* nbytes);

 protected:
  std::lock_guard<std::mutex> lock_;
  Context& context_;
  const uint64_t slot_;
  const uint64_t rank_;

  // Every operation that requires the ContextMutator will access the
  // pending operation count. Therefore we can perform lookup of this
  // object at construction time.
  PendingOpCountIterator it_;

  // If the existing iterator does not exists, insert new pending
  // operation count object and return the new iterator.
  PendingOpCountIterator insertIfNotExists();
};

class Context final : public ::gloo::transport::Context,
                      public std::enable_shared_from_this<Context> {
 public:
  Context(std::shared_ptr<Device> device, int rank, int size);

  virtual ~Context();

  std::unique_ptr<transport::Pair>& createPair(int rank) override;

  std::unique_ptr<transport::UnboundBuffer> createUnboundBuffer(
      void* ptr,
      size_t size) override;

 private:
  std::shared_ptr<Device> device_;

  std::mutex m_;

  using pendingRecvTuple = std::tuple<
      WeakNonOwningPtr<UnboundBuffer>,
      size_t,
      size_t,
      std::unordered_set<int>>;

  // Buffers with pending receive operation by slot.
  std::unordered_map<uint64_t, std::deque<pendingRecvTuple>> pendingRecv_;

  // Pending remote operation count by slot number.
  std::unordered_map<uint64_t, PendingOpCount> remotePendingOp_;

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
      WeakNonOwningPtr<UnboundBuffer>* buf,
      size_t* offset,
      size_t* nbytes);

  friend class ContextMutator;

  friend class UnboundBuffer;
};

} // namespace uv
} // namespace transport
} // namespace gloo
