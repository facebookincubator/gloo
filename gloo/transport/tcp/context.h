/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
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

// PendingOpTally keeps track of the number of remote pending
// operations (both pending send and receive operations) for a single
// slot, across all pairs. This used to be tracked in the pairs
// themselves, but to support receive-from-any we need a centralized
// view into all pending send operations. This class facilitaties that
// centralized view for both pending send and receive operations.
//
// The tally for a slot is accessed through the transport context
// object defined further down in this file.
//
class PendingOpTally {
 public:
  using tally_count_t = int8_t;

 private:
  // Counts either pending send or pending recv operations.
  // Keeps track of number of non-zero entries such that
  // we can remove tallies that are no longer needed.
  class Tally {
   public:
    explicit Tally(size_t length) : tally(length), nz(0) {}

    tally_count_t get(size_t rank) {
      return tally[rank];
    }

    tally_count_t update(size_t rank, tally_count_t v) {
      auto cur = tally[rank];
      if (cur == 0) {
        cur += v;
        if (cur != 0) {
          nz++;
        }
      } else {
        cur += v;
        if (cur == 0) {
          nz--;
        }
      }
      tally[rank] = cur;
      return cur;
    }

    std::vector<int8_t> tally;
    ssize_t nz;
  };

 public:
  explicit PendingOpTally(size_t length) : send_(length), recv_(length) {}

  bool empty() {
    return send_.nz == 0 && recv_.nz == 0;
  }

  tally_count_t getSend(size_t rank) {
    return send_.get(rank);
  }

  tally_count_t getRecv(size_t rank) {
    return recv_.get(rank);
  }

  tally_count_t updateSend(size_t rank, tally_count_t v) {
    return send_.update(rank, v);
  }

  tally_count_t updateRecv(size_t rank, tally_count_t v) {
    return recv_.update(rank, v);
  }

 protected:
  Tally send_;
  Tally recv_;
};

// Forward declaration
class Context;
class Device;
class Pair;
class UnboundBuffer;

// Short lived object that is returned to Pair functions. It has a
// lock on the context object so that it can atomically retrieve and
// mutate the pending operation tally as well as check for pending
// send or receive operations.
//
// It is expected to be destructed as soon as it leaves scope.
//
class ContextMutator {
  using tally_count_t = PendingOpTally::tally_count_t;
  using PendingOpTallyMap = std::unordered_map<uint64_t, PendingOpTally>;
  using PendingOpTallyIterator = PendingOpTallyMap::iterator;

 public:
  ContextMutator(Context& context, uint64_t slot, uint64_t rank);

  ~ContextMutator();

  // Current number of remote pending recv operations for rank.
  tally_count_t getRemotePendingRecv();

  // Current number of remote pending send operations for rank.
  tally_count_t getRemotePendingSend();

  // Update number of remote pending recv operations by `v`.
  tally_count_t updateRemotePendingRecv(tally_count_t v);

  // Update number of remote pending send operations by `v`.
  tally_count_t updateRemotePendingSend(tally_count_t v);

  // Find buffer for which we should execute a recv operation.
  UnboundBuffer* findRecvFromAny(size_t* offset, size_t* nbytes);

 protected:
  std::unique_lock<std::mutex> lock_;
  Context& context_;
  const uint64_t slot_;
  const uint64_t rank_;

  // Every operation that requires the ContextMutator will access the
  // pending operation tally. Therefore we can perform lookup of this
  // object at construction time.
  PendingOpTallyIterator it_;

  // If the existing iterator does not exists, insert new pending
  // operation tally object and return the new iterator.
  PendingOpTallyIterator insertIfNotExists();
};

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

  std::mutex m_;

  using pendingRecvTuple =
      std::tuple<UnboundBuffer*, size_t, size_t, std::unordered_set<int>>;

  // Buffers with pending receive operation by slot.
  std::unordered_map<uint64_t, std::deque<pendingRecvTuple>> pendingRecv_;

  // Pending remote operation tally by slot number.
  std::unordered_map<uint64_t, PendingOpTally> remotePendingOp_;

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
      std::vector<int> srcRanks);

  UnboundBuffer* findRecvFromAny(
      uint64_t slot,
      int rank,
      size_t* offset,
      size_t* nbytes);

  // Set exception on every pair in this context. This is called when
  // waiting for a send or recv operation on an unbound buffer times
  // out. All pairs should be signaled and closed in that event.
  void signalException(const std::string& msg);

  friend class ContextMutator;

  friend class UnboundBuffer;
};

} // namespace tcp
} // namespace transport
} // namespace gloo
