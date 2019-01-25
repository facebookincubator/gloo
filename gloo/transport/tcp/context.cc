/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/transport/tcp/context.h"

#include "gloo/common/error.h"
#include "gloo/common/logging.h"
#include "gloo/transport/tcp/device.h"
#include "gloo/transport/tcp/pair.h"
#include "gloo/transport/tcp/unbound_buffer.h"

namespace gloo {
namespace transport {
namespace tcp {

ContextMutator::ContextMutator(Context& context, size_t slot, size_t rank)
    : lock_(context.m_),
      context_(context),
      slot_(slot),
      rank_(rank),
      it_(context_.remotePendingOp_.find(slot)) {}

ContextMutator::~ContextMutator() {
  if (it_ != context_.remotePendingOp_.end() && it_->second.empty()) {
    context_.remotePendingOp_.erase(it_);
  }
}

int8_t ContextMutator::getRemotePendingRecv() {
  if (it_ == context_.remotePendingOp_.end()) {
    return 0;
  }
  return it_->second.getRecv(rank_);
}

int8_t ContextMutator::getRemotePendingSend() {
  if (it_ == context_.remotePendingOp_.end()) {
    return 0;
  }
  return it_->second.getSend(rank_);
}

int8_t ContextMutator::updateRemotePendingRecv(int8_t v) {
  auto it = insertIfNotExists();
  return it->second.updateRecv(rank_, v);
}

int8_t ContextMutator::updateRemotePendingSend(int8_t v) {
  auto it = insertIfNotExists();
  return it->second.updateSend(rank_, v);
}

ContextMutator::PendingOpTallyIterator ContextMutator::insertIfNotExists() {
  if (it_ == context_.remotePendingOp_.end()) {
    std::tie(it_, std::ignore) =
        context_.remotePendingOp_.emplace(slot_, PendingOpTally(context_.size));
  }
  return it_;
}

UnboundBuffer* ContextMutator::findRecvFromAny(size_t* offset, size_t* nbytes) {
  return context_.findRecvFromAny(slot_, rank_, offset, nbytes);
}

Context::Context(std::shared_ptr<Device> device, int rank, int size)
    : ::gloo::transport::Context(rank, size), device_(device) {}

Context::~Context() {}

std::unique_ptr<transport::Pair>& Context::createPair(int rank) {
  pairs_[rank] = std::unique_ptr<transport::Pair>(
      new tcp::Pair(device_, shared_from_this(), rank, getTimeout()));
  return pairs_[rank];
}

std::unique_ptr<transport::UnboundBuffer> Context::createUnboundBuffer(
    void* ptr,
    size_t size) {
  auto buf = new tcp::UnboundBuffer(shared_from_this(), ptr, size);
  return std::unique_ptr<transport::UnboundBuffer>(buf);
}

void Context::recvFromAny(
    UnboundBuffer* buf,
    uint64_t slot,
    size_t offset,
    size_t nbytes,
    std::vector<int> srcRanks) {
  for (;;) {
    // Find rank of pair we can attempt a recv from
    auto rank = recvFromAnyFindRank(buf, slot, offset, nbytes, srcRanks);
    if (rank == -1) {
      return;
    }
    // Try recv from returned rank
    auto ptr = pairs_[rank].get();
    GLOO_ENFORCE(ptr != nullptr);
    auto pair = dynamic_cast<Pair*>(ptr);
    GLOO_ENFORCE(pair != nullptr);
    if (pair->tryRecv(buf, slot, offset, nbytes)) {
      return;
    }
  }
}

int Context::recvFromAnyFindRank(
    UnboundBuffer* buf,
    uint64_t slot,
    size_t offset,
    size_t nbytes,
    std::vector<int> srcRanks) {
  std::unique_lock<std::mutex> lock(m_);

  // See if there is a pending remote send that can fulfill this recv.
  auto it = remotePendingOp_.find(slot);
  if (it != remotePendingOp_.end()) {
    auto& remotePendingOps = it->second;

    // Doing a linear search to find eligible ranks is suboptimal in
    // terms of performance but is functionally correct.
    for (const auto& srcRank : srcRanks) {
      if (remotePendingOps.getSend(srcRank) > 0) {
        // We've found a rank that could fulfill this recv.
        //
        // The caller of this function will try and attempt a recv
        // which will decrement the remote pending sends counter.
        //
        return srcRank;
      }
    }
  }

  // No candidates; register buffer for recv
  auto set = std::unordered_set<int>(srcRanks.begin(), srcRanks.end());
  pendingRecv_[slot].emplace_back(std::make_tuple(buf, offset, nbytes, set));
  return -1;
}

// Allowed to be called only by ContextMutator::findRecvFromAny.
// The context lock is already held by the caller.
UnboundBuffer* Context::findRecvFromAny(
    uint64_t slot,
    int rank,
    size_t* offset,
    size_t* nbytes) {
  // See if there is a pending recv for this slot.
  auto pit = pendingRecv_.find(slot);
  if (pit != pendingRecv_.end()) {
    auto& recvs = pit->second;

    // Iterate over available buffers to find a match.
    for (auto rit = recvs.begin(); rit != recvs.end(); rit++) {
      const auto& ranks = std::get<3>(*rit);

      // If the rank of this peer is in the set of acceptable ranks for
      // this slot we can proceed and return the buffer to recv into.
      if (ranks.count(rank) > 0) {
        // Capture values to return.
        auto buf = std::get<0>(*rit);
        *offset = std::get<1>(*rit);
        *nbytes = std::get<2>(*rit);
        // Cleanup.
        recvs.erase(rit);
        if (recvs.empty()) {
          pendingRecv_.erase(pit);
        }
        return buf;
      }
    }
  }

  return nullptr;
}

void Context::signalException(const std::string& msg) {
  std::unique_lock<std::mutex> lock(m_);
  for (auto& pair : pairs_) {
    if (pair) {
      reinterpret_cast<tcp::Pair*>(pair.get())->signalExceptionExternal(msg);
    }
  }
}

} // namespace tcp
} // namespace transport
} // namespace gloo
