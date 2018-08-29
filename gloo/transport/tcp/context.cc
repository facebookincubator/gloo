/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/transport/tcp/context.h"

#include "gloo/common/error.h"
#include "gloo/transport/tcp/device.h"
#include "gloo/transport/tcp/pair.h"
#include "gloo/transport/tcp/unbound_buffer.h"

namespace gloo {
namespace transport {
namespace tcp {

Context::Context(std::shared_ptr<Device> device, int rank, int size)
    : ::gloo::transport::Context(rank, size), device_(device) {
}

Context::~Context() {
}

std::unique_ptr<transport::Pair>& Context::createPair(
    int rank,
    std::chrono::milliseconds timeout) {
  if (timeout < std::chrono::milliseconds::zero()) {
    GLOO_THROW_INVALID_OPERATION_EXCEPTION("Invalid timeout", timeout.count());
  }
  auto pair = new tcp::Pair(
      device_,
      timeout,
      [=] (uint64_t slot) -> UnboundBuffer* {
        return recvFromAnyCallback(rank, slot);
      });
  pairs_[rank] = std::unique_ptr<transport::Pair>(pair);
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
    std::vector<int> srcRanks) {
  for (;;) {
    // Find rank of pair we can attempt a recv from
    auto rank = recvFromAnyFindRank(buf, slot, srcRanks);
    if (rank == -1) {
      return;
    }
    // Try recv from returned rank
    auto pair = dynamic_cast<Pair*>(pairs_[rank].get());
    if (pair->tryRecv(buf, slot)) {
      return;
    }
  }
}

int Context::recvFromAnyFindRank(
    UnboundBuffer* buf,
    uint64_t slot,
    std::vector<int> srcRanks) {
  std::unique_lock<std::mutex> lock(m_);

  // See if there is a pending remote send that can fulfill this recv.
  auto it = pendingRemoteSend_.find(slot);
  if (it != pendingRemoteSend_.end()) {
    auto& ranks = it->second;
    for (const auto& srcRank : srcRanks) {
      if (ranks.count(srcRank) > 0) {
        // We've found a rank that could fulfill this recv.
        //
        // An entry is only valid once and since the caller of this
        // function will try and attempt the recv, we can now remove
        // the rank from the set (or remove the set itself).
        //
        if (ranks.size() == 1) {
          pendingRemoteSend_.erase(slot);
        } else {
          ranks.erase(srcRank);
        }
        return srcRank;
      }
    }
  }

  // No candidates; register buffer for recv
  auto set = std::unordered_set<int>(srcRanks.begin(), srcRanks.end());
  pendingRecv_.emplace(slot, std::make_tuple(buf, set));
  return -1;
}

UnboundBuffer* Context::recvFromAnyCallback(
    int rank,
    uint64_t slot) {
  std::unique_lock<std::mutex> lock(m_);

  // See if there is a pending recv for this slot.
  auto it = pendingRecv_.find(slot);
  if (it != pendingRecv_.end()) {
    auto buf = std::get<0>(it->second);
    const auto& ranks = std::get<1>(it->second);
    // If the rank of this peer is in the set of acceptable ranks for
    // this slot we can proceed and return the buffer to recv into.
    if (ranks.count(rank) > 0) {
      pendingRecv_.erase(it);
      return buf;
    }
  }

  // No candidates; register rank for pending remote send.
  pendingRemoteSend_[slot].insert(rank);
  return nullptr;
}

} // namespace tcp
} // namespace transport
} // namespace gloo
