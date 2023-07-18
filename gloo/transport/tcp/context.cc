/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/transport/tcp/context.h"

#include <cstring>

#include "gloo/common/error.h"
#include "gloo/common/logging.h"
#include "gloo/common/utils.h"
#include "gloo/transport/tcp/device.h"
#include "gloo/transport/tcp/pair.h"
#include "gloo/transport/tcp/unbound_buffer.h"

namespace gloo {
namespace transport {
namespace tcp {

Context::Context(std::shared_ptr<Device> device, int rank, int size)
    : ::gloo::transport::Context(rank, size), device_(std::move(device)) {}

Context::~Context() {
  // Pairs refer to device by raw pointer.
  // Ensure they are destructed before the device.
  pairs_.clear();
  device_.reset();
}

void Context::createAndConnectAllPairs(IStore &store) {
  // Here instead of sending N addresses to store,
  // we send only 1 device address (since they are all the same)
  // and N sequence numbers to differentiate them.
  // This reduces the load on store from cubic down to quadratic.
  //
  // In theory if we can further flatten this to linear if we use
  // self rank number during initial peer mesh connection identification
  // (just so we don't need to obtain the expected seq num from remote peer
  // to identify self) but that would require a somewhat more major change
  // to the seq num allocation logic and due to the layering in this lib
  // it's not super straightforward so left for folks having more bandwidth
  // later on.

  int localRank = 0;
  bool localRankSet = false;
  auto localHostName = getHostname();

  // We will create all the pairs including self
  // the self pair will not be connected
  // it's just to keep the later seq num matching logic simple
  std::vector<ssize_t> pairIdentifiers;
  for (int i = 0; i < size; i++) {
    auto& pair = createPair(i);
    pairIdentifiers.emplace_back(
        static_cast<Pair*>(pair.get())->address().getSeq());
  }

  // Obtain the pair object for this rank
  // and tack on all the pair identifiers used for the remote side
  // to identify themselves use during the rendezvous process
  // TODO:
  // Seems like logically the remote rank should really just report its rank
  // during rendezvous instead of the device seq number.
  // That would allow it to bypass the seemingly unnecessary O(n) store access,
  // just to get the seq number the other side is expecting.
  // However this seems to require some major surgery to the stack that
  // Pieter originally wrote (since the connect() is at `Device` level,
  // which does not have the rank info hosted at a higher `Pair` level).
  // So better safe than sorry for now we try to minimize the changeset needed.
  const auto& currentRankPair = getPair(rank);
  auto deviceAddress = Address(
      static_cast<const Pair*>(currentRankPair.get())->address().getSockaddr());
  Rank currentRankInfo(
      localHostName, deviceAddress.bytes(), std::move(pairIdentifiers));
  store.set(std::to_string(rank), currentRankInfo.bytes());

  // Connect every pair
  for (int i = 0; i < size; i++) {
    if (i == rank) {
      // at this point we have enumerated all the ranks located on this host
      // up to the current rank, so the current `localRank` number is
      // what we'll set to the pairs.
      localRankSet = true;
      // We are not going to connect self.
      continue;
    }

    // Wait for address of other side of this pair to become available
    std::ostringstream key;
    key << i;
    store.wait({key.str()}, getTimeout());

    // Connect to other side of this pair
    std::vector<char> rankInfoBytes = store.get(key.str());
    Rank remoteRankInfo(rankInfoBytes);
    const auto& remoteHostname = remoteRankInfo.hostname;
    if (!localRankSet && remoteHostname == localHostName) {
      ++localRank;
    }

    const auto& pair = getPair(i);
    auto remoteDeviceAddr = Address(remoteRankInfo.addressBytes).getSockaddr();
    auto remoteAddr =
        Address(remoteDeviceAddr, remoteRankInfo.pairIdentifiers[rank]);
    pair->connect(remoteAddr.bytes());
  }

  // Set the local rank info for all mesh pairs involving current rank
  for (int i = 0; i < size; i++) {
    if (i == rank) {
      continue;
    }
    const auto& pair = getPair(i);
    pair->setLocalRank(localRank);
  }
}

std::unique_ptr<transport::Pair>& Context::createPair(int rank) {
  pairs_[rank] = std::unique_ptr<transport::Pair>(
      new tcp::Pair(this, device_.get(), rank, getTimeout()));
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
    const std::vector<int>& srcRanks) {
  std::unique_lock<std::mutex> lock(mutex_);

  // See if there is a remote pending send that can fulfill this recv.
  auto it = findPendingOperations(slot);
  if (it != pendingOperations_.end()) {
    auto& pendingOperation = *it;

    // Out of all remote pending sends, find the first one
    // that exists in the set of eligible ranks.
    for (const auto rank : pendingOperation.getSendList()) {
      for (const auto srcRank : srcRanks) {
        if (rank == srcRank) {
          // We've found a rank that could fulfill this recv.
          //
          // The caller of this function will try and attempt a recv,
          // which will remove this remote pending send operation,
          // if it's still around.
          //
          return rank;
        }
      }
    }
  }

  // No candidates; register buffer for recv
  pendingRecv_[slot].emplace_back(
      buf->getWeakNonOwningPtr(),
      offset,
      nbytes,
      std::unordered_set<int>(srcRanks.begin(), srcRanks.end()));
  return -1;
}

// Allowed to be called only by ContextMutator::findRecvFromAny,
// where the context lock is already held.
bool Context::findRecvFromAny(
    uint64_t slot,
    int rank,
    WeakNonOwningPtr<UnboundBuffer>* buf,
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
        *buf = std::get<0>(*rit);
        *offset = std::get<1>(*rit);
        *nbytes = std::get<2>(*rit);
        // Cleanup.
        recvs.erase(rit);
        if (recvs.empty()) {
          pendingRecv_.erase(pit);
        }
        return true;
      }
    }
  }

  return false;
}

void Context::signalException(const std::string& msg) {
  // The `pairs_` vector is logically constant. After the context and
  // all of its pairs have been created it is not mutated until the
  // context is destructed. Therefore, we don't need to acquire this
  // context's instance lock before looping over `pairs_`.
  for (auto& pair : pairs_) {
    if (pair) {
      reinterpret_cast<tcp::Pair*>(pair.get())->signalExceptionExternal(msg);
    }
  }
}

Rank::Rank(const std::vector<char>& bytes) {
  size_t bytesOffset = 0;
  // hostname
  size_t hostnameSz;
  std::memcpy(&hostnameSz, bytes.data(), sizeof(hostnameSz));
  hostname = std::string(bytes.data() + sizeof(hostnameSz), hostnameSz);
  bytesOffset += sizeof(hostnameSz) + hostnameSz;
  // address
  size_t addrSz;
  std::memcpy(&addrSz, bytes.data() + bytesOffset, sizeof(addrSz));
  auto beginIter = bytes.begin() + bytesOffset + sizeof(addrSz);
  auto endIter = beginIter + addrSz;
  addressBytes = std::vector<char>(beginIter, endIter);
  bytesOffset += sizeof(addrSz) + addrSz;
  // pair identifiers
  size_t pairIdChunkSz = bytes.size() - bytesOffset;
  GLOO_ENFORCE_EQ(
      pairIdChunkSz % sizeof(ssize_t),
      0,
      "Remaining bytes do not map to entire chunk of pair identifiers");
  size_t numPairs = pairIdChunkSz / sizeof(ssize_t);
  pairIdentifiers.resize(numPairs);
  std::memcpy(pairIdentifiers.data(), bytes.data() + bytesOffset, pairIdChunkSz);
}

std::vector<char> Rank::bytes() const {
  size_t hostnameSz = hostname.size();
  size_t addrSz = addressBytes.size();
  size_t numPairIds = pairIdentifiers.size();
  size_t pairIdSz = sizeof(ssize_t);
  size_t pairIdChunkSz = pairIdSz * numPairIds;
  size_t totalSz = sizeof(hostnameSz) + hostnameSz + sizeof(addrSz) + addrSz +
      pairIdChunkSz;
  std::vector<char> buf(totalSz);
  auto bufOffset = buf.data();
  // hostname
  std::memcpy(bufOffset, &hostnameSz, sizeof(hostnameSz));
  bufOffset += sizeof(hostnameSz);
  std::memcpy(bufOffset, hostname.data(), hostnameSz);
  bufOffset += hostnameSz;
  // address
  std::memcpy(bufOffset, &addrSz, sizeof(addrSz));
  bufOffset += sizeof(addrSz);
  std::memcpy(bufOffset, addressBytes.data(), addressBytes.size());
  bufOffset += addrSz;
  // pair identifiers
  std::memcpy(bufOffset, pairIdentifiers.data(), pairIdChunkSz);
  return buf;
}

} // namespace tcp
} // namespace transport
} // namespace gloo
