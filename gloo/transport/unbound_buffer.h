/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gloo {
namespace transport {

// The unbound buffer class represents a chunk of memory.
// It can either be used as a source for send operations or a
// destination for receive operations, or both. There should only be a
// single pending operation against an unbound buffer at any given
// time, or resulting behavior is undefined.
//
// It is called unbound to contrast with the bound buffers that have
// been available since the inception of Gloo. It is unbound in that
// it is not tied to a particular pair.
//
class UnboundBuffer {
 public:
  UnboundBuffer(void* ptr, size_t size)
      : ptr(ptr), size(size) {}
  virtual ~UnboundBuffer() = 0;

  void* const ptr;
  const size_t size;

  // If specified, the source of this recv is stored in the rank pointer.
  virtual void waitRecv(int* rank = nullptr) = 0;

  // If specified, the destination of this send is stored in the rank pointer.
  virtual void waitSend(int* rank = nullptr) = 0;

  virtual void send(int dstRank, uint64_t slot) = 0;

  virtual void recv(int srcRank, uint64_t slot) = 0;

  virtual void recv(std::vector<int> srcRanks, uint64_t slot) = 0;
};

} // namespace transport
} // namespace gloo
