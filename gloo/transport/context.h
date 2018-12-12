/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <memory>

#include "gloo/transport/pair.h"
#include "gloo/transport/unbound_buffer.h"

namespace gloo {
namespace transport {

// The context represents a set of pairs that belong to the same
// group. Its equivalent class at the top level namespace represents
// the same group but cannot represent transport specifics.
//
// While implementing the recv-from-any functionality we realized we
// realized we needed some transport-specific state shared between all
// pairs in a group, to arbitrate between multiple pairs attempting to
// send to the same buffer. To avoid over-generalization, transports
// can implement this however they want in their own subclass.
//
class Context {
 public:
  Context(int rank, int size);

  virtual ~Context();

  const int rank;
  const int size;

  virtual std::unique_ptr<Pair>& getPair(int rank);

  virtual std::unique_ptr<Pair>& createPair(int rank) = 0;

  // Creates unbound buffer to be used with the ranks in this context.
  // It is not bound to a specific rank, but still bound to this
  // context. This is needed to support recv-from-any semantics, where
  // the context is used as shared arbiter between pairs that are
  // ready to send and buffers that are ready to receive.
  virtual std::unique_ptr<transport::UnboundBuffer> createUnboundBuffer(
      void* ptr,
      size_t size) = 0;

  void setTimeout(std::chrono::milliseconds timeout) {
    timeout_ = timeout;
  }

  std::chrono::milliseconds getTimeout() const {
    return timeout_;
  }

 protected:
  // Lifecycle of the pairs is managed by a std::unique_ptr of the
  // base class. This is done because the public context API dictates
  // that getPair() returns a reference to this type. Functions
  // internal to this class can cast these points to the native
  // transport specific type.
  std::vector<std::unique_ptr<Pair>> pairs_;

  // Default timeout for new pairs (e.g. during initialization) and
  // any kind of send/recv operation.
  std::chrono::milliseconds timeout_;
};

} // namespace transport
} // namespace gloo
