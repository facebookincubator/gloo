/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "gloo/transport/address.h"
#include "gloo/transport/buffer.h"
#include "gloo/transport/unbound_buffer.h"

namespace gloo {
namespace transport {

class Pair {
 public:
  virtual ~Pair() = 0;

  virtual const Address& address() const = 0;

  virtual void connect(const std::vector<char>& bytes) = 0;

  virtual void close() = 0;

  virtual void setSync(bool enable, bool busyPoll) = 0;

  virtual std::unique_ptr<Buffer>
  createSendBuffer(int slot, void* ptr, size_t size) = 0;

  virtual std::unique_ptr<Buffer>
  createRecvBuffer(int slot, void* ptr, size_t size) = 0;

  // Send from the specified buffer to remote side of pair.
  virtual void send(
      UnboundBuffer* buf,
      uint64_t tag,
      size_t offset = 0,
      size_t nbytes = 0) = 0;

  // Receive into the specified buffer from the remote side of pair.
  virtual void recv(
      UnboundBuffer* buf,
      uint64_t tag,
      size_t offset = 0,
      size_t nbytes = 0) = 0;
};

} // namespace transport
} // namespace gloo
