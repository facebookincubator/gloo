/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <chrono>
#include <memory>

#include "gloo/transport/pair.h"

namespace gloo {
namespace transport {

// Forward declarations
class Pair;
class Buffer;
class UnboundBuffer;

// The device abstraction can be considered as a factory for all
// communication pairs. A communication pair can be associated with
// send and receive buffers. Send buffers serve as the source for one
// sided writes and receive buffers serve as the target of one sided
// writes. Both ends of the pair can create either type of buffer, as
// long as it is paired with the opposite type on the remote end of
// the pair; every receive buffer must be paired with a corresponding
// send buffer and vice versa. The device abstraction may start a
// background thread to handle I/O multiplexing (not configurable).
class Device {
 public:
  virtual ~Device() = 0;

  virtual std::string str() const = 0;

  virtual const std::string& getPCIBusID() const = 0;

  virtual int getInterfaceSpeed() const { return 0; }

  virtual bool hasGPUDirect() const { return false; }

  virtual std::unique_ptr<Pair> createPair(
      std::chrono::milliseconds timeout) = 0;

  // Factory function to create an unbound buffer for use with the
  // transport used for this context. Use this function to avoid tying
  // downstream code to a specific transport.
  // The return value is not tied to this device and can be used with
  // any devices/pairs of the same transport.
  virtual std::unique_ptr<UnboundBuffer> createUnboundBuffer(
      void* ptr, size_t size) = 0;
};

} // namespace transport
} // namespace gloo
