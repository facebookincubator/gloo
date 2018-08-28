/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/transport/context.h"

#include <memory>

namespace gloo {
namespace transport {
namespace tcp {

// Forward declaration
class Device;

class Context : public ::gloo::transport::Context {
 public:
  Context(std::shared_ptr<Device> device, int rank, int size);

  virtual ~Context();

  std::unique_ptr<transport::Pair>& createPair(
      int rank,
      std::chrono::milliseconds timeout) override;

 protected:
  std::shared_ptr<Device> device_;
};

} // namespace tcp
} // namespace transport
} // namespace gloo
