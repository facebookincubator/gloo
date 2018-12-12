/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "gloo/transport/context.h"

#include <memory>

namespace gloo {
namespace transport {
namespace ibverbs {

// Forward declaration
class Device;
class Pair;

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

  friend class Pair;
};

} // namespace ibverbs
} // namespace transport
} // namespace gloo
