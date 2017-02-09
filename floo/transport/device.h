/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <memory>

#include "floo/transport/pair.h"

namespace floo {
namespace transport {

// Forward declarations
class Pair;
class Buffer;

class Device {
 public:
  virtual ~Device() = 0;

  virtual std::unique_ptr<Pair> createPair() = 0;
};

} // namespace transport
} // namespace floo
