/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <vector>

namespace gloo {
namespace transport {

class Address {
 public:
  // Upper bound for an address' byte representation.
  static constexpr auto kMaxByteSize = 192;

  virtual ~Address() = 0;

  virtual std::string str() const = 0;
  virtual std::vector<char> bytes() const = 0;
};

} // namespace transport
} // namespace gloo
