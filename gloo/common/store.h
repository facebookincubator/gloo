/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace gloo {

class IStore {
 public:
  virtual ~IStore() = default;

  virtual void set(const std::string& key, const std::vector<char>& data) = 0;

  virtual std::vector<char> get(const std::string& key) = 0;

  virtual void wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) = 0;
};

} // namespace gloo
