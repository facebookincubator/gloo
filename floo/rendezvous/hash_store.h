/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "floo/rendezvous/store.h"

#include <condition_variable>
#include <mutex>
#include <unordered_map>

namespace floo {
namespace rendezvous {

class HashStore : public Store {
 public:
  virtual ~HashStore() {}

  virtual void set(const std::string& key, const std::vector<char>& data)
      override;

  virtual std::vector<char> get(const std::string& key) override;

  virtual void wait(const std::vector<std::string>& keys) override;

 protected:
  std::unordered_map<std::string, std::vector<char>> map_;
  std::mutex m_;
  std::condition_variable cv_;
};

} // namespace rendezvous
} // namespace floo
