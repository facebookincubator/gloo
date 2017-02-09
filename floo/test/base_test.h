/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <thread>
#include <vector>

#include "floo/rendezvous/hash_store.h"
#include "floo/transport/tcp/device.h"

namespace floo {
namespace test {

class BaseTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    device_ = ::floo::transport::tcp::CreateDevice("localhost");
    store_ = std::unique_ptr<::floo::rendezvous::Store>(
        new ::floo::rendezvous::HashStore);
  }

  void spawnThreads(int size, std::function<void(int)> fn) {
    std::vector<std::thread> threads;
    std::vector<std::string> errors;
    for (int rank = 0; rank < size; rank++) {
      threads.push_back(std::thread([&, rank]() {
        try {
          fn(rank);
        } catch (const std::exception& ex) {
          errors.push_back(ex.what());
        }
      }));
    }

    // Wait for threads to complete
    for (auto& thread : threads) {
      thread.join();
    }

    // Re-throw first exception if there is one
    if (errors.size() > 0) {
      throw(std::runtime_error(errors[0]));
    }
  }

  std::shared_ptr<::floo::transport::Device> device_;
  std::unique_ptr<::floo::rendezvous::Store> store_;
};

} // namespace test
} // namespace floo
