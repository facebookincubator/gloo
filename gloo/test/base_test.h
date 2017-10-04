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

#include <exception>
#include <functional>
#include <thread>
#include <vector>

#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/hash_store.h"
#include "gloo/transport/tcp/device.h"
#include "gloo/types.h"

namespace gloo {
namespace test {

class Barrier {
 public:
  explicit Barrier(std::size_t count) : count_(count) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (--count_ == 0) {
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this] { return count_ == 0; });
    }
  }

 private:
  std::size_t count_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

class BaseTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    device_ = ::gloo::transport::tcp::CreateDevice("localhost");
    store_ = std::unique_ptr<::gloo::rendezvous::Store>(
        new ::gloo::rendezvous::HashStore);
  }

  void spawnThreads(int size, std::function<void(int)> fn) {
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors;
    for (int rank = 0; rank < size; rank++) {
      threads.push_back(std::thread([&, rank]() {
        try {
          fn(rank);
        } catch (const std::exception&) {
          errors.push_back(std::current_exception());
        }
      }));
    }

    // Wait for threads to complete
    for (auto& thread : threads) {
      thread.join();
    }

    // Re-throw first exception if there is one
    if (errors.size() > 0) {
      std::rethrow_exception(errors[0]);
    }
  }

  void spawn(
      int size,
      std::function<void(std::shared_ptr<Context>)> fn) {
    Barrier barrier(size);
    spawnThreads(size, [&](int rank) {
        auto context =
          std::make_shared<::gloo::rendezvous::Context>(rank, size);
        if (size > 1) {
          context->connectFullMesh(*store_, device_);
        }
        fn(context);

        // Since the test suite only deals with threads within a
        // process, we can cheat and use an in-process barrier to
        // ensure all threads have finished before explicitly closing
        // all pairs. Instead of relying on the pair's destructor
        // closing the underlying connection, we explicitly call
        // close(). This sets the SO_LINGER socket option (in case of
        // the tcp transport) to avoid the TIME_WAIT connection state.
        // This test suite contains many tests and we risk running out
        // of ports to bind to otherwise.
        barrier.wait();
        if (size > 1) {
          context->closeConnections();
        }
      });
  }

  std::shared_ptr<::gloo::transport::Device> device_;
  std::unique_ptr<::gloo::rendezvous::Store> store_;
};

template <typename T>
class Fixture {
 public:
  Fixture(const std::shared_ptr<Context> context, int ptrs, int count)
      : context(context),
        inputs(ptrs),
        count(count) {
    for (int i = 0; i < ptrs; i++) {
      std::unique_ptr<T[]> ptr(new T[count]);
      srcs.push_back(std::move(ptr));
    }
  }

  Fixture(Fixture&& other) noexcept
    : context(other.context),
      inputs(other.inputs),
      count(other.count) {
    srcs = std::move(other.srcs);
  }

  void assignValues() {
    const auto stride = context->size * srcs.size();
    for (auto i = 0; i < srcs.size(); i++) {
      auto val = (context->rank * srcs.size()) + i;
      for (auto j = 0; j < count; j++) {
        srcs[i][j] = (j * stride) + val;
      }
    }
  }

  void checkAllreduceResult() {
    const auto stride = context->size * srcs.size();
    for (auto i = 0; i < srcs.size(); i++) {
      for (auto j = 0; j < count; j++) {
        auto expected = T((j * stride * stride) + (stride * (stride - 1)) / 2);
        ASSERT_EQ(expected, srcs[i][j])
          << "Mismatch in srcs[" << i << "][" << j << "]";
      }
    }
  }

  std::vector<T*> getPointers() const {
    std::vector<T*> out;
    for (const auto& src : srcs) {
      out.push_back(src.get());
    }
    return out;
  }

  std::shared_ptr<Context> context;
  const int inputs;
  const int count;
  std::vector<std::unique_ptr<T[]> > srcs;
};

} // namespace test
} // namespace gloo
