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
    if (count_ > 0) {
      if (--count_ == 0) {
        cv_.notify_all();
      } else {
        cv_.wait(lock, [this] { return count_ == 0; });
      }
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

  void spawn(int size, std::function<void(std::shared_ptr<Context>)> fn) {
    Barrier barrier(size);
    spawnThreads(size, [&](int rank) {
      auto context = std::make_shared<::gloo::rendezvous::Context>(rank, size);
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
      : context(context), inputs(ptrs), count(count) {
    for (int i = 0; i < ptrs; i++) {
      std::unique_ptr<T[]> ptr(new T[count]);
      srcs.push_back(std::move(ptr));
    }
  }

  Fixture(Fixture&& other) noexcept
      : context(other.context), inputs(other.inputs), count(other.count) {
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

  void checkBroadcastResult(Fixture<T>& fixture, int root, int rootPointer) {
    // Expected is set to the expected value at ptr[0]
    const auto expected = root * fixture.srcs.size() + rootPointer;
    // Stride is difference between values at subsequent indices
    const auto stride = fixture.srcs.size() * fixture.context->size;
    // Verify all buffers passed by this instance
    for (const auto& ptr : fixture.getPointers()) {
      for (auto i = 0; i < fixture.count; i++) {
        ASSERT_EQ(T((i * stride) + expected), ptr[i])
            << "Mismatch at index " << i;
      }
    }
  }

  void checkAllreduceResult() {
    const auto stride = context->size * srcs.size();
    for (auto i = 0; i < srcs.size(); i++) {
      for (auto j = 0; j < count; j++) {
        auto expected = T((j * stride * stride) + (stride * (stride - 1)) / 2);
        GLOO_ENFORCE_EQ(
            expected,
            srcs[i][j],
            "Mismatch in srcs[",
            i,
            "][",
            j,
            "] expected:",
            expected,
            " actual: ",
            srcs[i][j],
            " difference: ",
            expected - srcs[i][j]);
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

  std::vector<const T*> getConstPointers() const {
    std::vector<const T*> out;
    for (const auto& src : srcs) {
      out.push_back(src.get());
    }
    return out;
  }

  std::shared_ptr<Context> context;
  const int inputs;
  const int count;
  std::vector<std::unique_ptr<T[]>> srcs;
};

template <>
class Fixture<float16> {
 public:
  Fixture(const std::shared_ptr<Context> context, int ptrs, int count)
      : context(context), inputs(ptrs), count(count) {
    for (int i = 0; i < ptrs; i++) {
      std::unique_ptr<float16[]> ptr(new float16[count]);
      srcs.push_back(std::move(ptr));
    }
  }

  Fixture(Fixture&& other) noexcept
      : context(other.context), inputs(other.inputs), count(other.count) {
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

  void
  checkBroadcastResult(Fixture<float16>& fixture, int root, int rootPointer) {
    // Expected is set to the expected value at ptr[0]
    const auto expected = root * fixture.srcs.size() + rootPointer;
    // Stride is difference between values at subsequent indices
    const auto stride = fixture.srcs.size() * fixture.context->size;
    // Verify all buffers passed by this instance
    for (const auto& ptr : fixture.getPointers()) {
      for (auto i = 0; i < fixture.count; i++) {
        ASSERT_EQ(float16((i * stride) + expected), ptr[i])
            << "Mismatch at index " << i;
      }
    }
  }

  void checkAllreduceResult() {
    // roundoff error for float16 math can be high (> 1 in some cases) so we
    // will test for relative equality

    // next smallest representable float16 value > 1 is 1.00097625, so we pick
    // threshold accordingly
    static float16 floatEqThreshold = static_cast<float16>(0.001);
    const auto stride = context->size * srcs.size();
    for (auto i = 0; i < srcs.size(); i++) {
      for (auto j = 0; j < count; j++) {
        auto expected =
            float16((j * stride * stride) + (stride * (stride - 1)) / 2);
        // use direct comparison for small numbers, where
        // testing for relative equality can be inaccurate; this also prevents
        // cases that would lead to division by 0
        if (expected > float16(-1) && expected < float16(1)) {
          GLOO_ENFORCE(
              expected - srcs[i][j] <= floatEqThreshold &&
                  srcs[i][j] - expected <= floatEqThreshold,
              "Mismatch in srcs[",
              i,
              "][",
              j,
              "] expected:",
              expected,
              " actual: ",
              srcs[i][j],
              " difference: ",
              expected - srcs[i][j]);
        } else {
          GLOO_ENFORCE(
              (expected - srcs[i][j]) / expected <= floatEqThreshold &&
                  (srcs[i][j] - expected) / expected <= floatEqThreshold,
              "Mismatch in srcs[",
              i,
              "][",
              j,
              "] expected:",
              expected,
              " actual: ",
              srcs[i][j],
              " difference: ",
              expected - srcs[i][j]);
        }
      }
    }
  }

  std::vector<float16*> getPointers() const {
    std::vector<float16*> out;
    for (const auto& src : srcs) {
      out.push_back(src.get());
    }
    return out;
  }

  std::vector<const float16*> getConstPointers() const {
    std::vector<const float16*> out;
    for (const auto& src : srcs) {
      out.push_back(src.get());
    }
    return out;
  }

  std::shared_ptr<Context> context;
  const int inputs;
  const int count;
  std::vector<std::unique_ptr<float16[]>> srcs;
};

template <>
class Fixture<float> {
 public:
  Fixture(const std::shared_ptr<Context> context, int ptrs, int count)
      : context(context), inputs(ptrs), count(count) {
    for (int i = 0; i < ptrs; i++) {
      std::unique_ptr<float[]> ptr(new float[count]);
      srcs.push_back(std::move(ptr));
    }
  }

  Fixture(Fixture&& other) noexcept
      : context(other.context), inputs(other.inputs), count(other.count) {
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

  void
  checkBroadcastResult(Fixture<float>& fixture, int root, int rootPointer) {
    // Expected is set to the expected value at ptr[0]
    const auto expected = root * fixture.srcs.size() + rootPointer;
    // Stride is difference between values at subsequent indices
    const auto stride = fixture.srcs.size() * fixture.context->size;
    // Verify all buffers passed by this instance
    for (const auto& ptr : fixture.getPointers()) {
      for (auto i = 0; i < fixture.count; i++) {
        ASSERT_EQ(float((i * stride) + expected), ptr[i])
            << "Mismatch at index " << i;
      }
    }
  }

  void checkAllreduceResult() {
    // roundoff error for float math can be high (> 1 in some cases) so we
    // will test for relative equality
    static float floatEqThreshold = static_cast<float>(0.0001);
    const auto stride = context->size * srcs.size();
    for (auto i = 0; i < srcs.size(); i++) {
      for (auto j = 0; j < count; j++) {
        auto expected =
            float((j * stride * stride) + (stride * (stride - 1)) / 2);
        // use direct comparison for small numbers, where
        // testing for relative equality can be inaccurate; this also prevents
        // cases that would lead to division by 0
        if (expected > float(-1) && expected < float(1)) {
          GLOO_ENFORCE(
              expected - srcs[i][j] <= floatEqThreshold &&
                  srcs[i][j] - expected <= floatEqThreshold,
              "Mismatch in srcs[",
              i,
              "][",
              j,
              "] expected:",
              expected,
              " actual: ",
              srcs[i][j],
              " difference: ",
              expected - srcs[i][j]);
        } else {
          GLOO_ENFORCE(
              (expected - srcs[i][j]) / expected <= floatEqThreshold &&
                  (srcs[i][j] - expected) / expected <= floatEqThreshold,
              "Mismatch in srcs[",
              i,
              "][",
              j,
              "] expected:",
              expected,
              " actual: ",
              srcs[i][j],
              " difference: ",
              expected - srcs[i][j]);
        }
      }
    }
  }

  std::vector<float*> getPointers() const {
    std::vector<float*> out;
    for (const auto& src : srcs) {
      out.push_back(src.get());
    }
    return out;
  }

  std::vector<const float*> getConstPointers() const {
    std::vector<const float*> out;
    for (const auto& src : srcs) {
      out.push_back(src.get());
    }
    return out;
  }

  std::shared_ptr<Context> context;
  const int inputs;
  const int count;
  std::vector<std::unique_ptr<float[]>> srcs;
};

} // namespace test
} // namespace gloo
