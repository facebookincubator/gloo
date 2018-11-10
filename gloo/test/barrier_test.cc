/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <functional>
#include <thread>
#include <vector>

#include "gloo/barrier_all_to_all.h"
#include "gloo/barrier_all_to_one.h"
#include "gloo/test/base_test.h"

namespace gloo {
namespace test {
namespace {

// Function to instantiate and run algorithm.
using Func = void(std::shared_ptr<::gloo::Context>);

// Test parameterization.
using Param = std::tuple<int, std::function<Func>>;

// Test fixture.
class BarrierTest : public BaseTest,
                    public ::testing::WithParamInterface<Param> {};

TEST_P(BarrierTest, SinglePointer) {
  auto contextSize = std::get<0>(GetParam());
  auto fn = std::get<1>(GetParam());

  spawn(contextSize, [&](std::shared_ptr<Context> context) {
    fn(context);
  });
}

static std::function<Func> barrierAllToAll =
    [](std::shared_ptr<::gloo::Context> context) {
      ::gloo::BarrierAllToAll algorithm(context);
      algorithm.run();
    };

INSTANTIATE_TEST_CASE_P(
    BarrierAllToAll,
    BarrierTest,
    ::testing::Combine(
        ::testing::Range(2, 16),
        ::testing::Values(barrierAllToAll)));

static std::function<Func> barrierAllToOne =
    [](std::shared_ptr<::gloo::Context> context) {
      ::gloo::BarrierAllToOne algorithm(context);
      algorithm.run();
    };

INSTANTIATE_TEST_CASE_P(
    BarrierAllToOne,
    BarrierTest,
    ::testing::Combine(
        ::testing::Range(2, 16),
        ::testing::Values(barrierAllToOne)));

using NewParam = std::tuple<int>;

class BarrierNewTest : public BaseTest,
                       public ::testing::WithParamInterface<NewParam> {};

TEST_P(BarrierNewTest, Default) {
  auto contextSize = std::get<0>(GetParam());
  spawn(contextSize, [&](std::shared_ptr<Context> context) {
    BarrierOptions opts(context);

    // Run barrier to synchronize processes after starting.
    barrier(opts);

    // Take turns in sleeping for a bit and checking that all processes
    // saw that artificial delay through the barrier.
    auto singleProcessDelay = std::chrono::milliseconds(10);
    for (size_t i = 0; i < context->size; i++) {
      const auto start = std::chrono::high_resolution_clock::now();
      if (i == context->rank) {
        /* sleep override */
        std::this_thread::sleep_for(singleProcessDelay);
      }

      barrier(opts);

      // Expect all processes to have taken at least as long as the sleep
      auto stop = std::chrono::high_resolution_clock::now();
      auto delta = std::chrono::duration_cast<decltype(singleProcessDelay)>(
          stop - start);
      ASSERT_GE(delta.count(), singleProcessDelay.count());
    }
  });
}

INSTANTIATE_TEST_CASE_P(
    BarrierNewDefault,
    BarrierNewTest,
    ::testing::Values(2, 4, 7));

} // namespace
} // namespace test
} // namespace gloo
