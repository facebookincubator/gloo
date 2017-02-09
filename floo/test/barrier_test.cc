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

#include "floo/barrier_all_to_all.h"
#include "floo/barrier_all_to_one.h"
#include "floo/test/base_test.h"

namespace floo {
namespace test {
namespace {

// Function to instantiate and run algorithm.
using Func = void(std::shared_ptr<::floo::Context>&);

// Test parameterization.
using Param = std::tuple<int, std::function<Func>>;

// Test fixture.
class BarrierTest : public BaseTest,
                    public ::testing::WithParamInterface<Param> {};

TEST_P(BarrierTest, SinglePointer) {
  auto contextSize = std::get<0>(GetParam());
  auto fn = std::get<1>(GetParam());

  spawnThreads(contextSize, [&](int contextRank) {
    auto context =
        std::make_shared<::floo::Context>(contextRank, contextSize);
    context->connectFullMesh(*store_, device_);
    fn(context);
  });
}

static std::function<Func> barrierAllToAll =
    [](std::shared_ptr<::floo::Context>& context) {
      ::floo::BarrierAllToAll algorithm(context);
      algorithm.run();
    };

INSTANTIATE_TEST_CASE_P(
    BarrierAllToAll,
    BarrierTest,
    ::testing::Combine(
        ::testing::Range(2, 16),
        ::testing::Values(barrierAllToAll)));

static std::function<Func> barrierAllToOne =
    [](std::shared_ptr<::floo::Context>& context) {
      ::floo::BarrierAllToOne algorithm(context);
      algorithm.run();
    };

INSTANTIATE_TEST_CASE_P(
    BarrierAllToOne,
    BarrierTest,
    ::testing::Combine(
        ::testing::Range(2, 16),
        ::testing::Values(barrierAllToOne)));

} // namespace
} // namespace test
} // namespace floo
