/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <functional>
#include <thread>
#include <vector>

#include "gloo/reduce_scatter.h"
#include "gloo/test/base_test.h"

namespace gloo {
namespace test {
namespace {

// RAII handle for aligned buffer
template <typename T>
std::vector<T, aligned_allocator<T, kBufferAlignment>> newBuffer(int size) {
  return std::vector<T, aligned_allocator<T, kBufferAlignment>>(size);
}

// Function to instantiate and run algorithm.
using Func = void(
    std::shared_ptr<::gloo::Context>,
    std::vector<float*> dataPtrs,
    int dataSize,
    std::vector<int> recvCounts);

using Func16 = void(
    std::shared_ptr<::gloo::Context>,
    std::vector<float16*> dataPtrs,
    int dataSize,
    std::vector<int> recvCounts);

// Test parameterization.
using Param = std::tuple<int, int, std::function<Func>, int>;
using ParamHP = std::tuple<int, int, std::function<Func16>>;

template <typename Algorithm>
class ReduceScatterConstructorTest : public BaseTest {
};

static std::function<Func> reduceScatterHalvingDoubling = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float*> dataPtrs,
    int dataSize,
    std::vector<int> recvCounts) {
  ::gloo::ReduceScatterHalvingDoubling<float> algorithm(
      context, dataPtrs, dataSize, recvCounts);
  algorithm.run();
};

static std::function<Func16> reduceScatterHalvingDoublingHP = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float16*> dataPtrs,
    int dataSize,
    std::vector<int> recvCounts) {
  ::gloo::ReduceScatterHalvingDoubling<float16> algorithm(
      context, dataPtrs, dataSize, recvCounts);
  algorithm.run();
};

// Test fixture.
class ReduceScatterTest : public BaseTest,
                          public ::testing::WithParamInterface<Param> {};

class ReduceScatterTestHP : public BaseTest,
                            public ::testing::WithParamInterface<ParamHP> {};

TEST_P(ReduceScatterTest, SinglePointer) {
  auto contextSize = std::get<0>(GetParam());
  auto dataSize = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());
  auto base = std::get<3>(GetParam());

  spawnThreads(contextSize, [&](int contextRank) {
    auto context = std::make_shared<::gloo::rendezvous::Context>(
        contextRank, contextSize, base);
    context->connectFullMesh(*store_, device_);

    auto buffer = newBuffer<float>(dataSize);
    auto* ptr = buffer.data();
    for (int i = 0; i < dataSize; i++) {
      ptr[i] = contextRank;
    }
    std::vector<int> recvCounts;
    auto rem = dataSize;
    auto chunkSize = (dataSize + contextSize - 1) / contextSize;
    for (int i = 0; i < contextSize; ++i) {
      recvCounts.push_back(std::min(chunkSize, rem));
      rem = rem > chunkSize ? rem - chunkSize : 0;
    }

    fn(context, std::vector<float*>{ptr}, dataSize, recvCounts);

    auto expected = (contextSize * (contextSize - 1)) / 2;
    for (int i = 0; i < recvCounts[contextRank]; i++) {
      ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i;
    }
  });
}

TEST_F(ReduceScatterTest, MultipleAlgorithms) {
  auto contextSize = 4;
  auto dataSize = 1000;
  auto fns = {reduceScatterHalvingDoubling};

  spawnThreads(contextSize, [&](int contextRank) {
    auto context =
        std::make_shared<::gloo::rendezvous::Context>(contextRank, contextSize);
    context->connectFullMesh(*store_, device_);

    auto buffer = newBuffer<float>(dataSize);
    auto* ptr = buffer.data();
    std::vector<int> recvCounts;
    auto rem = dataSize;
    auto chunkSize = (dataSize + contextSize - 1) / contextSize;
    for (int i = 0; i < contextSize; ++i) {
      recvCounts.push_back(std::min(chunkSize, rem));
      rem = rem > chunkSize ? rem - chunkSize : 0;
    }
    for (const auto& fn : fns) {
      for (int i = 0; i < dataSize; i++) {
        ptr[i] = contextRank;
      }

      fn(context, std::vector<float*>{ptr}, dataSize, recvCounts);

      auto expected = (contextSize * (contextSize - 1)) / 2;
      for (int i = 0; i < recvCounts[contextRank]; i++) {
        ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i;
      }

      for (int i = 0; i < dataSize; i++) {
        ptr[i] = contextRank;
      }

      fn(context, std::vector<float*>{ptr}, dataSize, recvCounts);

      expected = (contextSize * (contextSize - 1)) / 2;
      for (int i = 0; i < recvCounts[contextRank]; i++) {
        ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i;
      }
    }
  });
}

TEST_F(ReduceScatterTestHP, HalfPrecisionTest) {
  int contextSize = 4;
  auto dataSize = 1024;
  auto fns = {reduceScatterHalvingDoublingHP};

  spawnThreads(contextSize, [&](int contextRank) {
    auto context =
        std::make_shared<::gloo::rendezvous::Context>(contextRank, contextSize);
    context->connectFullMesh(*store_, device_);

    auto buffer = newBuffer<float16>(dataSize);
    auto* ptr = buffer.data();
    std::vector<int> recvCounts;
    auto rem = dataSize;
    auto chunkSize = (dataSize + contextSize - 1) / contextSize;
    for (int i = 0; i < contextSize; ++i) {
      recvCounts.push_back(std::min(chunkSize, rem));
      rem = rem > chunkSize ? rem - chunkSize : 0;
    }
    for (const auto& fn : fns) {
      for (int i = 0; i < dataSize; i++) {
        ptr[i] = contextRank;
      }

      fn(context, std::vector<float16*>{ptr}, dataSize, recvCounts);

      float16 expected(contextSize * (contextSize - 1) / 2);
      for (int i = 0; i < recvCounts[contextRank]; i++) {
        ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i;
      }
    }
  });
}

INSTANTIATE_TEST_CASE_P(
    ReduceScatterHalvingDoubling,
    ReduceScatterTest,
    ::testing::Combine(
        ::testing::ValuesIn(
          std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 16, 24, 32})),
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(reduceScatterHalvingDoubling),
        ::testing::Values(0)));

} // namespace
} // namespace test
} // namespace gloo
