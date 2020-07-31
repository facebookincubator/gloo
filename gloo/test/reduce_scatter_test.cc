/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <functional>
#include <thread>
#include <vector>

#include "gloo/common/aligned_allocator.h"
#include "gloo/reduce_scatter.h"
#include "gloo/test/base_test.h"

namespace gloo {
namespace test {
namespace {

// RAII handle for aligned buffer
template <typename T>
#ifdef _WIN32
std::vector<T> newBuffer(int size) {
  return std::vector<T>(size);
#else
std::vector<T, aligned_allocator<T, kBufferAlignment>> newBuffer(int size) {
  return std::vector<T, aligned_allocator<T, kBufferAlignment>>(size);
#endif
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
using Param = std::tuple<Transport, int, int, std::function<Func>, int>;
using ParamHP = std::tuple<Transport, int, int, std::function<Func16>>;

template <typename Algorithm>
class ReduceScatterConstructorTest : public BaseTest {};

static std::function<Func> reduceScatterHalvingDoubling =
    [](std::shared_ptr<::gloo::Context> context,
       std::vector<float*> dataPtrs,
       int dataSize,
       std::vector<int> recvCounts) {
      ::gloo::ReduceScatterHalvingDoubling<float> algorithm(
          context, dataPtrs, dataSize, recvCounts);
      algorithm.run();
    };

static std::function<Func16> reduceScatterHalvingDoublingHP =
    [](std::shared_ptr<::gloo::Context> context,
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
  const auto transport = std::get<0>(GetParam());
  const auto contextSize = std::get<1>(GetParam());
  const auto dataSize = std::get<2>(GetParam());
  const auto fn = std::get<3>(GetParam());
  const auto base = std::get<4>(GetParam());

  spawn(transport, contextSize, [&](std::shared_ptr<Context> context) {
    const auto contextRank = context->rank;
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
  const auto transport = Transport::TCP;
  const auto contextSize = 4;
  const auto dataSize = 1000;
  const auto fns = {
      reduceScatterHalvingDoubling,
  };

  spawn(transport, contextSize, [&](std::shared_ptr<Context> context) {
    const auto contextRank = context->rank;
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
  const auto transport = Transport::TCP;
  const auto contextSize = 4;
  const auto dataSize = 1024;
  const auto fns = {
      reduceScatterHalvingDoublingHP,
  };

  spawn(transport, contextSize, [&](std::shared_ptr<Context> context) {
    const auto contextRank = context->rank;
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
        ::testing::ValuesIn(kTransportsForClassAlgorithms),
        ::testing::Values(1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 16, 24, 32),
        ::testing::Values(4, 100, 1000, 10000),
        ::testing::Values(reduceScatterHalvingDoubling),
        ::testing::Values(0)));

} // namespace
} // namespace test
} // namespace gloo
