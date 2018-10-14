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

#include "gloo/allgather.h"
#include "gloo/allgather_ring.h"
#include "gloo/common/common.h"
#include "gloo/test/base_test.h"

namespace gloo {
namespace test {
namespace {

// Test parameterization.
using Param = std::tuple<int, int, int>;

// Test fixture.
class AllgatherTest : public BaseTest,
                      public ::testing::WithParamInterface<Param> {};

TEST_P(AllgatherTest, VarNumPointer) {
  auto contextSize = std::get<0>(GetParam());
  auto dataSize = std::get<1>(GetParam());
  auto numPtrs = std::get<2>(GetParam());

  spawn(contextSize, [&](std::shared_ptr<Context> context) {
    Fixture<float> inFixture(context, numPtrs, dataSize);
    inFixture.assignValues();

    std::unique_ptr<float[]> outPtr =
        gloo::make_unique<float[]>(numPtrs * dataSize * contextSize);

    AllgatherRing<float> algorithm(
        context, inFixture.getConstPointers(), outPtr.get(), dataSize);

    algorithm.run();

    auto stride = contextSize * numPtrs;
    for (int i = 0; i < contextSize; ++i) {
      auto val = i * numPtrs;
      for (int j = 0; j < dataSize; j++) {
        float exp = j * stride + val;
        for (int k = 0; k < numPtrs; ++k) {
          ASSERT_EQ(
              outPtr.get()[i * dataSize * numPtrs + k * dataSize + j], exp + k)
              << "Mismatch at index [" << i << ", " << j + dataSize << "]";
        }
      }
    }
  });
}

TEST_F(AllgatherTest, MultipleAlgorithms) {
  auto contextSize = 4;
  auto dataSize = 1000;
  auto numPtrs = 8;

  spawn(contextSize, [&](std::shared_ptr<Context> context) {
    Fixture<float> inFixture(context, numPtrs, dataSize);
    inFixture.assignValues();

    std::unique_ptr<float[]> outPtr =
        gloo::make_unique<float[]>(numPtrs * dataSize * contextSize);

    for (int alg = 0; alg < 2; alg++) {
      AllgatherRing<float> algorithm(
          context, inFixture.getConstPointers(), outPtr.get(), dataSize);
      algorithm.run();

      auto stride = contextSize * numPtrs;
      for (int i = 0; i < contextSize; ++i) {
        auto val = i * numPtrs;
        for (int j = 0; j < dataSize; j++) {
          float exp = j * stride + val;
          for (int k = 0; k < numPtrs; ++k) {
            ASSERT_EQ(
                outPtr.get()[i * dataSize * numPtrs + k * dataSize + j],
                exp + k)
                << "Mismatch at index [" << i << ", " << j + dataSize << "]";
          }
        }
      }
    }
  });
}

std::vector<int> genMemorySizes() {
  std::vector<int> v;
  v.push_back(sizeof(float));
  v.push_back(100);
  v.push_back(1000);
  v.push_back(10000);
  return v;
}

INSTANTIATE_TEST_CASE_P(
    AllgatherRing,
    AllgatherTest,
    ::testing::Combine(
        ::testing::Range(2, 10),
        ::testing::ValuesIn(genMemorySizes()),
        ::testing::Range(1, 4)));

using NewParam = std::tuple<int, int>;

class AllgatherNewTest : public BaseTest,
                         public ::testing::WithParamInterface<NewParam> {};

TEST_P(AllgatherNewTest, Default) {
  auto contextSize = std::get<0>(GetParam());
  auto dataSize = std::get<1>(GetParam());

  auto validate = [dataSize](
      const std::shared_ptr<Context>& context,
      Fixture<uint64_t>& output) {
    const auto ptr = output.getPointer();
    const auto stride = context->size;
    for (auto j = 0; j < context->size; j++) {
      for (auto k = 0; k < dataSize; k++) {
        ASSERT_EQ(j + k * stride, ptr[k + j * dataSize])
          << "Mismatch at index " << (k + j * dataSize);
      }
    }
  };

  spawn(contextSize, [&](std::shared_ptr<Context> context) {
      auto input = Fixture<uint64_t>(context, 1, dataSize);
      auto output = Fixture<uint64_t>(context, 1, contextSize * dataSize);

      // Run with raw pointers and sizes in options
      {
        input.assignValues();
        output.clear();

        AllgatherOptions opts;
        opts.inPtr = input.getPointer();
        opts.inElements = dataSize;
        opts.outPtr = output.getPointer();
        opts.outElements = contextSize * dataSize;
        opts.elementSize = sizeof(uint64_t);
        input.assignValues();
        output.clear();
        allgather(context, opts);
        validate(context, output);
      }

      // Run with (optionally cached) unbound buffers in options
      {
        input.assignValues();
        output.clear();

        AllgatherOptions opts;
        opts.inBuffer = context->createUnboundBuffer(
            input.getPointer(),
            dataSize * sizeof(uint64_t));
        opts.outBuffer = context->createUnboundBuffer(
            output.getPointer(),
            contextSize * dataSize * sizeof(uint64_t));
        opts.elementSize = sizeof(uint64_t);
        allgather(context, opts);
        validate(context, output);
      }
    });
}

INSTANTIATE_TEST_CASE_P(
    AllgatherNewDefault,
    AllgatherNewTest,
    ::testing::Combine(
        ::testing::Values(2, 4, 7),
        ::testing::ValuesIn(genMemorySizes())));


} // namespace
} // namespace test
} // namespace gloo
