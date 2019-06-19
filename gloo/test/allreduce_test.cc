/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>

#include <functional>
#include <thread>
#include <vector>

#include "gloo/allreduce.h"
#include "gloo/allreduce_bcube.h"
#include "gloo/allreduce_halving_doubling.h"
#include "gloo/allreduce_ring.h"
#include "gloo/allreduce_ring_chunked.h"
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
    int dataSize);

using Func16 = void(
    std::shared_ptr<::gloo::Context>,
    std::vector<float16*> dataPtrs,
    int dataSize);

// Test parameterization.
using Param = std::tuple<int, int, std::function<Func>, int>;
using ParamHP = std::tuple<int, int, std::function<Func16>>;

template <typename Algorithm>
class AllreduceConstructorTest : public BaseTest {
};

typedef ::testing::Types<
  AllreduceRing<float>,
  AllreduceRingChunked<float> > AllreduceTypes;
TYPED_TEST_CASE(AllreduceConstructorTest, AllreduceTypes);

TYPED_TEST(AllreduceConstructorTest, InlinePointers) {
  this->spawn(2, [&](std::shared_ptr<Context> context) {
      float f = 1.0f;
      TypeParam algorithm(
        context,
        {&f},
        1);
    });
}

TYPED_TEST(AllreduceConstructorTest, SpecifyReductionFunction) {
  this->spawn(2, [&](std::shared_ptr<Context> context) {
      float f = 1.0f;
      std::vector<float*> ptrs = {&f};
      TypeParam algorithm(
        context,
        ptrs,
        ptrs.size(),
        ReductionFunction<float>::product);
    });
}

static std::function<Func> allreduceRing = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float*> dataPtrs,
    int dataSize) {
  ::gloo::AllreduceRing<float> algorithm(context, dataPtrs, dataSize);
  algorithm.run();
};

static std::function<Func16> allreduceRingHP = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float16*> dataPtrs,
    int dataSize) {
  ::gloo::AllreduceRing<float16> algorithm(context, dataPtrs, dataSize);
  algorithm.run();
};

static std::function<Func> allreduceRingChunked = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float*> dataPtrs,
    int dataSize) {
  ::gloo::AllreduceRingChunked<float> algorithm(
      context, dataPtrs, dataSize);
  algorithm.run();
};

static std::function<Func16> allreduceRingChunkedHP = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float16*> dataPtrs,
    int dataSize) {
  ::gloo::AllreduceRingChunked<float16> algorithm(
      context, dataPtrs, dataSize);
  algorithm.run();
};

static std::function<Func> allreduceHalvingDoubling = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float*> dataPtrs,
    int dataSize) {
  ::gloo::AllreduceHalvingDoubling<float> algorithm(
      context, dataPtrs, dataSize);
  algorithm.run();
};

static std::function<Func> allreduceBcube = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float*> dataPtrs,
    int dataSize) {
  ::gloo::AllreduceBcube<float> algorithm(context, dataPtrs, dataSize);
  algorithm.run();
};

static std::function<Func16> allreduceHalvingDoublingHP = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float16*> dataPtrs,
    int dataSize) {
  ::gloo::AllreduceHalvingDoubling<float16> algorithm(
      context, dataPtrs, dataSize);
  algorithm.run();
};

// Test fixture.
class AllreduceTest : public BaseTest,
                      public ::testing::WithParamInterface<Param> {};

class AllreduceTestHP : public BaseTest,
                        public ::testing::WithParamInterface<ParamHP> {};

TEST_P(AllreduceTest, SinglePointer) {
  auto contextSize = std::get<0>(GetParam());
  auto dataSize = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());
  auto base = std::get<3>(GetParam());

  spawn(contextSize, [&](std::shared_ptr<Context> context) {
    const auto contextRank = context->rank;
    auto buffer = newBuffer<float>(dataSize);
    auto* ptr = buffer.data();
    for (int i = 0; i < dataSize; i++) {
      ptr[i] = contextRank;
    }

    fn(context, std::vector<float*>{ptr}, dataSize);

    auto expected = (contextSize * (contextSize - 1)) / 2;
    for (int i = 0; i < dataSize; i++) {
      ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i;
    }
  }, base);
}

TEST_F(AllreduceTest, MultipleAlgorithms) {
  auto contextSize = 4;
  auto dataSize = 1000;
  auto fns = {allreduceRing,
              allreduceRingChunked,
              allreduceHalvingDoubling,
              allreduceBcube};

  spawn(contextSize, [&](std::shared_ptr<Context> context) {
    const auto contextRank = context->rank;
    auto buffer = newBuffer<float>(dataSize);
    auto* ptr = buffer.data();
    for (const auto& fn : fns) {
      for (int i = 0; i < dataSize; i++) {
        ptr[i] = contextRank;
      }

      fn(context, std::vector<float*>{ptr}, dataSize);

      auto expected = (contextSize * (contextSize - 1)) / 2;
      for (int i = 0; i < dataSize; i++) {
        ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i;
      }

      for (int i = 0; i < dataSize; i++) {
        ptr[i] = contextRank;
      }

      fn(context, std::vector<float*>{ptr}, dataSize);

      expected = (contextSize * (contextSize - 1)) / 2;
      for (int i = 0; i < dataSize; i++) {
        ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i;
      }
    }
  });
}

TEST_F(AllreduceTestHP, HalfPrecisionTest) {
  int contextSize = 4;
  auto dataSize = 1024;
  auto fns = {
      allreduceRingHP, allreduceRingChunkedHP, allreduceHalvingDoublingHP};

  spawn(contextSize, [&](std::shared_ptr<Context> context) {
    const auto contextRank = context->rank;
    auto buffer = newBuffer<float16>(dataSize);
    auto* ptr = buffer.data();
    for (const auto& fn : fns) {
      for (int i = 0; i < dataSize; i++) {
        ptr[i] = contextRank;
      }

      fn(context, std::vector<float16*>{ptr}, dataSize);

      float16 expected(contextSize * (contextSize - 1) / 2);
      for (int i = 0; i < dataSize; i++) {
        ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i;
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
    AllreduceRing,
    AllreduceTest,
    ::testing::Combine(
        ::testing::Range(1, 16),
        ::testing::ValuesIn(genMemorySizes()),
        ::testing::Values(allreduceRing),
        ::testing::Values(0)));

INSTANTIATE_TEST_CASE_P(
    AllreduceRingChunked,
    AllreduceTest,
    ::testing::Combine(
        ::testing::Range(1, 16),
        ::testing::ValuesIn(genMemorySizes()),
        ::testing::Values(allreduceRingChunked),
        ::testing::Values(0)));

INSTANTIATE_TEST_CASE_P(
    AllreduceHalvingDoubling,
    AllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(
          std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 16, 24, 32})),
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(allreduceHalvingDoubling),
        ::testing::Values(0)));

INSTANTIATE_TEST_CASE_P(
    AllreduceBcubeBase2,
    AllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(
          std::vector<int>({1, 2, 4, 8, 16})),
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(allreduceBcube),
        ::testing::Values(2)));

INSTANTIATE_TEST_CASE_P(
    AllreduceBcubeBase3,
    AllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(std::vector<int>({1, 3, 9, 27})),
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(allreduceBcube),
        ::testing::Values(3)));

INSTANTIATE_TEST_CASE_P(
    AllreduceBcubeBase4,
    AllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(
          std::vector<int>({1, 4, 16})),
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(allreduceBcube),
        ::testing::Values(4)));

using Algorithm = AllreduceOptions::Algorithm;
using NewParam = std::tuple<int, int, int, bool, Algorithm>;

class AllreduceNewTest : public BaseTest,
                         public ::testing::WithParamInterface<NewParam> {};

TEST_P(AllreduceNewTest, Default) {
  auto contextSize = std::get<0>(GetParam());
  auto numPointers = std::get<1>(GetParam());
  auto dataSize = std::get<2>(GetParam());
  auto inPlace = std::get<3>(GetParam());
  auto algorithm = std::get<4>(GetParam());

  spawn(contextSize, [&](std::shared_ptr<Context> context) {
    Fixture<uint64_t> inputs(context, numPointers, dataSize);
    Fixture<uint64_t> outputs(context, numPointers, dataSize);

    AllreduceOptions opts(context);
    opts.setAlgorithm(algorithm);
    opts.setOutputs(outputs.getPointers(), dataSize);
    if (inPlace) {
      outputs.assignValues();
    } else {
      opts.setInputs(inputs.getPointers(), dataSize);
      inputs.assignValues();
      outputs.clear();
    }

    opts.setReduceFunction([](void* a, const void* b, const void* c, size_t n) {
      auto ua = static_cast<uint64_t*>(a);
      const auto ub = static_cast<const uint64_t*>(b);
      const auto uc = static_cast<const uint64_t*>(c);
      for (size_t i = 0; i < n; i++) {
        ua[i] = ub[i] + uc[i];
      }
    });

    // A small maximum segment size triggers code paths where we'll
    // have a number of segments larger than the lower bound of
    // twice the context size.
    opts.setMaxSegmentSize(128);

    allreduce(opts);

    const auto stride = contextSize * numPointers;
    const auto base = (stride * (stride - 1)) / 2;
    const auto out = outputs.getPointers();
    for (auto j = 0; j < numPointers; j++) {
      for (auto k = 0; k < dataSize; k++) {
        ASSERT_EQ(k * stride * stride + base, out[j][k])
          << "Mismatch at out[" << j << "][" << k << "]";
      }
    }
  });
}

INSTANTIATE_TEST_CASE_P(
    AllreduceNewRing,
    AllreduceNewTest,
    ::testing::Combine(
        ::testing::Values(1, 2, 4, 7),
        ::testing::Values(1, 2, 3),
        ::testing::Values(1, 10, 100, 1000, 10000),
        ::testing::Values(true, false),
        ::testing::Values(Algorithm::RING)));

INSTANTIATE_TEST_CASE_P(
    AllreduceNewBcube,
    AllreduceNewTest,
    ::testing::Combine(
        ::testing::Values(1, 2, 4, 7),
        ::testing::Values(1, 2, 3),
        ::testing::Values(1, 10, 100, 1000, 10000),
        ::testing::Values(true, false),
        ::testing::Values(Algorithm::BCUBE)));

template <typename T>
AllreduceOptions::Func getFunction() {
  void (*func)(void*, const void*, const void*, size_t) = &::gloo::sum<T>;
  return AllreduceOptions::Func(func);
}

TEST_F(AllreduceNewTest, TestTimeout) {
  spawn(2, [&](std::shared_ptr<Context> context) {
    Fixture<uint64_t> outputs(context, 1, 1);
    AllreduceOptions opts(context);
    opts.setOutputs(outputs.getPointers(), 1);
    opts.setReduceFunction(getFunction<uint64_t>());
    opts.setTimeout(std::chrono::milliseconds(10));
    if (context->rank == 0) {
      try {
        allreduce(opts);
        FAIL() << "Expected exception to be thrown";
      } catch (::gloo::IoException& e) {
        ASSERT_NE(std::string(e.what()).find("Timed out"), std::string::npos);
      }
    }
  });
}

} // namespace
} // namespace test
} // namespace gloo
