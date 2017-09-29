/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/test/cuda_base_test.h"

#include "gloo/allreduce_builder.h"

namespace gloo {
namespace test {
namespace {

template <typename T>
class CudaAllreduceBuilderTest : public BaseTest {
};

using CudaAllreduceBuilderTypes = ::testing::Types<float, float16>;

TYPED_TEST_CASE(CudaAllreduceBuilderTest, CudaAllreduceBuilderTypes);

TYPED_TEST(CudaAllreduceBuilderTest, Test) {
  std::vector<enum ::gloo::AllreduceBuilder<TypeParam>::Implementation> impls =
    {
      ::gloo::AllreduceBuilder<TypeParam>::HalvingDoubling,
      ::gloo::AllreduceBuilder<TypeParam>::HalvingDoublingPipelined,
      ::gloo::AllreduceBuilder<TypeParam>::Ring,
      ::gloo::AllreduceBuilder<TypeParam>::RingChunked,
    };

  // Only test with 10 elements; this is not an algorithm implementation test
  auto count = 10;

  // Works for context size 1 or > 1
  for (auto size = 1; size <= 2; size++) {
    this->spawn(size, [&](std::shared_ptr<Context> context) {
        ::gloo::AllreduceBuilder<TypeParam> builder;

        for (const auto impl : impls) {
          // Works for 1 or > 1 inputs
          for (auto inputs = 1; inputs <= 2; inputs++) {
            CudaFixture<TypeParam> f(context, inputs, count);
            f.assignValues();

            // Build and run algorithm
            auto algorithm = builder.
              setInputs(f.getCudaPointers()).
              setCount(count).
              setImplementation(impl).
              getAlgorithm(context);
            algorithm->run();

            // Verify results
            f.copyToHost();
            f.checkAllreduceResult();
          }
        }
      });
  }
}

TYPED_TEST(CudaAllreduceBuilderTest, TestAsync) {
  std::vector<enum ::gloo::AllreduceBuilder<TypeParam>::Implementation> impls =
    {
      ::gloo::AllreduceBuilder<TypeParam>::HalvingDoubling,
      ::gloo::AllreduceBuilder<TypeParam>::HalvingDoublingPipelined,
      ::gloo::AllreduceBuilder<TypeParam>::Ring,
      ::gloo::AllreduceBuilder<TypeParam>::RingChunked,
    };

  // Only test with 10 elements; this is not an algorithm implementation test
  auto count = 10;

  // Works for context size 1 or > 1
  for (auto size = 1; size <= 2; size++) {
    this->spawn(size, [&](std::shared_ptr<Context> context) {
        ::gloo::AllreduceBuilder<TypeParam> builder;

        for (const auto impl : impls) {
          // Works for 1 or > 1 inputs
          for (auto inputs = 1; inputs <= 2; inputs++) {
            CudaFixture<TypeParam> f(context, inputs, count);
            f.assignValuesAsync();

            // Build and run algorithm
            auto algorithm = builder.
              setInputs(f.getCudaPointers()).
              setStreams(f.getCudaStreams()).
              setCount(count).
              setImplementation(impl).
              getAlgorithm(context);
            algorithm->run();

            // Verify results
            f.copyToHost();
            f.checkAllreduceResult();
          }
        }
      });
  }
}

} // namespace
} // namespace test
} // namespace gloo
