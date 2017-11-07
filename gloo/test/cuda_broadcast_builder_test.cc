/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/test/cuda_base_test.h"

#include "gloo/broadcast_builder.h"

namespace gloo {
namespace test {
namespace {

template <typename T>
class CudaBroadcastBuilderTest : public BaseTest {
};

using CudaBroadcastBuilderTypes = ::testing::Types<float, float16>;

TYPED_TEST_CASE(CudaBroadcastBuilderTest, CudaBroadcastBuilderTypes);

TYPED_TEST(CudaBroadcastBuilderTest, TestAsync) {
  // Only test with 10 elements; this is not an algorithm implementation test
  auto count = 10;
  // Works for context size 1 or > 1
  for (auto size = 1; size <= 2; size++) {
    this->spawn(size, [&](std::shared_ptr<Context> context) {
      ::gloo::BroadcastBuilder<TypeParam> builder;
      // Run with varying root
      // TODO(PN): go up to processCount
      for (auto rootProcessRank = 0; rootProcessRank < 1; rootProcessRank++) {
        // TODO(PN): go up to pointerCount
        for (auto rootPointerRank = 0; rootPointerRank < 1; rootPointerRank++) {
          // Test for Non-GPU Direct
          /* TODO (gains): For GPU Direct, we need IB. See if we can run IB on
          single Machine without leaving the adapter. */
          for (bool gpuDirect : { false }) {
            for (bool withStream : { false, true }) {
            // Works for 1 or > 1 inputs
              for (auto inputs = 1; inputs <= 2; inputs++) {
                CudaFixture<TypeParam> f(context, inputs, count);
                f.assignValuesAsync();
                builder.
                  setInputs(f.getCudaPointers()).
                  setCount(count).
                  setRootRank(rootProcessRank).
                  setRootPointerRank(rootPointerRank).
                  setGPUDirect(gpuDirect);

                if (withStream) {
                  builder.setStreams(f.getCudaStreams());
                } else {
                  builder.setStreams({});
                }

                // Run the algorithm
                auto algorithm = builder.getAlgorithm(context);
                algorithm->run();

                // If using streams, then synchronize.
                if (withStream) {
                  f.synchronizeCudaStreams();
                }

                // Copy the results to host then verify them.
                f.copyToHost();
                f.checkBroadcastResult(f, rootProcessRank, rootPointerRank);
              }
            }
          }
        }
      }
    });
  }
}

} // namespace
} // namespace test
} // namespace gloo
