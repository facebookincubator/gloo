/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/broadcast_builder.h"
#include "gloo/test/base_test.h"

namespace gloo {
namespace test {
namespace {

template <typename T>
class BroadcastBuilderTest : public BaseTest {
};

using BroadcastBuilderTypes =
  ::testing::Types<
    int8_t, uint8_t, int32_t, int64_t, uint64_t, float, double, float16
  >;

TYPED_TEST_CASE(BroadcastBuilderTest, BroadcastBuilderTypes);

TYPED_TEST(BroadcastBuilderTest, Test) {
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
          // Works for 1 or > 1 inputs
          for (auto inputs = 1; inputs <= 2; inputs++) {
            Fixture<TypeParam> f(context, inputs, count);
            f.assignValues();

            // Build and run algorithm
            auto algorithm = builder.
              setInputs(f.getPointers()).
              setCount(count).
              setRootRank(rootProcessRank).
              setRootPointerRank(rootPointerRank).
              getAlgorithm(context);
            algorithm->run();

            // Verify results
            f.checkBroadcastResult(f, rootProcessRank, rootPointerRank);
          }
        }
      }
    });
  }
}

} // namespace
} // namespace test
} // namespace gloo
