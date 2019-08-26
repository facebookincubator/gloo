/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <functional>
#include <memory>
#include <vector>

#include "gloo/cuda_broadcast_one_to_all.h"
#include "gloo/test/cuda_base_test.h"

namespace gloo {
namespace test {
namespace {

// Function to instantiate and run algorithm.
using Func = std::unique_ptr<::gloo::Algorithm>(
    std::shared_ptr<::gloo::Context>&,
    std::vector<float*> ptr,
    int count,
    int rootRank,
    int rootPointerRank,
    std::vector<cudaStream_t> streams);

// Test parameterization.
using Param = std::tuple<Transport, int, int, int, std::function<Func>>;

// Test fixture.
class CudaBroadcastTest : public CudaBaseTest,
                          public ::testing::WithParamInterface<Param> {
 public:
  void assertResult(CudaFixture<float>& fixture, int root, int rootPointer) {
    fixture.copyToHost();

    // Expected is set to the expected value at ptr[0]
    const auto expected = root * fixture.srcs.size() + rootPointer;
    // Stride is difference between values at subsequent indices
    const auto stride = fixture.srcs.size() * fixture.context->size;
    // Verify all buffers passed by this instance
    for (const auto& ptr : fixture.getPointers()) {
      for (auto i = 0; i < fixture.count; i++) {
        ASSERT_EQ((i * stride) + expected, ptr[i]) << "Mismatch at index " << i;
      }
    }
  }
};

TEST_P(CudaBroadcastTest, Default) {
  const auto transport = std::get<0>(GetParam());
  const auto contextSize = std::get<1>(GetParam());
  const auto numPointers = std::get<2>(GetParam());
  const auto dataSize = std::get<3>(GetParam());
  const auto fn = std::get<4>(GetParam());

  spawn(transport, contextSize, [&](std::shared_ptr<Context> context) {
    auto fixture = CudaFixture<float>(context, numPointers, dataSize);
    auto ptrs = fixture.getCudaPointers();

    // Run with varying root
    // TODO(PN): go up to processCount
    for (auto rootProcessRank = 0; rootProcessRank < 1; rootProcessRank++) {
      // TODO(PN): go up to pointerCount
      for (auto rootPointerRank = 0; rootPointerRank < 1; rootPointerRank++) {
        fixture.assignValues();
        auto algorithm =
            fn(context, ptrs, dataSize, rootProcessRank, rootPointerRank, {});
        algorithm->run();

        // Verify result
        assertResult(fixture, rootProcessRank, rootPointerRank);
      }
    }
  });
}

TEST_P(CudaBroadcastTest, DefaultAsync) {
  const auto transport = std::get<0>(GetParam());
  const auto contextSize = std::get<1>(GetParam());
  const auto numPointers = std::get<2>(GetParam());
  const auto dataSize = std::get<3>(GetParam());
  const auto fn = std::get<4>(GetParam());

  spawn(transport, contextSize, [&](std::shared_ptr<Context> context) {
    auto fixture = CudaFixture<float>(context, numPointers, dataSize);
    auto ptrs = fixture.getCudaPointers();
    auto streams = fixture.getCudaStreams();

    // Run with varying root
    // TODO(PN): go up to contextSize
    for (auto rootProcessRank = 0; rootProcessRank < numPointers;
         rootProcessRank++) {
      // TODO(PN): go up to pointerCount
      for (auto rootPointerRank = 0; rootPointerRank < 1; rootPointerRank++) {
        fixture.assignValuesAsync();
        auto algorithm = fn(
            context, ptrs, dataSize, rootProcessRank, rootPointerRank, streams);
        algorithm->run();

        // Verify result
        fixture.synchronizeCudaStreams();
        assertResult(fixture, rootProcessRank, rootPointerRank);
      }
    }
  });
}

static std::function<Func> broadcastOneToAll =
    [](std::shared_ptr<::gloo::Context>& context,
       std::vector<float*> ptrs,
       int count,
       int rootProcessRank,
       int rootPointerRank,
       std::vector<cudaStream_t> streams) {
      return std::unique_ptr<::gloo::Algorithm>(
          new ::gloo::CudaBroadcastOneToAll<float>(
              context, ptrs, count, rootProcessRank, rootPointerRank, streams));
    };

INSTANTIATE_TEST_CASE_P(
    OneToAllBroadcast,
    CudaBroadcastTest,
    ::testing::Combine(
        ::testing::ValuesIn(kTransportsForClassAlgorithms),
        ::testing::Values(2, 3, 4, 5),
        ::testing::Values(1, cudaNumDevices()),
        ::testing::Values(4, 100, 1000, 10000),
        ::testing::Values(broadcastOneToAll)));

} // namespace
} // namespace test
} // namespace gloo
