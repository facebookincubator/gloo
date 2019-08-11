/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <functional>
#include <thread>
#include <vector>

#include "gloo/broadcast.h"
#include "gloo/broadcast_one_to_all.h"
#include "gloo/test/base_test.h"

namespace gloo {
namespace test {
namespace {

// Function to instantiate and run algorithm.
using Func = std::unique_ptr<::gloo::Algorithm>(
    std::shared_ptr<::gloo::Context>&,
    std::vector<float*> ptrs,
    size_t count,
    int rootRank,
    int rootPointerRank);

// Test parameterization.
using Param = std::tuple<int, int, size_t, std::function<Func>>;

// Test fixture.
class BroadcastTest : public BaseTest,
                      public ::testing::WithParamInterface<Param> {
 public:
  void assertResult(Fixture<float>& fixture, int root, int rootPointer) {
    // Expected is set to the expected value at ptr[0]
    const auto expected = root * fixture.srcs.size() + rootPointer;
    // Stride is difference between values at subsequent indices
    const auto stride = fixture.srcs.size() * fixture.context->size;
    // Verify all buffers passed by this instance
    for (const auto& ptr : fixture.getPointers()) {
      for (auto i = 0; i < fixture.count; i++) {
        ASSERT_EQ((i * stride) + expected, ptr[i])
          << "Mismatch at index " << i;
      }
    }
  }
};

TEST_P(BroadcastTest, Default) {
  auto processCount = std::get<0>(GetParam());
  auto pointerCount = std::get<1>(GetParam());
  auto elementCount = std::get<2>(GetParam());
  auto fn = std::get<3>(GetParam());

  spawn(processCount, [&](std::shared_ptr<Context> context) {
      auto fixture = Fixture<float>(context, pointerCount, elementCount);
      auto ptrs = fixture.getPointers();

      // Run with varying root
      // TODO(PN): go up to processCount
      for (auto rootProcessRank = 0;
           rootProcessRank < 1;
           rootProcessRank++) {
        // TODO(PN): go up to pointerCount
        for (auto rootPointerRank = 0;
             rootPointerRank < 1;
             rootPointerRank++) {
          fixture.assignValues();
          auto algorithm = fn(context,
                              ptrs,
                              elementCount,
                              rootProcessRank,
                              rootPointerRank);
          algorithm->run();

          // Verify result
          assertResult(fixture, rootProcessRank, rootPointerRank);
        }
      }
    });
}

std::vector<size_t> genMemorySizes() {
  std::vector<size_t> v;
  v.push_back(sizeof(float));
  v.push_back(100);
  v.push_back(1000);
  v.push_back(10000);
  return v;
}

static std::function<Func> broadcastOneToAll = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float*> ptrs,
    size_t count,
    int rootProcessRank,
    int rootPointerRank) {
  return std::unique_ptr<::gloo::Algorithm>(
    new ::gloo::BroadcastOneToAll<float>(
      context, ptrs, count, rootProcessRank, rootPointerRank));
};

INSTANTIATE_TEST_CASE_P(
    OneToAllBroadcast,
    BroadcastTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 4, 5),
        ::testing::Values(1, 2),
        ::testing::ValuesIn(genMemorySizes()),
        ::testing::Values(broadcastOneToAll)));

INSTANTIATE_TEST_CASE_P(
    LargeBroadcast,
    BroadcastTest,
    ::testing::Combine(
        ::testing::Values(2),
        ::testing::Values(1),
        ::testing::Values((size_t)512 * 1024 * 1024),
        ::testing::Values(broadcastOneToAll)));

using NewParam = std::tuple<int, int, bool, bool>;

class BroadcastNewTest : public BaseTest,
                         public ::testing::WithParamInterface<NewParam> {};

TEST_P(BroadcastNewTest, Default) {
  auto contextSize = std::get<0>(GetParam());
  auto dataSize = std::get<1>(GetParam());
  auto passBuffers = std::get<2>(GetParam());
  auto inPlace = std::get<3>(GetParam());

  spawn(contextSize, [&](std::shared_ptr<Context> context) {
      auto input = Fixture<uint64_t>(context, 1, dataSize);
      auto output = Fixture<uint64_t>(context, 1, dataSize);

      // Take turns being root
      for (auto root = 0; root < context->size; root++) {
        BroadcastOptions opts(context);
        opts.setRoot(root);

        input.clear();
        output.clear();

        if (context->rank == root) {
          if (inPlace) {
            // If in place, use output as input
            output.assignValues();
          } else {
            // If not in place, use separate input
            input.assignValues();
            if (passBuffers) {
              opts.setInput<uint64_t>(context->createUnboundBuffer(
                  input.getPointer(),
                  dataSize * sizeof(uint64_t)));
            } else {
              opts.setInput(input.getPointer(), dataSize);
            }
          }
        }

        if (passBuffers) {
          opts.setOutput<uint64_t>(context->createUnboundBuffer(
              output.getPointer(),
              dataSize * sizeof(uint64_t)));
        } else {
          opts.setOutput(output.getPointer(), dataSize);
        }

        broadcast(opts);

        // Validate output
        const auto ptr = output.getPointer();
        const auto stride = context->size;
        for (auto k = 0; k < dataSize; k++) {
          ASSERT_EQ(root + k * stride, ptr[k])
            << "Mismatch at index " << k;
        }
      }
    });
}

INSTANTIATE_TEST_CASE_P(
    BroadcastNewDefault,
    BroadcastNewTest,
    ::testing::Combine(
        ::testing::Values(1, 2, 4, 7),
        ::testing::Values(1, 10, 100),
        ::testing::Values(false, true),
        ::testing::Values(false, true)));

TEST_F(BroadcastTest, TestTimeout) {
  spawn(2, [&](std::shared_ptr<Context> context) {
    Fixture<uint64_t> output(context, 1, 1);
    BroadcastOptions opts(context);
    opts.setOutput(output.getPointer(), 1);
    opts.setRoot(0);
    opts.setTimeout(std::chrono::milliseconds(10));
    if (context->rank == 0) {
      try {
        broadcast(opts);
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
