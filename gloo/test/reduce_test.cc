/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/math.h"
#include "gloo/reduce.h"
#include "gloo/test/base_test.h"

namespace gloo {
namespace test {
namespace {

// Test parameterization.
using Param = std::tuple<int, size_t, bool>;

// Test fixture.
class ReduceTest : public BaseTest,
                   public ::testing::WithParamInterface<Param> {};

TEST_P(ReduceTest, Default) {
  auto contextSize = std::get<0>(GetParam());
  auto dataSize = std::get<1>(GetParam());
  auto inPlace = std::get<2>(GetParam());

  spawn(contextSize, [&](std::shared_ptr<Context> context) {
    auto input = Fixture<uint64_t>(context, 1, dataSize);
    auto output = Fixture<uint64_t>(context, 1, dataSize);

    ReduceOptions opts;

    if (inPlace) {
      opts.outPtr = output.getPointer();
    } else {
      opts.inPtr = input.getPointer();
      opts.outPtr = output.getPointer();
    }

    opts.elements = dataSize;
    opts.elementSize = sizeof(uint64_t);
    opts.reduce = [](void* a, const void* b, const void* c, size_t n) {
      auto ua = static_cast<uint64_t*>(a);
      const auto ub = static_cast<const uint64_t*>(b);
      const auto uc = static_cast<const uint64_t*>(c);
      for (size_t i = 0; i < n; i++) {
        ua[i] = ub[i] + uc[i];
      }
    };

    // A small maximum segment size triggers code paths where we'll
    // have a number of segments larger than the lower bound of
    // twice the context size.
    opts.maxSegmentSize = 128;

    // Take turns being root
    for (opts.root = 0; opts.root < context->size; opts.root++) {
      if (inPlace) {
        output.assignValues();
      } else {
        input.assignValues();
        output.clear();
      }
      reduce(context, opts);

      // Validate result if this process was root
      if (context->rank == opts.root) {
        const auto base = (contextSize * (contextSize - 1)) / 2;
        const auto ptr = output.getPointer();
        const auto stride = context->size;
        for (auto j = 0; j < dataSize; j++) {
          ASSERT_EQ(j * stride * stride + base, ptr[j])
              << "Mismatch at index " << j;
        }
      }
    }
  });
}

std::vector<size_t> genMemorySizes() {
  std::vector<size_t> v;
  v.push_back(1);
  v.push_back(10);
  v.push_back(100);
  v.push_back(1000);
  v.push_back(10000);
  return v;
}

INSTANTIATE_TEST_CASE_P(
    ReduceDefault,
    ReduceTest,
    ::testing::Combine(
        ::testing::Values(2, 4, 7),
        ::testing::ValuesIn(genMemorySizes()),
        ::testing::Values(true, false)));

} // namespace
} // namespace test
} // namespace gloo
