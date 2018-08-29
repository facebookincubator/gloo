/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/test/base_test.h"

#include "gloo/transport/tcp/unbound_buffer.h"

namespace gloo {
namespace test {
namespace {

// Test parameterization (context size, buffer size).
using Param = std::tuple<int, int>;

// Test fixture.
class SendRecvTest : public BaseTest,
                     public ::testing::WithParamInterface<Param> {};

TEST_P(SendRecvTest, AllToAll) {
  auto contextSize = std::get<0>(GetParam());
  auto dataSize = std::get<1>(GetParam());

  spawn(contextSize, [&](std::shared_ptr<Context> context) {
      using buffer_ptr = std::unique_ptr<::gloo::transport::UnboundBuffer>;
      std::vector<int> input(contextSize);
      std::vector<int> output(contextSize);
      std::vector<buffer_ptr> inputBuffers(contextSize);
      std::vector<buffer_ptr> outputBuffers(contextSize);

      // Initialize
      for (auto i = 0; i < context->size; i++) {
        input[i] = context->rank;
        output[i] = -1;
        inputBuffers[i] =
          context->createUnboundBuffer(&input[i], sizeof(input[i]));
        outputBuffers[i] =
          context->createUnboundBuffer(&output[i], sizeof(output[i]));
      }

      // Send a message with the local rank to every other rank
      for (auto i = 0; i < context->size; i++) {
        if (i == context->rank) {
          continue;
        }
        inputBuffers[i]->send(i, context->rank);
      }

      // Receive message from every other rank
      for (auto i = 0; i < context->size; i++) {
        if (i == context->rank) {
          continue;
        }
        outputBuffers[i]->recv(i, i);
      }

      // Wait for send and recv to complete
      for (auto i = 0; i < context->size; i++) {
        if (i == context->rank) {
          continue;
        }
        inputBuffers[i]->waitSend();
        outputBuffers[i]->waitRecv();
      }

      // Verify output
      for (auto i = 0; i < context->size; i++) {
        if (i == context->rank) {
          continue;
        }
        ASSERT_EQ(i, output[i]) << "Mismatch at index " << i;
      }
    });
}

INSTANTIATE_TEST_CASE_P(
    SendRecvDefault,
    SendRecvTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 4, 5, 6, 7, 8),
        ::testing::Values(1)));


} // namespace
} // namespace test
} // namespace gloo
