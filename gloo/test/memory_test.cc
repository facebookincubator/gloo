/**
 * Copyright (c) 2019-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/test/base_test.h"

#include <fstream>
#include <sstream>

namespace gloo {
namespace test {
namespace {

size_t readResidentSetSize() {
  std::stringstream path;
  path << "/proc/" << getpid() << "/statm";
  std::ifstream f(path.str());
  size_t size;
  size_t resident;
  f >> size >> resident;
  return (getpagesize() * resident);
}

class MemoryTest : public BaseTest {};

TEST_F(MemoryTest, ManySlotsNoLeaks) {
  spawn(Transport::TCP, 2, [&](std::shared_ptr<Context> context) {
    size_t tmp0;
    size_t tmp1;
    auto buf0 = context->createUnboundBuffer(&tmp0, sizeof(tmp0));
    auto buf1 = context->createUnboundBuffer(&tmp1, sizeof(tmp1));
    auto step = [&](size_t slot) {
      const auto peer = 1 - context->rank;
      if (context->rank == 0) {
        buf0->send(peer, slot);
        buf1->recv(peer, slot);
        buf0->waitSend();
        buf1->waitRecv();
      } else {
        buf0->recv(peer, slot);
        buf1->send(peer, slot);
        buf0->waitRecv();
        buf1->waitSend();
      }
    };

    // Prime processes with a few send/recv ping/pongs
    size_t slot = 0;
    for (auto i = 0; i < 10; i++) {
      step(slot++);
    }

    // Read current memory usage and run for a while
    auto baselineResidentSetSize = readResidentSetSize();
    for (auto i = 0; i < 10000; i++) {
      step(slot++);
    }

    // Ensure memory usage didn't increase
    auto newResidentSetSize = readResidentSetSize();
    ASSERT_EQ(baselineResidentSetSize, newResidentSetSize);
  });
}

} // namespace
} // namespace test
} // namespace gloo
