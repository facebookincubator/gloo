/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include "gloo/transport/unbound_buffer.h"

namespace gloo {
namespace transport {

// Have to provide implementation for pure virtual destructor.
UnboundBuffer::~UnboundBuffer() {}
bool UnboundBuffer::testRecv()  {
  abort();
}
bool UnboundBuffer::testSend() {
  abort();
}

} // namespace transport
} // namespace gloo
