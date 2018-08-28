/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/transport/ibverbs/context.h"

#include "gloo/transport/ibverbs/device.h"

namespace gloo {
namespace transport {
namespace ibverbs {

Context::Context(std::shared_ptr<Device> device, int rank, int size)
    : ::gloo::transport::Context(rank, size), device_(device) {
}

Context::~Context() {
}

} // namespace ibverbs
} // namespace transport
} // namespace gloo
