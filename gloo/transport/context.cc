/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/transport/context.h"

namespace gloo {
namespace transport {

Context::Context(int rank, int size) : rank(rank), size(size) {
  pairs_.resize(size);
}

// Have to provide implementation for pure virtual destructor.
Context::~Context() {}

std::unique_ptr<transport::Pair>& Context::getPair(int rank) {
  return pairs_.at(rank);
}

} // namespace transport
} // namespace gloo
