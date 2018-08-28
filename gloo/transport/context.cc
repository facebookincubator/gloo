/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/transport/context.h"

namespace gloo {
namespace transport {

Context::Context(int rank, int size) : rank(rank), size(size) {}

// Have to provide implementation for pure virtual destructor.
Context::~Context() {}

} // namespace transport
} // namespace gloo
