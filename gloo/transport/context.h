/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

namespace gloo {
namespace transport {

// The context represents a set of pairs that belong to the same
// group. Its equivalent class at the top level namespace represents
// the same group but cannot represent transport specifics.
//
// While implementing the recv-from-any functionality we realized we
// realized we needed some transport-specific state shared between all
// pairs in a group, to arbitrate between multiple pairs attempting to
// send to the same buffer. To avoid over-generalization, transports
// can implement this however they want in their own subclass.
//
class Context {
 public:
  Context(int rank, int size);

  virtual ~Context();

  const int rank;
  const int size;
};

} // namespace transport
} // namespace gloo
