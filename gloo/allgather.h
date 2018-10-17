/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/context.h"
#include "gloo/transport/unbound_buffer.h"

namespace gloo {

struct AllgatherOptions {
  // The input and output can either be specified as a unbound buffer
  // (that can be cached and reused by the caller), or a literal
  // pointer and number of elements stored at that pointer.
  //
  // The operation is executed in place on the output if the input is
  // set to null. The input for this process is assumed to be at the
  // location in the output buffer where it would otherwise be.
  std::unique_ptr<transport::UnboundBuffer> inBuffer;
  void* inPtr;
  size_t inElements;
  std::unique_ptr<transport::UnboundBuffer> outBuffer;
  void* outPtr;
  size_t outElements;

  // Number of bytes per element.
  size_t elementSize;

  // Tag for this gather operation.
  // Must be unique across operations executing in parallel.
  uint32_t tag;
};

void allgather(const std::shared_ptr<Context>& context, AllgatherOptions& opts);

} // namespace gloo
