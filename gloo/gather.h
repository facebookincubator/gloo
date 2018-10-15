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

struct GatherOptions {
  // The input and output buffers can either be specified as an unbound
  // buffer (that can be cached and reused by the caller), or a
  // literal pointer and number of elements stored at that pointer.
  std::unique_ptr<transport::UnboundBuffer> inBuffer;
  void* inPtr;
  size_t inElements;
  std::unique_ptr<transport::UnboundBuffer> outBuffer;
  void* outPtr;
  size_t outElements;

  // Number of bytes per element.
  size_t elementSize;

  // Rank of receiving process.
  int root;

  // Tag for this gather operation.
  // Must be unique across operations executing in parallel.
  uint32_t tag;
};

void gather(const std::shared_ptr<Context>& context, GatherOptions& opts);

} // namespace gloo
