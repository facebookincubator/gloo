/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <functional>
#include <memory>

#include "gloo/context.h"
#include "gloo/transport/unbound_buffer.h"

namespace gloo {

struct ReduceOptions {
  // The input and output buffers can either be specified as a unbound
  // buffer (that can be cached and reused by the caller), or a
  // literal pointer and number of elements stored at that pointer.
  std::unique_ptr<transport::UnboundBuffer> inBuffer;
  void* inPtr = nullptr;
  std::unique_ptr<transport::UnboundBuffer> outBuffer;
  void* outPtr = nullptr;

  // Number of elements.
  size_t elements = 0;

  // Number of bytes per element.
  size_t elementSize = 0;

  // Rank of process to reduce to.
  int root = 0;

  // Reduction function (output, input 1, input 2, number of elements).
  std::function<void(void*, const void*, const void*, size_t)> reduce;

  // Tag for this gather operation.
  // Must be unique across operations executing in parallel.
  uint32_t tag = 0;

  // This is the maximum size of each I/O operation (send/recv) of which
  // two are in flight at all times. A smaller value leads to more
  // overhead and a larger value leads to poor cache behavior.
  static constexpr size_t kMaxSegmentSize = 1024 * 1024;

  // Internal use only. This is used to exercise code paths where we
  // have more than 2 segments per rank without making the tests slow
  // (because they would require millions of elements if the default
  // were not configurable).
  size_t maxSegmentSize = kMaxSegmentSize;
};

void reduce(const std::shared_ptr<Context>& context, ReduceOptions& opts);

} // namespace gloo
