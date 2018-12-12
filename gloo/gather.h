/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "gloo/context.h"
#include "gloo/transport/unbound_buffer.h"

namespace gloo {

class GatherOptions {
 public:
  explicit GatherOptions(const std::shared_ptr<Context>& context)
      : context(context), timeout(context->getTimeout()) {}

  template <typename T>
  void setInput(std::unique_ptr<transport::UnboundBuffer> buf) {
    elementSize = sizeof(T);
    in = std::move(buf);
  }

  template <typename T>
  void setInput(T* ptr, size_t elements) {
    elementSize = sizeof(T);
    in = context->createUnboundBuffer(ptr, elements * sizeof(T));
  }

  template <typename T>
  void setOutput(std::unique_ptr<transport::UnboundBuffer> buf) {
    elementSize = sizeof(T);
    out = std::move(buf);
  }

  template <typename T>
  void setOutput(T* ptr, size_t elements) {
    elementSize = sizeof(T);
    out = context->createUnboundBuffer(ptr, elements * sizeof(T));
  }

  void setRoot(int root) {
    this->root = root;
  }

  void setTag(uint32_t tag) {
    this->tag = tag;
  }

  void setTimeout(std::chrono::milliseconds timeout) {
    this->timeout = timeout;
  }

 protected:
  std::shared_ptr<Context> context;
  std::unique_ptr<transport::UnboundBuffer> in;
  std::unique_ptr<transport::UnboundBuffer> out;

  // Number of bytes per element.
  size_t elementSize = 0;

  // Rank of receiving process.
  int root = -1;

  // Tag for this operation.
  // Must be unique across operations executing in parallel.
  uint32_t tag = 0;

  // End-to-end timeout for this operation.
  std::chrono::milliseconds timeout;

  friend void gather(GatherOptions&);
};

void gather(GatherOptions& opts);

} // namespace gloo
