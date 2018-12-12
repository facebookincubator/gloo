/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include "gloo/context.h"
#include "gloo/transport/unbound_buffer.h"

namespace gloo {

class ScatterOptions {
 public:
  explicit ScatterOptions(const std::shared_ptr<Context>& context)
      : context(context), timeout(context->getTimeout()) {}

  template <typename T>
  void setInputs(std::vector<std::unique_ptr<transport::UnboundBuffer>> bufs) {
    this->elementSize = sizeof(T);
    this->in = std::move(bufs);
  }

  template <typename T>
  void setInputs(std::vector<T*> ptrs, size_t elements) {
    setInputs(ptrs.data(), ptrs.size(), elements);
  }

  template <typename T>
  void setInputs(T** ptrs, size_t len, size_t elements) {
    this->elementSize = sizeof(T);
    this->in.reserve(len);
    for (size_t i = 0; i < len; i++) {
      this->in.push_back(
          context->createUnboundBuffer(ptrs[i], elements * sizeof(T)));
    }
  }

  template <typename T>
  void setOutput(std::unique_ptr<transport::UnboundBuffer> buf) {
    this->elementSize = sizeof(T);
    this->out = std::move(buf);
  }

  template <typename T>
  void setOutput(T* ptr, size_t elements) {
    this->elementSize = sizeof(T);
    this->out = context->createUnboundBuffer(ptr, elements * sizeof(T));
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

  // Scatter has N input buffers where each one in its
  // entirety gets sent to a rank. The input(s) only need to
  // be set on the root process.
  std::vector<std::unique_ptr<transport::UnboundBuffer>> in;

  // Scatter only has a single output buffer per rank.
  std::unique_ptr<transport::UnboundBuffer> out;

  // Number of bytes per element.
  size_t elementSize = 0;

  // Rank of process to scatter from.
  int root = -1;

  // Tag for this operation.
  // Must be unique across operations executing in parallel.
  uint32_t tag = 0;

  // End-to-end timeout for this operation.
  std::chrono::milliseconds timeout;

  friend void scatter(ScatterOptions&);
};

void scatter(ScatterOptions& opts);

} // namespace gloo
