/**
 * Copyright (c) 2019-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "gloo/context.h"
#include "gloo/transport/unbound_buffer.h"

namespace gloo {

class GathervOptions {
 public:
  explicit GathervOptions(const std::shared_ptr<Context>& context)
      : context(context), timeout(context->getTimeout()) {}

  template <typename T>
  void setInput(std::unique_ptr<transport::UnboundBuffer> buf) {
    setInput(std::move(buf), sizeof(T));
  }

  template <typename T>
  void setInput(T* ptr, size_t elements) {
    setInput(static_cast<void*>(ptr), elements, sizeof(T));
  }

  template <typename T>
  void setOutput(
      std::unique_ptr<transport::UnboundBuffer> buf,
      std::vector<size_t> elementsPerRank) {
    setOutput(std::move(buf), std::move(elementsPerRank), sizeof(T));
  }

  template <typename T>
  void setOutput(T* ptr, std::vector<size_t> elementsPerRank) {
    setOutput(static_cast<void*>(ptr), std::move(elementsPerRank), sizeof(T));
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

  // Number of elements per rank in the output.
  std::vector<size_t> elementsPerRank;

  // Number of bytes per element.
  size_t elementSize = 0;

  // Rank of receiving process.
  int root = -1;

  // Tag for this operation.
  // Must be unique across operations executing in parallel.
  uint32_t tag = 0;

  // End-to-end timeout for this operation.
  std::chrono::milliseconds timeout;

  // Set element size, or check the argument is equal to the current value.
  void setElementSize(size_t elementSize);

  // Untemplated implementation of setInput on unbound buffer.
  void setInput(
      std::unique_ptr<transport::UnboundBuffer> buf,
      size_t elementSize);

  // Untemplated implementation of setInput on opaque pointer.
  void setInput(void* ptr, size_t elements, size_t elementSize);

  // Untemplated implementation of setOutput on unbound buffer.
  void setOutput(
      std::unique_ptr<transport::UnboundBuffer> buf,
      std::vector<size_t> elements,
      size_t elementSize);

  // Untemplated implementation of setOutput on opaque pointer.
  void setOutput(void* ptr, std::vector<size_t> elements, size_t elementSize);

  friend void gatherv(GathervOptions&);
};

void gatherv(GathervOptions& opts);

} // namespace gloo
