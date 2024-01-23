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

class AllgathervOptions {
 public:
  explicit AllgathervOptions(const std::shared_ptr<Context>& context)
      : context(context), timeout(context->getTimeout()) {}

  template <typename T>
  void setInput(std::unique_ptr<transport::UnboundBuffer> buf) {
    setInput(std::move(buf), sizeof(T));
  }

  template <typename T>
  void setInput(T* ptr, size_t elements_2) {
    setInput(static_cast<void*>(ptr), elements_2, sizeof(T));
  }

  template <typename T>
  void setOutput(
      std::unique_ptr<transport::UnboundBuffer> buf,
      std::vector<size_t> elements_2) {
    setOutput(std::move(buf), std::move(elements_2), sizeof(T));
  }

  template <typename T>
  void setOutput(T* ptr, std::vector<size_t> elements_2) {
    setOutput(static_cast<void*>(ptr), std::move(elements_2), sizeof(T));
  }

  void setTag(uint32_t tag_2) {
    this->tag = tag_2;
  }

  void setTimeout(std::chrono::milliseconds timeout_2) {
    this->timeout = timeout_2;
  }

 protected:
  std::shared_ptr<Context> context;
  std::unique_ptr<transport::UnboundBuffer> in;
  std::unique_ptr<transport::UnboundBuffer> out;

  // Number of elements per rank in the output.
  std::vector<size_t> elements;

  // Number of bytes per element.
  size_t elementSize = 0;

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

  friend void allgatherv(AllgathervOptions&);
};

void allgatherv(AllgathervOptions& opts);

} // namespace gloo
