/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "gloo/context.h"
#include "gloo/transport/unbound_buffer.h"

namespace gloo {

class AllreduceOptions {
 public:
  using Func = std::function<void(void*, const void*, const void*, size_t)>;

  explicit AllreduceOptions(const std::shared_ptr<Context>& context)
      : context(context), timeout(context->getTimeout()) {}

  template <typename T>
  void setInput(std::unique_ptr<transport::UnboundBuffer> buf) {
    std::vector<std::unique_ptr<transport::UnboundBuffer>> bufs(1);
    bufs[0] = std::move(buf);
    setInputs<T>(std::move(bufs));
  }

  template <typename T>
  void setInputs(std::vector<std::unique_ptr<transport::UnboundBuffer>> bufs) {
    this->elements = bufs[0]->size / sizeof(T);
    this->elementSize = sizeof(T);
    this->in = std::move(bufs);
  }

  template <typename T>
  void setInput(T* ptr, size_t elements) {
    setInputs(&ptr, 1, elements);
  }

  template <typename T>
  void setInputs(std::vector<T*> ptrs, size_t elements) {
    setInputs(ptrs.data(), ptrs.size(), elements);
  }

  template <typename T>
  void setInputs(T** ptrs, size_t len, size_t elements) {
    this->elements = elements;
    this->elementSize = sizeof(T);
    this->in.reserve(len);
    for (size_t i = 0; i < len; i++) {
      this->in.push_back(
          context->createUnboundBuffer(ptrs[i], elements * sizeof(T)));
    }
  }

  template <typename T>
  void setOutput(std::unique_ptr<transport::UnboundBuffer> buf) {
    std::vector<std::unique_ptr<transport::UnboundBuffer>> bufs(1);
    bufs[0] = std::move(buf);
    setOutputs<T>(std::move(bufs));
  }

  template <typename T>
  void setOutputs(std::vector<std::unique_ptr<transport::UnboundBuffer>> bufs) {
    this->elements = bufs[0]->size / sizeof(T);
    this->elementSize = sizeof(T);
    this->out = std::move(bufs);
  }

  template <typename T>
  void setOutput(T* ptr, size_t elements) {
    setOutputs(&ptr, 1, elements);
  }

  template <typename T>
  void setOutputs(std::vector<T*> ptrs, size_t elements) {
    setOutputs(ptrs.data(), ptrs.size(), elements);
  }

  template <typename T>
  void setOutputs(T** ptrs, size_t len, size_t elements) {
    this->elements = elements;
    this->elementSize = sizeof(T);
    this->out.reserve(len);
    for (size_t i = 0; i < len; i++) {
      this->out.push_back(
          context->createUnboundBuffer(ptrs[i], elements * sizeof(T)));
    }
  }

  void setReduceFunction(Func fn) {
    this->reduce = fn;
  }

  void setTag(uint32_t tag) {
    this->tag = tag;
  }

  void setMaxSegmentSize(size_t maxSegmentSize) {
    this->maxSegmentSize = maxSegmentSize;
  }

  void setTimeout(std::chrono::milliseconds timeout) {
    this->timeout = timeout;
  }

 protected:
  std::shared_ptr<Context> context;
  std::vector<std::unique_ptr<transport::UnboundBuffer>> in;
  std::vector<std::unique_ptr<transport::UnboundBuffer>> out;

  // Number of elements.
  size_t elements = 0;

  // Number of bytes per element.
  size_t elementSize = 0;

  // Reduction function.
  Func reduce;

  // Tag for this operation.
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

  // End-to-end timeout for this operation.
  std::chrono::milliseconds timeout;

  friend void allreduce(AllreduceOptions&);
};

void allreduce(AllreduceOptions& opts);

} // namespace gloo
