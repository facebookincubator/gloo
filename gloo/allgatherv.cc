/**
 * Copyright (c) 2019-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/allgatherv.h"

#include <cstring>
#include <numeric>

#include "gloo/common/logging.h"
#include "gloo/types.h"

namespace gloo {

void AllgathervOptions::setElementSize(size_t elementSize) {
  if (this->elementSize == 0) {
    this->elementSize = elementSize;
  } else {
    GLOO_ENFORCE_EQ(
        elementSize,
        this->elementSize,
        "Element size does not match existing value. ",
        "Please double check that the input and output types match.");
  }
}

void AllgathervOptions::setInput(
    std::unique_ptr<transport::UnboundBuffer> buf,
    size_t elementSize) {
  setElementSize(elementSize);
  this->in = std::move(buf);
}

void AllgathervOptions::setInput(
    void* ptr,
    size_t elements,
    size_t elementSize) {
  setElementSize(elementSize);
  this->in = context->createUnboundBuffer(ptr, elements * elementSize);
}

void AllgathervOptions::setOutput(
    std::unique_ptr<transport::UnboundBuffer> buf,
    std::vector<size_t> elements,
    size_t elementSize) {
  const auto totalElements =
      std::accumulate(elements.begin(), elements.end(), size_t(0));
  setElementSize(elementSize);
  GLOO_ENFORCE_EQ(elements.size(), context->size);
  this->elements = std::move(elements);
  GLOO_ENFORCE_EQ(totalElements * elementSize, buf->size);
  this->out = std::move(buf);
}

void AllgathervOptions::setOutput(
    void* ptr,
    std::vector<size_t> elements,
    size_t elementSize) {
  const auto totalElements =
      std::accumulate(elements.begin(), elements.end(), size_t(0));
  setElementSize(elementSize);
  GLOO_ENFORCE_EQ(elements.size(), context->size);
  this->elements = std::move(elements);
  this->out = context->createUnboundBuffer(ptr, totalElements * elementSize);
}

void allgatherv(AllgathervOptions& opts) {
  const auto& context = opts.context;
  transport::UnboundBuffer* in = opts.in.get();
  transport::UnboundBuffer* out = opts.out.get();
  const auto slot = Slot::build(kAllgatherSlotPrefix, opts.tag);

  // Sanity checks
  GLOO_ENFORCE(opts.elementSize > 0);
  const auto recvRank = (context->size + context->rank - 1) % context->size;
  GLOO_ENFORCE(
      context->getPair(recvRank),
      "missing connection between rank " + std::to_string(context->rank) +
          " (this process) and rank " + std::to_string(recvRank));
  const auto sendRank = (context->size + context->rank + 1) % context->size;
  GLOO_ENFORCE(
      context->getPair(sendRank),
      "missing connection between rank " + std::to_string(context->rank) +
          " (this process) and rank " + std::to_string(sendRank));

  // Compute byte counts and offsets into output buffer.
  std::vector<size_t> byteCounts;
  std::vector<size_t> byteOffsets;
  byteCounts.reserve(context->size);
  byteOffsets.reserve(context->size);
  size_t offset = 0;
  for (const auto& elements : opts.elements) {
    const auto bytes = elements * opts.elementSize;
    byteCounts.push_back(bytes);
    byteOffsets.push_back(offset);
    offset += bytes;
  }

  // If the input buffer is specified, the output buffer needs to be primed.
  if (in != nullptr) {
    GLOO_ENFORCE_EQ(byteCounts[context->rank], in->size);
    memcpy(
        static_cast<uint8_t*>(out->ptr) + byteOffsets[context->rank],
        static_cast<uint8_t*>(in->ptr),
        in->size);
  }

  const auto baseIndex = context->size + context->rank;
  for (auto i = 0; i < context->size - 1; i++) {
    const size_t sendIndex = (baseIndex - i) % context->size;
    const size_t recvIndex = (baseIndex - i - 1) % context->size;

    if (i == 0) {
      out->send(sendRank, slot, byteOffsets[sendIndex], byteCounts[sendIndex]);
      out->recv(recvRank, slot, byteOffsets[recvIndex], byteCounts[recvIndex]);
      continue;
    }

    // Wait for previous operations to complete before kicking off new ones.
    out->waitSend(opts.timeout);
    out->waitRecv(opts.timeout);
    out->send(sendRank, slot, byteOffsets[sendIndex], byteCounts[sendIndex]);
    out->recv(recvRank, slot, byteOffsets[recvIndex], byteCounts[recvIndex]);
  }

  // Wait for final operations to complete.
  out->waitSend(opts.timeout);
  out->waitRecv(opts.timeout);
}

} // namespace gloo
