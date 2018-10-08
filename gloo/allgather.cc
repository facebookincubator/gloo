/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/allgather.h"

#include <cstring>

#include "gloo/common/logging.h"
#include "gloo/types.h"

namespace gloo {

void allgather(const std::shared_ptr<Context>& context, AllgatherOptions& opts) {
  std::unique_ptr<transport::UnboundBuffer> tmpInBuffer;
  std::unique_ptr<transport::UnboundBuffer> tmpOutBuffer;
  transport::UnboundBuffer* in = nullptr;
  transport::UnboundBuffer* out = nullptr;
  const auto slot = Slot::build(kAllgatherSlotPrefix, opts.tag);

  // Sanity checks
  GLOO_ENFORCE(opts.elementSize > 0);
  const auto recvRank = (context->size + context->rank - 1) % context->size;
  GLOO_ENFORCE(context->getPair(recvRank), "pair missing (rank ", recvRank, ")");
  const auto sendRank = (context->size + context->rank + 1) % context->size;
  GLOO_ENFORCE(context->getPair(sendRank), "pair missing (rank ", sendRank, ")");

  // Figure out pointer to input buffer
  if (opts.inBuffer) {
    in = opts.inBuffer.get();
  } else if (opts.inPtr != nullptr) {
    GLOO_ENFORCE(opts.inElements > 0);
    tmpInBuffer =
      context->createUnboundBuffer(opts.inPtr, opts.inElements * opts.elementSize);
    in = tmpInBuffer.get();
  }

  // Figure out pointer to output buffer
  if (opts.outBuffer) {
    out = opts.outBuffer.get();
  } else {
    GLOO_ENFORCE(opts.outPtr != nullptr);
    GLOO_ENFORCE(opts.outElements > 0);
    tmpOutBuffer =
      context->createUnboundBuffer(opts.outPtr, opts.outElements * opts.elementSize);
    out = tmpOutBuffer.get();
  }

  GLOO_ENFORCE_EQ(out->size, in->size * context->size);

  // If the input buffer is specified, this is NOT an in place operation,
  // and the output buffer needs to be primed with the input.
  if (in != nullptr) {
    memcpy(
        (uint8_t*) out->ptr + context->rank * opts.inElements * opts.elementSize,
        (uint8_t*) in->ptr,
        opts.inElements * opts.elementSize);
  }

  // The chunk size may not be divisible by 2; use dynamic lookup.
  std::array<size_t, 2> chunkSize;
  chunkSize[0] = (opts.inElements * opts.elementSize) / 2;
  chunkSize[1] = (opts.inElements * opts.elementSize) - chunkSize[0];
  std::array<size_t, 2> chunkOffset;
  chunkOffset[0] = 0;
  chunkOffset[1] = chunkSize[0];

  for (auto i = 0; i < (context->size - 1) * 2; i++) {
    size_t sendOffset =
      (((context->size + context->rank - (i / 2))
        * opts.inElements
        * opts.elementSize)
       + chunkOffset[i & 0x1])
      % (opts.outElements * opts.elementSize);
    size_t recvOffset =
      (((context->size + context->rank - 1 - (i / 2))
        * opts.inElements
        * opts.elementSize)
       + chunkOffset[i & 0x1])
      % (opts.outElements * opts.elementSize);
    size_t size = chunkSize[i & 0x1];
    if (i < 2) {
      out->send(sendRank, slot, sendOffset, size);
      out->recv(recvRank, slot, recvOffset, size);
      continue;
    }

    // Wait for pending operations to complete to synchronize with the
    // previous iteration. Because we kick off two operations before
    // getting here we always wait for the next-to-last operation.
    out->waitSend();
    out->waitRecv();
    out->send(sendRank, slot, sendOffset, size);
    out->recv(recvRank, slot, recvOffset, size);
  }

  // Wait for completes
  for (auto i = 0; i < 2; i++) {
    out->waitSend();
    out->waitRecv();
  }
}

} // namespace gloo
