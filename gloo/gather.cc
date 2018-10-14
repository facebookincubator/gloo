/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/gather.h"

#include <cstring>

#include "gloo/common/logging.h"
#include "gloo/types.h"

namespace gloo {

void gather(const std::shared_ptr<Context>& context, GatherOptions& opts) {
  std::unique_ptr<transport::UnboundBuffer> tmpInBuffer;
  std::unique_ptr<transport::UnboundBuffer> tmpOutBuffer;
  transport::UnboundBuffer* in = nullptr;
  transport::UnboundBuffer* out = nullptr;
  const auto slot = Slot::build(kGatherSlotPrefix, opts.tag);

  // Sanity checks
  GLOO_ENFORCE(opts.elementSize > 0);

  // Figure out pointer to input buffer
  if (opts.inBuffer) {
    in = opts.inBuffer.get();
  } else {
    GLOO_ENFORCE(opts.inPtr != nullptr);
    GLOO_ENFORCE(opts.inElements > 0);
    tmpInBuffer =
      context->createUnboundBuffer(opts.inPtr, opts.inElements * opts.elementSize);
    in = tmpInBuffer.get();
  }

  if (context->rank == opts.root) {
    const size_t chunkSize = in->size;

    // Figure out pointer to output buffer (only for root rank)
    if (opts.outBuffer) {
      out = opts.outBuffer.get();
    } else {
      GLOO_ENFORCE(opts.outPtr != nullptr);
      GLOO_ENFORCE(opts.outElements > 0);
      tmpOutBuffer =
        context->createUnboundBuffer(opts.outPtr, opts.outElements * opts.elementSize);
      out = tmpOutBuffer.get();
    }

    // Ensure the output buffer has the right size.
    GLOO_ENFORCE(in->size * context->size == out->size);

    // Post receive operations from peers into out buffer
    for (size_t i = 0; i < context->size; i++) {
      if (i == context->rank) {
        continue;
      }
      out->recv(i, slot, i * chunkSize, chunkSize);
    }

    // Copy local input to output
    memcpy((char*) out->ptr + (context->rank * chunkSize), in->ptr, chunkSize);

    // Wait for receive operations to complete
    for (size_t i = 0; i < context->size; i++) {
      if (i == context->rank) {
        continue;
      }
      out->waitRecv();
    }
  } else {
    in->send(opts.root, slot);
    in->waitSend();
  }
}

} // namespace gloo
