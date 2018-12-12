/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/barrier.h"

namespace gloo {

BarrierOptions::BarrierOptions(const std::shared_ptr<Context>& context)
    : context(context),
      buffer(context->createUnboundBuffer(nullptr, 0)),
      timeout(context->getTimeout()) {}

void barrier(BarrierOptions& opts) {
  const auto& context = opts.context;
  auto& buffer = opts.buffer;
  const auto slot = Slot::build(kBarrierSlotPrefix, opts.tag);

  // Below implements a dissemination barrier, described in "Two algorithms
  // for barrier synchronization (1988)" by Hensgen, Finkel and Manber.
  // PDF: https://www.inf.ed.ac.uk/teaching/courses/ppls/BarrierPaper.pdf
  // DOI: 10.1007/BF01379320

  // Instead of iterating over i up to log2(context->size), we immediately
  // compute 2^i and compare with context->size.
  for (size_t d = 1; d < context->size; d <<= 1) {
    buffer->recv((context->size + context->rank - d) % context->size, slot);
    buffer->send((context->size + context->rank + d) % context->size, slot);
    buffer->waitRecv(opts.timeout);
    buffer->waitSend(opts.timeout);
  }
}

} // namespace gloo
