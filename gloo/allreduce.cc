/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/allreduce.h"

#include <algorithm>
#include <cstring>

#include "gloo/common/logging.h"
#include "gloo/math.h"
#include "gloo/types.h"

namespace gloo {

namespace {

using BufferVector = std::vector<std::unique_ptr<transport::UnboundBuffer>>;
using ReductionFunction = AllreduceOptions::Func;
using ReduceRangeFunction = std::function<void(size_t, size_t)>;
using BroadcastRangeFunction = std::function<void(size_t, size_t)>;

// Returns function that computes local reduction over inputs and
// stores it in the output for a given range in those buffers.
// This is done prior to either sending a region to a neighbor, or
// reducing a region received from a neighbor.
ReduceRangeFunction genLocalReduceFunction(
    BufferVector& in,
    BufferVector& out,
    size_t elementSize,
    ReductionFunction fn) {
  if (in.size() > 0) {
    if (in.size() == 1) {
      return [&in, &out](size_t offset, size_t length) {
        memcpy(
            static_cast<uint8_t*>(out[0]->ptr) + offset,
            static_cast<const uint8_t*>(in[0]->ptr) + offset,
            length);
      };
    } else {
      return [&in, &out, elementSize, fn](size_t offset, size_t length) {
        fn(static_cast<uint8_t*>(out[0]->ptr) + offset,
           static_cast<const uint8_t*>(in[0]->ptr) + offset,
           static_cast<const uint8_t*>(in[1]->ptr) + offset,
           length / elementSize);
        for (size_t i = 2; i < in.size(); i++) {
          fn(static_cast<uint8_t*>(out[0]->ptr) + offset,
             static_cast<const uint8_t*>(out[0]->ptr) + offset,
             static_cast<const uint8_t*>(in[i]->ptr) + offset,
             length / elementSize);
        }
      };
    }
  } else {
    return [&out, elementSize, fn](size_t offset, size_t length) {
      for (size_t i = 1; i < out.size(); i++) {
        fn(static_cast<uint8_t*>(out[0]->ptr) + offset,
           static_cast<const uint8_t*>(out[0]->ptr) + offset,
           static_cast<const uint8_t*>(out[i]->ptr) + offset,
           length / elementSize);
      }
    };
  }
}

// Returns function that performs a local broadcast over outputs for a
// given range in the buffers. This is executed after receiving every
// globally reduced chunk.
BroadcastRangeFunction genLocalBroadcastFunction(BufferVector& out) {
  return [&out](size_t offset, size_t length) {
    for (size_t i = 1; i < out.size(); i++) {
      memcpy(
          static_cast<uint8_t*>(out[i]->ptr) + offset,
          static_cast<const uint8_t*>(out[0]->ptr) + offset,
          length);
    }
  };
}

} // namespace

void allreduce(AllreduceOptions& opts) {
  const auto& context = opts.context;
  std::vector<std::unique_ptr<transport::UnboundBuffer>>& in = opts.in;
  std::vector<std::unique_ptr<transport::UnboundBuffer>>& out = opts.out;
  const auto slot = Slot::build(kAllreduceSlotPrefix, opts.tag);

  // Sanity checks
  GLOO_ENFORCE_GT(out.size(), 0);
  GLOO_ENFORCE(opts.elements > 0);
  GLOO_ENFORCE(opts.elementSize > 0);
  GLOO_ENFORCE(opts.reduce != nullptr);

  // Assert the size of all inputs and outputs is identical.
  const size_t totalBytes = opts.elements * opts.elementSize;
  for (size_t i = 0; i < out.size(); i++) {
    GLOO_ENFORCE_EQ(out[i]->size, totalBytes);
  }
  for (size_t i = 0; i < in.size(); i++) {
    GLOO_ENFORCE_EQ(in[i]->size, totalBytes);
  }

  // Initialize local reduction and broadcast functions.
  // Note that these are a no-op if only a single output is specified
  // and is used as both input and output.
  const auto reduceInputs =
      genLocalReduceFunction(in, out, opts.elementSize, opts.reduce);
  const auto broadcastOutputs = genLocalBroadcastFunction(out);

  // Simple circuit if there is only a single process.
  if (context->size == 1) {
    reduceInputs(0, totalBytes);
    broadcastOutputs(0, totalBytes);
    return;
  }

  // Note: context->size > 1
  const auto recvRank = (context->size + context->rank + 1) % context->size;
  const auto sendRank = (context->size + context->rank - 1) % context->size;
  GLOO_ENFORCE(
      context->getPair(recvRank),
      "missing connection between rank " + std::to_string(context->rank) +
      " (this process) and rank " + std::to_string(recvRank));
  GLOO_ENFORCE(
      context->getPair(sendRank),
      "missing connection between rank " + std::to_string(context->rank) +
      " (this process) and rank " + std::to_string(sendRank));

  // The ring algorithm works as follows.
  //
  // The given input is split into a number of chunks equal to the
  // number of processes. Once the algorithm has finished, every
  // process hosts one chunk of reduced output, in sequential order
  // (rank 0 has chunk 0, rank 1 has chunk 1, etc.). As the input may
  // not be divisible by the number of processes, the chunk on the
  // final ranks may have partial output or may be empty.
  //
  // As a chunk is passed along the ring and contains the reduction of
  // successively more ranks, we have to alternate between performing
  // I/O for that chunk and computing the reduction between the
  // received chunk and the local chunk. To avoid this alternating
  // pattern, we split up a chunk into multiple segments (>= 2), and
  // ensure we have one segment in flight while computing a reduction
  // on the other. The segment size has an upper bound to minimize
  // memory usage and avoid poor cache behavior. This means we may
  // have many segments per chunk when dealing with very large inputs.
  //
  // The nomenclature here is reflected in the variable naming below
  // (one chunk per rank and many segments per chunk).
  //

  // Ensure that maximum segment size is a multiple of the element size.
  // Otherwise, the segment size can exceed the maximum segment size after
  // rounding it up to the nearest multiple of the element size.
  // For example, if maxSegmentSize = 10, and elementSize = 4,
  // then after rounding up: segmentSize = 12;
  const size_t maxSegmentSize =
      opts.elementSize * (opts.maxSegmentSize / opts.elementSize);

  // The number of bytes per segment must be a multiple of the bytes
  // per element for the reduction to work; round up if necessary.
  const size_t segmentBytes = roundUp(
      std::min(
          // Rounded division to have >= 2 segments per chunk.
          (totalBytes + (context->size * 2 - 1)) / (context->size * 2),
          // Configurable segment size limit
          maxSegmentSize),
      opts.elementSize);

  // Compute how many segments make up the input buffer.
  //
  // Round up to the nearest multiple of the context size such that
  // there is an equal number of segments per process and execution is
  // symmetric across processes.
  //
  // The minimum is twice the context size, because the algorithm
  // below overlaps sending/receiving a segment with computing the
  // reduction of the another segment.
  //
  const size_t numSegments = roundUp(
      std::max(
          (totalBytes + (segmentBytes - 1)) / segmentBytes,
          (size_t)context->size * 2),
      (size_t)context->size);
  GLOO_ENFORCE_EQ(numSegments % context->size, 0);
  GLOO_ENFORCE_GE(numSegments, context->size * 2);
  const size_t numSegmentsPerRank = numSegments / context->size;

  // Allocate scratch space to hold two chunks
  std::unique_ptr<uint8_t[]> tmpAllocation(new uint8_t[segmentBytes * 2]);
  std::unique_ptr<transport::UnboundBuffer> tmpBuffer =
      context->createUnboundBuffer(tmpAllocation.get(), segmentBytes * 2);
  transport::UnboundBuffer* tmp = tmpBuffer.get();

  // Use dynamic lookup for chunk offset in the temporary buffer.
  // With two operations in flight we need two offsets.
  // They can be indexed using the loop counter.
  std::array<size_t, 2> segmentOffset;
  segmentOffset[0] = 0;
  segmentOffset[1] = segmentBytes;

  // Function computes the offsets and lengths of the segments to be
  // sent and received for a given iteration during reduce/scatter.
  auto computeReduceScatterOffsets = [&](size_t i) {
    struct {
      size_t sendOffset;
      size_t recvOffset;
      ssize_t sendLength;
      ssize_t recvLength;
    } result;

    // Compute segment index to send from (to rank - 1) and segment
    // index to receive into (from rank + 1). Multiply by the number
    // of bytes in a chunk to get to an offset. The offset is allowed
    // to be out of range (>= totalBytes) and this is taken into
    // account when computing the associated length.
    result.sendOffset =
        ((((context->rank + 1) * numSegmentsPerRank) + i) * segmentBytes) %
        (numSegments * segmentBytes);
    result.recvOffset =
        ((((context->rank + 2) * numSegmentsPerRank) + i) * segmentBytes) %
        (numSegments * segmentBytes);

    // If the segment is entirely in range, the following statement is
    // equal to segmentBytes. If it isn't, it will be less, or even
    // negative. This is why the ssize_t typecasts are needed.
    result.sendLength = std::min(
        (ssize_t)segmentBytes,
        (ssize_t)totalBytes - (ssize_t)result.sendOffset);
    result.recvLength = std::min(
        (ssize_t)segmentBytes,
        (ssize_t)totalBytes - (ssize_t)result.recvOffset);

    return result;
  };

  for (auto i = 0; i < numSegments; i++) {
    if (i >= 2) {
      // Compute send and receive offsets and lengths two iterations
      // ago. Needed so we know when to wait for an operation and when
      // to ignore (when the offset was out of bounds), and know where
      // to reduce the contents of the temporary buffer.
      auto prev = computeReduceScatterOffsets(i - 2);
      if (prev.recvLength > 0) {
        // Prepare out[0]->ptr to hold the local reduction
        reduceInputs(prev.recvOffset, prev.recvLength);
        // Wait for segment from neighbor.
        tmp->waitRecv(opts.timeout);
        // Reduce segment from neighbor into out->ptr.
        opts.reduce(
            static_cast<uint8_t*>(out[0]->ptr) + prev.recvOffset,
            static_cast<const uint8_t*>(out[0]->ptr) + prev.recvOffset,
            static_cast<const uint8_t*>(tmp->ptr) + segmentOffset[i & 0x1],
            prev.recvLength / opts.elementSize);
      }
      if (prev.sendLength > 0) {
        out[0]->waitSend(opts.timeout);
      }
    }

    // Issue new send and receive operation in all but the final two
    // iterations. At that point we have already sent all data we
    // needed to and only have to wait for the final segments to be
    // reduced into the output.
    if (i < (numSegments - 2)) {
      // Compute send and receive offsets and lengths for this iteration.
      auto cur = computeReduceScatterOffsets(i);
      if (cur.recvLength > 0) {
        tmp->recv(recvRank, slot, segmentOffset[i & 0x1], cur.recvLength);
      }
      if (cur.sendLength > 0) {
        // Prepare out[0]->ptr to hold the local reduction for this segment
        if (i < numSegmentsPerRank) {
          reduceInputs(cur.sendOffset, cur.sendLength);
        }
        out[0]->send(sendRank, slot, cur.sendOffset, cur.sendLength);
      }
    }
  }

  // Function computes the offsets and lengths of the segments to be
  // sent and received for a given iteration during allgather.
  auto computeAllgatherOffsets = [&](size_t i) {
    struct {
      size_t sendOffset;
      size_t recvOffset;
      ssize_t sendLength;
      ssize_t recvLength;
    } result;

    result.sendOffset =
        ((((context->rank) * numSegmentsPerRank) + i) * segmentBytes) %
        (numSegments * segmentBytes);
    result.recvOffset =
        ((((context->rank + 1) * numSegmentsPerRank) + i) * segmentBytes) %
        (numSegments * segmentBytes);

    // If the segment is entirely in range, the following statement is
    // equal to segmentBytes. If it isn't, it will be less, or even
    // negative. This is why the ssize_t typecasts are needed.
    result.sendLength = std::min(
        (ssize_t)segmentBytes,
        (ssize_t)totalBytes - (ssize_t)result.sendOffset);
    result.recvLength = std::min(
        (ssize_t)segmentBytes,
        (ssize_t)totalBytes - (ssize_t)result.recvOffset);

    return result;
  };

  // Ring allgather.
  //
  // Beware: totalBytes <= (numSegments * segmentBytes), which is
  // incompatible with the generic allgather algorithm where the
  // contribution is identical across processes.
  //
  for (auto i = 0; i < numSegments; i++) {
    if (i >= 2) {
      auto prev = computeAllgatherOffsets(i - 2);
      if (prev.recvLength > 0) {
        out[0]->waitRecv(opts.timeout);
        // Broadcast received segments to output buffers.
        broadcastOutputs(prev.recvOffset, prev.recvLength);
      }
      if (prev.sendLength > 0) {
        out[0]->waitSend(opts.timeout);
      }
    }

    // Issue new send and receive operation in all but the final two
    // iterations. At that point we have already sent all data we
    // needed to and only have to wait for the final segments to be
    // sent to the output.
    if (i < (numSegments - 2)) {
      auto cur = computeAllgatherOffsets(i);
      if (cur.recvLength > 0) {
        out[0]->recv(recvRank, slot, cur.recvOffset, cur.recvLength);
      }
      if (cur.sendLength > 0) {
        out[0]->send(sendRank, slot, cur.sendOffset, cur.sendLength);
        // Broadcast first segments to outputs buffers.
        if (i < numSegmentsPerRank) {
          broadcastOutputs(cur.sendOffset, cur.sendLength);
        }
      }
    }
  }
}

} // namespace gloo
