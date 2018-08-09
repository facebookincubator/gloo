/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <algorithm>
#include <cmath>

#include "gloo/common/common.h"
#include "gloo/common/logging.h"
#include "gloo/hip.h"
#include "gloo/hip_private.h"

namespace gloo {

// Below works both for HipHostPointer and HipDevicePointer
template <typename T, typename Dst>
class HipLocalNativeReduce : public LocalOp<T> {
 public:
  HipLocalNativeReduce(
      std::vector<HipStream>& streams,
      std::vector<HipDevicePointer<T> >& devicePtrs,
      Dst& targetPtr,
      const HipReductionFunction<T>* fn,
      size_t offset,
      size_t count)
      : streams_(streams),
        targetPtr_(targetPtr.range(offset, count)),
        fn_(fn),
        numPtrs_(devicePtrs.size()),
        steps_(log2(numPtrs_)) {
    // Only works with power-of-2 number of pointers
    GLOO_ENFORCE(1 << steps_, streams.size(), "Not power of two");

    // Incorporate offset/count into devicePtrs
    devicePtrs_.reserve(devicePtrs.size());
    for (const auto& ptr : devicePtrs) {
      devicePtrs_.push_back(ptr.range(offset, count));
    }

    // Add level of indirection so that we can shuffle this instead
    // of shuffling BOTH the streams and device ptr vectors.
    for (auto i = 0; i < numPtrs_; i++) {
      indices_.push_back(i);
    }

    // Shuffle order in an attempt to evenly spread work across devices when
    // dealing with multiple instances of this operation.
    std::random_shuffle(indices_.begin(), indices_.end());

    // Initialize
    HipDeviceGuard guard;
    for (auto i = 0; i < steps_; i++) {
      auto sz = 1 << i;
      for (auto j = 0; j < numPtrs_; j += sz * 2) {
        auto indexA = indices_[j];
        auto indexB = indices_[j + sz];
        auto devA = devicePtrs_[indexA].getDeviceID();
        auto devB = devicePtrs_[indexB].getDeviceID();

        // Number of elements must be equal
        GLOO_ENFORCE_EQ(
            devicePtrs_[indexA].getCount(),
            devicePtrs_[indexB].getCount());

        // Devices must be able to access each others memory
        int canAccessPeer = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, devA, devB));
        GLOO_ENFORCE_EQ(
            1,
            canAccessPeer,
            "GPU ",
            devA,
            " does not have peer access to GPU ",
            devB);

        // Enable peer access for devA to memory on devB
        HIP_CHECK(hipSetDevice(devA));
        hipDeviceEnablePeerAccess(devB, 0);

        // Use hipGetLastError so that any error is cleared.
        auto err = hipGetLastError();
        if (err != hipErrorPeerAccessAlreadyEnabled) {
          HIP_CHECK(err);
        }
      }
    }
  }

  virtual void runAsync() {
    HipDeviceGuard guard;
    for (auto i = 0; i < steps_; i++) {
      auto sz = 1 << i;
      for (auto j = 0; j < numPtrs_; j += sz * 2) {
        const auto indexA = indices_[j];
        const auto indexB = indices_[j + sz];
        auto& streamA = streams_[indexA];
        auto& streamB = streams_[indexB];

        // Record event on secondary stream
        HIP_CHECK(hipSetDevice(devicePtrs_[indexB].getDeviceID()));
        HIP_CHECK(hipEventRecord(
                       streamB.getEvent(),
                       streamB.getStream()));

        // Make primary stream wait for secondary stream.
        // This ensures any operations on the source pointer
        // have finished before we start the reduction.
        HIP_CHECK(hipSetDevice(devicePtrs_[indexA].getDeviceID()));
        HIP_CHECK(hipStreamWaitEvent(
                       streamA.getStream(),
                       streamB.getEvent(),
                       0));

        // Queue reduction
        fn_->call(
            devicePtrs_[indexA],
            devicePtrs_[indexB],
            devicePtrs_[indexA].getCount(),
            streamA);
      }
    }

    // Queue copy to target on the root stream
    auto root = indices_[0];
    streams_[root].copyAsync(targetPtr_, devicePtrs_[root]);
  }

  virtual void wait() {
    // Wait for the final memory copy to complete
    auto root = indices_[0];
    streams_[root].wait();
  }

 protected:
  std::vector<HipStream>& streams_;
  std::vector<HipDevicePointer<T> > devicePtrs_;
  Dst targetPtr_;
  const HipReductionFunction<T>* fn_;
  const int numPtrs_;
  const int steps_;
  std::vector<int> indices_;
};

// Below works both for HipHostPointer and HipDevicePointer
template <typename T, typename Src>
class HipLocalNativeBroadcast : public LocalOp<T> {
 public:
  HipLocalNativeBroadcast(
      std::vector<HipStream>& streams,
      std::vector<HipDevicePointer<T> >& devicePtrs,
      Src& sourcePtr,
      size_t offset,
      size_t count)
      : streams_(streams),
        sourcePtr_(sourcePtr.range(offset, count)),
        count_(count),
        numPtrs_(devicePtrs.size()),
        steps_(log2(numPtrs_)) {
    // Only works with power-of-2 number of pointers
    GLOO_ENFORCE(1 << steps_, streams.size(), "Not power of two");

    // Incorporate offset/count into devicePtrs
    devicePtrs_.reserve(devicePtrs.size());
    for (const auto& ptr : devicePtrs) {
      devicePtrs_.push_back(ptr.range(offset, count));
    }

    // Initialize
    HipDeviceGuard guard;
    for (auto i = steps_ - 1; i >= 0; i--) {
      auto sz = 1 << i;
      for (auto j = 0; j < numPtrs_; j += sz * 2) {
        auto indexA = j;
        auto indexB = j + sz;
        auto devA = devicePtrs_[indexA].getDeviceID();
        auto devB = devicePtrs_[indexB].getDeviceID();

        // Number of elements must be equal
        GLOO_ENFORCE_EQ(
            devicePtrs_[indexA].getCount(),
            devicePtrs_[indexB].getCount());

        // Devices must be able to access each others memory
        int canAccessPeer = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, devA, devB));
        GLOO_ENFORCE_EQ(
            1,
            canAccessPeer,
            "GPU ",
            devA,
            " does not have peer access to GPU ",
            devB);

        // Enable peer access for devA to memory on devB
        HIP_CHECK(hipSetDevice(devA));
        hipDeviceEnablePeerAccess(devB, 0);

        // Use hipGetLastError so that any error is cleared.
        auto err = hipGetLastError();
        if (err != hipErrorPeerAccessAlreadyEnabled) {
          HIP_CHECK(err);
        }
      }
    }
  }

  virtual void runAsync() {
    HipDeviceGuard guard;

    // Copy from source ptr to first device ptr
    streams_[0].copyAsync(devicePtrs_[0], sourcePtr_);

    // Tree broadcast
    for (auto i = steps_ - 1; i >= 0; i--) {
      auto sz = 1 << i;
      for (auto j = 0; j < numPtrs_; j += sz * 2) {
        const auto indexA = j;
        const auto indexB = j + sz;
        auto& streamA = streams_[indexA];
        auto& streamB = streams_[indexB];

        // Record event on target stream
        HIP_CHECK(hipSetDevice(
                       devicePtrs_[indexB].getDeviceID()));
        HIP_CHECK(hipEventRecord(
                       streamB.getEvent(),
                       streamB.getStream()));

        // Make source stream wait on target stream.
        // This ensures any operations on the target pointer
        // have finished before we start the copy.
        HIP_CHECK(hipSetDevice(
                       devicePtrs_[indexA].getDeviceID()));
        HIP_CHECK(hipStreamWaitEvent(
                       streamA.getStream(),
                       streamB.getEvent(),
                       0));

        // Execute copy and wait for it to complete on the target
        // stream. This ensures that in the next iteration of this
        // loop the target can be used as source while knowing the
        // previous copy has completed.
        HIP_CHECK(hipMemcpyAsync(
                       *devicePtrs_[indexB],
                       *devicePtrs_[indexA],
                       count_ * sizeof(T),
                       hipMemcpyDeviceToDevice,
                       streamA.getStream()));
        HIP_CHECK(hipEventRecord(
                       streamA.getEvent(),
                       streamA.getStream()));
        HIP_CHECK(hipSetDevice(
                       devicePtrs_[indexB].getDeviceID()));
        HIP_CHECK(hipStreamWaitEvent(
                       streamB.getStream(),
                       streamA.getEvent(),
                       0));

        // Emit event on the target stream so we can wait on all
        // events in the wait() function. Otherwise waiting on
        // this event would NOT indicate completion.
        HIP_CHECK(hipEventRecord(
                       streamB.getEvent(),
                       streamB.getStream()));
      }
    }
  }

  virtual void wait() {
    // Wait for all memory copies on the source streams and receipt
    // confirmation on the target streams to complete.
    for (auto& stream : streams_) {
      stream.wait();
    }
  }

 protected:
  std::vector<HipStream>& streams_;
  std::vector<HipDevicePointer<T> > devicePtrs_;
  Src sourcePtr_;
  const int count_;
  const int numPtrs_;
  const int steps_;
};

} // namespace gloo
