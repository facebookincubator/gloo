/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/common/common.h"
#include "gloo/common/logging.h"
#include "gloo/rccl/rccl.h"

namespace gloo {

template <typename T>
std::vector<rccl::RCCLElement<T> > toDeviceElements(
    std::vector<HipStream>& streams,
    const std::vector<HipDevicePointer<T> >& ptrs,
    size_t offset,
    size_t count) {
  std::vector<rccl::RCCLElement<T> > elements;
  elements.reserve(ptrs.size());
  for (auto i = 0; i < ptrs.size(); i++) {
    elements.push_back(
        rccl::RCCLElement<T>(
            ptrs[i].range(offset, count),
            streams[i],
            ptrs[i].range(offset, count),
            streams[i]));
  }
  return elements;
}

// Forward declaration
template <typename T, typename Dst>
class HipLocalRCCLReduce;

// Partial specialization for device pointer target
template <typename T>
class HipLocalRCCLReduce<T, HipDevicePointer<T> > : public LocalOp<T> {
 public:
  HipLocalRCCLReduce(
      std::vector<HipStream>& streams,
      std::vector<HipDevicePointer<T> >& devicePtrs,
      HipDevicePointer<T>& targetPtr,
      const HipReductionFunction<T>* fn,
      size_t offset,
      size_t count) {
    // The targetPtr must be one of devicePtrs.
    auto root = -1;
    for (auto i = 0; i < devicePtrs.size(); i++) {
      if (devicePtrs[i] == targetPtr) {
        root = i;
        break;
      }
    }
    GLOO_ENFORCE_GE(root, 0, "targetPtr must be one of devicePtrs");

    // Only if we have multiple device pointers does this
    // operation need to execute.
    if (devicePtrs.size() > 1) {
      reduceOp_ = make_unique<rccl::ReduceOp<T> >(
          toDeviceElements(streams, devicePtrs, offset, count),
          fn,
          root);
    }
  }

  virtual ~HipLocalRCCLReduce() {}

  virtual void runAsync() {
    if (reduceOp_) {
      reduceOp_->runAsync();
    }
  }

  virtual void wait() {
    if (reduceOp_) {
      reduceOp_->wait();
    }
  }

 protected:
  std::unique_ptr<rccl::ReduceOp<T> > reduceOp_;
};

// Partial specialization for host pointer target
template <typename T>
class HipLocalRCCLReduce<T, HipHostPointer<T> > : public LocalOp<T> {
 public:
  HipLocalRCCLReduce(
      std::vector<HipStream>& streams,
      std::vector<HipDevicePointer<T> >& devicePtrs,
      HipHostPointer<T>& targetPtr,
      const HipReductionFunction<T>* fn,
      size_t offset,
      size_t count)
      : root_(0),
        stream_(streams[root_]),
        devicePtr_(devicePtrs[root_].range(offset, count)),
        hostPtr_(targetPtr.range(offset, count)) {
    if (devicePtrs.size() > 1) {
      reduceOp_ = make_unique<rccl::ReduceOp<T> >(
          toDeviceElements(streams, devicePtrs, offset, count),
          fn,
          root_);
    }
  }

  virtual ~HipLocalRCCLReduce() {}

  virtual void runAsync() {
    if (reduceOp_) {
      reduceOp_->runAsync();
    }

    // The stream for operations on devicePtrs_[0] now includes an
    // asynchronous wait for completion of the reduce operation, if it
    // was executed. This means we can sequence an asynchronous memory
    // copy and wait on completion of that to signal completion of
    // both operations.
    stream_.copyAsync(hostPtr_, devicePtr_);
  }

  virtual void wait() {
    stream_.wait();
  }

 protected:
  const int root_;
  HipStream& stream_;
  HipDevicePointer<T> devicePtr_;
  HipHostPointer<T> hostPtr_;
  std::unique_ptr<rccl::ReduceOp<T> > reduceOp_;
};

// Forward declaration
template <typename T, typename Src>
class HipLocalRCCLBroadcast;

// Specialization for device pointer source
template <typename T>
class HipLocalRCCLBroadcast<T, HipDevicePointer<T> > : public LocalOp<T> {
 public:
  HipLocalRCCLBroadcast(
      std::vector<HipStream>& streams,
      std::vector<HipDevicePointer<T> >& devicePtrs,
      HipDevicePointer<T>& sourcePtr,
      size_t offset,
      size_t count) {
    // The sourcePtr must be one of devicePtrs.
    auto root = -1;
    for (auto i = 0; i < devicePtrs.size(); i++) {
      if (devicePtrs[i] == sourcePtr) {
        root = i;
        break;
      }
    }
    GLOO_ENFORCE_GE(root, 0, "sourcePtr must be one of devicePtrs");

    // Only if we have multiple device pointers does this
    // operation need to execute.
    if (devicePtrs.size() > 1) {
      broadcastOp_ = make_unique<rccl::BroadcastOp<T> >(
          toDeviceElements(streams, devicePtrs, offset, count),
          root);
    }
  }

  virtual ~HipLocalRCCLBroadcast() {}

  virtual void runAsync() {
    if (broadcastOp_) {
      broadcastOp_->runAsync();
    }
  }

  virtual void wait() {
    if (broadcastOp_) {
      broadcastOp_->wait();
    }
  }

 protected:
  std::unique_ptr<rccl::BroadcastOp<T> > broadcastOp_;
};

// Specialization for host pointer source
template <typename T>
class HipLocalRCCLBroadcast<T, HipHostPointer<T> > : public LocalOp<T> {
 public:
  HipLocalRCCLBroadcast(
      std::vector<HipStream>& streams,
      std::vector<HipDevicePointer<T> >& devicePtrs,
      HipHostPointer<T>& sourcePtr,
      size_t offset,
      size_t count)
      : root_(0),
        stream_(streams[root_]),
        devicePtr_(devicePtrs[root_].range(offset, count)),
        sourcePtr_(sourcePtr.range(offset, count)) {
    if (devicePtrs.size() > 1) {
      broadcastOp_ = make_unique<rccl::BroadcastOp<T> >(
          toDeviceElements(streams, devicePtrs, offset, count),
          root_);
    }
  }

  virtual ~HipLocalRCCLBroadcast() {}

  virtual void runAsync() {
    // Since we run an asynchronous memcpy to devicePtr_ which is
    // executed on the stream associated with that device pointer, the
    // broadcast operation will only start after the memcpy completes.
    stream_.copyAsync(devicePtr_, sourcePtr_);
    if (broadcastOp_) {
      broadcastOp_->runAsync();
    }
  }

  virtual void wait() {
    stream_.wait();
    if (broadcastOp_) {
      broadcastOp_->wait();
    }
  }

 protected:
  const int root_;
  HipStream& stream_;
  HipDevicePointer<T> devicePtr_;
  HipHostPointer<T> sourcePtr_;
  std::unique_ptr<rccl::BroadcastOp<T> > broadcastOp_;
};

template <typename T, typename Dst>
std::unique_ptr<LocalOp<T> > hipRCCLReduce(
    std::vector<HipStream>& streams,
    std::vector<HipDevicePointer<T> >& devicePtrs,
    Dst& targetPtr,
    const HipReductionFunction<T>* fn,
    size_t offset,
    size_t count) {
  GLOO_ENFORCE_EQ(streams.size(), devicePtrs.size());
  return make_unique<HipLocalRCCLReduce<T, Dst> >(
      streams, devicePtrs, targetPtr, fn, offset, count);
}

template <typename T, typename Src>
std::unique_ptr<LocalOp<T> > hipRCCLBroadcast(
    std::vector<HipStream>& streams,
    std::vector<HipDevicePointer<T> >& devicePtrs,
    Src& sourcePtr,
    size_t offset,
    size_t count) {
  GLOO_ENFORCE_EQ(streams.size(), devicePtrs.size());
  return make_unique<HipLocalRCCLBroadcast<T, Src> >(
      streams, devicePtrs, sourcePtr, offset, count);
}

} // namespace gloo
