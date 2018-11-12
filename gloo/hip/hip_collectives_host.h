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
#include "gloo/hip.h"

namespace gloo {

// Forward declaration
template <typename T, typename Dst>
class HipLocalHostReduce;

// Partial specialization for device pointer target
template <typename T>
class HipLocalHostReduce<T, HipDevicePointer<T> > : public LocalOp<T> {
 public:
  HipLocalHostReduce(
      std::vector<HipStream>& streams,
      std::vector<HipDevicePointer<T> >& devicePtrs,
      HipDevicePointer<T>& targetPtr,
      const HipReductionFunction<T>* fn,
      size_t offset,
      size_t count)
      : streams_(streams),
        targetPtr_(targetPtr.range(offset, count)),
        offset_(offset),
        count_(count),
        fn_(fn) {
    // Incorporate offset/count into devicePtrs
    devicePtrs_.reserve(devicePtrs.size());
    for (const auto& ptr : devicePtrs) {
      devicePtrs_.push_back(ptr.range(offset, count));
    }
    // Allocate N temporary buffers to async copy device ptrs into.
    for (auto i = 0; i < devicePtrs_.size(); i++) {
      tmpPtrs_.push_back(HipHostPointer<T>::alloc(count));
    }
  }

  virtual void runAsync() {
    // Asynchronously copy device memory to host
    for (auto i = 0; i < devicePtrs_.size(); i++) {
      streams_[i].copyAsync(tmpPtrs_[i], devicePtrs_[i]);
    }
    // Reduce specified pointers into tmpPtrs_[0]
    streams_[0].wait();
    for (auto i = 1; i < devicePtrs_.size(); i++) {
      streams_[i].wait();
      fn_->call(tmpPtrs_[0], tmpPtrs_[i], count_, streams_[i]);
    }
    // Copy final reduction back to device
    streams_[0].copyAsync(targetPtr_, tmpPtrs_[0]);
  }

  virtual void wait() {
    // Reduction happens on CPU but we still have to wait for the
    // memory copy of the result back to device.
    streams_[0].wait();
  }

 protected:
  std::vector<HipStream>& streams_;
  std::vector<HipDevicePointer<T> > devicePtrs_;
  HipDevicePointer<T> targetPtr_;
  const size_t offset_;
  const size_t count_;
  const HipReductionFunction<T>* fn_;

  // Temporary buffers used for async memory copies
  std::vector<HipHostPointer<T> > tmpPtrs_;
};

// Partial specialization for host pointer target
template <typename T>
class HipLocalHostReduce<T, HipHostPointer<T> > : public LocalOp<T> {
 public:
  HipLocalHostReduce(
      std::vector<HipStream>& streams,
      std::vector<HipDevicePointer<T> >& devicePtrs,
      HipHostPointer<T>& targetPtr,
      const HipReductionFunction<T>* fn,
      size_t offset,
      size_t count)
      : streams_(streams),
        targetPtr_(targetPtr.range(offset, count)),
        offset_(offset),
        count_(count),
        fn_(fn) {
    // Incorporate offset/count into devicePtrs
    devicePtrs_.reserve(devicePtrs.size());
    for (const auto& ptr : devicePtrs) {
      devicePtrs_.push_back(ptr.range(offset, count));
    }
    // Allocate N-1 temporary buffers to async copy device ptrs into.
    for (auto i = 1; i < devicePtrs_.size(); i++) {
      tmpPtrs_.push_back(HipHostPointer<T>::alloc(count));
    }
  }

  virtual void runAsync() {
    // Asynchronously copy device memory to host
    streams_[0].copyAsync(targetPtr_, devicePtrs_[0]);
    for (auto i = 1; i < devicePtrs_.size(); i++) {
      streams_[i].copyAsync(tmpPtrs_[i-1], devicePtrs_[i]);
    }
    // Reduce specified pointers into targetPtr_
    streams_[0].wait();
    for (auto i = 1; i < devicePtrs_.size(); i++) {
      streams_[i].wait();
      fn_->call(targetPtr_, tmpPtrs_[i-1], count_, streams_[i]);
    }
  }

  virtual void wait() {
    // Because reduction happens on CPU, this op is synchronous.
  }

 protected:
  std::vector<HipStream>& streams_;
  std::vector<HipDevicePointer<T> > devicePtrs_;
  HipHostPointer<T> targetPtr_;
  const size_t offset_;
  const size_t count_;
  const HipReductionFunction<T>* fn_;

  // Temporary buffers used for async memory copies
  std::vector<HipHostPointer<T> > tmpPtrs_;
};

// Forward declaration
template <typename T, typename Src>
class HipLocalHostBroadcast;

// Specialization for device pointer source
template <typename T>
class HipLocalHostBroadcast<T, HipDevicePointer<T> > : public LocalOp<T> {
 public:
  HipLocalHostBroadcast(
      std::vector<HipStream>& streams,
      std::vector<HipDevicePointer<T> >& devicePtrs,
      HipDevicePointer<T>& sourcePtr,
      size_t offset,
      size_t count)
      : streams_(streams),
        sourcePtr_(sourcePtr.range(offset, count)) {
    // Incorporate offset/count into devicePtrs
    devicePtrs_.reserve(devicePtrs.size());
    for (const auto& ptr : devicePtrs) {
      devicePtrs_.push_back(ptr.range(offset, count));
    }
  }

  virtual void runAsync() {
    // Asynchronously copy source to device ptrs
    for (auto i = 0; i < devicePtrs_.size(); i++) {
      streams_[i].copyAsync(devicePtrs_[i], sourcePtr_);
    }
  }

  virtual void wait() {
    for (auto i = 0; i < devicePtrs_.size(); i++) {
      streams_[i].wait();
    }
  }

 protected:
  std::vector<HipStream>& streams_;
  std::vector<HipDevicePointer<T> > devicePtrs_;
  HipDevicePointer<T> sourcePtr_;
};

// Specialization for host pointer source
template <typename T>
class HipLocalHostBroadcast<T, HipHostPointer<T> > : public LocalOp<T> {
 public:
  HipLocalHostBroadcast(
      std::vector<HipStream>& streams,
      std::vector<HipDevicePointer<T> >& devicePtrs,
      HipHostPointer<T>& sourcePtr,
      size_t offset,
      size_t count)
      : streams_(streams),
        sourcePtr_(sourcePtr.range(offset, count)) {
    // Incorporate offset/count into devicePtrs
    devicePtrs_.reserve(devicePtrs.size());
    for (const auto& ptr : devicePtrs) {
      devicePtrs_.push_back(ptr.range(offset, count));
    }
  }

  virtual void runAsync() {
    // Asynchronously copy host memory to device
    for (auto i = 0; i < devicePtrs_.size(); i++) {
      streams_[i].copyAsync(devicePtrs_[i], sourcePtr_);
    }
  }

  virtual void wait() {
    for (auto i = 0; i < devicePtrs_.size(); i++) {
      streams_[i].wait();
    }
  }

 protected:
  std::vector<HipStream>& streams_;
  std::vector<HipDevicePointer<T> > devicePtrs_;
  HipHostPointer<T> sourcePtr_;
};

template <typename T, typename Dst>
std::unique_ptr<LocalOp<T> > hipHostReduce(
    std::vector<HipStream>& streams,
    std::vector<HipDevicePointer<T> >& devicePtrs,
    Dst& targetPtr,
    const HipReductionFunction<T>* fn,
    size_t offset,
    size_t count) {
  GLOO_ENFORCE_EQ(streams.size(), devicePtrs.size());
  // Simple copy operation if there is only a single device pointer.
  if (devicePtrs.size() == 1) {
    return make_unique<
      HipLocalMemcpy<T, HipDevicePointer<T>, Dst> >(
          streams[0],
          devicePtrs[0],
          targetPtr,
          offset,
          count);
  }
  return make_unique<HipLocalHostReduce<T, Dst> >(
      streams,
      devicePtrs,
      targetPtr,
      fn,
      offset,
      count);
}

template <typename T, typename Src>
std::unique_ptr<LocalOp<T> > hipHostBroadcast(
    std::vector<HipStream>& streams,
    std::vector<HipDevicePointer<T> >& devicePtrs,
    Src& sourcePtr,
    size_t offset,
    size_t count) {
  GLOO_ENFORCE_EQ(streams.size(), devicePtrs.size());
  // Simple copy operation if there is only a single device pointer.
  if (devicePtrs.size() == 1) {
    return make_unique<
      HipLocalMemcpy<T, Src, HipDevicePointer<T> > >(
          streams[0],
          sourcePtr,
          devicePtrs[0],
          offset,
          count);
  }
  return make_unique<HipLocalHostBroadcast<T, Src> >(
      streams,
      devicePtrs,
      sourcePtr,
      offset,
      count);
}

} // namespace gloo
