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
#include "gloo/cuda.h"
#include "gloo/cuda_private.h"

namespace gloo {

// Below works both for CudaHostPointer and CudaDevicePointer
template <typename T, typename Dst>
class CudaLocalDeviceReduce : public LocalOp<T> {
 public:
  CudaLocalDeviceReduce(
      std::vector<CudaStream>& streams,
      std::vector<CudaDevicePointer<T> >& devicePtrs,
      Dst& targetPtr,
      const CudaReductionFunction<T>* fn,
      size_t offset,
      size_t count)
      : streams_(streams),
        targetPtr_(targetPtr.range(offset, count)),
        fn_(fn) {
    // Incorporate offset/count into devicePtrs
    devicePtrs_.reserve(devicePtrs.size());
    for (const auto& ptr : devicePtrs) {
      devicePtrs_.push_back(ptr.range(offset, count));
    }

    // Initialize reduction sequence and setup peer access
    computeIndices();
    initializePeerAccess();
  }

  void queueReduction(int indexA, int indexB) {
    auto& streamA = streams_[indexA];
    auto& streamB = streams_[indexB];

    // Record event on secondary stream
    CUDA_CHECK(cudaSetDevice(devicePtrs_[indexB].getDeviceID()));
    CUDA_CHECK(cudaEventRecord(
                   streamB.getEvent(),
                   streamB.getStream()));

    // Make primary stream wait for secondary stream.
    // This ensures any operations on the source pointer
    // have finished before we start the reduction.
    CUDA_CHECK(cudaSetDevice(devicePtrs_[indexA].getDeviceID()));
    CUDA_CHECK(cudaStreamWaitEvent(
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

  virtual void runAsync() {
    CudaDeviceGuard guard;

    // Queue reduction within groups
    for (const auto& group : indices_) {
      auto steps = log2(group.size());
      for (auto i = 0; i < steps; i++) {
        auto sz = 1 << i;
        for (auto j = 0; j < group.size(); j += sz * 2) {
          const auto indexA = group[j];
          const auto indexB = group[j + sz];
          queueReduction(indexA, indexB);
        }
      }
    }

    // Queue reduction across groups (if applicable)
    if (indices_.size() > 1) {
      queueReduction(indices_[0][0], indices_[1][0]);
    }

    // Queue copy to target on the root stream
    auto root = indices_[0][0];
    streams_[root].copyAsync(targetPtr_, devicePtrs_[root]);
  }

  virtual void wait() {
    // Wait for the final memory copy to complete
    auto root = indices_[0][0];
    streams_[root].wait();
  }

 protected:
  int findGroup(const std::vector<std::vector<int>>& groups, int device) {
    for (auto j = 0; j < groups.size(); j++) {
      auto& group = groups[j];
      if (std::find(group.begin(), group.end(), device) != group.end()) {
        return j;
      }
    }
    // Sanity check
    GLOO_ENFORCE(
        false,
        "Expected device ",
        device,
        " to be present in groupFullyConnected()");
  }

  // Ensures that the devices in group 0 have peer access to the device holding
  // the target pointer. Then we know the final memcpy can be performed using
  // peer access as well.
  template <typename Dst1 = Dst>
  void alignIndicesWithTarget(
      const std::vector<std::vector<int>>& groups,
      typename std::enable_if<
          std::is_same<Dst1, CudaDevicePointer<T>>::value>::type* = 0) {
    auto targetGroup = findGroup(groups, targetPtr_.getDeviceID());
    if (targetGroup != 0) {
      auto it = indices_.begin() + targetGroup;
      auto group = std::move(*it);
      indices_.erase(it);
      indices_.insert(indices_.begin(), std::move(group));
    }
  }

  // Ensures that the devices in group 0 are closest to the memory in
  // the target pointer. The final memcpy will then be as fast as possible.
  template <typename Dst1 = Dst>
  void alignIndicesWithTarget(
      const std::vector<std::vector<int>>& groups,
      typename std::enable_if<
          std::is_same<Dst1, CudaHostPointer<T>>::value>::type* = 0) {
    // TODO
  }

  void computeIndices() {
    // Add level of indirection so that we can shuffle this instead
    // of shuffling BOTH the streams and device pointer vectors.
    // Group indices by peer to peer access.
    const auto& groups = groupFullyConnected();
    GLOO_ENFORCE_LE(groups.size(), 2, "Expected <= 2 peer access groups");
    indices_.resize(groups.size());
    for (auto i = 0; i < devicePtrs_.size(); i++) {
      indices_[findGroup(groups, devicePtrs_[i].getDeviceID())].push_back(i);
    }
    // Remove empty groups
    for (auto it = indices_.begin(); it != indices_.end();) {
      if (it->empty()) {
        it = indices_.erase(it);
      } else {
        it++;
      }
    }
    // Best effort alignment so that the group in indices[0] is close to target
    if (indices_.size() > 1) {
      alignIndicesWithTarget(groups);
    }

    // Shuffle order in an attempt to evenly spread work across devices when
    // dealing with multiple instances of this operation.
    auto& groupA = indices_[0];
    std::random_shuffle(groupA.begin(), groupA.end());

    // Make sure that the devices associated with the first element in
    // each vector have peer to peer access, so we can reduce between groups.
    // This is possible if this host uses the canonical NVLink topology.
    if (indices_.size() > 1) {
      auto devA = devicePtrs_[groupA.front()].getDeviceID();
      auto& groupB = indices_[1];

      // Find device with peer access to devA
      auto ok = false;
      for (auto it = groupB.begin(); it != groupB.end(); it++) {
        if (canAccessPeer(devA, devicePtrs_[*it].getDeviceID())) {
          // Move this index to front of vector
          auto index = *it;
          groupB.erase(it);
          groupB.insert(groupB.begin(), index);
          // Shuffle the remainder
          std::random_shuffle(groupB.begin() + 1, groupB.end());
          ok = true;
          break;
        }
      }
      GLOO_ENFORCE(ok, "Expected to find peer access between groups");
    }
  }

  void enablePeerAccess(int indexA, int indexB) {
    auto devA = devicePtrs_[indexA].getDeviceID();
    auto devB = devicePtrs_[indexB].getDeviceID();

    // Number of elements must be equal
    GLOO_ENFORCE_EQ(
        devicePtrs_[indexA].getCount(),
        devicePtrs_[indexB].getCount());

    // Devices must be able to access each others memory
    GLOO_ENFORCE_EQ(
        1,
        canAccessPeer(devA, devB),
        "GPU ",
        devA,
        " does not have peer access to GPU ",
        devB);

    // Enable peer access for devA to memory on devB
    CudaDeviceScope scope(devA);
    cudaDeviceEnablePeerAccess(devB, 0);

    // Use cudaGetLastError so that any error is cleared.
    auto err = cudaGetLastError();
    if (err != cudaErrorPeerAccessAlreadyEnabled) {
      CUDA_CHECK(err);
    }
  }

  void initializePeerAccess() {
    // Initialize peer access within groups
    for (const auto& group : indices_) {
      int steps = log2(group.size());
      GLOO_ENFORCE(
          1 << steps,
          group.size(),
          "Number of pointers in group not a power of two");
      for (auto i = 0; i < steps; i++) {
        auto sz = 1 << i;
        for (auto j = 0; j < group.size(); j += sz * 2) {
          enablePeerAccess(group[j], group[j + sz]);
        }
      }
    }
    // Initialize peer access across groups (if applicable)
    if (indices_.size() > 1) {
      enablePeerAccess(indices_[0].front(), indices_[1].front());
    }
  }

  std::vector<CudaStream>& streams_;
  std::vector<CudaDevicePointer<T> > devicePtrs_;
  Dst targetPtr_;
  const CudaReductionFunction<T>* fn_;
  std::vector<std::vector<int>> indices_;
};

// Below works both for CudaHostPointer and CudaDevicePointer
template <typename T, typename Src>
class CudaLocalDeviceBroadcast : public LocalOp<T> {
 public:
  CudaLocalDeviceBroadcast(
      std::vector<CudaStream>& streams,
      std::vector<CudaDevicePointer<T> >& devicePtrs,
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
    CudaDeviceGuard guard;
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
        CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, devA, devB));
        GLOO_ENFORCE_EQ(
            1,
            canAccessPeer,
            "GPU ",
            devA,
            " does not have peer access to GPU ",
            devB);

        // Enable peer access for devA to memory on devB
        CUDA_CHECK(cudaSetDevice(devA));
        cudaDeviceEnablePeerAccess(devB, 0);

        // Use cudaGetLastError so that any error is cleared.
        auto err = cudaGetLastError();
        if (err != cudaErrorPeerAccessAlreadyEnabled) {
          CUDA_CHECK(err);
        }
      }
    }
  }

  virtual void runAsync() {
    CudaDeviceGuard guard;

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
        CUDA_CHECK(cudaSetDevice(
                       devicePtrs_[indexB].getDeviceID()));
        CUDA_CHECK(cudaEventRecord(
                       streamB.getEvent(),
                       streamB.getStream()));

        // Make source stream wait on target stream.
        // This ensures any operations on the target pointer
        // have finished before we start the copy.
        CUDA_CHECK(cudaSetDevice(
                       devicePtrs_[indexA].getDeviceID()));
        CUDA_CHECK(cudaStreamWaitEvent(
                       streamA.getStream(),
                       streamB.getEvent(),
                       0));

        // Execute copy and wait for it to complete on the target
        // stream. This ensures that in the next iteration of this
        // loop the target can be used as source while knowing the
        // previous copy has completed.
        CUDA_CHECK(cudaMemcpyAsync(
                       *devicePtrs_[indexB],
                       *devicePtrs_[indexA],
                       count_ * sizeof(T),
                       cudaMemcpyDeviceToDevice,
                       streamA.getStream()));
        CUDA_CHECK(cudaEventRecord(
                       streamA.getEvent(),
                       streamA.getStream()));
        CUDA_CHECK(cudaSetDevice(
                       devicePtrs_[indexB].getDeviceID()));
        CUDA_CHECK(cudaStreamWaitEvent(
                       streamB.getStream(),
                       streamA.getEvent(),
                       0));

        // Emit event on the target stream so we can wait on all
        // events in the wait() function. Otherwise waiting on
        // this event would NOT indicate completion.
        CUDA_CHECK(cudaEventRecord(
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
  std::vector<CudaStream>& streams_;
  std::vector<CudaDevicePointer<T> > devicePtrs_;
  Src sourcePtr_;
  const int count_;
  const int numPtrs_;
  const int steps_;
};

template <typename T, typename Dst>
std::unique_ptr<LocalOp<T> > cudaDeviceReduce(
    std::vector<CudaStream>& streams,
    std::vector<CudaDevicePointer<T> >& devicePtrs,
    Dst& targetPtr,
    const CudaReductionFunction<T>* fn,
    size_t offset,
    size_t count) {
  GLOO_ENFORCE_EQ(streams.size(), devicePtrs.size());

  // Simple copy operation if there is only a single device pointer.
  if (devicePtrs.size() == 1) {
    return make_unique<
      CudaLocalMemcpy<T, CudaDevicePointer<T>, Dst> >(
          streams[0],
          devicePtrs[0],
          targetPtr,
          offset,
          count);
  }

  return make_unique<CudaLocalDeviceReduce<T, Dst> >(
      streams,
      devicePtrs,
      targetPtr,
      fn,
      offset,
      count);
}

template <typename T, typename Src>
std::unique_ptr<LocalOp<T> > cudaDeviceBroadcast(
    std::vector<CudaStream>& streams,
    std::vector<CudaDevicePointer<T> >& devicePtrs,
    Src& sourcePtr,
    size_t offset,
    size_t count) {
  GLOO_ENFORCE_EQ(streams.size(), devicePtrs.size());

  // Simple copy operation if there is only a single device pointer.
  if (devicePtrs.size() == 1) {
    return make_unique<
      CudaLocalMemcpy<T, Src, CudaDevicePointer<T> > >(
          streams[0],
          sourcePtr,
          devicePtrs[0],
          offset,
          count);
  }

  return make_unique<CudaLocalDeviceBroadcast<T, Src> >(
      streams,
      devicePtrs,
      sourcePtr,
      offset,
      count);
}

} // namespace gloo
