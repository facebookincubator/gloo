/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>

#include "gloo/common/linux.h"
#include "gloo/common/logging.h"
#include "gloo/hip.h"
#include "gloo/transport/device.h"

namespace gloo {

#define HIP_CHECK(condition)                   \
  do {                                          \
    hipError_t error = condition;              \
    GLOO_ENFORCE_EQ(                            \
      error,                                    \
      hipSuccess,                              \
      "Error at: ",                             \
      __FILE__,                                 \
      ":",                                      \
      __LINE__,                                 \
      ": ",                                     \
      hipGetErrorString(error));               \
  } while (0)

inline int getCurrentGPUID() {
  int id = 0;
  HIP_CHECK(hipGetDevice(&id));
  return id;
}

inline int getGPUIDForPointer(const void* ptr) {
  hipPointerAttributes attr;
  HIP_CHECK(hipPointerGetAttributes(&attr, ptr));
  return attr.device;
}

inline int getDeviceCount() {
  int count;
  HIP_CHECK(hipGetDeviceCount(&count));
  return count;
}

const std::string& getHipPCIBusID(int device);

template<typename T>
int findHipDevicePointerClosestToDevice(
    std::vector<HipDevicePointer<T> >& ptrs,
    std::shared_ptr<transport::Device>& dev) {
  // Compute distance between every pointer
  auto devBusID = dev->getPCIBusID();
  std::vector<int> distance(ptrs.size());
  int minDistance = INT_MAX;
  int minDistanceCount = 0;
  for (auto i = 0; i < ptrs.size(); i++) {
    auto hipBusID = getHipPCIBusID(ptrs[i].getDeviceID());
    distance[i] = pciDistance(devBusID, hipBusID);
    if (distance[i] <= minDistance) {
      if (distance[i] < minDistance) {
        minDistance = distance[i];
        minDistanceCount = 0;
      }
      minDistanceCount++;
    }
  }
  // Choose random pointer closest to device;
  auto minOffset = rand() % minDistanceCount;
  int minIndex = 0;
  for (auto i = 0; i < ptrs.size(); i++) {
    if (distance[i] == minDistance) {
      if (minOffset == 0) {
        minIndex = i;
      }
      minOffset--;
    }
  }
  return minIndex;
}

class HipDeviceGuard {
 public:
  HipDeviceGuard() : previous_(getCurrentGPUID()) {
  }

  ~HipDeviceGuard() {
    HIP_CHECK(hipSetDevice(previous_));
  }

 private:
  int previous_;
};

class HipDeviceScope {
 public:
  explicit HipDeviceScope(int device) : guard_() {
    HIP_CHECK(hipSetDevice(device));
  }

 private:
  HipDeviceGuard guard_;
};

// Managed chunk of GPU memory.
// Convenience class used for tests and benchmarks.
template<typename T>
class HipMemory {
 public:
  explicit HipMemory(size_t elements);
  HipMemory(HipMemory&&) noexcept;
  ~HipMemory();

  void set(int val, size_t stride = 0, hipStream_t stream = kStreamNotSet);

  T* operator*() const {
    return ptr_;
  }

  std::unique_ptr<T[]> copyToHost() const;

  const size_t elements;
  const size_t bytes;

 protected:
  HipMemory(const HipMemory&) = delete;
  HipMemory& operator=(const HipMemory&) = delete;

  int device_;
  T* ptr_;
};

// Container class for a set of per-device streams
class HipDeviceStreams {
 public:
  HipDeviceStreams() {
    const int numDevices = getDeviceCount();
    streams_.reserve(numDevices);
    for (auto i = 0; i < numDevices; i++) {
      streams_.push_back(HipStream(i));
    }
  }
  hipStream_t operator[](const int i) {
    GLOO_ENFORCE_LT(i, streams_.size());
    return *streams_[i];
  }

 protected:
  HipDeviceStreams(const HipDeviceStreams&) = delete;
  HipDeviceStreams& operator=(const HipDeviceStreams&) = delete;

  std::vector<HipStream> streams_;
};

} // namespace gloo
