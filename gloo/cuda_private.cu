/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda_private.h"

#include <algorithm>
#include <map>

#include <cuda_fp16.h>

#include "gloo/common/common.h"
#include "gloo/types.h"

namespace gloo {

template<typename T>
__global__ void initializeMemory(
    T* ptr,
    const int val,
    const size_t count,
    const size_t stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < count; i += blockDim.x) {
    ptr[i] = (i * stride) + val;
  }
}

template<>
__global__ void initializeMemory<float16>(
    float16* ptr,
    const int val,
    const size_t count,
    const size_t stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  half* ptrAsHalf = (half*) ptr;
  for (; i < count; i += blockDim.x) {
    ptrAsHalf[i] = __float2half(static_cast<float>((i * stride) + val));
  }
}

template<typename T>
CudaMemory<T>::CudaMemory(size_t elements)
    : elements(elements),
      bytes(elements * sizeof(T)) {
  CUDA_CHECK(cudaGetDevice(&device_));
  // Sychronize memory allocation with NCCL operations
  std::lock_guard<std::mutex> lock(CudaShared::getMutex());
  CUDA_CHECK(cudaMalloc(&ptr_, bytes));
}

template<typename T>
CudaMemory<T>::CudaMemory(CudaMemory<T>&& other) noexcept
  : elements(other.elements),
    bytes(other.bytes),
    device_(other.device_),
    ptr_(other.ptr_) {
  // Nullify pointer on move source
  other.ptr_ = nullptr;
}

template<typename T>
CudaMemory<T>::~CudaMemory() {
  CudaDeviceScope scope(device_);
  if (ptr_ != nullptr) {
    // Sychronize memory allocation with NCCL operations
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    CUDA_CHECK(cudaFree(ptr_));
  }
}

template<typename T>
void CudaMemory<T>::set(int val, size_t stride, cudaStream_t stream) {
  CudaDeviceScope scope(device_);
  if (stream == kStreamNotSet) {
    initializeMemory<T><<<1, 32>>>(ptr_, val, elements, stride);
  } else {
    initializeMemory<T><<<1, 32, 0, stream>>>(ptr_, val, elements, stride);
  }
}

template<typename T>
std::unique_ptr<T[]> CudaMemory<T>::copyToHost() const {
  auto host = make_unique<T[]>(elements);
  cudaMemcpy(host.get(), ptr_, bytes, cudaMemcpyDefault);
  return host;
}

// Instantiate template
template class CudaMemory<float>;
template class CudaMemory<float16>;

// Lookup PCI bus IDs for device.
// As the number of available devices won't change at
// runtime we can seed this cache on the first call.
const std::string& getCudaPCIBusID(int device) {
  static std::once_flag once;
  static std::vector<std::string> busIDs;

  std::call_once(once, [](){
    std::array<char, 16> buf;
    auto count = getDeviceCount();
    busIDs.resize(count);
    for (auto i = 0; i < count; i++) {
      CUDA_CHECK(cudaDeviceGetPCIBusId(buf.data(), buf.size(), i));
      busIDs[i] = buf.data();
    }
  });

  return busIDs[device];
}

bool canAccessPeer(int deviceA, int deviceB) {
  static std::once_flag once;
  static std::vector<int> canAccess;
  static int count;

  std::call_once(once, [](){
    count = getDeviceCount();
    canAccess.resize(count * count);
    for (auto i = 0; i < count; i++) {
      for (auto j = 0; j < count; j++) {
        if (i == j) {
          continue;
        }

        CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess[i * count + j], i, j));

        // Verify symmetry
        if (i > j) {
          GLOO_ENFORCE_EQ(
            canAccess[i * count + j],
            canAccess[j * count + i],
            "Expecting peer access to be symmetric");
        }
      }
    }
  });

  return canAccess[deviceA * count + deviceB] == 1;
}

const std::vector<std::vector<int>>& groupFullyConnected() {
  static std::once_flag once;
  static std::vector<std::vector<int>> result;

  std::call_once(once, [](){
    // Generate grouping for all visible devices
    auto count = getDeviceCount();
    std::set<int> devices;
    for (auto i = 0; i < count; i++) {
      devices.insert(i);
    }

    // As long as there are ungrouped devices...
    while (!devices.empty()) {
      std::set<int> tmp;

      // Use any elements from devices set as start
      auto it = devices.begin();
      auto root = *it;
      tmp.insert(root);

      // Fill tmp with devices with peer access to "root"
      for (const auto& it : devices) {
        if (canAccessPeer(root, it)) {
          tmp.insert(it);
        }
      }

      // Remove devices until we have a fully connected set
      std::map<int, int> links;
      while (tmp.size() > 2) {
        links.clear();
        for (const auto& i : tmp) {
          for (const auto& j : tmp) {
            if (canAccessPeer(i, j)) {
              links[i]++;
            }
          }
        }

        // If tmp is fully connected, every links[X] == tmp.size() - 1
        auto linkPredicate = [&tmp](const decltype(links)::value_type& it) {
          return it.second == (tmp.size() - 1);
        };
        if (std::all_of(links.begin(), links.end(), linkPredicate)) {
          break;
        }

        // Remove least connected device and try again
        auto linkCompare = [](
            const decltype(links)::value_type& l,
            const decltype(links)::value_type& r) {
          return l.second < r.second;
        };
        auto min = min_element(links.begin(), links.end(), linkCompare);
        tmp.erase(min->first);
      }

      // All devices in tmp now have a home; remove them from devices set
      for (const auto& it : tmp) {
        devices.erase(it);
      }

      // Add to result
      result.push_back(std::vector<int>(tmp.begin(), tmp.end()));
    }
  });

  return result;
}

} // namespace gloo
