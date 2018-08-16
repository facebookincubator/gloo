/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/hip_private.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
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
HipMemory<T>::HipMemory(size_t elements)
    : elements(elements),
      bytes(elements * sizeof(T)) {
  HIP_CHECK(hipGetDevice(&device_));
  // Sychronize memory allocation with RCCL operations
  std::lock_guard<std::mutex> lock(HipShared::getMutex());
  HIP_CHECK(hipMalloc(&ptr_, bytes));
}

template<typename T>
HipMemory<T>::HipMemory(HipMemory<T>&& other) noexcept
  : elements(other.elements),
    bytes(other.bytes),
    device_(other.device_),
    ptr_(other.ptr_) {
  // Nullify pointer on move source
  other.ptr_ = nullptr;
}

template<typename T>
HipMemory<T>::~HipMemory() {
  HipDeviceScope scope(device_);
  if (ptr_ != nullptr) {
    // Sychronize memory allocation with RCCL operations
    std::lock_guard<std::mutex> lock(HipShared::getMutex());
    HIP_CHECK(hipFree(ptr_));
  }
}

template<typename T>
void HipMemory<T>::set(int val, size_t stride, hipStream_t stream) {
  HipDeviceScope scope(device_);
  if (stream == kStreamNotSet) {
   hipLaunchKernelGGL( initializeMemory<T>, dim3(1), dim3(32), 0, 0, ptr_, val, elements, stride);
  } else {
   hipLaunchKernelGGL( initializeMemory<T>, dim3(1), dim3(32), 0, stream, ptr_, val, elements, stride);
  }
}

template<typename T>
std::unique_ptr<T[]> HipMemory<T>::copyToHost() const {
  HipDeviceScope scope(device_);
  auto host = make_unique<T[]>(elements);
  // Synchronize to ensure that the copy has completed.
  // The caller needs to be able to use the result immediately.
  HIP_CHECK(hipMemcpyAsync(host.get(), ptr_, bytes, hipMemcpyDefault, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  return host;
}

// Instantiate template
template class HipMemory<float>;
template class HipMemory<float16>;

// Lookup PCI bus IDs for device.
// As the number of available devices won't change at
// runtime we can seed this cache on the first call.
const std::string& getHipPCIBusID(int device) {
  static std::once_flag once;
  static std::vector<std::string> busIDs;

  std::call_once(once, [](){
    std::array<char, 16> buf;
    auto count = getDeviceCount();
    busIDs.resize(count);
    for (auto i = 0; i < count; i++) {
      HIP_CHECK(hipDeviceGetPCIBusId(buf.data(), buf.size(), i));
      busIDs[i] = buf.data();
    }
  });

  return busIDs[device];
}

} // namespace gloo
