/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/hip_private.h"
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

#ifdef ROCM_FP16_SUPPORT
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
#endif

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

// Instantiate template
template class HipMemory<float>;
#ifdef ROCM_FP16_SUPPORT
template class HipMemory<float16>;
#endif
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
