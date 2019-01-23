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

namespace gloo {

const hipStream_t kStreamNotSet = (hipStream_t)(-1);
const int kInvalidDeviceId = -1;

// Default mutex to synchronize contentious HIP and RCCL operations
static std::mutex defaultHipMutex;
std::atomic<std::mutex*> HipShared::mutex_(&defaultHipMutex);

HipStream::HipStream(int deviceId, hipStream_t stream)
    : deviceId_(deviceId),
      stream_(stream),
      streamOwner_(false) {
  HipDeviceScope scope(deviceId_);

  // Create new stream if it wasn't specified
  if (stream_ == kStreamNotSet) {
    #ifdef ROCM_2
      int loPri, hiPri;
      //Note hipStreamCreateWithPriority() supported in rocm 2.0 + 
      HIP_CHECK(hipDeviceGetStreamPriorityRange(&loPri, &hiPri));
      HIP_CHECK(hipStreamCreateWithPriority(
                   &stream_, hipStreamNonBlocking, hiPri));
    #else               
      HIP_CHECK(hipStreamCreate(&stream_)); 
    #endif
    streamOwner_ = true;
  }

  // Create new event to synchronize operations against
  HIP_CHECK(hipEventCreateWithFlags(&event_, hipEventDisableTiming));
}

HipStream::HipStream(HipStream&& other) noexcept
    : deviceId_(other.deviceId_),
      stream_(other.stream_),
      streamOwner_(other.streamOwner_),
      event_(other.event_) {
  other.deviceId_ = kInvalidDeviceId;
  other.stream_ = nullptr;
  other.event_ = nullptr;
}

HipStream::~HipStream() {
  if (deviceId_ == kInvalidDeviceId) {
    return;
  }

  if (event_ != nullptr) {
    // Make sure outstanding operations are complete. If the event
    // hasn't been queued this call will return immediately.
    HIP_CHECK(hipEventSynchronize(event_));
    HIP_CHECK(hipEventDestroy(event_));
  }
  if (streamOwner_ && stream_ != nullptr) {
    HIP_CHECK(hipStreamDestroy(stream_));
  }
}

template <typename T>
void HipStream::copyAsync(
    HipHostPointer<T>& dst,
    HipDevicePointer<T>& src) {
  HipDeviceScope scope(deviceId_);
  GLOO_ENFORCE_LE(dst.getCount(), src.getCount());
  HIP_CHECK(hipMemcpyAsync(
                 *dst,
                 *src,
                 dst.getCount() * sizeof(T),
                 hipMemcpyDeviceToHost,
                 stream_));
  HIP_CHECK(hipEventRecord(event_, stream_));
}

template <typename T>
void HipStream::copyAsync(
    HipHostPointer<T>& dst,
    HipHostPointer<T>& src) {
  HipDeviceScope scope(deviceId_);
  GLOO_ENFORCE_LE(dst.getCount(), src.getCount());
  HIP_CHECK(hipMemcpyAsync(
                 *dst,
                 *src,
                 dst.getCount() * sizeof(T),
                 hipMemcpyHostToHost,
                 stream_));
  HIP_CHECK(hipEventRecord(event_, stream_));
}

template <typename T>
void HipStream::copyAsync(
    HipDevicePointer<T>& dst,
    HipDevicePointer<T>& src) {
  HipDeviceScope scope(deviceId_);
  GLOO_ENFORCE_LE(dst.getCount(), src.getCount());
  HIP_CHECK(hipMemcpyAsync(
                 *dst,
                 *src,
                 dst.getCount() * sizeof(T),
                 hipMemcpyDeviceToDevice,
                 stream_));
  HIP_CHECK(hipEventRecord(event_, stream_));
}

template <typename T>
void HipStream::copyAsync(
    HipDevicePointer<T>& dst,
    HipHostPointer<T>& src) {
  HipDeviceScope scope(deviceId_);
  GLOO_ENFORCE_LE(dst.getCount(), src.getCount());
  HIP_CHECK(hipMemcpyAsync(
                 *dst,
                 *src,
                 dst.getCount() * sizeof(T),
                 hipMemcpyHostToDevice,
                 stream_));
  HIP_CHECK(hipEventRecord(event_, stream_));
}

void HipStream::record() {
  HIP_CHECK(hipEventRecord(event_, stream_));
}

void HipStream::wait() {
  HipDeviceScope scope(deviceId_);
  HIP_CHECK(hipEventSynchronize(event_));
}

template <typename T>
HipDevicePointer<T> HipDevicePointer<T>::alloc(
    size_t count) {
  T* ptr = nullptr;
  size_t bytes = count * sizeof(T);
  {
    std::lock_guard<std::mutex> lock(HipShared::getMutex());
    HIP_CHECK(hipMalloc(&ptr, bytes));
  }
  auto p = create(ptr, count);
  p.owner_ = true;
  return p;
}

template<typename T>
HipDevicePointer<T> HipDevicePointer<T>::create(
    T* ptr,
    size_t count) {
  HipDevicePointer p(ptr, count, false);
  return p;
}

template<typename T>
HipDevicePointer<T>::HipDevicePointer(T* ptr, size_t count, bool owner)
    : device_(ptr),
      count_(count),
      owner_(owner),
      deviceId_(getGPUIDForPointer(device_)) {
}

template<typename T>
HipDevicePointer<T>::HipDevicePointer(HipDevicePointer<T>&& other) noexcept
    : device_(other.device_),
      count_(other.count_),
      owner_(other.owner_),
      deviceId_(other.deviceId_) {
  // Nullify fields that would otherwise be destructed
  other.device_ = nullptr;
  other.owner_ = false;
  other.deviceId_ = kInvalidDeviceId;
}

template<typename T>
HipDevicePointer<T>& HipDevicePointer<T>::operator=(
    HipDevicePointer<T>&& other) {
  device_ = other.device_;
  count_ = other.count_;
  owner_ = other.owner_;
  deviceId_ = other.deviceId_;

  // Nullify fields that would otherwise be destructed
  other.device_ = nullptr;
  other.owner_ = false;
  other.deviceId_ = kInvalidDeviceId;

  return *this;
}

template<typename T>
HipDevicePointer<T>::~HipDevicePointer() {
  if (deviceId_ == kInvalidDeviceId) {
    return;
  }
  HipDeviceScope scope(deviceId_);
  if (owner_ && device_ != nullptr) {
    std::lock_guard<std::mutex> lock(HipShared::getMutex());
    HIP_CHECK(hipFree(device_));
  }
}

template <typename T>
HipHostPointer<T> HipHostPointer<T>::alloc(size_t count) {
  T* ptr = nullptr;
  size_t bytes = count * sizeof(T);
  {
    std::lock_guard<std::mutex> lock(HipShared::getMutex());
    HIP_CHECK(hipHostMalloc(&ptr, bytes));
  }
  return HipHostPointer<T>(ptr, count, true);
}

template <typename T>
HipHostPointer<T>::HipHostPointer(T* ptr, size_t count, bool owner)
    : host_(ptr),
      count_(count),
      owner_(owner) {}

template <typename T>
HipHostPointer<T>::HipHostPointer(HipHostPointer&& other) noexcept
    : host_(other.host_),
      count_(other.count_),
      owner_(other.owner_) {
  other.host_ = nullptr;
  other.count_ = 0;
  other.owner_ = false;
}

template<typename T>
HipHostPointer<T>& HipHostPointer<T>::operator=(HipHostPointer<T>&& other) {
  host_ = other.host_;
  count_ = other.count_;
  owner_ = other.owner_;
  other.host_ = nullptr;
  other.count_ = 0;
  other.owner_ = false;
  return *this;
}

template<typename T>
HipHostPointer<T>::~HipHostPointer() {
  if (owner_) {
    std::lock_guard<std::mutex> lock(HipShared::getMutex());
    HIP_CHECK(hipHostFree(host_));
  }
}

// Instantiate templates
#define INSTANTIATE_COPY_ASYNC(T)                                       \
  template class HipDevicePointer<T>;                                  \
  template class HipHostPointer<T>;                                    \
                                                                        \
  template void HipStream::copyAsync<T>(                               \
      HipHostPointer<T>& dst,                                          \
      HipDevicePointer<T>& src);                                       \
                                                                        \
  template void HipStream::copyAsync<T>(                               \
      HipHostPointer<T>& dst,                                          \
      HipHostPointer<T>& src);                                         \
                                                                        \
  template void HipStream::copyAsync<T>(                               \
      HipDevicePointer<T>& dst,                                        \
      HipDevicePointer<T>& src);                                       \
                                                                        \
  template void HipStream::copyAsync<T>(                               \
      HipDevicePointer<T>& dst,                                        \
      HipHostPointer<T>& src);

INSTANTIATE_COPY_ASYNC(int8_t);
INSTANTIATE_COPY_ASYNC(uint8_t);
INSTANTIATE_COPY_ASYNC(int32_t);
INSTANTIATE_COPY_ASYNC(int64_t);
INSTANTIATE_COPY_ASYNC(uint64_t);
INSTANTIATE_COPY_ASYNC(float16);
INSTANTIATE_COPY_ASYNC(float);
INSTANTIATE_COPY_ASYNC(double);

// Borrowed limits from Caffe2 code (see core/hip/common_hip.h)
constexpr static int kHipNumThreads = 512;
constexpr static int kHipMaximumNumBlocks = 4096;

static inline int hipGetBlocks(const int N) {
  return std::min((N + kHipNumThreads - 1) / kHipNumThreads,
                  kHipMaximumNumBlocks);
}

#define DELEGATE_SIMPLE_HIP_BINARY_OPERATOR(T, Funcname, op)           \
  __global__                                                            \
  void _Kernel_##T##_##Funcname(T* dst, const T* src, const int n) {    \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;                 \
         i < (n);                                                       \
         i += blockDim.x * gridDim.x) {                                 \
      dst[i] = dst[i] op src[i];                                        \
    }                                                                   \
  }                                                                     \
  template <>                                                           \
  void Funcname<T>(                                                     \
    T* dst,                                                             \
    const T* src,                                                       \
    size_t n,                                                           \
    const hipStream_t stream) {                                        \
    hipLaunchKernelGGL(_Kernel_##T##_##Funcname,                                          \
      hipGetBlocks(n),                                                 \
      kHipNumThreads,                                                  \
      0,                                                                \
      stream,                                                         \
        dst, src, n);                                                   \
  }

#define DELEGATE_HALF_PRECISION_HIP_BINARY_OPERATOR(Funcname, op)             \
  __global__ void _Kernel_half_##Funcname(                                     \
      half* dst, const half* src, const int n) {                               \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);               \
         i += blockDim.x * gridDim.x) {                                        \
      float r = __half2float(dst[i]) op __half2float(src[i]);                  \
      dst[i] = __float2half(r);                                                \
    }                                                                          \
  }                                                                            \
  template <>                                                                  \
  void Funcname<float16>(                                                      \
      float16* dst,                                                            \
      const float16* src,                                                      \
      size_t n,                                                                \
      const hipStream_t stream) {                                             \
    hipLaunchKernelGGL(_Kernel_half_##Funcname, dim3(hipGetBlocks(n)), dim3(kHipNumThreads), 0, stream,  \
        (half*)dst, (half*)src, n);                                            \
  }

DELEGATE_SIMPLE_HIP_BINARY_OPERATOR(int8_t, hipSum, +);
DELEGATE_SIMPLE_HIP_BINARY_OPERATOR(int8_t, hipProduct, *);
DELEGATE_SIMPLE_HIP_BINARY_OPERATOR(uint8_t, hipSum, +);
DELEGATE_SIMPLE_HIP_BINARY_OPERATOR(uint8_t, hipProduct, *);
DELEGATE_SIMPLE_HIP_BINARY_OPERATOR(int32_t, hipSum, +);
DELEGATE_SIMPLE_HIP_BINARY_OPERATOR(int32_t, hipProduct, *);
DELEGATE_SIMPLE_HIP_BINARY_OPERATOR(int64_t, hipSum, +);
DELEGATE_SIMPLE_HIP_BINARY_OPERATOR(int64_t, hipProduct, *);
DELEGATE_SIMPLE_HIP_BINARY_OPERATOR(uint64_t, hipSum, +);
DELEGATE_SIMPLE_HIP_BINARY_OPERATOR(uint64_t, hipProduct, *);
DELEGATE_SIMPLE_HIP_BINARY_OPERATOR(float, hipSum, +);
DELEGATE_SIMPLE_HIP_BINARY_OPERATOR(float, hipProduct, *);
DELEGATE_SIMPLE_HIP_BINARY_OPERATOR(double, hipSum, +);
DELEGATE_SIMPLE_HIP_BINARY_OPERATOR(double, hipProduct, *);
DELEGATE_HALF_PRECISION_HIP_BINARY_OPERATOR(hipSum, +);
DELEGATE_HALF_PRECISION_HIP_BINARY_OPERATOR(hipProduct, *);

#define DELEGATE_SIMPLE_HIP_BINARY_COMPARE(T, Funcname, op)            \
  __global__                                                            \
  void _Kernel_##T##_##Funcname(T* dst, const T* src, const int n) {    \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;                 \
         i < (n);                                                       \
         i += blockDim.x * gridDim.x) {                                 \
      if (src[i] op dst[i]) {                                           \
        dst[i] = src[i];                                                \
      }                                                                 \
    }                                                                   \
  }                                                                     \
  template <>                                                           \
  void Funcname<T>(                                                     \
    T* dst,                                                             \
    const T* src,                                                       \
    size_t n,                                                           \
    const hipStream_t stream) {                                        \
    hipLaunchKernelGGL(_Kernel_##T##_##Funcname,                                          \
      hipGetBlocks(n),                                                 \
      kHipNumThreads,                                                  \
      0,                                                                \
      stream,                                                         \
        dst, src, n);                                                   \
  }

#define DELEGATE_HALF_PRECISION_HIP_BINARY_COMPARE(Funcname, op)              \
  __global__ void _Kernel_half_##Funcname(                                     \
      half* dst, const half* src, const int n) {                               \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);               \
         i += blockDim.x * gridDim.x) {                                        \
      if (__half2float(src[i]) op __half2float(dst[i])) {                      \
        dst[i] = src[i];                                                       \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  template <>                                                                  \
  void Funcname<float16>(                                                      \
      float16* dst,                                                            \
      const float16* src,                                                      \
      size_t n,                                                                \
      const hipStream_t stream) {                                             \
    hipLaunchKernelGGL(_Kernel_half_##Funcname, dim3(hipGetBlocks(n)), dim3(kHipNumThreads), 0, stream,  \
        (half*)dst, (half*)src, n);                                            \
  }

DELEGATE_SIMPLE_HIP_BINARY_COMPARE(int8_t, hipMin, <);
DELEGATE_SIMPLE_HIP_BINARY_COMPARE(int8_t, hipMax, >);
DELEGATE_SIMPLE_HIP_BINARY_COMPARE(uint8_t, hipMin, <);
DELEGATE_SIMPLE_HIP_BINARY_COMPARE(uint8_t, hipMax, >);
DELEGATE_SIMPLE_HIP_BINARY_COMPARE(int32_t, hipMin, <);
DELEGATE_SIMPLE_HIP_BINARY_COMPARE(int32_t, hipMax, >);
DELEGATE_SIMPLE_HIP_BINARY_COMPARE(int64_t, hipMin, <);
DELEGATE_SIMPLE_HIP_BINARY_COMPARE(int64_t, hipMax, >);
DELEGATE_SIMPLE_HIP_BINARY_COMPARE(uint64_t, hipMin, <);
DELEGATE_SIMPLE_HIP_BINARY_COMPARE(uint64_t, hipMax, >);
DELEGATE_SIMPLE_HIP_BINARY_COMPARE(float, hipMin, <);
DELEGATE_SIMPLE_HIP_BINARY_COMPARE(float, hipMax, >);
DELEGATE_SIMPLE_HIP_BINARY_COMPARE(double, hipMin, <);
DELEGATE_SIMPLE_HIP_BINARY_COMPARE(double, hipMax, >);
DELEGATE_HALF_PRECISION_HIP_BINARY_COMPARE(hipMin, <);
DELEGATE_HALF_PRECISION_HIP_BINARY_COMPARE(hipMax, >);

} // namespace gloo
